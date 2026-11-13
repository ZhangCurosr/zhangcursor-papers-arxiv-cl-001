# Do the Details Help the Whole? One-Way Transfer from Person Actions to Group Intent in VLMs on JRDB-Social

Chengguang Gan<sup>1</sup>, Zhixi Cai<sup>2</sup>, Yunhao Liang<sup>3</sup>, Hanjun Wei<sup>3</sup>,

QingHao Zhang<sup>4</sup>, Shiwen Ni<sup>5</sup>

<sup>1</sup>Independent Researcher <sup>2</sup>Monash University <sup>3</sup>University of Chinese Academy of Sciences <sup>4</sup>Pusan National University <sup>5</sup>Shenzhen University of Advanced Technology ganchengguan@yahoo.co.jp, zhixi.cai@monash.edu, liangyunhao22@mails.ucas.ac.cn, weihanjun23@mails.ucas.ac.cn, zhangqinghao@pusan.ac.kr, nishiwen@suat-sz.edu.cn

## Abstract

Social scene datasets label people at two levels: what each person is doing and what the group is there for. Training on both levels at once is often assumed to help each of them. We test this assumption for vision-language models (VLMs) on JRDB-Social, treating person actions as the details and group intent as the whole. We fine-tune Qwen3.5 4B under eight training arms with three seeds, repeat the core arms on Qwen3.5 9B and Qwen3-VL 4B, and judge every arm with a verdict rule fixed before testing. Writing each member’s actions before the intent raises intent macro-F1 from 30.1 to 31.0, and writing the intent first reaches 31.2, while action F1 stays within 0.2 points of an action-only model. Under the selected recipe, no arm trained with intent improves actions. The gain requires correct, image-grounded actions: empty action lists or actions borrowed from another group with the same intent lower intent F1. Forced-prefix tests, occlusion, member masking, linear probes and component grafting show that the intent follows the written actions, relies more on people and partly resides in the vision tower. Supplying actions as input instead of as targets hurts both levels.

## 1 Introduction

A robot that moves through a station or a campus has to read people at two levels (Mavrogiannis et al., 2023). It needs to know what each person is doing, such as standing, walking or looking at the robot, and it needs to know what a group is there for, such as waiting for a bus or heading to class. Datasets recorded from robots annotate both levels (Martin-Martin et al., 2021; Jahangard et al., 2024; Ehsanpour et al., 2022; Han et al., 2022), and models for group activity have long assumed that the person level supports the group level (Ibrahim et al., 2016; Ehsanpour et al., 2020). Several recent systems go further and describe the two levels as mutually reinforcing (Lee et al., 2024; Gan et al., 2024). In text, sentence-level and word-level labels have been reported to reinforce each other in the same way (Gan et al., 2023, 2026).

![](images/d7f2dfa04885460331d5574556b459e95dc83aa3073e4518e3883c53cb03b9fd.jpg)  
Figure 1: Three people at a bus stop seen by a mobile robot. A model trained only on group intent guesses commuting; a model trained to first write each member’s actions answers waiting.

Vision-language models (VLMs) make this question concrete. A single model can be trained to name each member’s actions, to name the group’s intent, or to write both in one answer, with every other part of the recipe held fixed. This lets us ask three questions that are hard to separate in handbuilt hierarchical models. Do the details help the whole? Does the whole help the details? And if one direction helps, what exactly in the details carries the gain?

We study these questions on JRDB-Social (Jahangard et al., 2024), where each annotated group comes with per-member action labels and a group intent label (Figure 1). We fine-tune Qwen3.5 4B with three seeds under eight training arms, two single-task references and six arms that combine, mix or condition the two levels, and repeat the core arms on Qwen3.5 9B and Qwen3-VL 4B. A verdict rule written before any test run calls an arm reinforcing only when it beats both single-task references. Around this grid we run controls for prompt wording, learning rate, label budget and the content of the action targets, we extend the comparison to a tool-using agent and to GRPO rewards (Shao et al., 2024b), and we open the models with five mechanistic analyses.

The answer is one-directional. Writing each member’s actions before the intent (J-FC) raises intent macro-F1 from 30.1 to 31.0, and writing the intent first (J-CF) reaches 31.2, while action F1 stays within 0.2 points of the action-only model. Under the selected learning rate, no arm trained with intent improves actions. The gain needs correct actions that match the image: an empty action field lowers intent F1 to 29.8, and actions borrowed from another group with the same intent lower it to 28.1. Forced prefixes show that the intent follows the actions the model writes, occlusion and member masking show that the joint model relies more on people and on each individual member, and grafting places part of the gain in the vision tower. When the same actions are supplied as input during training instead of as targets, both levels drop.

These results matter for anyone who builds social perception on top of a VLM. Asking the model to write out what each person is doing is a cheap way to improve group-level understanding, it costs at most 0.2 action points on Qwen3.5 4B, and it helps most when intent labels are scarce. The same information given as a hint in the prompt does harm. For research on multi-granularity learning, the results show that a gain in one direction can coexist with no gain in the other, so both directions need their own matched reference. Our contributions are:

• A study of cross-granularity transfer between person actions and group intent in VLMs, with eight training arms, three seeds, a preregistered verdict rule and controls for prompt, learning rate, label budget and target content.

• Evidence that the transfer runs from details to the whole only, that it needs correct and visually grounded details, and that it is largest when intent labels are scarce.

• Analyses that trace the transfer to the written actions, to a stronger reliance on people and on individual members, and partly to the vision tower, together with its limits under conditioning, a second backbone, a tool-using agent and RL rewards.

## 2 Related Work

Person and group levels in social scenes. Group activity recognition builds group labels from person features, either through temporal hierarchies (Ibrahim et al., 2016), relation graphs (Wu et al., 2019) and transformers (Gavrilyuk et al., 2020; Zhou et al., 2022), or without any person labels at all (Kim et al., 2022). Adding a person action loss raises group accuracy on volleyball and collective activity data (Ibrahim et al., 2016; Choi et al., 2009; Ehsanpour et al., 2020), while on JRDB-PAR coupling the global level to the person level lowers global F1 (Han et al., 2022). JRDB-Social (Jahangard et al., 2024) evaluates VLMs zero shot at the person, pair and group levels and finds that accuracy falls from the person to the group level. A recent VLM system injects a coarse description of an interaction as a prior for fine answers (Su et al., 2026).

Mutual reinforcement across granularities. Multi-task learning shares structure across related tasks (Caruana, 1997; Ruder, 2017), and the benefit can be uneven between task pairs (Standley et al., 2020; Fifty et al., 2021), partly because task gradients conflict (Yu et al., 2020). In multimodal models, region-level grounding and image-level answering have been reported to help each other (You et al., 2024; Chen et al., 2023; Shao et al., 2024a), though joint training can also trade one ability for the other (Zhang et al., 2024; Fan et al., 2026). Writing intermediate steps before an answer helps language and vision-language models reason (Wei et al., 2022; Zhang et al., 2023), training on such steps as targets helps small models (Hsieh et al., 2023), and editing the steps tests whether the answer depends on them (Lanham et al., 2023). The mutual reinforcement effect (MRE) states that word-level and text-level labels improve each other when learned together, first for Japanese sentence classification and entity recognition (Gan et al., 2023) and then across languages and extraction tasks (Gan et al., 2026).

Our position. Earlier studies either test one direction or compare against single-task models trained with a different recipe. We test both directions between person actions and group intent inside the same VLM, under one shared recipe, three seeds and a verdict rule fixed in advance, and we follow the result through content controls, mechanistic analyses, agents and rewards. This turns a common modelling assumption into a measured, directionspecific effect that practitioners can act on.

## 3 Task Formulation: Person Actions, Group Intent and Eight Training Arms

## 3.1 Person Actions as Details and Group Intent as the Whole

A sample s is one keyframe of one social group. The input X holds up to three frames about two seconds apart, with the n members marked by numbered red boxes, and a zoomed crop of the group in the target frame. Each member $m \in \{ 1 , \ldots , n \}$ carries a set of actions $A _ { m } \subseteq { \mathcal { A } }$ drawn from 27 action labels, and the group carries an intent set $Y \subseteq { \mathcal { C } }$ drawn from eight headline intents (commuting, eating, socializing, navigating, studying or working, discussing, waiting, leisure). Groups with $n = 1$ are single people with an intent of their own.

The model answers in JSON. We write $\tau _ { F } ( s )$ for the serialized member list $\{ ( m , A _ { m } ) \} _ { m = 1 } ^ { n }$ and $\tau _ { C } ( s )$ for the serialized intent list Y . A prompt q states which fields to produce. Every arm minimizes a token-level negative log-likelihood of the form

$$
\mathcal { L } ( \theta ) = - \mathbb { E } _ { s } \sum _ { ( q , t ) \in \mathcal { T } ( s ) } \log p _ { \theta } ( t \mid X , q ) ,\tag{1}
$$

where $\boldsymbol { \mathcal { T } } ( s )$ is the set of prompt and target pairs that the arm builds from s.

## 3.2 Eight Training Arms that Combine, Separate or Condition the Two Granularities

The arms differ only in $\boldsymbol { \mathcal { T } } ( \boldsymbol { s } )$ , as summarized in Figure 2a. The two single-task references are

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { S - F } } ( s ) = \{ ( q F , \tau _ { F } ) \} , } \end{array}\tag{2}
$$

$$
{ \mathcal { T } } _ { \mathrm { S - C } } ( s ) = \{ ( q _ { C } , \tau _ { C } ) \} .\tag{3}
$$

The joint arms put both levels into one answer, in either order:

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { J - F C } } ( s ) = \{ ( q _ { F C } , \tau _ { F } \oplus \tau _ { C } ) \} , } \end{array}
$$

$$
\mathcal { T } _ { \mathrm { J - C F } } ( s ) = \{ ( q _ { C F } , \tau _ { C } \oplus \tau _ { F } ) \} ,\tag{4}
$$

(5)

where ⊕ concatenates the two JSON fields. Under J-FC, the chain rule splits the loss into log $p ( \tau _ { F }$ X)+log $p ( \tau _ { C } \mid X , \tau _ { F } )$ , so the intent is learned and decoded after the model has written the actions; J-CF reverses this order. MIX keeps the two levels as separate samples, ${ \mathcal { T } } _ { \mathrm { M I X } } ( s ) = { \mathcal { T } } _ { \mathrm { S - F } } ( s ) \cup { \mathcal { T } } _ { \mathrm { S - C } } ( s )$

The conditioned arms place the other level in the prompt during training:

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { C O N D } } ( s ) = \{ ( { q } _ { F } \oplus { c } ( \tau _ { C } ) , \tau _ { F } ) , } \\ { ( { q } _ { C } \oplus { c } ( \tau _ { F } ) , \tau _ { C } ) \} , } \end{array}\tag{6}
$$

and are tested with the plain prompts $q _ { F }$ and $q _ { C }$ COND uses the gold labels of the same group for $c ( \cdot )$ , COND-N fills the slot with “N/A”, and COND-S copies the labels of a donor group from another location with a different intent and, when possible, the same number of members.

## 3.3 A Pre-registered Verdict Rule for Mutual Reinforcement

Let $F _ { a }$ be the action score and $C _ { a }$ the intent score of arm a (§4), averaged over seeds. For every arm other than the references we compute

$$
\Delta _ { F } ( a ) = F _ { a } - F _ { \mathrm { S - F } } , \qquad \Delta _ { C } ( a ) = C _ { a } - C _ { \mathrm { S - C } } ,\tag{7}
$$

and assign the verdict

$$
v ( a ) = \left\{ \begin{array} { l l } { { \mathrm { R E I N F O R C E S } } } & { \Delta _ { F } > 0 , \Delta _ { C } > 0 , } \\ { { \mathrm { H A R M S } } } & { \Delta _ { F } < 0 , \Delta _ { C } < 0 , } \\ { { \mathrm { T R A D E S } } } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{8}
$$

We fixed this rule, the references, the metrics and the test split before any test evaluation, and we kept a dated record of every later addition.

## 4 Experimental Setup

## 4.1 JRDB-Social, Backbones and Evaluation Metrics

Data. We use the official JRDB-Social split (Jahangard et al., 2024) of 20 training, 7 development and 27 test sequences recorded by a mobile robot in 19 test locations. Each group contributes up to six keyframes for training and up to five for evaluation, which gives 4,540 training, 1,397 development and $^ { 5 , 4 2 7 }$ test samples. The test set contains 1,136 groups, 250 with several members and 886 single people. After selecting the learning rate on the development set, every final model is trained on training and development data together.

Backbones and training. The main backbone is Qwen3.5 4B (Team, 2026), trained with seeds 42, 1 and 2. We repeat the core arms on Qwen3.5 9B (seed 42) and on Qwen3-VL 4B Instruct (Bai et al., 2025) (three seeds). All arms share one recipe: full fine-tuning for three epochs with learning rate $1 0 ^ { - 5 }$ , chosen on the development set by the summed scores of S-F and S-C, and the same batch size, schedule and image resolution. Decoding is greedy, and every test answer of every supervised arm parses as valid JSON.

![](images/bb43180a9fd95f18c035fac0a5d1d5fbadc1e877b1196ca2cdfbad094c726081.jpg)  
Figure 2: Overview of the study. (a) The training arms differ only in which levels appear as targets and in which order. (b) Controls on the joint target, the prompt, the learning rate, the intent labels and the training steps. (c) Other backbones, a tool-using agent and GRPO rewards. (d) Analyses inside the trained models.

Metrics. The action score F is the macro-F1 over the set $\boldsymbol { A } _ { 1 2 }$ of the 12 actions with at least 20 test instances,

$$
F = \frac { 1 } { \left| \mathcal { A } _ { 1 2 } \right| } \sum _ { c \in \mathcal { A } _ { 1 2 } } \frac { 2 \mathrm { T P } _ { c } } { 2 \mathrm { T P } _ { c } + \mathrm { F P } _ { c } + \mathrm { F N } _ { c } } ,\tag{9}
$$

where the counts run over all (member, action) pairs of the test samples and actions that the annotators marked as hard are ignored on both sides. The intent score C is a group-level macro-F1 over the eight headline intents. A group’s predicted intent set is the majority vote over its keyframes,

$$
\begin{array} { r } { \hat { Y } _ { g } = \Big \{ c : \sum _ { k = 1 } ^ { K _ { g } } \mathbf { 1 } [ c \in \hat { Y } _ { g , k } ] \geq K _ { g } / 2 \Big \} , } \end{array}\tag{10}
$$

where $K _ { g }$ is the number of keyframes of group g. We also report C on multi-member and singleperson groups, on five merged intent families, and on 336 groups whose intent a human or a strong VLM judged recoverable from the frames, a subset whose definition was fixed before any Qwen model was evaluated (Appendix A).

## 4.2 Controls for Prompt Wording, Learning Rate, Label Budget and Detail Content

Five controls, shown in Figure 2b, test what the joint target adds. Content controls keep the J-FC prompt and output format and change only the action part of the target: J-NFC writes an empty action list for every member, J-SFC copies the actions of a donor group with another intent, and

J-AFC copies the actions of a donor group from another location with the same intent. These arms separate the effect of correct actions from the effect of the output format and from the effect of intentconsistent action text. Prompt matching trains S-C-P, an intent-only model that receives the J-FC instruction. Learning rate repeats S-F, S-C, J-FC and COND at $2 \times 1 0 ^ { - 5 }$ . Label budget keeps intent labels for 25% or 50% of the training groups, with two independent draws per budget, while all action labels stay. Training steps trains S-F and S-C for six epochs to match the number of updates of MIX and COND.

## 5 Results: Person Actions Lift Group Intent, but Not the Reverse

## 5.1 Joint Training Lifts Group Intent while Person Actions Stay Level

Table 1 gives the main grid. Both joint arms raise intent F1 over S-C, by 0.9 points for J-FC and 1.1 points for J-CF, and keep action F1 within 0.2 points of S-F. MIX also gains on intent (+0.5) but loses 2.2 points on actions, so presenting the two levels as separate samples costs the details without adding much to the whole. Under the selected recipe no arm reaches REINFORCES: every arm that sees intent labels scores below S-F on actions. The transfer therefore runs in one direction, from the details to the whole.

On the 336 groups whose intent is judged visible in the frames, J-FC, J-CF and MIX add 1.7, 2.0 and 1.5 points over S-C. On the full test set the gain comes from single people (+0.9 and +1.1 for J-FC and J-CF), where the intent of the “group” is the intent of one person, while multi-member groups stay level; on single people J-FC pays for it with 1.5 action points (Appendix C). On five merged intent families the joint arms stay within 0.4 points of S-C, so the gain sits in the finer distinctions within a family, which §5.2 examines class by class. Qwen3.5 9B shows the same pattern with a single seed: J-FC gains 1.5 intent points and loses 1.5 action points.

<table><tr><td rowspan="2">Arm</td><td colspan="2">Person actions</td><td colspan="6">Group intent</td><td rowspan="2">Verdict</td></tr><tr><td>F1</td><td> $\Delta _ { F }$ </td><td>F1</td><td> $\Delta _ { C }$ </td><td>Multi</td><td>Single</td><td>5-class</td><td>Visible</td></tr><tr><td>Zero shot</td><td>34.8</td><td></td><td>26.5</td><td></td><td>23.6</td><td>25.9</td><td>37.5</td><td>29.1</td><td>zero shot</td></tr><tr><td>S-F</td><td>59.8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>reference</td></tr><tr><td>S-C</td><td>一</td><td></td><td>30.1</td><td></td><td>31.6</td><td>25.0</td><td>48.7</td><td>31.8</td><td>reference</td></tr><tr><td>MIX</td><td>57.6</td><td>-2.2</td><td>30.6</td><td>+0.5</td><td>32.0</td><td>25.5</td><td>48.9</td><td>33.3</td><td>TRADES</td></tr><tr><td>J-FC</td><td>59.7</td><td>-0.1</td><td>31.0</td><td>+0.9</td><td>31.6</td><td>25.9</td><td>48.3</td><td>33.4</td><td>TRADES</td></tr><tr><td>J-CF</td><td>59.6</td><td>-0.2</td><td>31.2</td><td>+1.1</td><td>31.4</td><td>26.1</td><td>48.8</td><td>33.8</td><td>TRADES</td></tr><tr><td>COND</td><td>57.0</td><td>-2.8</td><td>27.8</td><td>-2.3</td><td>28.4</td><td>24.3</td><td>45.5</td><td>29.0</td><td>HARMS</td></tr><tr><td>COND-N</td><td>58.6</td><td>-1.2</td><td>30.1</td><td>+0.0</td><td>31.7</td><td>24.5</td><td>48.6</td><td>32.4</td><td>TRADES</td></tr><tr><td>COND-S</td><td>58.3</td><td>-1.5</td><td>29.5</td><td>-0.6</td><td>30.1</td><td>25.1</td><td>47.4</td><td>32.2</td><td>HARMS</td></tr></table>

Table 1: Main results on the JRDB-Social test set with Qwen3.5-4B (macro-F1 ×100, mean of three seeds). $\Delta _ { F }$ and $\Delta _ { C }$ compare each arm with S-F and S-C. Multi and Single split the intent score by group size, 5-class merges the intent labels into five families, and Visible restricts it to the 336 groups whose intent can be judged from the frames (§4).

Three controls test whether something other than the action targets explains the gain. Prompt matching: S-C-P, which reads the J-FC instruction but answers only the intent, scores 29.6, so the instruction alone does not help and J-FC is 1.4 points above it; its counterpart S-F-P scores 60.4 on actions, the same as S-F (seed 42). Training steps: six epochs of single-task training (seed 42) lower both references (S-F 59.3, S-C 28.7), so the extra updates of MIX and COND do not explain their scores; against these step-matched references MIX still trades $( - 0 . 9 \ : / \ : + 2 . 2 )$ and COND still harms $( - 1 . 6 / - 0 . 7 )$ . Learning rate: at $2 \times 1 0 ^ { - 5 }$ (seed 42) the single-task references fall to 57.8 and 27.9 while J-FC falls only to 58.5 on actions and holds 31.2 on intent, which makes J-FC better than both references and turns its verdict into REINFORCES for that learning rate. COND also moves from HARMS to TRADES $( - 0 . 7 / + 0 . 4 )$ : at this learning rate the references degrade more than the other arms.

Label budget. When intent labels are kept for only 25% of the training groups (seed 42), S-C drops to 25.3 on average over two draws while J-FC, which still sees every action label, holds 27.9 (Figure 3). The gap is 2.6 points at 25%, 1.5 at 50% and 1.9 at full labels, and on the five merged families it grows from 0.7 to 1.0 to 2.5 as labels become scarce. J-FC never falls below S-C in any

![](images/220a911c57f9b7bb0c981fcac01800cd87a5b7c4eaf21d9bbe2a6238fb190820.jpg)

![](images/c11106e3b6fe6e04730dd51aff427b48e0c839a54e4b8015f36bba35b1e090f8.jpg)  
Figure 3: Intent macro-F1 when intent labels are kept for 25%, 50% or all training groups (seed 42; two draws per budget, averaged). Action labels are always complete.

![](images/6199b9acab79a2972096258c44147502fec3b22e59eb226eef727a5afcafc8af.jpg)

![](images/ffaed7e82d05347993024cb87d55ead0634b7a62925a0336ed954195cf823c1c.jpg)  
Figure 4: Intent macro-F1 (three seeds) when the action part of the joint target is empty (J-nFC), taken from another group with the same intent (J-aFC), taken from another group with a different intent (J-sFC), or correct (J-FC).

of the five comparisons.

## 5.2 Only Correct and Visually Grounded Details Carry the Gain

The content controls in Figure 4 keep the J-FC instruction and output format and change only what the action field contains. Writing an empty action list for every member gives 29.8, slightly below S-C. Copying the actions of another group with the same intent gives 28.1, the lowest of the content controls. Only the correct actions of the group in the image reach 31.0. The gain of J-FC thus

decomposes as

$$
\begin{array} { r } { \underbrace { C _ { \mathrm { J - F C } } - C _ { \mathrm { S - C } } } _ { + 0 . 9 } = \underbrace { C _ { \mathrm { J - F C } } - C _ { \mathrm { J - N F C } } } _ { + 1 . 2 \mathrm { c o n t e n t } } } \\ { + \underbrace { C _ { \mathrm { J - N F C } } - C _ { \mathrm { S - C } } } _ { - 0 . 3 \mathrm { f o r m a t } } , } \end{array}\tag{11}
$$

so the output format contributes nothing and the action content carries the whole gain. The comparison with J-AFC matters most: those targets are consistent with the intent label and differ from the correct ones only in not describing the people in the image, and they cost 2.9 points against J-FC. What helps the whole is a description of what these particular people are doing, not a list of actions that usually go with the intent. Actions from a group with a different intent (J-SFC) end in between at 30.4. In every subset, correct actions score above empty actions and above same-intent donor actions (Appendix Table 10).

Which intents gain. The gain falls on a few small intents (Figure 5). Per class, J-FC raises navigating from 6.1 to 9.9, waiting from 11.3 to 14.3 and discussing from 0.0 to 6.8, while commuting moves from 80.4 to 80.6 and eating rises by 1.5 points. Of the macro average, discussing adds 0.85 points, navigating 0.48 and waiting 0.37, while socializing takes away 0.78. S-C predicts on average 6.3 distinct intents over the test set against 7.0 for J-FC, so single-task training collapses part of the rare intents and action supervision keeps them alive. Socializing loses recall (Appendix D).

## 5.3 Details as Input, a Second Backbone, Tool-Using Agents and RL Rewards

Table 2 moves the comparison away from the main setting. Details as input. The COND arm gives the model the gold actions in its training prompt and tests it without them, much like learning with privileged information (Vapnik and Vashist, 2009). It lowers both levels on all three backbones, by 2.8 and 2.3 points on Qwen3.5 4B, and the harm follows the quality of the hint: gold labels cost 2.3 intent points, a donor group’s labels 0.6 and “N/A” nothing. A model that learns to read the answer from its prompt stops looking for it in the image, a shortcut in the sense of Geirhos et al. (2020).

A second backbone. On Qwen3-VL 4B, MIX and J-CF still raise intent over S-C (by 0.5 and 0.05 points) while losing actions, so the direction of the transfer holds, but its size shrinks and J-FC falls on both levels.

<table><tr><td>Setting</td><td>Arm</td><td> $\Delta _ { F }$ </td><td> $\Delta { _ { C } }$ </td><td>Verdict</td></tr><tr><td>Qwen3-VL 4B</td><td>MIX</td><td>-2.0</td><td>+0.5</td><td>TRADES</td></tr><tr><td></td><td>J-FC</td><td>-0.8</td><td>-1.2</td><td>HARMS</td></tr><tr><td></td><td>J-CF</td><td>-1.7</td><td>+0.0</td><td>TRADES</td></tr><tr><td></td><td>COND</td><td>-4.2</td><td>-1.1</td><td>HARMS</td></tr><tr><td>Qwen3.5 9B†</td><td>MIX</td><td>-3.5</td><td>+1.0</td><td>TRADES</td></tr><tr><td></td><td>J-FC</td><td>-1.5</td><td>+1.5</td><td>TRADES</td></tr><tr><td></td><td>COND</td><td>-5.4</td><td>-2.7</td><td>HARMS</td></tr><tr><td>Agent (4B)</td><td>AG-FC</td><td>-0.3</td><td>-0.5</td><td>HARMS</td></tr><tr><td></td><td>AG-CF</td><td>-1.6</td><td>-2.1</td><td>HARMS</td></tr><tr><td>GRPO (4B)‡</td><td> $R _ { C }$ </td><td>-0.1</td><td>+0.8</td><td></td></tr><tr><td></td><td> $R _ { F }$ </td><td>-1.1</td><td>+0.0</td><td>一</td></tr><tr><td></td><td> $R _ { C } + { \textstyle { \frac { 1 } { 2 } } } R _ { F }$ </td><td>-1.0</td><td>+0.1</td><td>一</td></tr></table>

Table 2: The transfer outside the main setting (×100). Deltas use the single-task arms of the same setting as references; the agent rows compare with AG-F and AG-C. <sup>†</sup>One seed. <sup>‡</sup>Two seeds, deltas relative to the J-FC starting point.

Tool-using agents. We give the 4B model an offline harness with zoom and frame tools in the style of ReAct (Yao et al., 2022) and train four agent arms on scripted trajectories that differ only in the notes they write (Appendix E). The tools alone already lift intent: the intent-only agent AG-C reaches 30.9 against 30.1 for S-C. The joint agent arms then lose to the single-task agents on both levels over all groups, while on single people AG-FC still gains 1.0 intent points. A likely reason is that once the agent zooms into the members, the image evidence that J-FC writes down is already in its context.

RL rewards. GRPO started from J-FC raises intent by 0.8 points with an intent reward and by 0.1 when half an action reward is added, and an action reward alone lowers actions by 1.1 points. When the supervised model is trained without the development split and GRPO runs on development prompts (seed 42), every reward with an intent term lowers intent, by 1.8 to 3.3 points, although the rewards rise on the RL prompts. The transfer comes from the supervised targets, and an action reward adds nothing to intent (Appendix F), in line with Yue et al. (2026).

## 6 Analysis: How Group Intent Draws on Person Actions

## 6.1 Group Intent Follows the Actions the Model Writes

A joint model writes one level before the other, which lets us intervene on the first field and read the second, a mediation test (Vig et al., 2020) of the kind used to check whether written reasoning drives an answer (Lanham et al., 2023; Turpin et al., 2023). For a prefix variant $v \in$ {correct, wrong, empty} we force the first field, let the model complete the answer, and measure

![](images/5c146e9e8b11e240868efefa3a1b691a060ae6005436310b389c06467c044e0d.jpg)  
Figure 5: Per-class intent F1 of S-C and J-FC, averaged over three seeds. Leisure is 0 for both and is not shown.

![](images/39006a78ffa989d83bae4ba0d9aac3769b13746ff0eafec7c8443b09987ac752.jpg)  
Figure 6: Forced prefixes. Blue: the model’s action field is replaced by the correct actions, another group’s actions or empty lists, and the intent is decoded. Orange: the intent field is replaced and the actions are decoded. Direct generation uses J-FC and J-CF (three seeds); the agent uses AG-FC and AG-CF with all earlier notes rewritten (seed 42).

$$
\delta _ { v } = S \big ( \mathrm { a n s w e r } \mid v \big ) - S \big ( \mathrm { o w n \ a n s w e r } \big ) ,\tag{12}
$$

where S is the intent score when actions are forced in J-FC and the action score when the intent is forced in J-CF. Wrong prefixes come from a donor group with a different intent, mapped onto the member ids. Forcing the model’s own answer as the prefix changes the scores by at most 0.2 points, so the procedure itself is neutral.

Figure 6 shows a clear asymmetry. With actions forced, correct actions raise intent F1 by 2.3 points, wrong actions lower it by 3.2 and empty actions lower it by 10.1; two seeds lose about 4 points without actions, and the third loses 22.6 because it then answers commuting for almost every sample. With the intent forced, the correct intent raises action F1 by 1.3 and a wrong intent lowers it by 3.3, but an empty intent changes it by only 0.5. The intent depends on the written actions, while the actions do not need the intent and only suffer when it is wrong.

The two directions also differ in how far a perfect prefix can carry the other level (seed 42). With correct actions written, J-FC reaches 33.9 intent F1, 2.9 points above its own answers and close to the 35.1 that a linear classifier reaches from the gold actions and the group size (Appendix A). With the correct intent written, J-CF reaches 60.4 action F1, the same as S-F: even a perfect whole adds nothing to the details. The agent behaves the same way once all earlier notes are rewritten as well: empty actions cost the intent 2.3 points and an empty intent costs the actions 0.5.

<table><tr><td>Measure</td><td>S-C</td><td>J-FC</td></tr><tr><td>Person share ρ of the occlusion loss</td><td>0.40</td><td>0.46</td></tr><tr><td>Intent change, one member hidden</td><td>0.163</td><td>0.194</td></tr><tr><td>Marked member hit rate</td><td>0.545</td><td>0.555</td></tr><tr><td>chance level</td><td>0.471</td><td>0.477</td></tr><tr><td>Action probe gain (S-F: 2.7)</td><td>0.5</td><td>2.6</td></tr><tr><td>Intent probe gain (S-F: 1.1)</td><td>2.0</td><td>1.8</td></tr></table>

Table 3: How the two models use people and members. ρ is a three-seed mean; the masking and probe rows use seed 42. The hit rate is the share of the human-marked member among the members whose removal deletes a correct intent. Probe gains are balanced accuracy ×100 over the zero-shot model at the best layer, averaged over eight targets.

The decoding order is only part of the story. J-CF writes the intent first, cannot read its own actions when it does so, and still gains 1.1 points, so training on the actions also changes how the model sees the scene. The next two analyses look at that change.

## 6.2 Joint Models Rely More on People and on Each Member

People or place. We blank either the members (keeping the background) or everything around them (keeping the members) at test time and measure the intent loss in each case. The person share of the loss is

$$
\rho = \frac { C - C _ { \mathrm { n o } \mathrm { p e o p l e } } } { \left( C - C _ { \mathrm { n o } \mathrm { p e o p l e } } \right) + \left( C - C _ { \mathrm { n o } \mathrm { p l a c e } } \right) } .\tag{13}
$$

Averaged over three seeds, ρ rises from 0.40 for S-C to 0.46 for J-FC, 0.50 for MIX, 0.51 for

J-CF and 0.59 for COND (Table 3). The zeroshot model sits at 0.42, so intent supervision alone leaves the balance between people and place where it was, and every arm that learns from actions moves it toward the people. A blurred fill gives the same order (Appendix Table 17).

Which members. We then hide one member at a time in multi-member samples (1,926 member and sample pairs, seed 42) and record how much the predicted intent set changes,

$$
\Delta _ { k } = 1 - \mathrm { F } 1 \big ( \hat { Y } , \hat { Y } ^ { ( - k ) } \big ) ,\tag{14}
$$

where $\hat { Y } ^ { ( - k ) }$ is the prediction with member k hidden. Hiding a single member changes the intent of J-FC by 0.194 on average and that of S-C by 0.163. A human reviewer marked, for each correct intent, the member who shows it most clearly and rated the others as strong or weak evidence. Among the members whose removal deletes a correct intent, the marked member is found 55.5% of the time for J-FC and 54.5% for S-C, against chance levels of 47.7% and 47.1%, and hiding strong members moves J-FC more than hiding weak ones (0.233 against 0.197). Joint training makes the intent depend more on each member while keeping the same preference for the members a person would point to.

## 6.3 The Transfer Appears in Hidden States and in the Vision Tower

Linear probes. We feed 3,000 test samples with the same J-FC prompt to every model, take the last prompt token at each of the 33 layers, and train logistic probes (Alain and Bengio, 2016) for eight intent targets and eight action targets with folds split by sequence. For arm a and target set $\mathcal { G }$ we report the gain over the zero-shot model $R _ { 0 }$ at the best layer,

$$
g _ { a } ( \mathcal { G } ) = \frac { 1 } { \vert \mathcal { G } \vert } \sum _ { t \in \mathcal { G } } \bigl ( \mathrm { B A } _ { a } ^ { \star } ( t ) - \mathrm { B A } _ { R _ { 0 } } ^ { \star } ( t ) \bigr ) ,\tag{15}
$$

where $\mathrm { B A } ^ { \star } ( t )$ is the balanced accuracy of target t at the best of the 33 layers. On average over the eight targets, all models are far above a locationonly probe and above label permutations (Hewitt and Liang, 2019) (Appendix G). Action supervision adds 2.7 points of action information, while intent supervision adds 0.5 (Table 3). Both kinds of supervision add intent information, by 1.1 points for S-F and 2.0 for S-C, and S-F reaches 3.4 points when probed with its own prompt. Learning the details makes the hidden state richer in both levels; learning the whole leaves the details where the zero-shot model had them.

Component grafting. Fine-tuned skills can often be localized in a small part of the weights (Panigrahi et al., 2023; Ilharco et al., 2022). We copy one group of weights from S-C into J-FC, or the reverse, and evaluate the result. The groups are the 24 linear-attention layers, the 8 softmax-attention layers, the MLPs, the vision tower and the vision merger. Each graft has three random controls that perturb the same tensors by the same norm in a random direction. Moving the vision tower of S-C into J-FC lowers intent F1 by 1.05 points, while all three random controls raise it slightly; moving the vision tower of J-FC into S-C raises intent F1 by 1.13 points, more than seven times the largest random effect (Appendix Figure 8). Measured as the share of the gap between the two models that a graft moves,

$$
m = \frac { C _ { \mathrm { g r a f t } } - C _ { \mathrm { b a s e } } } { C _ { \mathrm { d o n o r } } - C _ { \mathrm { b a s e } } } ,\tag{16}
$$

the vision tower moves 56% of the gap in one direction and 60% in the other. No other component moves intent F1 toward the donor in both directions, and waiting is the only class that follows the vision tower both ways (from 9.7 to 14.3 and from 15.5 to 10.6). Part of the advantage of J-FC is stored in how it encodes the image, which fits the content controls: the actions help only when they describe what the image shows.

## 7 Conclusion

Trained on JRDB-Social under one recipe and a verdict rule fixed in advance, Qwen3.5 4B models that write each member’s actions together with the group intent gain on intent and keep their action score: the details help the whole, while the whole does not help the details. The gain needs correct actions of the people in the image, the intent reads the actions the model writes, and part of the advantage sits in the vision tower; the same actions given as a training hint lower both levels. For social perception with VLMs this suggests a simple recipe: ask the model what each person is doing before asking what the group is doing, and keep person labels as targets rather than inputs.

## Limitations

Our study covers one dataset, JRDB-Social, recorded on one university campus, and two model families of 4B and 9B parameters. The 9B model was trained with one seed, and the grafting, probing and member masking analyses use seed 42 only. Group intent in JRDB-Social is hard to judge from a few frames: human agreement with the gold labels reaches a kappa of 0.33, so absolute intent scores are low for every model and the gains we report are about one point on the full test set. Split by test location, the gain of J-FC sits in the 58 groups recorded at a training location on another day; it is level with S-C on sequences from the same day and location as training data, and 1.0 point below S-C on the 427 groups from unseen locations (Appendix D). The agent and GRPO experiments use one harness and one reward design each.

## References

Guillaume Alain and Yoshua Bengio. 2016. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631.

Rich Caruana. 1997. Multitask learning. Machine learning, 28(1):41–75.

Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. 2023. Shikra: Unleashing multimodal LLM’s referential dialogue magic. arXiv preprint arXiv:2306.15195.

Wongun Choi, Khuram Shahid, and Silvio Savarese. 2009. What are they doing?: Collective activity classification using spatio-temporal relationship among people. In 2009 IEEE 12th international conference on computer vision workshops, ICCV Workshops, pages 1282–1289. IEEE.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and psychological measurement, 20(1):37–46.

Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith. 2020. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. arXiv preprint arXiv:2002.06305.

Mahsa Ehsanpour, Alireza Abedin, Fatemeh Saleh, Javen Shi, Ian Reid, and Hamid Rezatofighi. 2020. Joint learning of social groups, individuals action and sub-group activities in videos. In European Conference on Computer Vision, pages 177–195. Springer.

Mahsa Ehsanpour, Fatemeh Saleh, Silvio Savarese, Ian Reid, and Hamid Rezatofighi. 2022. JRDB-Act: A

large-scale dataset for spatio-temporal action, social group and activity detection. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20951–20960. IEEE.

Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Xinze Guan, and Xin Wang. 2026. GRIT: Teaching MLLMs to think with images. Advances in Neural Information Processing Systems, 38:116522–116543.

Chris Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. 2021. Efficiently identifying task groupings for multi-task learning. Advances in Neural Information Processing Systems, 34:27503– 27516.

Chengguang Gan, Sunbowen Lee, Qingyu Yin, Yunhao Liang, Xinyang He, Hanjun Wei, Younghun Lim, Shijian Wang, Hexiang Huang, Qinghao Zhang, et al. 2026. A multilingual dataset and empirical validation for the mutual reinforcement effect in information extraction. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 1810–1829.

Chengguang Gan, Qinghao Zhang, and Tatsunori Mori. 2023. Sentence-to-label generation framework for multi-task learning of Japanese sentence classification and named entity recognition. In International Conference on Applications of Natural Language to Information Systems, pages 257–270. Springer.

Wenqing Gan, Yan Sun, Feiran Liu, and Xiangfeng Luo. 2024. MPT-PAR: Mix-parameters transformer for panoramic activity recognition. arXiv preprint arXiv:2408.00420.

Kirill Gavrilyuk, Ryan Sanford, Mehrsan Javan, and Cees GM Snoek. 2020. Actor-transformers for group activity recognition. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 836–845. IEEE.

Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. 2020. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. 2025. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948.

Ruize Han, Haomin Yan, Jiacheng Li, Songmiao Wang, Wei Feng, and Song Wang. 2022. Panoramic human activity recognition. In European Conference on Computer Vision, pages 244–261. Springer.

John Hewitt and Percy Liang. 2019. Designing and interpreting probes with control tasks. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (emnlp-ijcnlp), pages 2733–2743.

Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. 2023. Distilling step-by-step! Outperforming larger language models with less training data and smaller model sizes. In Findings of the association for computational linguistics: ACL 2023, pages 8003–8017.

Mostafa S Ibrahim, Srikanth Muralidharan, Zhiwei Deng, Arash Vahdat, and Greg Mori. 2016. A hierarchical deep temporal model for group activity recognition. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 1971–1980.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2022. Editing models with task arithmetic. arXiv preprint arXiv:2212.04089.

Simindokht Jahangard, Zhixi Cai, Shiki Wen, and Hamid Rezatofighi. 2024. JRDB-Social: A multifaceted robotic dataset for understanding of context and dynamics of human interactions within social groups. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22087–22097. IEEE.

Dongkeun Kim, Jinsung Lee, Minsu Cho, and Suha Kwak. 2022. Detector-free weakly supervised group activity recognition. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20051–20061. IEEE.

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, et al. 2023. Measuring faithfulness in chain-of-thought reasoning. arXiv preprint arXiv:2307.13702.

Sumin Lee, Yooseung Wang, Sangmin Woo, and Changick Kim. 2024. Spatio-temporal proximity-aware dual-path model for panoramic activity recognition. In European Conference on Computer Vision, pages 19–36. Springer.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems, 36:34892– 34916.

Roberto Martin-Martin, Mihir Patel, Hamid Rezatofighi, Abhijeet Shenoi, JunYoung Gwak, Eric Frankel, Amir Sadeghian, and Silvio Savarese. 2021. JRDB: A dataset and benchmark of egocentric robot visual perception of humans in built environments. IEEE transactions on pattern analysis and machine intelligence, 45(6):6748–6765.

Christoforos Mavrogiannis, Francesca Baldini, Allan Wang, Dapeng Zhao, Pete Trautman, Aaron Steinfeld, and Jean Oh. 2023. Core challenges of social robot navigation: A survey. ACM Transactions on Human-Robot Interaction, 12(3):1–39.

Abhishek Panigrahi, Nikunj Saunshi, Haoyu Zhao, and Sanjeev Arora. 2023. Task-specific skill localization in fine-tuned language models. In International Conference on Machine Learning, pages 27011–27033. PMLR.

Sebastian Ruder. 2017. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098.

Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. 2024a. Visual CoT: Advancing multimodal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing Systems, 37:8612–8642.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024b. DeepSeek-Math: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Carlos N Silla Jr and Alex A Freitas. 2011. A survey of hierarchical classification across different application domains. Data mining and knowledge discovery, 22(1):31–72.

Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. 2020. Which tasks should be learned together in multi-task learning? In International conference on machine learning, pages 9120–9132. PMLR.

Yuejiao Su, Xinshen Zhang, Zhen Ye, Lei Yao, Lap-Pui Chau, and Yi Wang. 2026. EARL: Towards a unified analysis-guided reinforcement learning framework for egocentric interaction reasoning and pixel grounding. arXiv preprint arXiv:2605.14742.

Qwen Team. 2026. Qwen3.5-Omni technical report. arXiv preprint arXiv:2604.15804.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. 2023. Language models don’t always say what they think: Unfaithful explanations in chain-ofthought prompting. Advances in Neural Information Processing Systems, 36:74952–74965.

Vladimir Vapnik and Akshay Vashist. 2009. A new learning paradigm: Learning using privileged information. Neural networks, 22(5-6):544–557.

Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. 2020. Investigating gender bias in language models using causal mediation analysis. Advances in neural information processing systems, 33:12388– 12401.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837.

Jianchao Wu, Limin Wang, Li Wang, Jie Guo, and Gangshan Wu. 2019. Learning actor relation graphs for group activity recognition. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9956–9966. IEEE.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. ReAct: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. 2024. Ferret: Refer and ground anything anywhere at any granularity. In International Conference on Learning Representations, volume 2024, pages 57153–57180.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. 2020. Gradient surgery for multi-task learning. Advances in neural information processing systems, 33:5824– 5836.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. 2026. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? Advances in Neural Information Processing Systems, 38:57654–57689.

Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Leizhang, Chunyuan Li, et al. 2024. LLaVA-Grounding: Grounded visual chat with large multimodal models. In European Conference on Computer Vision, pages 19–35. Springer.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. 2023. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923.

Ziwei Zheng, Minghao Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, and Chao Shen. 2026. DeepEyes: Incentivizing “thinking with images” via reinforcement learning. In International Conference on Learning Representations, volume 2026, pages 126775–126798.

Honglu Zhou, Asim Kadav, Aviv Shamsian, Shijie Geng, Farley Lai, Long Zhao, Ting Liu, Mubbasir Kapadia, and Hans Peter Graf. 2022. COMPOSER: Compositional reasoning of group activity in videos with keypoint-only modality. In European Conference on Computer Vision, pages 249–266. Springer.

## A Data, Labels and Evaluation

Rendering. JRDB provides a stitched 3760×480 panorama per timestep. For each group we cut a 1,128 × 480 window centred on its members; a group that spans more than 1,028 pixels is cut with a 100 pixel margin on each side and scaled to that size. We draw a numbered red box around every member and add a zoomed crop of the group. Multi-frame samples hold the frames about two seconds before and after the target frame, so the model sees up to four images. Member ids follow the left-to-right order in the target frame, and keyframes in which a group has more than eight members are skipped.

Labels. JRDB-Social gives every member action a difficulty from 1 to 3. We count actions of difficulty 1 or 2 as positives and treat difficulty 3 as don’t care: such an action is removed from both the gold and the predicted set of that member. The 12 actions with at least 20 test positives are walking, standing, sitting, holding something, listening to someone, talking to someone, looking at the robot, looking into something, cycling, going upstairs, bending and interacting with a door. The model may answer with any of the 27 actions and nine intents; the ninth intent, attending class, has two test groups and is left out of the macro average. The five merged families are movement (commuting, navigating, leisure), social (socializing, discussing), task (studying or working, attending class), eating and waiting. Table 4 lists how many test groups carry each intent.

Human review and the visual subset. One annotator reviewed all renderable multi-member test groups (251 by the group size in the labels) and 200 single-person test groups sampled by class, 451 in total. For a blind subset the annotator first named the intent without seeing the gold label or any model output; 89 of these items have a blind answer. On them, agreement between the annotator and the gold label is κ = 0.33 (Cohen, 1960), and agreement between the annotator and Claude Opus 5.5 given the same frames is κ = 0.54 (Table 4). The three intents where J-FC gains most (navigating, discussing, waiting) have a human agreement with the gold label of at most 0.11. Where the blind answer or the Opus answer differed from the gold label, the annotator also judged the gold intent as correct, plausible alongside another answer, wrong, or not judgeable. The visual subset contains the 336 groups (205 multi-member, 131 single) where the annotator found the gold intent correct or plausible, or where Opus named a gold intent and the annotator did not mark the gold label as wrong or not judgeable. Its definition was fixed before any Qwen model was evaluated. The subset is 61% multi-member, against 22% on the full test set.

<table><tr><td></td><td colspan="2">Test groups</td><td colspan="2">Oracle F1</td><td colspan="3">Cohen&#x27;s κ on blind items</td></tr><tr><td>Intent</td><td>all</td><td>multi</td><td>actions</td><td>actions + location</td><td>human vs. gold</td><td>human vs. Opus</td><td>Opus vs. gold</td></tr><tr><td>commuting</td><td>624</td><td>90</td><td>83.6</td><td>79.4</td><td>0.61</td><td>0.71</td><td>0.45</td></tr><tr><td>eating</td><td>211</td><td>76</td><td>47.0</td><td>53.6</td><td>0.66</td><td>0.69</td><td>0.63</td></tr><tr><td>socializing</td><td>147</td><td>95</td><td>47.9</td><td>47.9</td><td>0.46</td><td>0.50</td><td>0.36</td></tr><tr><td>navigating</td><td>100</td><td>16</td><td>16.6</td><td>27.2</td><td>0.00</td><td>0.00</td><td>-0.04</td></tr><tr><td>studying/working</td><td>98</td><td>36</td><td>38.4</td><td>43.6</td><td>0.69</td><td>0.61</td><td>0.53</td></tr><tr><td>discussing</td><td>24</td><td>21</td><td>13.8</td><td>12.1</td><td>-0.03</td><td>0.79</td><td>0.22</td></tr><tr><td>waiting</td><td>79</td><td>16</td><td>25.4</td><td>31.9</td><td>0.11</td><td>0.46</td><td>0.16</td></tr><tr><td>leisure</td><td>33</td><td>10</td><td>8.3</td><td>12.3</td><td>0.11</td><td>0.56</td><td>0.19</td></tr><tr><td>macro</td><td>1,136</td><td>250</td><td>35.1</td><td>38.5</td><td>0.33</td><td>0.54</td><td>0.31</td></tr></table>

Table 4: The eight headline intents. Oracle F1: a logistic regression on the gold actions and the group size, alone or with the location. κ: 89 blind items (59 multi-member, 30 single); the human never used the label navigating on them, so its κ of 0 is degenerate.

Label priors and oracles. Predicting the most frequent intent for every group gives 8.9 intent F1 (6.6 on multi-member groups, 9.4 on single people, 16.0 on the five families), and predicting the most frequent intent of each training location gives 15.9 (16.8, 15.1 and 26.8); ten of the 19 test locations do not occur in training. The same two priors reach 4.8 and 8.2 action F1. The zero-shot Qwen3.5 4B reaches 26.5 intent F1. To see how much of the intent is readable from the actions, we fit one class-balanced logistic regression per intent (Silla Jr and Freitas, 2011) on the training and development groups, using the share of members with each gold action and the group size. This oracle reaches 35.1 (35.1 on multi-member groups, 27.7 on single people), and adding the location raises it to 38.5 (43.2 and 29.3). Only 17% of the test groups get their full intent set right from gold actions, and 36% with the location added, so actions constrain the intent without fixing it.

## B Prompts and Training Recipe

## All arms share one system prompt:

You analyse social groups in frames from a   
mobile robot's 360-degree camera. The people   
marked with numbered red boxes form one social   
group; unmarked people are not in the group.   
Member actions (a member can have several):   
<27 actions>.   
Group aims (a group can have several):   
<9 intents>.   
Answer with JSON only.

The user turn starts with a sentence on the images and the number of members, followed by one of four requests:

F: For every member, list their actions in   
the target frame. Output:   
{"members": [{"id": 1, "actions":   
["..."]}, ...]}   
C: What is the group's aim? Output:

J-FC: For every member, list their actions in   
the target frame, then give the group's   
aim. Output: {"member $\ s ^ { \prime \prime } \colon \ [ \ldots ]$   
"aim": ["..."]}

J-CF: Give the group's aim, then for every   
member list their actions in the target   
frame. Output: {"aim": ["..."],   
"members": [...]}

The conditioned arms append one sentence during training, “The group’s aim is: . . . ” to the F request and “Member actions in the target frame: 1: . . . ; 2: . . . ” to the C request. S-C-P and S-F-P use the J-FC request with an intent-only or action-only target. For the content controls, the donor of J-SFC and COND-S is a group from another location with a different intent and, when possible, the same number of members; the donor of J-AFC shares the intent instead. Donor members are mapped onto the ids 1, . . . , n cyclically, and donors are drawn deterministically. In 14.7% of the J-SFC training samples the donor actions happen to equal the group’s own gold actions.

All supervised runs follow visual instruction tuning (Liu et al., 2023) with full fine-tuning in bf16 and the loss on the answer tokens only, three epochs, a cosine schedule with 3% warmup and an effective batch of 8. We chose the learning rate from $\{ 1 0 ^ { - 5 } , 2 \times 1 0 ^ { - 5 } \}$ by training S-F and S-C on the training split and scoring them on the development split: $1 0 ^ { - 5 }$ gives 64.1 action F1 and 41.8 intent F1, against 59.7 and 40.6 for $2 \times 1 0 ^ { - 5 }$ . Answers are decoded greedily with at most 400 new tokens, and only the first JSON object in an answer is parsed. The answers of the zero-shot model average about 83 tokens, a short reasoning followed by the JSON, and 5,409 of its 5,427 intent answers parse.

<table><tr><td rowspan="2">Arm</td><td colspan="2">Action F</td><td colspan="3">Intent C</td></tr><tr><td>42</td><td>1 2</td><td>42</td><td>1</td><td>2</td></tr><tr><td>S-F S-C</td><td>60.4 59.1</td><td>59.9</td><td>29.1</td><td>30.2</td><td>31.0</td></tr><tr><td>MIX J-FC</td><td>58.3 56.8 59.5 58.5</td><td>57.6 61.1</td><td>30.9 31.0</td><td>29.4 30.5</td><td>31.4 31.5</td></tr><tr><td>J-CF</td><td>59.4 58.6</td><td>60.8</td><td>31.8 28.1</td><td>30.5</td><td>31.5 27.5</td></tr><tr><td>COND COND-N COND-S</td><td>57.7 57.2 58.3 58.5 58.7 57.9</td><td>56.1 59.1</td><td>30.8 29.5</td><td>27.7 30.3</td><td>29.3 30.1</td></tr><tr><td></td><td></td><td>58.4</td><td></td><td>29.0</td><td></td></tr><tr><td>S-C-P</td><td>一</td><td></td><td>29.4</td><td>29.5</td><td>29.9</td></tr><tr><td>J-nFC</td><td></td><td></td><td>29.8</td><td>29.2</td><td>30.4</td></tr><tr><td>J-aFC</td><td>一</td><td>一</td><td>26.9</td><td>29.4</td><td>28.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>J-sFC</td><td>一</td><td>一 一</td><td>29.8</td><td>30.8</td><td>30.7</td></tr></table>

Table 5: Scores (×100) of each Qwen3.5 4B arm for seeds 42, 1 and 2. The content controls write donor or empty actions by design, so their action scores are not meaningful (4.4 to 4.7 for J-SFC, below the majority prior).

## C Per-Seed Scores, Strata and Other Backbones

Fine-tuning results move with the random seed (Dodge et al., 2020), so Table 5 lists the action and intent scores of every Qwen3.5 4B arm for each seed, and Table 6 splits the three-seed means by group size. The verdicts in the main text use the means of these columns.

Seeds and subsets. Per seed, J-FC beats S-C on intent by 1.9, 0.3 and 0.5 points on all groups, by 2.0, −0.8 and −1.4 on multi-member groups and by 0.2, 1.1 and 1.2 on single people. On the visual subset S-C scores 30.5, 31.1 and 33.7 and J-FC adds 3.1, 1.0 and 0.8. On the five families the visual-subset gains are 0.9 for J-FC, 1.1 for J-CF and 1.5 for MIX, and on the eight intents without discussing 0.7, 1.2 and 0.9: about 60% of the J-FC gain on the subset comes from discussing (0 to 8.1), since 22 of the 24 discussing groups fall into it. Inside the subset, single people gain 2.4, 2.6 and 2.0 points, for J-FC mostly on navigating (5.8 to 13.2) and waiting (10.7 to 20.5), and multi-member groups gain $0 . 0 , - 0 . 1$ and 0.7. On the 800 groups outside the subset the three arms stay within 0.6 points of S-C.

Controls by stratum. J-FC is 1.6, 1.0 and 1.6 points above the prompt-matched S-C-P for the three seeds, 0.4 on multi-member groups and 1.1 on single people. $\mathrm { A t } 2 \times 1 0 ^ { - 5 }$ (seed 42) J-FC is $. 0 . 6 \ I + 2 . 7$ on multi-member groups and +1.2 $I + 2 . 8$ on single people against the references of the same learning rate. The six-epoch S-C (seed 42) reaches 46.2 on the five families, 1.3 below its

three-epoch score.

Other backbones. Table 7 gives the absolute scores behind the boundary rows of Table 2. The 9B model starts lower on intent than the 4B model (17.8 against 26.5 zero shot) and gains more from fine-tuning (8.7 points for S-C against 3.6). On 9B, J-FC is $- 0 . 1 / + 1 . 0$ on multi-member groups and −1.2 / +0.7 on single people. On Qwen3-VL, MIX is $- 2 . 3 / - 0 . 2$ on multi-member groups and $- 1 . 4 / + 1 . 2$ on single people, $\mathrm { J - F C } - 1 . 7 / - 2 . 1$ and $- 1 . 3 / + 0 . 5 , \mathrm { J } \mathrm { - C F - 1 . 9 / + 0 . 2 }$ and −2.4 / −0.5, and $\mathrm { C O N D - 3 . 7 / - 1 . 8 }$ and $- 3 . 2 / + 0 . 1 $

## D Intent Classes, Content Controls, Label Budget and Test Locations

Per class. Table 8 extends Figure 5 to the other arms. S-C never predicts discussing correctly in any seed. J-FC recovers about seven fewer socializing groups per seed, and of the 441 socializing group and seed pairs only 2 are predicted as discussing, so the drop is a loss of socializing recall rather than a shift between the two social classes. Over three seeds J-FC finds 3 of the 72 discussing group and seed pairs and predicts discussing for 11 pairs outside the social family, 9 of them single people; J-SFC finds 4. Leaving discussing out of the macro average gives 34.4 for S-C and 34.5 for J-FC; leaving out discussing and socializing gives 35.2 and 36.3 (J-CF 34.7 and 36.2, MIX 34.2 and 35.8). The per-class gain of J-FC is barely related to how much room the action oracle leaves above S-C (rank correlation 0.10 over the eight classes): socializing has the most room (18 points) and loses most.

Collapse to frequent intents. S-C predicts 7, 6 and 6 distinct intents in the three seeds, J-FC 7 in each, J-CF 8, 7 and 7, and COND-S 7, 9 and 8. The entropy of the predicted intent distribution is 1.54, 1.46 and 1.58 bits for S-C, 1.60, 1.63 and 1.57 for J-FC, and 1.39 to 1.40 for COND, which answers commuting for 738 test groups against 642 for S-C (seed 42). Every arm over-predicts eating (288 to 382 predicted groups against 211 gold, seed 42) and none recalls leisure.

Where the cue lies. Opus also saw each test group in the full view, with only the people and with only the place. A gold intent enters a stratum only when Opus ranks it first (ties allowed) in the full view; it then counts as recoverable from a partial view when Opus also ranks it first there and does not answer that it cannot tell. All other gold intents form the last row. Table 9 gives the recall per stratum. The gain of the joint arms does not sit on the intents that only people reveal: J-FC recalls 3.3 points fewer of them than S-C, and 5.7 fewer on multi-member groups.

<table><tr><td rowspan="2">Arm</td><td colspan="2">Action F</td><td colspan="2">Intent C</td><td colspan="3">Multi-member groups</td><td colspan="3">Single people</td></tr><tr><td>multi</td><td>single</td><td>multi</td><td>single</td><td> $\Delta _ { F }$ </td><td> $\Delta _ { C }$ </td><td>verdict</td><td> $\Delta _ { F }$ </td><td> $\Delta _ { C }$ </td><td>verdict</td></tr><tr><td>S-F</td><td>56.8</td><td>52.1</td><td></td><td></td><td></td><td></td><td>reference</td><td></td><td></td><td>reference</td></tr><tr><td>S-C</td><td></td><td></td><td>31.6</td><td>25.0</td><td></td><td></td><td>reference</td><td></td><td></td><td>reference</td></tr><tr><td>MIX</td><td>54.2</td><td>50.6</td><td>32.0</td><td>25.5</td><td>-2.5</td><td>+0.4</td><td>TRADES</td><td>-1.5</td><td>+0.5</td><td>TRADES</td></tr><tr><td>J-FC</td><td>56.9</td><td>50.6</td><td>31.6</td><td>25.9</td><td>+0.2</td><td>-0.1</td><td>TRADES</td><td>-1.5</td><td>+0.9</td><td>TRADES</td></tr><tr><td>J-CF</td><td>56.0</td><td>52.0</td><td>31.4</td><td>26.1</td><td>-0.8</td><td>-0.2</td><td>HARMS</td><td>-0.1</td><td>+1.1</td><td>TRADES</td></tr><tr><td>COND</td><td>54.4</td><td>48.9</td><td>28.4</td><td>24.3</td><td>-2.4</td><td>-3.2</td><td>HARMS</td><td>-3.2</td><td>-0.8</td><td>HARMS</td></tr><tr><td>COND-N</td><td>54.8</td><td>51.2</td><td>31.7</td><td>24.5</td><td>-2.0</td><td>+0.1</td><td>TRADES</td><td>-0.9</td><td>-0.5</td><td>HARMS</td></tr><tr><td>COND-S</td><td>55.4</td><td>49.7</td><td>30.1</td><td>25.1</td><td>-1.4</td><td>-1.5</td><td>HARMS</td><td>-2.4</td><td>+0.1</td><td>TRADES</td></tr></table>

Table 6: Three-seed means of Qwen3.5 4B split by group size, with the verdict rule applied inside each stratum (250 multimember groups, 886 single people).

<table><tr><td>Backbone</td><td>Arm</td><td>F</td><td>C</td><td> $\Delta _ { F }$ </td><td> $\Delta _ { C }$ </td></tr><tr><td>Qwen3.5 9B (seed 42)</td><td>zero shot S-F S-C MIX J-FC</td><td>36.1 57.6 54.0 56.0</td><td>17.8 26.5 27.4 28.0</td><td> $- 3 . 5 \quad + 1 . 0 \quad$ </td><td> $- 1 . 5 \ \textrm { } + 1 . 5$ </td></tr><tr><td>Qwen3-VL 4B (3 seeds)</td><td>COND zero shot S-F S-C MIX J-FC J-CF COND 51.4</td><td>52.2 34.5 55.6 53.6 54.8 53.9 28.8</td><td>23.8 24.4 28.7 29.2 27.6</td><td>-2.0 -0.8</td><td>-5.4-2.7 +0.5 -1.2</td></tr></table>

Table 7: Absolute scores on the other two backbones. Deltas are computed before rounding.

<table><tr><td>Arm</td><td>com. eat.</td><td>SOC. nav.</td><td>stu.</td><td>dis. wai.</td><td></td></tr><tr><td>S-C J-FC</td><td>80.4 64.9 80.6 66.4</td><td>29.8 6.1 23.6</td><td>48.3 9.9 46.3</td><td>0.01</td><td>11.3 6.814.3</td></tr><tr><td>J-CF MIX</td><td>80.1 65.3 81.1 66.5</td><td>25.8 24.6</td><td>8.848.4 4.9 48.9</td><td></td><td>7.014.6 5.013.5</td></tr><tr><td>J-sFC</td><td>64.6</td><td>15.0</td><td>3.6 47.0 16.6 38.4 13.8 25.4</td><td></td><td></td></tr><tr><td>COND</td><td>77.3 63.0 79.6</td><td>24.9</td><td>7.6 48.9</td><td>9.5 2.7</td><td>12.2 9.6</td></tr></table>

Table 8: Per-class intent F1, three-seed means. Leisure is 0 for every arm. The last row is the action oracle of Table 4.

Content controls. Table 10 repeats the content controls of Figure 4 on every subset. J-SFC reaches 9.5 on discussing, the highest of all arms, so wrong actions also teach the model to use that label; without discussing, J-SFC is 1.0 point below S-C. On the visual subset J-SFC equals J-FC.

<table><tr><td>Opus recovers it from</td><td>Pairs S-C</td><td>J-FC</td><td>J-CF</td><td>MIX</td></tr><tr><td>people only</td><td>101</td><td>55.8 -3.3</td><td>-0.3</td><td>+1.0</td></tr><tr><td>people and place</td><td>338</td><td>95.7 -1.5</td><td>-0.9</td><td>+0.3</td></tr><tr><td>place only</td><td>247</td><td>89.3 +1.2</td><td>+0.4</td><td>+0.9</td></tr><tr><td>neither</td><td>89</td><td>55.1 -1.1</td><td>-4.5</td><td>-3.0</td></tr><tr><td>not even the full view</td><td>543</td><td>23.4</td><td>-0.1</td><td>+0.1 +0.1</td></tr></table>

Table 9: Recall of gold (group, intent) pairs, split by the views from which Opus ranks that intent first, three-seed means. Arm columns are differences from S-C.
<table><tr><td>Arm</td><td>Actions</td><td>All Multi Single 5-cl. Vis.</td></tr><tr><td>S-C</td><td>none</td><td>30.1 31.6 25.0 48.7 31.8</td></tr><tr><td>J-nFC</td><td>empty lists</td><td>29.8 30.9 24.4 46.9 31.3</td></tr><tr><td>J-aFC</td><td>donor, same intent</td><td>28.1 30.0 23.6 45.2 30.4</td></tr><tr><td>J-sFC</td><td>donor, other intent</td><td>30.4 31.5 25.0 47.733.4</td></tr><tr><td>J-FC</td><td>correct</td><td>31.0 31.6 25.9 48.3 33.4</td></tr></table>

Table 10: Intent macro-F1 (×100, three seeds) when the action part of the joint target is emptied or replaced by the actions of a donor group, mostly from another location. All arms share the J-FC instruction and output format except S-C.

Label budget. All label budget runs use seed 42. Per draw, S-C scores 27.0 and 23.5 at 25% and 29.1 and 26.0 at 50%, while J-FC scores 27.1 and 28.6, then 30.0 and 28.1. With few intent labels S-C depends strongly on which groups were drawn, and J-FC, which still learns from the actions of every group, varies much less; in the second 25% draw S-C falls to 23.1 on multi-member groups, against 30.0 for J-FC. Without discussing, the gap is 1.6, 1.7 and 1.1 points at 25%, 50% and 100%. The action score of J-FC stays at 60.4 and 60.1 at 25% and 59.1 at 50%, close to S-F (60.4).

Test locations. Table 11 splits the test set by how close each sequence is to the training data. The gain of J-FC sits in the 58 groups recorded at a training location on another day. Unseen locations give higher intent scores than seen ones for most arms, probably because commuting, the easiest class, is more frequent there.

<table><tr><td>Arm</td><td>same day other day (651) (58)</td><td>unseen (427)</td></tr><tr><td>S-C</td><td>28.4 23.0</td><td>31.5</td></tr><tr><td>MIX</td><td>27.8</td><td>31.7</td></tr><tr><td>J-FC</td><td>28.6 28.2 33.2</td><td>30.5</td></tr><tr><td>J-CF</td><td>28.9 30.4</td><td>30.6</td></tr><tr><td>COND</td><td>25.7 23.9</td><td>28.3</td></tr><tr><td>AG-C</td><td>27.2 30.1</td><td>30.9</td></tr><tr><td>AG-FC</td><td>29.2 26.5</td><td>29.9</td></tr><tr><td>S-F (action)</td><td>52.9</td><td>54.4</td></tr><tr><td>J-FC (action)</td><td>58.5 53.0 58.2</td><td>54.0</td></tr></table>

Table 11: Intent F1 by test location (three seeds; number of groups in brackets): same day and location as a training sequence, a training location on another day, or an unseen location. The last two rows give action F1.

## E Tool-Using Agent

Harness. The agent receives the target window and the member list, and may call up to six tools before giving its final JSON: zoom(member), pair(member\_a, member\_b), frames() for the frames two seconds before and after, look\_around(direction) and overview() for the full panorama. The system prompt is the one above with one added sentence asking for a short sentence before each call. Unlike agents that learn when to look closer through RL (Zheng et al., 2026), ours follows a fixed calling plan, so the arms differ only in what they write.

Trajectories. Training trajectories are written by a script and use only the geometry: the agent first calls frames(), then zooms into up to four members in order of increasing box height, then answers. The four arms share these calls and images and differ only in the text: AG-F and AG-FC write a note “Member k: . . . ” with the gold actions after each zoom, AG-C writes only the short sentence before each call, and AG-CF additionally states a first impression of the intent before its first call. The final answers follow the F, C, J-FC and J-CF formats. Training uses the supervised recipe of Appendix B with the loss on assistant turns only.

Call order. A first version of the trajectories called frames() after the zooms, so every member note was written before the agent had seen the motion. With that order (seed 42) AG-FC reaches only 56.3 action F1, against 58.9 with the final order and 59.5 for J-FC; by member box height its action F1 is 56.9, 53.7 and 64.3 for the small, middle and large tercile, against 56.1, 62.9 and 69.8 with the final order and 56.3, 62.4 and 69.7 for J-FC. Against J-FC, the largest per-action losses of the first version are cycling (−10.4), talking (−7.5) and listening (−6.8), actions that need motion or a second person. Its verdicts were −0.2 / −0.9 for AG-FC and +0.5 / −4.3 for AG-CF.

<table><tr><td>Arm</td><td>F</td><td>C</td><td></td><td></td><td>multi single 5-class</td></tr><tr><td>AG-F</td><td>60.0</td><td></td><td></td><td></td><td></td></tr><tr><td>AG-C</td><td></td><td>30.9</td><td>32.7</td><td>24.6</td><td>48.3</td></tr><tr><td>AG-FC</td><td>59.7</td><td>30.4</td><td>31.0</td><td>25.6</td><td>49.4</td></tr><tr><td>AG-CF</td><td>58.4</td><td>28.8</td><td>30.3</td><td>23.4</td><td>44.8</td></tr></table>

Table 12: Agent arms on Qwen3.5 4B, three seeds. The last three columns split the intent score.

Results. Table 12 gives the agent scores. On multi-member groups AG-FC is −0.6 / −1.7 and AG-CF −0.2 / −2.5 against the single-task agents; on single people they are −0.8 / +1.0 and −2.5 / −1.3. AG-CF loses most, probably because its first impression is written before any evidence is seen; its intent change per seed is −4.9, −2.3 and +0.8. The tools help intent by themselves: for seed 42, AG-C reaches 31.4 against 29.1 for S-C, and AG-F 60.8 against 60.4 for S-F. Without training the tools do not help the 4B model: the zero-shot agent scores 33.8 / 20.5 against 34.8 / 26.5 for zeroshot direct answers, calls 2.46 tools per episode, mostly look\_around, and exceeds the budget in 109 episodes. The zero-shot 9B agent scores 36.3 / 22.3 against 36.1 / 17.8.

Teacher trajectories. A teacher run (seed 42) uses trajectories from Claude Opus 5.5, which ran the same harness on 3,210 training and development episodes. We keep the 1,617 episodes whose intent set matches the gold, whose member actions reach a set F1 of at least 0.5 and that stay within the tool budget (1,432 single, 185 multimember). Trained on these episodes alone, the student reaches 51.0 and 24.5, against 50.5 and 24.0 for scripted trajectories on the same episodes, while calling 1.81 tools per episode instead of 2.20 and using pair 767 times. Mixing the teacher episodes into the full scripted set gives 59.2 and 31.9, against 58.9 and 30.8 without them, with +1.8 intent on single people and −0.9 on multi-member groups; the mixed student keeps the scripted tool use.

Observation swaps. We replace the image returned by every zoom call while keeping the tool text (seed 42). With another member of the same group, the intent of AG-FC changes by 0.049 on multi-member groups and that of AG-C by 0.023; with a grey image by 0.109 and 0.066; with a member of another group by 0.284 and 0.231. The intent F1 of AG-FC drops from 30.8 to 29.5 with grey images, 30.6 with another member and 25.5 with another group, and its action F1 from 58.9 to 52.3, 58.4 and 43.3. Occluding people or places in every image the agent can request gives a person share $\rho$ of 0.49 for AG-FC and 0.50 for AG-C; on multimember groups hiding the people costs about 11 points and hiding the place about 6.

<table><tr><td>Direction</td><td></td><td>Replaced Correct</td><td>Wrong</td><td>Empty</td></tr><tr><td>actions → intent final turn</td><td></td><td>+1.7</td><td>-3.1</td><td>-1.9</td></tr><tr><td rowspan="3">intent → actions final turn</td><td>all turns</td><td>+1.9</td><td>-2.4</td><td>-2.3</td></tr><tr><td></td><td>+1.8</td><td>-3.4</td><td>-1.9</td></tr><tr><td>all turns</td><td>+1.4</td><td>-2.5</td><td>-0.5</td></tr></table>

Table 13: Forced prefixes in the agent (seed 42), relative to the forced own answer. With only the final turn replaced, the first impression of AG-CF still states an intent, and an empty intent then costs the actions 1.9 points.

Forced prefixes in the agent. Table 13 compares two versions of the agent prefix test. In the first only the final answer is replaced; in the second every earlier note is replaced as well, so the forced content is the only version of that level in the context. With only the final turn replaced, multi-member groups already show the asymmetry of direct generation $( + 1 . 2 / - 2 . 0 / - 1$ .8 from actions to intent, +0.9 / $- 0 . 3 / + 0 . 3$ from intent to actions). Dropping the note of the final turn costs 1.2 and 1.8 points by itself, so these numbers are relative to the forced own answer.

## F GRPO Training

Objective. Starting from J-FC (seed 42), we sample $G = 8$ answers $o _ { 1 } , \ldots , o _ { G }$ per prompt at temperature 1, score each with a reward $r _ { i }$ and use the group-centred advantage $\begin{array} { r } { \hat { A } _ { i } = r _ { i } - \frac { 1 } { G } \sum _ { j } r _ { j } } \end{array}$ (Shao et al., 2024b; Guo et al., 2025). Groups in which all rewards are equal are dropped. For a chunk of prompts with sampled answers O the loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G R P O } } = - \displaystyle \frac { 1 } { N } \sum _ { o _ { i } \in \mathcal { O } } \sum _ { t } \Bigl [ \hat { A } _ { i } \log \pi _ { \theta } ( o _ { i , t } \mid h _ { i , t } ) } \\ { - \beta \operatorname { K L } _ { i , t } \Bigr ] , \qquad } \end{array}\tag{17}
$$

where N is the number of generated tokens in the chunk, $h _ { i , t } = ( X , q , o _ { i , < t } )$ is the context of token $t , \mathrm { K L } _ { i , t }$ is the k3 estimate of the divergence from the frozen J-FC model and $\beta = 0 . 0 2$ . Answers that do not end within 400 tokens are left out of the loss. Each chunk of 16 prompts gives one update, so the importance ratio is 1. We use 1,024 prompts for two rounds, learning rate $2 \times 1 0 ^ { - 6 }$ with 5% warmup and a cosine schedule.

<table><tr><td rowspan="2">Arm</td><td colspan="2">train+dev prompts dev prompts</td><td colspan="2"></td></tr><tr><td> $\Delta _ { F }$ </td><td> $\Delta _ { C }$ </td><td> $\Delta _ { F }$ </td><td> $\Delta _ { C }$ </td></tr><tr><td>RC</td><td>-0.1</td><td>+0.8</td><td> $+ 0 . 1 \quad - 1 . 8 \quad$ </td><td></td></tr><tr><td>RF</td><td>-1.1</td><td>0.0</td><td> $- 2 . 5 \quad + 0 . 1 \hphantom { 0 }$ </td><td></td></tr><tr><td>RFC</td><td>-1.0</td><td>+0.1</td><td> $- 2 . 0 \quad - 3 . 3 \quad$ </td><td></td></tr><tr><td>RFC-gated</td><td>-1.0</td><td>-0.4</td><td> $+ 0 . 3 \quad - 2 . 7 \quad$ </td><td></td></tr></table>

Table 14: GRPO changes relative to the starting model. Left: start from J-FC trained on train and development data, RL on the same prompts (two seeds). Right: start from J-FC trained on the training split only (57.4 / 29.6), RL on development prompts (seed 42).
<table><tr><td>Run</td><td> $R _ { C }$   $R _ { F }$ </td><td>varying</td></tr><tr><td>RC, seen</td><td> $0 . 8 6 7  0 . 8 8 9$ </td><td> $0 . 4 3  0 . 3 0$ </td></tr><tr><td>RF, seen</td><td></td><td> $0 . 8 7 1  0 . 9 1 6$   $0 . 5 2  0 . 3 9$ </td></tr><tr><td>RFC, seen</td><td> $0 . 8 7 1  0 . 8 8 4$ </td><td> $0 . 8 7 4  0 . 9 0 5$   $0 . 6 9  0 . 5 9$ </td></tr><tr><td>RC, dev</td><td> $0 . 6 8 4  0 . 7 4 4$ </td><td> $0 . 5 3  0 . 3 6$ </td></tr><tr><td>RF, dev</td><td></td><td> $0 . 7 1 7  0 . 7 6 5$   $0 . 7 0  0 . 4 8$ </td></tr><tr><td>RFC, dev</td><td> $0 . 6 7 7  0 . 7 3 8$ </td><td> $0 . 7 1 5  0 . 7 5 4$   $0 . 8 2  0 . 6 4$ </td></tr><tr><td>RC, agent</td><td> $0 . 8 3 0  0 . 8 6 1$ </td><td> $0 . 4 7  0 . 2 8$ </td></tr><tr><td>RFC, agent</td><td> $0 . 8 3 9  0 . 8 6 3$ </td><td> $0 . 8 2 7  0 . 8 8 6$   $0 . 7 0  0 . 4 1$ </td></tr></table>

Table 15: Training rewards, first 16 steps → last 16 steps (seed 42), and the share of prompt groups whose samples differ in reward. “Seen”: RL prompts from the supervised training data; “dev”: unseen development prompts; “agent”: GRPO from AG-FC.

Rewards. Let $R _ { C }$ be the set F1 between predicted and gold intents and $R _ { F }$ the mean set F1 over members (difficulty-3 actions are ignored). With a format term $r _ { \mathrm { f m t } } \in \{ 0 , 1 \}$ for a parsable answer that holds both fields, the arms are

$$
r _ { \mathrm { R C } } = R _ { C } + r _ { \mathrm { f m t } } , \qquad r _ { \mathrm { R F } } = R _ { F } + r _ { \mathrm { f m t } } ,\tag{18}
$$

$$
r _ { \mathrm { R F C } } = R _ { C } + 0 . 5 R _ { F } + r _ { \mathrm { f m t } } ,\tag{19}
$$

and RFC-GATED adds 0.5 $R _ { F }$ only when $R _ { C } \geq$ 0.5.

Room for RL. Before training we checked that the rewards still vary. On 256 training prompts the greedy J-FC answers already score $R _ { C } = 0 . 8 8 5$ and $R _ { F } = 0 . 9 3 5$ , best-of-8 sampling reaches 0.970 and 0.972, and the eight samples differ in reward for 52% and 61% of the prompts, enough for groupcentred advantages.

Results. Table 14 shows that RF and RFC lower action F1 in both settings, and that an intent reward helps only when the RL prompts were already seen in supervised training. On seen prompts the intent gain of RC is 1.3 points for seed 42 and 0.3 for seed 1, the five-family deltas are +0.9, +0.4, +0.6 and +0.6 for the four arms, and RF and RFC lose 1.5 and 1.6 action points on multi-member groups. With unseen development prompts the five-family deltas are −4.4, −1.4, −5.1 and −4.0. Table 15 shows that the rewarded quantity rises in every run while the share of prompts with varying rewards falls, and the KL per token stays at or below 0.01. On held-out development groups the reward falls instead (0.82 to 0.78 for RC, 0.80 to 0.77 for RF), and so do the test scores. The number of actions per member moves only from 1.23 to 1.19 under RF (seed 42), so the reward is not gamed by empty answers.

GRPO on the agent. For the agent (seed 42), GRPO from AG-FC with 768 prompts gives +0.4 / −0.8 for RC, +0.4 / −0.3 for RFC and −0.8 /

−0.5 for a variant that adds 0.1 to a correct intent when the agent called a tool. The starting agent calls a tool in every episode, so this variant only rewards correct intents more. The tool calls per episode stay at 2.19.

Mechanisms after GRPO. GRPO leaves the mechanisms of §6 where they were (seed 42). The person share ρ of the four direct arms is 0.48, 0.41, 0.47 and 0.46 against 0.46 at the start, and 0.55, 0.52 and 0.49 for the agent arms against 0.49. Hiding one member changes their intent by 0.192 to 0.201, with marked-member hit rates of 0.527 to 0.546, and swapping a zoom image for another group’s member changes the agent intent on multimember groups by 0.269 to 0.292.

## G Mechanism Details

Occlusion. The “no people” view covers each member box grown by 25% (wider for cyclists and people pushing carts); the “no place” view keeps each box grown by 5% and covers the rest. Both are filled with grey; a blurred fill is the robustness variant. Table 17 lists ρ per seed. For seed 42, S-C scores 29.1 on the full view, 20.4 with only the people and 25.5 with only the place; COND scores 28.1, 22.5 and 14.6. The masks act on actions as expected: S-F keeps 49.4 action F1 with only the people and drops to 24.0 with only the place.

Forced prefixes per seed. Table 18 lists the prefix effects of direct generation per seed. The empty prefix of seed 2 makes J-FC answer commuting for

5,403 of the 5,427 samples. For seed 42, J-FC with gold actions written reaches 34.4 on multi-member groups, 27.4 on single people and 51.6 on the five families.

Member masking. Table 16 adds the remaining arms. “Others” is the change in the predicted actions of the members that stay visible. The singletask S-F shows this coupling as well (0.216), so it does not come from writing both levels in one answer. COND reacts most strongly to a single member but chooses members at chance level and reacts more to weak than to strong members. On the 738 pairs whose member box overlaps little with the others, hiding a member changes the intent of J-FC by 0.228 and that of S-C by 0.168, with hit rates of 0.607 and 0.631.

Probes. We sample 3,000 test samples once, reduce each layer’s hidden state to 256 dimensions with PCA and fit an L2 logistic regression per target in five folds split by sequence. The intent targets are the eight headline intents; the action targets are the eight most frequent actions, each true when some member performs it. The zero-shot model already reaches 66.1 balanced accuracy on intents and 81.8 on actions at its best layer, well above a probe on the location alone (56.7 and 57.0) and above a permutation baseline that swaps the labels of whole groups of the same size (95th percentile over 50 permutations of the best-layer score: 56.2 and 56.7). Two targets stay at the location level: on leisure no model beats the location probe (65.1), and on navigating the zero-shot model and COND do not (61.3). Table 19 gives the gains over the zero-shot model, and Figure 7 the curves over layers: on intent the trained models pull away from the zero-shot model mainly in the upper half of the network, while on actions S-C stays close to it at every depth.

Grafting. The five component groups together hold 657 tensors; embeddings and layer norms stay with the base model, and each grafted model is tested with the base model’s prompt. Figure 8 shows every graft with its random controls. Measured against the mean of the random controls instead of the base model, the vision tower moves intent F1 by −1.37 and +1.05 points. Grafting the MLPs of S-C into J-FC breaks the action field: the model keeps the joint format but writes intent labels as member actions, action F1 falls from 59.5 to 6.3 and intent F1 by 4.6, so this graft cannot locate the gain. The language-side attention layers of J-FC lower the intent score of S-C when moved alone (−1.1 for linear attention, −0.5 for softmax attention). Grafting the vision tower of S-C also lowers the action F1 of J-FC by 1.6 points, and the vision tower of J-FC raises the intent of S-C by 1.7 points on multi-member groups and lowers it by 0.6 on single people. The gap between J-FC and S-C that grafting works on is 1.9 points for seed 42, the largest of the three seeds (0.3 and 0.5 for the others).

<table><tr><td>Arm</td><td>Intent change</td><td>Gold-intent drop</td><td>Others</td><td>Pairs</td><td>Marked</td><td>Chance</td><td>Strong</td><td>Weak</td></tr><tr><td>S-F</td><td></td><td></td><td>0.216</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>S-C</td><td>0.163</td><td>0.056</td><td></td><td>162</td><td>0.545</td><td>0.471</td><td>0.190</td><td>0.159</td></tr><tr><td>J-FC</td><td>0.194</td><td>0.067</td><td>0.188</td><td>179</td><td>0.555</td><td>0.477</td><td>0.233</td><td>0.197</td></tr><tr><td>J-CF</td><td>0.177</td><td>0.050</td><td>0.229</td><td>151</td><td>0.535</td><td>0.478</td><td>0.202</td><td>0.159</td></tr><tr><td>MIX</td><td>0.174</td><td>0.054</td><td>一</td><td>170</td><td>0.567</td><td>0.480</td><td>0.186</td><td>0.171</td></tr><tr><td>COND</td><td>0.246</td><td>0.103</td><td>一</td><td>212</td><td>0.470</td><td>0.469</td><td>0.232</td><td>0.258</td></tr><tr><td>AG-F</td><td></td><td></td><td>0.234</td><td></td><td></td><td></td><td></td><td>一</td></tr><tr><td>AG-C</td><td>0.161</td><td>0.048</td><td></td><td>160</td><td>0.509</td><td>0.471</td><td>0.179</td><td>0.155</td></tr><tr><td>AG-FC</td><td>0.214</td><td>0.061</td><td>0.243</td><td>203</td><td>0.471</td><td>0.475</td><td>0.212</td><td>0.212</td></tr></table>

Table 16: Hiding one member (seed 42, 1,926 member and sample pairs). Intent change: change of the predicted intent set. Gold-intent drop: loss of set F1 against the gold intent. Pairs: (group, intent) pairs with a marked member that enter the hit rate. Marked: share of the marked member among the members whose removal deletes a correct intent. Strong and weak: intent change when a member rated as strong or weak evidence is hidden.

![](images/738ad43bfc7c9b25254d3a5b698a309eade7250cc12f4779625e1ed65f86b96a.jpg)

![](images/27b830855b7e8c1bf5f50c2cfe3e35fd4f568bf657920c24abbbceadc9e8cedc.jpg)  
Figure 7: Linear probe balanced accuracy per layer, averaged over eight intent targets (left) and eight action targets (right). Seed 42; every model reads the J-FC prompt.

![](images/a06714d8c1e417f53a10d17c117cc8b9e9788dc3c3614e20af7fe232fcffab81.jpg)  
Figure 8: Component grafting (seed 42). Bars: change in intent F1 when one component group of the donor replaces that of the base model; the vision tower is shown in blue. Dots: three random perturbations of the same tensors with the same norm.

<table><tr><td>Arm</td><td>42</td><td>1</td><td>2</td><td>Mean</td></tr><tr><td>zero shot</td><td></td><td>0.42</td><td>0.42</td><td>0.42</td></tr><tr><td>S-C</td><td>0.29</td><td>0.47</td><td>0.43</td><td>0.40</td></tr><tr><td>MIX</td><td>0.60</td><td>0.44</td><td>0.47</td><td>0.50</td></tr><tr><td>J-FC</td><td>0.46</td><td>0.45</td><td>0.48</td><td>0.46</td></tr><tr><td>J-CF</td><td>0.49</td><td>0.44</td><td>0.60</td><td>0.51</td></tr><tr><td>COND</td><td>0.71</td><td>0.43</td><td>0.64</td><td>0.59</td></tr><tr><td>S-C, blur</td><td>0.34</td><td>0.58</td><td>0.62</td><td>0.51</td></tr><tr><td>J-FC, blur</td><td>0.52</td><td>0.51</td><td>0.55</td><td>0.53</td></tr></table>

Table 17: Person share ρ of the occlusion loss per seed.

<table><tr><td>Direction</td><td>Prefix</td><td>42</td><td>1</td><td>2</td></tr><tr><td>actions → intent (J-FC)</td><td>correct wrong</td><td>+2.9 -2.4</td><td>+1.3 -4.0</td><td>+2.7 -3.2</td></tr><tr><td></td><td>empty</td><td>-4.1</td><td>-3.7</td><td>-22.6</td></tr><tr><td>intent → actions</td><td>correct</td><td>+1.0</td><td>+1.6</td><td>+1.3</td></tr><tr><td>(J-CF)</td><td>wrong empty</td><td>-3.4 0.0</td><td>-3.4 -0.6</td><td>-3.0 -0.8</td></tr></table>

Table 18: Forced prefixes in direct generation per seed, change of the decoded side’s macro-F1 (×100).

<table><tr><td colspan="3">Intent</td><td colspan="2">Action</td></tr><tr><td>Arm</td><td>Prompt</td><td>best late</td><td>best</td><td>late</td></tr><tr><td>S-F</td><td>J-FC</td><td>1.1 2.6</td><td>2.7</td><td>2.0</td></tr><tr><td>S-C</td><td>J-FC</td><td>2.0 2.7</td><td>0.5</td><td>0.1</td></tr><tr><td>MIX</td><td>J-FC</td><td>2.0 3.1</td><td>2.6</td><td>2.5</td></tr><tr><td>J-FC</td><td>J-FC</td><td>1.8 3.3</td><td>2.6</td><td>2.7</td></tr><tr><td>J-CF</td><td>J-FC</td><td>2.5 3.4</td><td>3.3</td><td>2.9</td></tr><tr><td>COND</td><td>J-FC</td><td>0.6 1.4</td><td>1.7</td><td>0.9</td></tr><tr><td>S-F</td><td>F</td><td>3.4</td><td>3.6</td><td>1.8 1.8</td></tr><tr><td>S-C</td><td>C</td><td>2.3</td><td>1.7 0.6</td><td>0.2</td></tr></table>

Table 19: Probe gain over the zero-shot model (balanced accuracy ×100, seed 42) at the best layer and averaged over layers 16 to 32. The last two rows feed each single-task model its own prompt and compare with the zero-shot model under the same prompt.