# Less Is Moral: A CHARMing Framework for Moral Foundations Detection in Endorsement Behaviour

Huixiang Fu, Marian-Andrei Rizoiu Behavioral Data Science, University of Technology Sydney Correspondence: huixiang.fu@student.uts.edu.au

## Abstract

Moral language plays a central role in shaping online endorsement and the diffusion of information, yet existing moral foundation detection systems often suffer from poor cross-domain generalization, weak rationale grounding, and reliance on costly prompting-based large language models (LLMs). We introduce CHARM, a MAC- and Hate-speech-Aware Rationalealigned Moral foundation detection framework built on a lightweight fine-tuned LLM, which integrates complementary moral grounding, rationale alignment, and polarity-aware hate speech signals to support more robust and faithful moral prediction. Unlike prior dictionary-, fine-tune-, or prompt-based detectors, which decouple computation from psychological theory, CHARM is built so that each component — MAC cross-attention, rationale alignment, and hate-speech modulation — operationalizes a distinct psychological construct. Using a 30% subsample of the MFTC, MFRC, and News training pools together with the richer supervision in MFTCXplain, CHARM improves AUC by up to 15.3% in-domain, surpasses the supervised baselines on every out-of-domain dataset in both AUC and F1, and offers a scalable, lowcost alternative to prompting-based LLM detectors. We further apply CHARM to large-scale COVID-19 discourse on Twitter and show that moral value alignment is strongly associated with online endorsement behavior. By making moral framing measurable at scale, CHARM offers a practical tool for studying the spread of morally charged misinformation<sup>1</sup>.

## 1 Introduction

In March 2021, the Center for Countering Digital Hate (2021) named a “Disinformation Dozen” responsible for up to 65% of anti-vaccine content on Facebook and Twitter — including Joseph Mercola, whose posts claiming hydrogen peroxide could treat COVID-19 were shared over 4,600 times on Facebook alone. Why does such content attract so much endorsement? Prior work links sharing to alignment between a post’s moral framing and the audience’s values (Abdurahman et al., 2025); Rao et al. (2025) further show that during COVID-19, pseudo-experts (the Disinformation Dozen among them) used markedly more negative moral framing than public health experts — especially along the care/harm and authority/subversion dimensions of Moral Foundations Theory (MFT) (Haidt, 2001; Graham et al., 2013). Two questions remain open: (i) is endorsement driven by the moral framing of content creators, and (ii) does the resulting endorsement network exhibit moral homophily between users and the creators they endorse?

Answering either at population scale requires a reliable way to detect moral framing in text. Computational moral foundation detection has evolved from lexicon-based methods (Graham et al., 2009b; Hopp et al., 2021) to supervised fine-tuning with pretrained language models (Nguyen et al., 2024; Preniqi et al., 2024), and more recently to prompting LLMs for moral classification (Chen et al., 2025; Skorski and Landowska, 2025).

Three gaps persist. (1) Detection decoupled from theory. Moral cognition is grounded in two complementary psychological frameworks — MFT and Morality-as-Cooperation (MAC) (Curry et al., 2019a,b; Graham et al., 2011; Haidt, 2012) — yet fine-tuned detectors inherit only the MFT label vocabulary, collapsing this richer structure into flat classification (Araque et al., 2022; Reinig et al., 2024). LLM-based detectors substitute statistical pretraining patterns for human moral deliberation (Chen et al., 2025; Hendrycks et al., 2021; Jiang et al., 2025; Jain et al., 2024; Lyu et al., 2024), producing labels with little auditable evidence. (2) Polarity and antisocial coupling ignored. Negative moral framing elevates endorsement (Rao et al., 2025), and moral language is tightly coupled to antisocial expression online (Kennedy et al., 2023; Brady et al., 2019) — yet most detectors collapse virtue and vice into a single foundation label and treat hate-speech detection as an unrelated task. (3) Robustness–efficiency tension. Fine-tuned models achieve strong in-domain accuracy but generalize poorly; prompt-based LLMs are more robust but expensive and depend on closed APIs.

We address these gaps with CHARM (MACand Hate-speech-Aware Rationale-aligned Moral foundation): a two-stage framework on a LoRAadapted LLaMA-3.1-8B backbone in which each architectural component operationalizes a distinct psychological construct — MAC cross-attention introduces cooperation-domain structure as a complementary inductive bias, rationale-aligned pooling grounds prediction in human moral-deliberation traces, and FiLM hate-speech modulation captures the documented moral–antisocial coupling. This work makes three contributions:

(1) Theory-integrated architecture. To our knowledge, CHARM is the first moral foundation detector to embed psychological structure in the model architecture rather than only the label space, departing from the dictionary-, fine-tune-, and prompt-based traditions that treat moral detection as classification disconnected from the theories whose labels they reuse. Ablations identify MAC grounding as the strongest cross-domain inductive bias, with the largest OOD F1 drops when removed (e.g., ARG .54→.43, VIG .67→.50).

(2) Efficient, polarity-aware, and faithful. CHARM is open-weight and uses a 30% subsample of the MFTC, MFRC, and News training pools, together with MFTCXplain, which provides polarity, rationale, and hate-speech supervision. This complete system improves AUC by up to 29.5% in-domain and 5.1% out-of-domain, surpasses the supervised baselines on every OOD dataset in both AUC and F1, and remains competitive with prompting-based LLMs at a fraction of the perinference cost. Its 10-dimensional polarity-aware head improves over the strongest polarity-aware baseline by .33 AUC. Token-level rationale spans align closely with human annotations and causally support predictions, supplying auditable evidence behind each decision.

(3) Moral framing predicts endorsement and network homophily at scale. On COVID-19 Twitter discourse, CHARM shows that producer-side moral framing predicts endorsement beyond behavioural and network features (author-only moral

AUC .84 vs. retweeter-only .72), and that the network is morally assortative across all foundations, most strongly along loyalty (r = .46).

## 2 Related Work

## 2.1 Moral Foundations Theory (MFT) and Morality-as-Cooperation (MAC)

MFT organizes human moral reasoning around five evolved foundations (Haidt and Joseph, 2004; Graham et al., 2009a, 2013): care/harm, fairness/cheating, loyalty/betrayal, authority/subversion, and purity/degradation — the first two termed individualizing and the latter three binding foundations — with a later extension adding liberty/oppression (Iyer et al., 2012).

MAC (Curry, 2016) grounds morality in evolutionary game theory, identifying seven cooperationbased domains: family values (kin altruism), group loyalty (coalition support), reciprocity (mutual exchange), heroism (costly altruism), deference (hierarchy), fairness (equitable allocation), and property rights (ownership). Although developed through distinct methodologies, the two frameworks are largely complementary: several MAC domains map onto MFT foundations (e.g., group loyalty onto loyalty/betrayal, deference onto authority/subversion), while MAC additionally captures cooperative moral types absent from MFT, such as kin altruism, heroism, and property rights (Curry et al., 2019a).

## 2.2 Moral Foundation Detection

Computational moral foundation detection spans three families. Lexicon-based methods rely on predefined moral dictionaries — MFD (Graham et al., 2009b), MFD 2.0 (Frimer et al., 2019), and eMFD (Hopp et al., 2021) — and scale well but struggle with contextual and implicit expressions. Fine-tuned models such as MoralBERT (Preniqi et al., 2024) and MFormer (Nguyen et al., 2024) improve contextual sensitivity but generalize poorly across domains; recent work addresses this through auxiliary supervision and domain adaptation, e.g., DAMF (Guo et al., 2023) and ME<sup>2</sup>-BERT (Zangari et al., 2025). Prompting-based LLMs such as MoVa (Chen et al., 2025) improve transferability but depend on closed, proprietary APIs. Across these families, moral detection remains a labelprediction task with limited theoretical grounding and auditable evidence. Recent work addresses the latter gap with rationale-rich hate-speech datasets.

MFTCXplain provides a multilingual benchmark with multi-hop explanations for LLM moral reasoning (Trager et al., 2025), while Vargas et al. (2026) provide Brazilian Portuguese moral rationales for self-explaining hate-speech detection. CHARM uses these resources for rationale-rich training and cross-lingual evaluation, respectively, and adds MAC grounding for cross-domain MFT detection.

## 3 CHARM Framework

Each component of CHARM operationalizes a distinct psychological construct: (i) the MFT foundation classifier targets the standard moral foundation taxonomy; (ii) the MAC cross-attention module introduces cooperation-domain structure as a complementary inductive bias; (iii) the rationalealignment module grounds prediction in human moral-deliberation traces; and (iv) the hate-speech modulation captures the documented coupling between moral language and antisocial expression. As shown in Figure 1, CHARM consists of two stages: (i) a LoRA-based encoder adaptation stage for learning foundation-level moral representations, and (ii) a polarity-aware classification stage that integrates rationale supervision, MAC grounding, and hate-speech modulation.

## 3.1 Problem Statement

Given a dataset $\mathcal { D } = \{ ( x _ { i } , y _ { i } ^ { ( m ) } , y _ { i } ^ { ( h ) } , r _ { i } , g _ { i } ) \} _ { i = 1 } ^ { N }$ where $y _ { i } ^ { ( m ) } \ \in \ \{ 0 , 1 \} ^ { 1 0 }$ denotes polarity-aware moral labels, $y _ { i } ^ { ( h ) } \in \{ 0 , 1 \}$ the hate-speech label, $r _ { i }$ the rationale mask, and $g _ { i } \in \mathbb { R } ^ { 7 }$ the MAC vector, the goal is to learn a polarity-aware moral classifier.

## 3.2 Foundation-Level Moral Encoding

Moral datasets differ substantially in annotation granularity: most provide only coarse-grained foundation labels, while a smaller subset additionally includes polarity annotations. To leverage all available supervision consistently, we adopt a twostage training strategy. Stage 1 learns transferable foundation-level moral representations using supervision from the five MFT foundations, while Stage 2 extends the model with polarity-aware signals built on top of the frozen Stage 1 encoder. The encoder is fine-tuned using LoRA (Hu et al., 2022). A lightweight classifier predicts five foundation logits $\hat { y } \in \mathbb { R } ^ { 5 }$ , optimized with:

$$
\mathcal { L } _ { \mathrm { m o r a l } } ^ { ( 5 d ) } = \frac { 1 } { 5 } \sum _ { k = 1 } ^ { 5 } \mathrm { B C E } ( \hat { y } _ { k } , y _ { k } ^ { ( 5 ) } ) .
$$

## 3.3 Polarity-Aware Moral Classification

Stage 2 builds on the frozen LoRA-adapted encoder from Stage 1 and performs polarity-aware moral classification using rationale alignment, MAC grounding, and hate-speech modulation.

## 3.3.1 Human Rationale Supervision

Rationale Selector. Given an input x, the LoRA encoder from Stage 1 produces text representations $H = \mathrm { E n c } ( x ) \in \mathbb { R } ^ { L \times d }$ , where L is the sequence length and d is the hidden dimension. A rationale selector predicts token-level rationale logit $z _ { t }$ and probability $p _ { t } ,$ , indicating whether the token belongs to a human-annotated rationale span. The selector is pretrained using token-level rationale supervision with binary cross-entropy:

$$
\mathcal { L } _ { \mathrm { { r a t } } } = \frac { 1 } { | \mathcal { V } | } \sum _ { t \in \mathcal { V } } \mathrm { B C E } ( \sigma ( z _ { t } ) , r _ { t } ) ,
$$

where $r _ { t } \in \{ 0 , 1 \}$ denotes the aligned rationale label and $\nu$ represents non-padding positions. To encourage contiguous rationale predictions, we additionally apply a total variation regularizer:

$$
\mathcal { L } _ { \mathrm { t v } } = \frac { 1 } { | \boldsymbol { A } | } \sum _ { t \in \boldsymbol { A } } ( p _ { t } - p _ { t - 1 } ) ^ { 2 } ,
$$

where $\mathcal { A }$ denotes adjacent valid token positions.

Rationale-Steered Attention Pooling. We combine the encoder attention distribution $a ^ { \mathrm { b a s e } }$ with the predicted rationale distribution $a ^ { \mathrm { { r a t } } }$ through a learnable interpolation $a = ( 1 - \alpha ) a ^ { \mathrm { b a s e } } + \alpha a ^ { \mathrm { r a t } }$ where $\alpha \in [ 0 , 1 ]$ is a learnable interpolation coefficient parameterized through a sigmoid function. The final text representation is obtained by weighted pooling over token representations, where $\begin{array} { r } { u ^ { \mathrm { r a t } } = \bar { \sum _ { t = 1 } ^ { L } } a _ { t } \bar { h } _ { t } } \end{array}$ . This encourages the model to focus on human-identified moral evidence while retaining contextual information from the encoder representations.

## 3.3.2 MAC-Theory Grounding

We introduce the Moral-As-Cooperation (MAC) grounding as an auxiliary supervision signal. Each sample is associated with seven MAC cooperation dimensions from eMACDscore (Malik et al., 2025), a lightweight dictionary-based moral mining tool that scores text over the seven MAC cooperation domains (see Section A.2 for details). Each dimension provides a probability score $\hat { p } _ { i }$ and a sentimentpolarity score $\hat { s } _ { i }$ that capture the strength and direction of the cooperative signal. After normalization, the two scores are multiplied as $g _ { i } = \tilde { p } _ { i } \cdot \tilde { s } _ { i }$ , forming a MAC vector $g \in \mathbb { R } ^ { 7 }$ . Each dimension of g is then projected into the hidden space to produce seven MAC tokens $M \in \mathbb { R } ^ { 7 \times d }$ . The MFT labels are represented as ten learnable queries $Q \in \mathbb { R } ^ { 1 0 \times d }$ corresponding to the virtue and vice poles of the five foundations. Cross-attention between Q and M produces ten moral representations:

![](images/a528d5209b6a72462ec348bab36fb910d3f2652a67711c54dd7c1342d10da347.jpg)  
Figure 1: Overview of CHARM, a two-stage framework for polarity-aware moral foundation detection. Stage 1 trains a LoRA adapter to learn foundation-level moral semantics. Stage 2 refines the text representation using rationale alignment, MAC-guided cross-attention, and hate-speech-aware FiLM modulation. The model jointly predicts 10 polarity-aware moral dimensions and hate speech.

$$
C = \mathrm { s o f t m a x } \bigg ( \frac { Q M ^ { \top } } { \sqrt { d } } \bigg ) M \in \mathbb { R } ^ { 1 0 \times d } ,
$$

along with an attention map $\begin{array} { r l r l } { W } & { { } } & { = } & { } \end{array}$ softmax $( Q M ^ { \top } / \sqrt { d } ) \quad \in \quad \mathbb { R } ^ { 1 0 \times 7 }$ , where $W _ { i j }$ denotes how strongly the i-th moral dimension attends to the j-th MAC dimension. The representations in C are then pooled into a single MAC vector c using learnable attention-based weighting. Specifically, each moral representation $C _ { i }$ is assigned a scalar attention score:

$$
a _ { i } = \frac { \exp ( w ^ { \top } C _ { i } ) } { \sum _ { k = 1 } ^ { 1 0 } \exp ( w ^ { \top } C _ { k } ) } ,
$$

where $w \in \mathbb { R } ^ { d }$ is a learnable parameter vector. The final pooled MAC representation is then computed as $\textstyle c = \sum _ { i = 1 } ^ { 1 0 } a _ { i } C _ { i } \in \mathbb { R } ^ { d }$ . Finally, the MAC vector c is fused with the rationale-guided text representation $u ^ { \mathrm { r a t } }$ to obtain the MAC-enhanced moral representation $u ^ { \mathrm { m a c } }$

$$
u ^ { \mathrm { m a c } } = \mathrm { F u s i o n M L P } ( [ u ^ { \mathrm { r a t } } \parallel c ] ) \in \mathbb { R } ^ { d } .
$$

This formulation injects cooperation-domain structure into the moral representation, enabling the model to capture complementary interactions between MAC signals and moral foundations.

## 3.3.3 Hate Speech Modulation

Prior work has shown that moral language and hate speech are closely related, with certain moral dimensions disproportionately associated with hateful rhetoric (Vargas et al., 2026). Motivated by this connection, we introduce hate speech as an auxiliary modulation signal for polarity-aware moral prediction. Given the MAC-grounded representation $u ^ { \mathrm { m a c } } \in \mathbb { R } ^ { d }$ , a hate-speech classifier first predicts a hate probability $\hat { y } ^ { ( h ) } = \sigma ( w _ { h } ^ { \top } u ^ { \mathrm { m a c } } )$ . The hate speech head is trained using binary cross-entropy loss. Following Feature-wise Linear Modulation (FiLM) (Perez et al., 2018), the predicted hate probability is transformed into channel-wise scaling and shifting parameters:

$$
\gamma = \operatorname { t a n h } ( f _ { \gamma } ( \hat { y } ^ { ( h ) } ) ) , \qquad \beta = f _ { \beta } ( \hat { y } ^ { ( h ) } ) ,
$$

where $\gamma \in [ - 1 , 1 ] ^ { d }$ and $\beta \in \mathbb { R } ^ { d }$ . We then modulate $u ^ { \mathrm { m a c } }$ through $u ^ { \mathrm { f i l m } } = u ^ { \mathrm { m a c } } \odot ( 1 + \gamma ) + \beta ,$ , allowing hate speech information to act as a soft conditioning signal over the moral representation. The final polarity-aware moral prediction is computed as a 10-dimensional probability vector, $\hat { y } ^ { ( m ) } =$ $\sigma ( W _ { m } u ^ { \mathrm { f i l m } } ) \in ( 0 , 1 ) ^ { 1 0 }$ . The corresponding classification objective is:

$$
\mathcal { L } _ { \mathrm { m o r a l } } ^ { ( 1 0 d ) } = \frac { 1 } { 1 0 } \sum _ { j = 1 } ^ { 1 0 } \mathrm { B C E } ( \hat { y } _ { j } ^ { ( m ) } , y _ { j } ^ { ( m ) } ) .
$$

## 4 Experiments

Datasets. We use nine datasets covering diverse forms of morally charged discourse. Dataset statistics and label distributions are provided in Table 1 and Section A.1. For all datasets, we precompute 7-dimensional MAC signals using eMACDscore. Training proceeds in two stages. Stage 1 jointly trains on four in-domain datasets using foundationlevel supervision, while Stage 2 further fine-tunes on MFTCXplain with polarity, rationale, and hatespeech annotations. During Stage 1, polarity labels in MFTCXplain are collapsed into foundation categories, while the same data split is retained across both stages to prevent leakage. Following MFORMER, we retain the original train/test splits and use 30% of the training data for the remaining in-domain datasets. We exclude liberty/oppression due to inconsistent supervision across datasets. The five out-of-domain datasets are reserved for evaluation under a shared validation setting.

<table><tr><td>Split</td><td>Dataset</td><td># Inst.</td><td>Text Type</td><td>Content Focus</td></tr><tr><td rowspan="4">In-domain</td><td>MFTC (Hoover et al., 2020)</td><td>34,987</td><td>Twitter posts</td><td>Socio-political discourse</td></tr><tr><td>MFRC (Trager et al., 2022)</td><td>17,886</td><td>Reddit comments</td><td>Politics and everyday morality</td></tr><tr><td>News (Hopp et al., 2021)</td><td>34,262</td><td>News articles</td><td>Diverse news topics from GDELT</td></tr><tr><td>†MFTCXplain (Trager et al., 2025)</td><td>3,245</td><td>Multilingual tweets</td><td>Moral reasoning over hate speech</td></tr><tr><td rowspan="5">OOD</td><td>SC (Forbes et al., 2020)</td><td>29,239</td><td>Rules-of-thumb</td><td>Everyday social and moral norms</td></tr><tr><td>VIG (Clifford et al., 2015)</td><td>132</td><td>Psychology vignettes</td><td>Foundation-violating scenarios</td></tr><tr><td>ARG (Kobbe et al., 2020)</td><td>320</td><td>Debate arguments</td><td>Pro/con stances on 16 topics</td></tr><tr><td>MIC (Ziems et al., 2022)</td><td>11,375</td><td>Chatbot RoTs</td><td>Human-chatbot moral biases</td></tr><tr><td>‡HATEBR (Vargas et al., 2026)</td><td>5,040</td><td>Instagram comments</td><td>Brazilian political hate speech</td></tr></table>

Table 1: Full profile of datasets used for training and evaluation. # Inst is the original dataset size. <sup>†</sup>MFTCXplain contains four languages (Portuguese, Persian, Italian, English). <sup>‡</sup>HateBR is in Brazilian Portuguese.

Backbone selection. We compare LLAMA-3.1- 8B<sup>2</sup> and QWEN3-8B<sup>3</sup>, and find that LLAMA-3.1- 8B consistently performs better across settings. We therefore use it as the backbone in all main experiments. Additional experimental details and QWEN3-8B results are provided in Section A.5 and Section B.3, respectively.

Baselines. We compare CHARM with five representative baselines spanning fine-tuned and zeroshot approaches. Because these models differ in their training objectives and available supervision, Table 2 explicitly summarizes their training paradigms and training corpora to contextualize the comparison.

Metrics. We evaluate the model along two axes: classification performance and rationale quality. For classification, we report F1 and AUC on both moral foundation prediction and hate speech detection. For rationale quality, we follow the ERASER protocol (DeYoung et al., 2020) to assess plausibility and faithfulness. Definitions of all evaluation metrics are provided in Section A.4.

## 4.1 Moral Detection Performance

In-domain performance. As shown in Table 3, CHARM performs strongly across the in-domain benchmarks, achieving the best AUC and F1 on MFRC, News, and MFTCXplain. On the three corpora shared with MFORMER—MFTC, MFRC, and News—CHARM uses only 30% of the available training instances, supplemented by the richer supervision from MFTCXplain. CHARM improves over MFORMER on MFRC and News by .07 and .11 AUC, respectively, with corresponding gains in F1. MFTC is the main exception: MFORMER achieves a higher F1 (.77 vs. .72) and slightly higher AUC (.89 vs. .87), with the F1 difference primarily concentrated in the Authority foundation (see Section A.10). Overall, CHARM’s gains are strongest on MFRC and News, while its performance on MFTC remains competitive rather than uniformly superior.

Generalization under distribution shift. CHARM also generalizes robustly to unseen domains, although its advantage varies across baseline families. Among fine-tuned baselines, CHARM consistently outperforms MFORMER across all five OOD datasets in both AUC and F1. Compared with Tuning-GPT4o-mini, CHARM matches or exceeds the reported AUC on four of the five OOD datasets, with VIG as the exception. Together, these results indicate that CHARM maintains strong cross-domain performance despite its limited supervision on MFTC, MFRC, and News. The comparison with zero-shot LLMs is more nuanced. Against MOVA, CHARM performs better on ARG (F1: .54 vs. .49; equal AUC), remains competitive on SC and HateBR, but trails on MIC and VIG. Similarly, CHARM outperforms Qwen-32B-Instruct in AUC across all nine datasets and in F1 on eight of nine, while Qwen shows a clear advantage on VIG (.79 vs. .67 F1). These results suggest that CHARM provides more consistent performance across domains, whereas zero-shot LLMs can be stronger on particular datasets.

<table><tr><td>Model</td><td>Training</td><td>Training data</td><td>Description</td></tr><tr><td>MORALBERT (Preniqi et al., 2024)</td><td>Fine-tuned</td><td>MFTC</td><td>BERT-based model for polarity-aware moral foundation classification.</td></tr><tr><td>MFORMER (Nguyen et al., 2024)</td><td>Fine-tuned</td><td>MFTC, MFRC, and News</td><td>RoBERTa-based classifiers trained for multi-domain moral foundation detection.</td></tr><tr><td>MoVA (Chen et al., 2025)</td><td>Zero-shot</td><td>None</td><td>LLM prompting framework for joint moral foundation prediction.</td></tr><tr><td>Qwen-32B-Instruct4</td><td>Zero-shot</td><td>None</td><td>Large instruction-tuned LLM evaluated through direct zero-shot prediction.</td></tr><tr><td>Tuning-GPT4o-mini (Chen et al., 2025)</td><td>Fine-tuned</td><td>30% of MFTC, MFRC, and News</td><td>Instruction-tuned LLM fine-tuned for moral foundation prediction.</td></tr><tr><td>CHARM</td><td>Fine-tuned</td><td>MFTCXplain and 30% of MFTC, MFRC, and News</td><td>Theory-grounded model with rationale and polarity-aware supervision.</td></tr></table>

Table 2: Overview of the compared models and their training configurations. The models differ in training paradigm and available training corpora; these differences are made explicit to contextualize the performance comparison.
<table><tr><td></td><td></td><td>MFTC MFRC</td><td>News</td><td></td><td>MFTCXpl.</td><td>ARG</td><td>SC</td><td>MIC</td><td>VIG</td><td>HateBR</td></tr><tr><td rowspan="6">AUC</td><td>CHARM</td><td>.87</td><td>.91</td><td>.83</td><td>.79</td><td>.87</td><td>.77</td><td>.73</td><td>.89</td><td>.82</td></tr><tr><td>MoVA†</td><td>.77</td><td>.80</td><td>.68</td><td>.75</td><td>.87</td><td>.76</td><td>.75</td><td>.96</td><td>.78</td></tr><tr><td>Qwen-32B-Inst.</td><td>.64</td><td>.63</td><td>.55</td><td>.62</td><td>.72</td><td>.66</td><td>.63</td><td>.88</td><td>.63</td></tr><tr><td>MFORMER†</td><td>.89</td><td>.84</td><td>.72</td><td>.61</td><td>.86</td><td>.70</td><td>.70</td><td>.81</td><td>.62</td></tr><tr><td>MORALBERT†</td><td>.75</td><td>.79</td><td>.61</td><td>.60</td><td>.77</td><td>.70</td><td>.69</td><td>.80</td><td>.54</td></tr><tr><td>Tuning-GPT4o-mini†</td><td>一</td><td>一</td><td>一</td><td>一</td><td>.85</td><td>.73</td><td>.73</td><td>.92</td><td>一</td></tr><tr><td rowspan="6">F</td><td>CHARM</td><td>.72</td><td>.72</td><td>.55</td><td>.59</td><td>.54</td><td>.46</td><td>.50</td><td>.67</td><td>.38</td></tr><tr><td>MOVA</td><td>.57</td><td>.48</td><td>.37</td><td>.52</td><td>.49</td><td>.52</td><td>.51</td><td>.69</td><td>.36</td></tr><tr><td>Qwen-32B-Inst.</td><td>.47</td><td>.40</td><td>.26</td><td>.41</td><td>.52</td><td>.42</td><td>.40</td><td>.79</td><td>.30</td></tr><tr><td>MFORMER</td><td>.77</td><td>.66</td><td>.53</td><td>.50</td><td>.51</td><td>.43</td><td>.44</td><td>.52</td><td>.31</td></tr><tr><td>MORALBERT</td><td>.56</td><td>.49</td><td>.33</td><td>.38</td><td>.47</td><td>.37</td><td>.41</td><td>.32</td><td>.24</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: F1 and AUC across nine datasets. Best results per dataset are shown in bold. Baseline AUC values (<sup>†</sup>) are taken from Chen et al. (2025); F1 scores are independently reproduced by running each model. “–” indicates results not reported and not reproducible. Per-foundation breakdowns are provided in Section A.10. Qwen-32B-Instruct and MOVA are evaluated using zero-shot prompting, whereas the other models use task-specific fine-tuning.

One possible explanation lies in input characteristics and inference strategy. SC and MIC contain relatively short inputs (14 and 17 tokens on average), whereas ARG contains substantially longer inputs (69.6 tokens on average), where moral evidence may be distributed across the context. Taskspecific fine-tuning and rationale supervision may therefore be particularly useful for capturing dispersed moral cues, while prompting-based models remain competitive on shorter inputs. Consistent with this interpretation, CHARM also outperforms MOVA on the Care foundation across SC and MIC, where its hate-speech-aware supervision may improve sensitivity to harm-related cues.

Moral polarity classification. As shown in Figure 2, CHARM clearly outperforms MORALBERT on both datasets, achieving AUC improvements of .33 on MFTCXplain and .21 on HateBR, alongside even larger gains in F1. Notably, CHARM also performs well on the Portuguese HateBR dataset, even though it was not designed for multilingual learning. This suggests that training on the multilingual MFTCXplain dataset may provide an unintended cross-lingual benefit.

Inference cost. Beyond predictive performance, CHARM and prompting-based approaches differ in deployment cost. MOVA incurs inference-time API costs of approximately \$.066 and \$.065 per 1K input tokens on SC and MIC, respectively, with costs incurred for each inference pass. By contrast,

![](images/3563e6b0ffb8c458cc79f1d7bc5cd1cb3390a1f0c8ed08b01f0e1f0e4efe600b.jpg)

CHARM is fine-tuned once on an open-source backbone and requires no per-example API calls at inference time.

## 4.2 Ablation Studies

We ablate MAC grounding, rationale supervision, and hate-speech supervision; F1 results are shown in Table 4, with AUC results in Section A.7. We also assess the contributions of CHARM’s training corpora through a leave-one-corpus-out analysis, with full results in Section A.8.

MAC provides the strongest structural inductive bias for cross-domain generalization. Removing MAC causes the largest overall degradation across OOD evaluation datasets, with especially severe drops on ARG (.54→.43) and VIG (.67→.50). Notably, compared with the other signals, SC and MIC are more strongly affected by removing MAC (SC: .46→.41; MIC: .50→.44). Both datasets primarily involve socially contextualized norm judgments and rule-of-thumb reasoning rather than explicit moral declarations, suggesting that MAC provides useful structural grounding for implicit moral interpretation under distribution shift.

Rationale supervision improves evidence grounding and prediction faithfulness. Removing rationale supervision leads to larger performance drops on datasets with relatively longer and more compositional inputs, particularly MFRC (.69→.65) and VIG (.67→.60), while having comparatively smaller effects on shorter-text datasets such as SC. Since rationale supervision is provided at the span level, this pattern suggests that rationale alignment is especially beneficial when morally relevant evidence is distributed across contextual spans rather than concentrated in a single local cue. This effect is further supported by the rationale evaluation results in Table 5. Compared with the variant without rationale supervision, CHARM consistently improves performance across all rationale evaluation metrics, indicating that rationale alignment enhances both plausibility and faithfulness, while also improving the causal relevance of the identified rationale spans for prediction. We further examine representative cases with and without rationale supervision, providing qualitative evidence that rationale supervision helps preserve label-relevant evidence and improves foundation-level predictions (see Section A.9).

Figure 2: AUC and Macro-F1 comparison between CHARM and MORALBERT for 10-dimension moral polarity classification on MFTCXplain and HateBR (only datasets with 10-dimensional moral annotations).

Hate-speech and moral supervision reinforce each other. Removing hate-speech supervision consistently reduces performance across multiple datasets, including MFTCXplain<sub>10d</sub> (.47→.45), SC (.46→.44), and VIG (.67→.65). The degradation on MFTCXplain<sub>10d</sub> suggesting that hate-speech supervision can help the model better distinguish fine-grained moral polarity. Table 6 further shows that CHARM achieves the strongest hate-speech detection performance, outperforming both prompted GPT-4o mini and a fine-tuned LLAMA model that is trained without moral supervision. These results indicate a mutually beneficial relationship between moral reasoning and harmful-content identification. Overall, all three supervision signals contribute complementary gains.

Leave-one-corpus-out analysis. The full configuration achieves the strongest average performance (AUC=.83; F1=.57), with MFRC making the largest contribution. MFTC and News provide complementary domain coverage. Removing MFTCXplain lowers average performance and produces the clearest declines on MFTCXplain and HateBR, confirming the value of its polarity, rationale, and hate-speech supervision. Variation across individual datasets suggests corpus-specific interactions rather than uniform gains from every training source. Overall, the results show that CHARM benefits from both corpus diversity and richer supervision. Complete results are reported in Tables 13 and 14 in Section A.8.

## 5 Moral Alignment and Endorsement Behaviour

We use CHARM to test, at scale, the two questions raised in Section 1: whether producer-side moralframing predicts endorsement, and whether the resulting endorsement network exhibits moral homophily between users and creators (Dehghani et al., 2016; Van Bavel et al., 2021). We use a subset of the COVID-19 Twitter dataset introduced by Chen et al. (2020), covering discourse from March 25 to April 16, 2020. Each user’s moral profile is constructed by aggregating tweetlevel moral scores inferred by CHARM. Although CHARM produces polarity-aware predictions over ten moral dimensions, we collapse scores into five foundation-level representations for endorsement analysis because several polarity dimensions are highly sparse in large-scale Twitter discourse. The full polarity distribution is provided in Section B.1. To reduce noise from weak moral expressions, we retain only tweets above the 90th percentile of overall moral intensity. We then construct a directed endorsement network based on retweet behavior, which is commonly treated as a proxy for endorsement (Metaxas et al., 2015). Mentions and quote tweets are excluded due to their ambiguous endorsement semantics. The pair-construction process is illustrated in Figure 4.

<table><tr><td>Variant</td><td>MFTCXpl.10d</td><td>MFTCXpl.5d</td><td>MFTC</td><td>MFRC</td><td>News</td><td>ARG</td><td>SC</td><td>MIC</td><td>VIG</td></tr><tr><td>baseline</td><td>.41</td><td>.56</td><td>.68</td><td>.56</td><td>.50</td><td>.45</td><td>.30</td><td>.43</td><td>.47</td></tr><tr><td>w/o hate speech</td><td>.45</td><td>.59</td><td>.64</td><td>.55</td><td>.55</td><td>.53</td><td>.44</td><td>.45</td><td>.65</td></tr><tr><td>w/o MAC</td><td>.46</td><td>.59</td><td>.66</td><td>.53</td><td>.47</td><td>.43</td><td>.41</td><td>.44</td><td>.50</td></tr><tr><td>w/o rationale</td><td>.42</td><td>.57</td><td>.68</td><td>.65</td><td>.55</td><td>.53</td><td>.46</td><td>.47</td><td>.60</td></tr><tr><td>CHARM</td><td>.47</td><td>.59</td><td>.72</td><td>.69</td><td>.55</td><td>.54</td><td>.46</td><td>.50</td><td>.67</td></tr></table>

Table 4: Ablation study reporting F1 scores across evaluation datasets. The baseline applies a task-specific classification head on top of the frozen Stage 1 LoRA encoder using only the 10-dimensional MFT supervision. AUC results show consistent trends and are reported in Section A.7.

<table><tr><td rowspan="2"></td><td colspan="2">Plausibility</td><td colspan="2">Faithfulness</td></tr><tr><td>IoU↑</td><td>TF1↑ AUP↑</td><td>Suf↓</td><td>Cmp↑</td></tr><tr><td>CHARM</td><td>.62</td><td>.63</td><td>.60</td><td>.02 .25</td></tr><tr><td>w/o rationale</td><td>.14</td><td>.12 .42</td><td>.20</td><td>.04</td></tr></table>

Table 5: Rationale quality evaluation on MFTCXplain. Arrows indicate whether higher (↑) or lower (↓) is better.
<table><tr><td></td><td>Bin. F1 Mac. F1 AUC</td><td></td><td></td></tr><tr><td>CHARM</td><td>.66</td><td>.70</td><td>.79</td></tr><tr><td>TUNING-LLAMA-3.1-8B</td><td>.61</td><td>.64</td><td>.71</td></tr><tr><td>GPT-4o mini (0-shot)</td><td>.43</td><td>.58</td><td>.61</td></tr><tr><td>GPT-4o mini (4-shot)</td><td>.47</td><td>.58</td><td>.61</td></tr></table>

Table 6: Hate speech detection results on MFTCXplain.

Finding 1: Moral content predicts endorsement, with content producer framing carrying stronger signal than consumer-side. To evaluate moral alignment in endorsement behavior, we formulate endorsement prediction as a binary classification task over directed user pairs from the retweet network. Positive samples correspond to repeated retweet interactions occurring more than five times, while negative samples are constructed using a two-hop non-endorsement constraint. Endorsed pairs exhibit higher moral cosine similarity than non-endorsed pairs (mean .934 vs. .892, ∆ = .042), with both Mann–Whitney and KS tests indicating significant differences $( p < 1 0 ^ { - 1 1 9 } )$ ; the separation is consistent across all five foundations, with loyalty and fairness showing the sharpest distributional gap (Section B.7). We further test whether this structure provides predictive utility beyond network and behavioral signals using a LightGBM classifier with moral-profile, interaction, behavioral, and semantic features. More training details are provided in Section B.6. As shown in Table 7, removing moral features consistently degrades performance, whereas models using only moral representations still retain substantial predictive power. Moreover, author-only moral features substantially outperform retweeteronly features, indicating that endorsement behavior is more strongly associated with the moral framing of content producers than with the inferred moral profiles of endorsers.

<table><tr><td>Feature Setting</td><td>AUC</td><td>Macro F1</td><td>Micro F1</td></tr><tr><td>full</td><td>.92</td><td>.84</td><td>.84</td></tr><tr><td>w/o moral features</td><td>.90</td><td>.82</td><td>.82</td></tr><tr><td>w/o interaction features</td><td>.85</td><td>.77</td><td>.77</td></tr><tr><td>author moral only</td><td>.84</td><td>.76</td><td>.76</td></tr><tr><td>retweeter moral only</td><td>.72</td><td>.66</td><td>.66</td></tr></table>

Table 7: Ablation study for endorsement prediction using interaction, behavioral, and moral-profile features.

SHAP analysis. In the full model (Figure 3a), behavioral and network features dominate overall, yet author purity (rank 3) and moral Euclidean distance (rank 5) appear among the top ten, confirming that moral framing adds predictive signal beyond engagement covariates. In the moral-only model (Figure 3b), author-side features hold seven of the top ten positions, with author purity and moral cosine similarity ranking first and second. Notably, author purity is the strongest individual moral predictor of endorsement despite purity having the lowest network assortativity $( r = . 2 7 )$ , suggesting that high purity framing in content attracts broad endorsement rather than congruent followers.

![](images/74cf3cb88f04df95b446958a56ba2f3531f94296db0b8683f3aa6ab8f4a60a1b.jpg)  
(a)

![](images/c2089b5808f326e118463e6ea9910d697fc61cd0c8c59804b2a6a46bcffd89ac.jpg)  
(b)

![](images/5b62391695c5ce7fe4210fc8b26c2f7493457de598472212632e594b5e97b219.jpg)  
(c)

Figure 3: Endorsement analysis. (a) SHAP feature importance for the full prediction model; (b) SHAP for the moral-only model; (c) moral assortativity coefficients compared with degree-preserving random networks (edges rewired while preserving each node’s in- and out-degree). Error bars in (c) indicate bootstrap confidence intervals.  
![](images/e608d081652d8587829bbbde1d1eedfd33b04f8521a417f17178b85df480817a.jpg)  
Figure 4: Construction of the endorsement network.

Finding 2: Endorsement networks exhibit robust moral homophily. We next test whether endorsement structure exhibits moral homophily. Following Newman (2002), we measure moral assortativity for each foundation as the edge-level Pearson correlation between the foundation scores of the two endpoints of each endorsement tie. The full computation is provided in Section B.8. As shown in Figure 3c, all five foundations are significantly positively assortative (all z > 21, p ≈ 0 against a degree-preserving null), indicating that users systematically endorse others with similar moral profiles. Assortativity is highest for loyalty (r = .46), followed by fairness and authority (r = .41), care (r = .37), and purity (r = .27). Per-foundation moral alignment distributions are

provided in Section B.7.

Human validation of case-study predictions. We validate the CHARM-inferred moral score reliability in the COVID-19 domain using 100 stratified tweets independently annotated by three annotators blinded to CHARM’s predictions. Annotation reliability is high overall, with an observed agreement of .83 and Gwet’s AC1 of .76. Against the majorityvoted human labels, CHARM achieves a macro AUC of .84 and a macro average precision of .56. These results support the use of CHARM-derived moral scores in the case study, although estimates for low-prevalence moral categories should be interpreted cautiously. Full annotation procedures, agreement statistics, and label-level alignment results are reported in Table 16 in Section B.2.

## 6 Conclusions

We introduced CHARM, a lightweight moral foundation detection framework that unifies rationale alignment, MAC-based cooperative signals, and hate-speech-aware polarity modeling within a finetuned LLM architecture. Across multiple datasets, CHARM demonstrates strong robustness, faithfulness, and sample efficiency, while remaining substantially more efficient than prompt-based LLM approaches. Using CHARM, we further show that online endorsement behavior is closely associated with latent moral alignment, particularly the moral framing of content authors. Notably, author purity is the strongest individual predictor of endorsement yet exhibits the weakest network assortativity $( r = . 2 7 )$ , suggesting that morally charged content can attract broad endorsement beyond moral congruence. By grounding model components in explicit psychological construct, CHARM bridges social-psychological theory and LLM practice.

## Limitations

Several limitations should be noted. First, polaritylevel moral annotations remain limited, and most existing datasets do not cover the liberty/oppression foundation. As a result, training and evaluation of our 10-dimensional setting were largely restricted to MFTCXplain and HateBRMoralXplain. Future work could improve this by developing larger datasets with polarity-level annotations and broader foundation coverage.

Second, although CHARM shows strong out-ofdomain performance on several datasets, results are weaker on VIG and MIC compared to promptbased approach, whose content focuses more on abstract social norms. In contrast, the in-domain training data mainly consists of concrete, opinionrich social media and news discourse. Since finetuning tends to adapt models toward the training distribution, this mismatch in content type may limit generalization to more abstract forms of moral reasoning.

Finally, MAC supervision relies on eMACDscore automatically generated labels because no publicly available human-annotated MAC dataset currently exists. Although the same process was applied consistently across all samples, the generated labels may still introduce noise into downstream training.

## Ethics Statement

This work studies moral reasoning discourse using publicly available datasets collected from social media, news, and online discussion platforms. Since moral annotations can reflect cultural and annotator-specific biases, model predictions should not be interpreted as objective judgments of morality. Our goal is to support the analysis of moral framing and moral conflict in language, rather than to determine whether individuals or viewpoints are morally right or wrong.

The proposed framework may also inherit biases present in the training data, especially in politically or culturally sensitive contexts. We therefore encourage careful use of such models and emphasize that they should not be deployed as standalone systems for high-stakes moderation or decisionmaking.

## Acknowledgments

This work was partially supported by the Defence Science and Technology Group (DSTG) and the Advanced Strategic Capabilities Accelerator (ASCA) through its Emerging and Disruptive Technologies Program, and by the Australian Academy of Science.

We thank Lin Tian for her guidance on experiment design and for her contributions to revising the manuscript.

The authors’ contributions are listed below. Huixiang Fu: Conceptual framing, experiment design, dataset sourcing, conducting experiments, result analysis, and writing.

Marian-Andrei Rizoiu: Conceptual framing, supervision, funding acquisition, manuscript revision, and writing.

## References

Suhaib Abdurahman, Nils K. Reimer, Preni Golazizian, Elisa Baek, Yixuan Shen, Jackson Trager, Roshni Lulla, Jonas Kaplan, Carolyn Parkinson, and Morteza Dehghani. 2025. Targeting audiences’ moral values shapes misinformation sharing. Journal of Experimental Psychology: General, 154(4):935–957.

Oscar Araque, Lorenzo Gatti, and Kyriaki Kalimeri. 2022. Libertymfd: A lexicon to assess the moral foundation of liberty. In Proceedings of the 2022 ACM Conference on Information Technologyfor Social Good, GoodIT ’22, page 154–160, New York, NY, USA. Association for Computing Machinery.

William J. Brady, Julian A. Wills, Dominic Burkart, John T. Jost, and Jay J. Van Bavel. 2019. An ideological asymmetry in the diffusion of moralized content on social media among political leaders. Journal of Experimental Psychology: General, 148(10):1802– 1813.

Center for Countering Digital Hate. 2021. The disinformation dozen. https://counterhate.com/ research/the-disinformation-dozen/. Accessed: 2026-05-24.

Emily Chen, Kristina Lerman, and Emilio Ferrara. 2020. Tracking social media discourse about the COVID-19 pandemic: Development of a public coronavirus twitter data set. JMIR Public Health and Surveillance, 6(2):e19273.

Ziyu Chen, Junfei Sun, Chenxi Li, Tuan Dung Nguyen, Jing Yao, Xiaoyuan Yi, Xing Xie, Chenhao Tan, and Lexing Xie. 2025. MoVa: Towards generalizable classification of human morals and values. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 33216– 33260, Suzhou, China. Association for Computational Linguistics.

Scott Clifford, Vijeth Iyengar, Roberto Cabeza, and Walter Sinnott-Armstrong. 2015. Moral foundations

vignettes: A standardized stimulus database of scenarios based on moral foundations theory. Behavior research methods, 47(4):1178–1198.

Oliver Scott Curry. 2016. Morality as Cooperation: A Problem-Centred Approach, pages 27–51. Springer International Publishing, Cham.

Oliver Scott Curry, Matthew Jones Chesters, and Caspar J Van Lissa. 2019a. Mapping morality with a compass: Testing the theory of ‘morality-ascooperation’with a new questionnaire. Journal of Research in Personality, 78:106–124.

Oliver Scott Curry, Daniel Austin Mullins, and Harvey Whitehouse. 2019b. Is it good to cooperate? testing the theory of morality-as-cooperation in 60 societies. Current Anthropology, 60(1):47–69.

Morteza Dehghani, Kate Johnson, Joe Hoover, Eyal Sagi, Justin Garten, Niki Jitendra Parmar, Stephen Vaisey, Rumen Iliev, and Jesse Graham. 2016. Purity homophily in social networks. Journal of Experimental Psychology: General, 145(3):366–375.

Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, and Byron C. Wallace. 2020. ERASER: A benchmark to evaluate rationalized NLP models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4443–4458, Online. Association for Computational Linguistics.

Maxwell Forbes, Jena D. Hwang, Vered Shwartz, Maarten Sap, and Yejin Choi. 2020. Social chemistry 101: Learning to reason about social and moral norms. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 653–670, Online. Association for Computational Linguistics.

Jeremy A. Frimer, Reihane Boghrati, Jonathan Haidt, Jesse Graham, and Morteza Dehghani. 2019. Moral foundations dictionary 2.0. https://osf.io/ ezn37.

Jesse Graham, Jonathan Haidt, Sena Koleva, Matt Motyl, Ravi Iyer, Sean P Wojcik, and Peter H Ditto. 2013. Moral foundations theory: The pragmatic validity of moral pluralism. In Advances in experimental social psychology, volume 47, pages 55–130. Elsevier.

Jesse Graham, Jonathan Haidt, and Brian A Nosek. 2009a. Liberals and conservatives rely on different sets of moral foundations. Journal ofpersonality and social psychology, 96(5):1029.

Jesse Graham, Jonathan Haidt, and Brian A Nosek. 2009b. Moral foundations dictionary. PsycTESTS Dataset.

Jesse Graham, Brian A. Nosek, Jonathan Haidt, Ravi Iyer, Spassena Koleva, and Peter H. Ditto. 2011. Mapping the moral domain. Journal of Personality and Social Psychology, 101(2):366–385.

Siyi Guo, Negar Mokhberian, and Kristina Lerman. 2023. A data fusion framework for multi-domain morality learning. Proceedings of the International AAAI Conference on Web and Social Media, 17(1):281–291.

Jonathan Haidt. 2001. The emotional dog and its rational tail: A social intuitionist approach to moral judgment. Psychological Review, 108(4):814–834.

Jonathan Haidt. 2012. The Righteous Mind: Why Good People Are Divided by Politics and Religion. Pantheon Books, New York.

Jonathan Haidt and Craig Joseph. 2004. Intuitive ethics: How innately prepared intuitions generate culturally variable virtues. Daedalus, 133(4):55–66.

Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. 2021. Aligning AI with shared human values. In International Conference on Learning Representations (ICLR).

Joe Hoover, Gwenyth Portillo-Wightman, Leigh Yeh, Shreya Havaldar, Aida Mostafazadeh Davani, Ying Lin, Brendan Kennedy, Mohammad Atari, Zahra Kamel, and Madelyn Mendlen. 2020. Moral foundations twitter corpus: A collection of 35k tweets annotated for moral sentiment. Social Psychological and Personality Science, 11(8):1057–1071.

Frederic R Hopp, Jacob T Fisher, Devin Cornell, Richard Huskey, and René Weber. 2021. The extended moral foundations dictionary (emfd): Development and applications of a crowd-sourced approach to extracting moral intuitions from text. Behavior research methods, 53(1):232–246.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR).

Ravi Iyer, Spassena Koleva, Jesse Graham, Peter Ditto, and Jonathan Haidt. 2012. Understanding libertarian morality: The psychological dispositions of selfidentified libertarians. PLOS ONE, 7(8):1–23.

Shomik Jain, Dan Calacci, and Ashia Wilson. 2024. "As an AI Language Model, Yes I Would Recommend Calling the Police": Norm inconsistency in LLM decision-making. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, pages 624– 633.

Liwei Jiang, Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, Jenny T. Liang, Sydney Levine, Jesse Dodge, Keisuke Sakaguchi, Maxwell Forbes, and Jack Hessel. 2025. Investigating machine moral judgement through the delphi experiment. Nature Machine Intelligence, 7(1):145–160.

Brendan Kennedy, Preni Golazizian, Jackson Trager, Mohammad Atari, Joe Hoover, Aida Mostafazadeh Davani, and Morteza Dehghani. 2023. The (moral) language of hate. PNAS Nexus, 2(7):pgad210.

Jonathan Kobbe, Ines Rehbein, Ioana Hulpus, and Heiner Stuckenschmidt. 2020. Exploring morality in argumentation. In Proceedings of the 7th Workshop on Argument Mining, pages 30–40.

Qing Lyu, Marianna Apidianaki, and Chris Callison-Burch. 2024. Towards faithful model explanation in NLP: A survey. Computational Linguistics, 50(2):657–723.

Musa Malik, Sungbin Youk, Frederic R. Hopp, Oliver Scott Curry, Marc Cheong, Mark Alfano, and René Weber. 2025. The extended morality as cooperation dictionary (emacd): A crowd-sourced approach via the moral narrative analyzer platform. Communication Methods and Measures, 19(3):201–231.

Panagiotis Metaxas, Eni Mustafaraj, Kily Wong, Laura Zeng, Megan O’Keefe, and Samantha Finn. 2015. What do retweets indicate? results from user survey and meta-review of research. In Proceedings ofthe International AAAI Conference on Web and Social Media, pages 658–661.

M. E. J. Newman. 2002. Assortative mixing in networks. Phys. Rev. Lett., 89:208701.

Tuan Dung Nguyen, Ziyu Chen, Nicholas George Carroll, Alasdair Tran, Colin Klein, and Lexing Xie. 2024. Measuring moral dimensions in social media with MFormer. In Proceedings of the International AAAI Conference on Web and Social Media, volume 18, pages 1134–1147.

Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. 2018. FiLM: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI Conference on Artificial Intelligence.

Vjosa Preniqi, Iacopo Ghinassi, Julia Ive, Charalampos Saitis, and Kyriaki Kalimeri. 2024. Moralbert: a finetuned language model for capturing moral values in social discussions. In Proceedings of the 2024 International Conference on Information Technology for Social Good, pages 433–442.

Ashwin Rao, Nazanin Sabri, Siyi Guo, Louiqa Raschid, and Kristina Lerman. 2025. Public health messaging on twitter during the covid-19 pandemic: Observational study. Journal ofMedical Internet Research, 27.

Ines Reinig, Maria Becker, Ines Rehbein, and Simone Ponzetto. 2024. A survey on modelling morality for text analysis. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 4136–4155, Bangkok, Thailand. Association for Computational Linguistics.

Maciej Skorski and Alina Landowska. 2025. The moral gap of large language models. arXiv preprint arXiv:2507.18523.

Jackson Trager, Francielle Vargas, Diego Alves, Matteo Guida, Mikel K. Ngueajio, Ameeta Agrawal, Yalda Daryani, Farzan Karimi Malekabadi, and Flor Miriam Plaza-del Arco. 2025. MFTCXplain: A multilingual benchmark dataset for evaluating the moral reasoning of LLMs through multi-hop hate speech explanation. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 15709–15740, Suzhou, China. Association for Computational Linguistics.

Jackson Trager, Alireza S. Ziabari, Aida Mostafazadeh Davani, Preni Golazizian, Farzan Karimi-Malekabadi, Ali Omrani, Zhihe Li, Brendan Kennedy, Nils Karl Reimer, and Melissa Reyes. 2022. The moral foundations reddit corpus. arXiv preprint arXiv:2208.05545.

Jay J. Van Bavel, Steve Rathje, Elizabeth Harris, Claire Robertson, and Anni Sternisko. 2021. How social media shapes polarization. Trends in Cognitive Sciences, 25(11):913–916.

Francielle Vargas, Jackson Trager, Diego Alves, Matteo Guida, Surendrabikram Thapa, Berk Atıl, Daryna Dementieva, Andrew J Smart, and Ameeta Agrawal. 2026. Self-explaining hate speech detection with moral rationales. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 34109– 34131, San Diego, California, United States. Association for Computational Linguistics.

Lorenzo Zangari, Candida M. Greco, Davide Picca, and Andrea Tagarelli. 2025. ME2-BERT: Are events and emotions what you need for moral foundation prediction? In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 9516– 9532, Abu Dhabi, UAE. Association for Computational Linguistics.

Caleb Ziems, Jane Yu, Yi-Chia Wang, Alon Halevy, and Diyi Yang. 2022. The moral integrity corpus: A benchmark for ethical dialogue systems. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3755–3773.

## A Additional Experimental Details

## A.1 Experimental Dataset Details

Tables 8 and 9 report label prevalence for the indomain training datasets, while Table 10 summarizes label prevalence for the out-of-domain evaluation datasets.

## A.2 Details of eMACDscore for MAC Grounding

Our MAC grounding signal is derived from eMACDscore, a Python-based moral mining tool introduced by Malik et al. (2025) as part of the extended Morality-as-Cooperation Dictionary (eMACD). Built upon the Morality-as-Cooperation framework (Curry, 2016), eMACD provides lexicon-based scores across seven cooperation domains: family, group, reciprocity, heroism, deference, fairness, and property. The lexicon itself was constructed through crowd-sourced annotation using the Moral Narrative Analyzer (MoNA) platform. We use eMACDscore to automatically generate MAC supervision signals for each training instance. Specifically, for every sample and MAC domain, the tool outputs a probability score $\hat { p } _ { i }$ and a sentiment-polarity score ${ \hat { s } } _ { i } ,$ which are combined into the MAC vector $g \in \mathbb { R } ^ { 7 }$ . We adopt eMACDscore because, to the best of our knowledge, there is currently no publicly available human-annotated MAC benchmark dataset.

Importantly, eMACD is developed independently of Moral Foundations Theory (MFT). Rather than being derived from MFT labels or annotations, it is grounded in a distinct theoretical framework and constructed through an independent crowdsourcing process. As a result, the MAC supervision provides a complementary moral signal instead of simply repackaging the MFT information already available to the model. We refer readers to Malik et al. (2025) for full details regarding the dictionary construction process, annotation protocol, and validation experiments.

## A.3 Baseline Implementation Details

For baseline comparisons, we reproduce each baseline under its original released configuration, including matching the random seed specified in their code, to ensure a fair comparison.

For evaluation, we follow the metrics commonly used in the original studies of each dataset, reporting macro-AUC and macro-F1. For F1, per-label decision thresholds are tuned on a shared validation set and applied unchanged to the test set. We release the trained checkpoint so that all reported scores can be exactly reproduced by running inference on the released model.

MFORMER. MFORMER (Nguyen et al., 2024) is a transformer-based moral foundation classification framework built on ROBERTA-base. The model trains five independent binary classifiers, each corresponding to one moral foundation in a one-vs-rest setting. It is trained on a mixture of annotated datasets spanning social media, news, and online discussion domains. During inference, predictions from the five classifiers are combined to produce the final multi-label moral output.

MOVA. MOVA (Chen et al., 2025) is a prompting-based framework for moral value classification using instruction-tuned GPT-4o mini. It formulates moral foundation detection as a multilabel generation task, where a single prompt is used to predict all moral dimensions jointly. The framework does not require task-specific fine-tuning and instead relies on prompt engineering to guide structured moral predictions. In our experiments, we use the official prompt templates released by the authors for reproduction, and treat MOVA as our primary zero-shot baseline for LLM-based moral reasoning.

Qwen-32B-Instruct. We include Qwen-32B-Instruct as a larger zero-shot LLM baseline. We use the same moral-label definitions and evaluation protocol as for the other models, without task-specific fine-tuning. This comparison provides an additional reference for the trade-off between model scale and task-specific supervision.

Fine-tuned GPT-4o mini. Following Chen et al. (2025), we include a supervised fine-tuned variant of GPT-4o mini trained on 30K labeled instances sampled from the MFORMER training corpus. The model is fine-tuned to generate structured moral labels directly from input text without rationale supervision or auxiliary moral signals. This baseline serves as a decoder-only LLM adapted for supervised moral classification. This setup allows us to evaluate whether fine-tuning a LLM improves over prompting-based inference under comparable supervision data.

MORALBERT. MORALBERT (Preniqi et al., 2024) is a BERT-based framework for polarityaware moral classification across ten virtue–vice dimensions. Unlike standard MFT classifiers that predict only foundation presence, MORALBERT explicitly models both positive (virtue) and negative (vice) polarities using a multi-label classification objective. Its explicit polarity modeling provides a relevant comparison for evaluating finegrained moral representation learning.

<table><tr><td>Dataset</td><td>Care</td><td>Fairness</td><td>Loyalty</td><td>Authority</td><td>Purity</td><td>Avg. Tokens</td><td>P90. Tokens</td><td>Total Size</td></tr><tr><td>MFTC</td><td>40.7%</td><td>41.8%</td><td>37.0%</td><td>41.5%</td><td>29.7%</td><td>31.9</td><td>50</td><td>13,995</td></tr><tr><td>MFRC</td><td>35.5%</td><td>44.2%</td><td>19.4%</td><td>32.5%</td><td>17.4%</td><td>47.5</td><td>92</td><td>7,155</td></tr><tr><td>News</td><td>26.5%</td><td>32.4%</td><td>33.1%</td><td>35.4%</td><td>30.4%</td><td>31.1</td><td>50</td><td>12,920</td></tr><tr><td>MFTCXplain</td><td>37.3%</td><td>36.0%</td><td>27.2%</td><td>23.2%</td><td>18.6%</td><td>49.6</td><td>85</td><td>3,245</td></tr><tr><td>Overall</td><td>39.9%</td><td>40.3%</td><td>33.4%</td><td>38.3%</td><td>27.0%</td><td>36.2</td><td>63</td><td>37,315</td></tr></table>

Table 8: Distribution of moral virtue dimensions and input length statistics across in-domain training datasets. The five moral dimensions correspond to virtue–vice axes. Mean token and P90 token denote the average and 90th-percentile token lengths, respectively.

<table><tr><td>Label</td><td>Percentage</td></tr><tr><td>Care</td><td>8.91%</td></tr><tr><td>Harm</td><td>29.34%</td></tr><tr><td>Fairness</td><td>13.50%</td></tr><tr><td>Cheating</td><td>24.35%</td></tr><tr><td>Loyalty</td><td>15.04%</td></tr><tr><td>Betrayal</td><td>13.50%</td></tr><tr><td>Authority</td><td>10.29%</td></tr><tr><td>Subversion</td><td>13.22%</td></tr><tr><td>Purity</td><td>7.95%</td></tr><tr><td>Degradation</td><td>11.31%</td></tr><tr><td>Hate Speech</td><td>38.92%</td></tr><tr><td>Total size</td><td>3,245</td></tr></table>

Table 9: Label distribution of MFTCXplain across moral polarity dimensions and hate speech annotations.

## A.4 Evaluation Metric Definitions

We report AUC and F1 scores for both hate speech detection and moral classification tasks.

Hate Speech Detection. For hate speech classification, we use the standard binary F1 score:

$$
F 1 = \frac { 2 \cdot \mathrm { P r e c i s i o n } \cdot \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } } ,
$$

where precision and recall are computed with respect to the hate speech class. We additionally report AUC, which measures the area under the receiver operating characteristic (ROC) curve and evaluates the model’s ranking quality across all classification thresholds.

Moral Classification. Moral classification is formulated as a multi-label prediction task, where each sample may express multiple moral foundations simultaneously. Let L denote the set of moral labels. For each label $l \in L$ , we compute a binary F1 score by treating the presence or absence of l as a separate binary classification problem:

$$
F 1 _ { l } = \frac { 2 \cdot \mathrm { P r e c i s i o n } _ { l } \cdot \mathrm { R e c a l l } _ { l } } { \mathrm { P r e c i s i o n } _ { l } + \mathrm { R e c a l l } _ { l } } .
$$

The reported Macro-F1 is then obtained by averaging over all moral labels:

$$
\mathrm { M a c r o - F 1 } = \frac { 1 } { | L | } \sum _ { l \in L } F 1 _ { l } .
$$

Similarly, AUC is computed as the average AUC across all moral labels. Because moral prediction is inherently multi-label and many instances activate only a small subset of moral dimensions, Macro-F1 can sometimes overestimate overall performance. In particular, labels with sparse positive instances or easier decision boundaries may inflate the averaged score despite limited holistic understanding of moral content. We therefore report both Macro-F1 and AUC to provide a more comprehensive assessment of model behaviour.

Unlike AUC, which is threshold-free, F1 depends on a decision threshold. To ensure a fair comparison, we adopt a unified threshold-selection protocol across all models. Using a held-out validation set shared by all methods, we select, for each label l, the threshold that maximizes the validation (F1\_l), and apply it unchanged to the test set. This isolates differences in model quality from differences in threshold choice.

Rationale Plausibility We compare the modelselected rationale tokens $R _ { k }$ against the humanannotated rationale spans $H _ { k }$ for each instance k. We report IoU-F1 and Token-F1:

<table><tr><td>Dataset</td><td>Authority</td><td>Care</td><td>Fairness</td><td>Loyalty</td><td>Purity</td><td>Avg. Tokens</td><td>P90. Tokens</td><td>Total Size</td></tr><tr><td>VIG</td><td>14.78%</td><td>27.83%</td><td>14.78%</td><td>13.91%</td><td>14.78%</td><td>17.0</td><td>19</td><td>132</td></tr><tr><td>SC</td><td>9.72%</td><td>44.44%</td><td>16.92%</td><td>17.80%</td><td>6.79%</td><td>12.1</td><td>17</td><td>2,9239</td></tr><tr><td>ARG</td><td>14.37%</td><td>41.56%</td><td>16.56%</td><td>8.12%</td><td>19.69%</td><td>69.6</td><td>120</td><td>320</td></tr><tr><td>MIC</td><td>17.15%</td><td>51.19%</td><td>20.72%</td><td>19.43%</td><td>10.86%</td><td>10.2</td><td>14</td><td>1,1375</td></tr><tr><td>HateBR</td><td>6.90%</td><td>28.97%</td><td>22.88%</td><td>4.21%</td><td>22.30%</td><td>23.5</td><td>47</td><td>5,040</td></tr></table>

Table 10: Distribution of moral virtue dimensions and input length statistics across out-of-domain evaluation datasets. The five moral dimensions correspond to virtue–vice axes. Avg. token and P90. token denote the average and 90th-percentile token lengths, respectively.

$$
\mathrm { I o U \mathrm { - } F 1 } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \frac { | R _ { k } \cap H _ { k } | } { | R _ { k } \cup H _ { k } | } ,
$$

$$
\mathrm { T o k e n - F 1 } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \frac { 2 | R _ { k } \cap H _ { k } | } { | R _ { k } | + | H _ { k } | } .
$$

We also report AUPRC to evaluate how well token-importance scores rank gold rationale tokens under severe class imbalance.

Rationale Faithfulness Following DeYoung et al. (2020), we report comprehensiveness and sufficiency. Let $p ( \cdot ) _ { y }$ denote the model probability for the gold label $y ,$ and let $e _ { k }$ denote the extracted rationale for instance $k .$

Comprehensiveness measures the confidence drop after removing rationale tokens:

$$
{ \mathrm { C o m p } } = { \frac { 1 } { N } } \sum _ { k = 1 } ^ { N } ( p ( x _ { k } ) _ { y } - p ( x _ { k } \setminus e _ { k } ) _ { y } ) .
$$

Sufficiency evaluates whether the rationale alone preserves enough evidence:

$$
\mathrm { S u f f } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \bigl ( p ( x _ { k } ) _ { y } - p ( e _ { k } ) _ { y } \bigr ) .
$$

Higher comprehensiveness indicates that the rationale contributes substantially to the prediction; lower sufficiency indicates that the rationale alone retains most of the predictive signal.

## A.5 Experimental Setup

All experiments were conducted on two NVIDIA A40 GPUs using PyTorch 2.10.0 (CUDA 12.8), Hugging Face Transformers 5.0.0, and PEFT 0.18.1. CHARM is built on LLaMA-3.1-8B and trained in two stages. In Stage 1, we attach LoRA adapters (rank 16, α = 32, dropout .1) to all attention and feed-forward projection matrices, and train the moral classifier and token scorer with a binary cross-entropy moral loss. The LoRA adapters introduce .56% trainable parameters of the full backbone, keeping the model lightweight. In Stage 2, we freeze the Stage 1 LoRA encoder and train the fusion heads, optimizing a combined objective that sums the moral loss with hate-speech, rationale, and total-variation (TV) smoothness terms, weighted by $\lambda _ { \mathrm { h a t e } } , \lambda _ { \mathrm { r a t } }$ , and $\lambda _ { \mathrm { T V } }$ (initialized to .1, .2, and .05, respectively; the moral term has a fixed weight of 1.0). These three λ values are learnable via log parameterisation. Both stages use the AdamW optimizer with a learning rate of $2 \times 1 0 ^ { - 4 }$ and a maximum sequence length of 128; we adopt these values following common practice without extensive tuning. Stage 1 is trained for 3 epochs and Stage 2 for 5 epochs, with a total training time of approximately one hour.

Stage 1 trains on MFTCXplain together with a 30% subsample of the MFRC, MFTC, and News training pools, while Stage 2 trains on MFTCXplain only. The 30% subsampling uses a fixed seed (42) with distribution-aware weighting toward the MFTCXplain label distribution, followed by a 9:1 stratified train/validation split and the same train/test split as MFORMER. The same MFTCXplain train/validation/test partition is used in both stages, so that no test instance is seen during either stage of training.

## A.6 Cross-Validation Robustness

To assess the stability of CHARM, we additionally perform cross-validation on the two smallest datasets, reporting the mean and standard deviation of macro-F1 and AUC across folds (Table 11). This is an independent robustness check using a different partitioning from the shared-split protocol in Table 3, and is therefore not directly comparable to the main results.

AUC remains stable across runs (std ≤ .033), consistent with our observation that threshold-free ranking is robust under variation. Macro-F1 exhibits larger variance, as expected given the small sample sizes (ARG: $n { = } 3 2 0 ;$ VIG: $n { = } 1 3 2 )$ and F1’s sensitivity to threshold selection.

<table><tr><td>Dataset</td><td>Macro-F1</td><td>AUC</td></tr><tr><td>ARG</td><td> $. 4 8 9 \pm . 0 7 9$ </td><td> $. 8 6 4 \pm . 0 2 6$ </td></tr><tr><td>VIG</td><td> $. 5 3 7 \pm . 0 4 8$ </td><td> $. 8 8 9 \pm . 0 3 3$ </td></tr></table>

Table 11: Stability of CHARM over 5-fold crossvalidation on the small-sample datasets (mean ± std).

## A.7 Ablation Results on AUC

Table 12 reports AUC-based ablation results across both in-domain and out-of-domain datasets. Overall, removing rationale supervision consistently degrades performance on most datasets, particularly under cross-domain evaluation, suggesting that rationale-guided representations improve robustness beyond token-level alignment.

The removal of MAC-enhanced representations leads to the largest performance drops on socially grounded datasets such as SC and MIC, indicating that cooperation-oriented moral structure contributes more strongly in open-ended social reasoning settings. In contrast, the hate-speech auxiliary objective primarily improves performance on politically charged or toxicity-related datasets, especially MFTC and MFRC.

These results suggest that the three components contribute complementary inductive biases: rationale supervision improves interpretability and robustness, MAC representations improve socialgeneralization capacity, and hate-speech supervision improves sensitivity to morally salient harmful content.

## A.8 Training-Corpus Contribution

We conduct a leave-one-corpus-out analysis to quantify the contribution of MFRC, MFTC, News, and MFTCXplain within CHARM. Each variant uses the same model configuration and evaluation protocol as the full model. This analysis measures the contribution of each corpus within CHARM, but it is not a controlled architecture-only comparison with the baselines.

As shown in Tables 13 and 14, the full configuration achieves the strongest average performance. MFRC makes the largest average contribution, while MFTC and News provide complementary domain coverage. Removing MFTCXplain lowers average performance and produces the clearest declines on MFTCXplain and HateBR, confirming the value of its polarity, rationale, and hate-speech supervision. Variation across individual datasets suggests corpus-specific interactions rather than uniform gains from every training source. Overall, the results reflect the complementary value of corpus diversity and richer supervision.

## A.9 Qualitative Rationale Analysis

We select cases in which CHARM exactly matches the reference foundation set, whereas the model without rationale supervision misses at least one reference foundation. The examples are therefore intended to clarify how rationale supervision affects evidence use rather than provide an additional performance evaluation. For visualization, subword rationale probabilities are averaged within each reconstructed word,

$$
p ( w ) = \frac { 1 } { | T ( w ) | } \sum _ { t \in T ( w ) } p _ { t } ,
$$

where $T ( w )$ is the set of subword tokens associated with word w. A threshold of $p ( w ) > . 5$ is used only for visualization and does not affect morallabel prediction.

As illustrated in Table 15, CHARM often retains evidence related to suffering, assistance, group loyalty, authority, and contamination. In Case 1, repeated references to “teamwork,” “her team,” and “everyone pitching in” make the Loyalty/Betrayal signal explicit, whereas the ablated model predicts Authority/Subversion. In Case 4, “Sandy victims” and their need for “attention and resources” emphasize suffering and assistance, helping CHARM recover Care/Harm. These examples complement the quantitative rationale ablation by showing how rationale supervision can preserve label-relevant evidence.

## A.10 Per-Foundation Performance Analysis

Figure 6 presents dataset-level Macro-F1 and Figure 7 AUC comparisons under both in-domain and out-of-domain settings. Across datasets, CHARM maintains relatively stable degradation under distribution shift, with larger robustness gaps emerging primarily on short-text and socially ambiguous datasets such as SC and MIC.

Compared with in-domain evaluation, out-ofdomain performance decreases are generally moderate, suggesting that the model captures transferable moral representations rather than relying solely on dataset-specific lexical cues. Notably, AUC scores remain comparatively stable even when F1 declines, indicating that the model preserves ranking consistency under domain shift despite threshold-sensitive classification degradation.

<table><tr><td>Variant</td><td>MFTCXplain10d</td><td>MFTCXplain5d</td><td>MFTC</td><td>MFRC</td><td>News</td><td>ARG</td><td>SC</td><td>MIC</td><td>VIG</td></tr><tr><td>baseline</td><td>.81</td><td>.77</td><td>.85</td><td>.84</td><td>.75</td><td>.78</td><td>.69</td><td>.68</td><td>.82</td></tr><tr><td>w/o hate speech</td><td>.83</td><td>.79</td><td>.88</td><td>.91</td><td>.84</td><td>.84</td><td>.75</td><td>.71</td><td>.87</td></tr><tr><td>w/o MAC</td><td>.83</td><td>.79</td><td>.84</td><td>.84</td><td>.73</td><td>.80</td><td>.79</td><td>.75</td><td>.82</td></tr><tr><td>w/o rationale</td><td>.82</td><td>.78</td><td>.87</td><td>.90</td><td>.82</td><td>.86</td><td>.77</td><td>.73</td><td>.89</td></tr><tr><td>CHARM</td><td>.83</td><td>.79</td><td>.87</td><td>.91</td><td>.83</td><td>.87</td><td>.77</td><td>.73</td><td>.89</td></tr></table>

Table 12: AUC across in-domain and out-of-domain datasets in ablation studies. Best result per dataset in bold.

<table><tr><td>Variant</td><td>Avg.</td><td>MFTC</td><td>MFRC</td><td>News</td><td>ARG</td><td>VIG</td><td>SC</td><td>MIC</td><td>MFTCXpl.</td><td>HateBR</td></tr><tr><td>CHARM</td><td>.83</td><td>.87</td><td>.91</td><td>.83</td><td>.87</td><td>.89</td><td>.77</td><td>.73</td><td>.79</td><td>.82</td></tr><tr><td>w/o MFRC</td><td>.63</td><td>.66</td><td>.55</td><td>.64</td><td>.59</td><td>.53</td><td>.56</td><td>.54</td><td>.84</td><td>.78</td></tr><tr><td>w/o MFTC</td><td>.70</td><td>.55</td><td>.78</td><td>.67</td><td>.73</td><td>.61</td><td>.69</td><td>.61</td><td>.85</td><td>.80</td></tr><tr><td>w/o News</td><td>.71</td><td>.70</td><td>.76</td><td>.52</td><td>.72</td><td>.69</td><td>.68</td><td>.62</td><td>.86</td><td>.81</td></tr><tr><td>w/o MFTCXplain</td><td>.80</td><td>.90</td><td>.90</td><td>.84</td><td>.82</td><td>.89</td><td>.74</td><td>.70</td><td>.70</td><td>.69</td></tr></table>

Table 13: AUC results for the leave-one-corpus-out analysis. The average is computed across all nine evaluation datasets. Best results per dataset are shown in bold.

![](images/e142a2c1bbb4055cbb8b9bfd694e6c98e39586ac375af475e28b35b67ba05189.jpg)  
Figure 5: Distribution of users across the ten polarityaware moral dimensions.

## B Moral Alignment and Endorsement Prediction

## B.1 Endorsement Network Construction

We construct a directed endorsement graph from the COVID-19 Twitter dataset introduced by Chen et al. (2020). Retweets are treated as endorsement signals following prior work (Metaxas et al., 2015), while mentions and quote tweets are excluded because they may reflect disagreement or contextual commentary.

For endorsement prediction, positive samples correspond to directed retweet interactions occurring more than five times between two users. Negative samples are constructed using a two-hop nonendorsement constraint to avoid trivially disconnected user pairs. The construction process is illustrated in Figure 4.

Figure 5 shows the distribution of users across the ten polarity-aware moral dimensions. Positive moral poles are substantially denser and more stable than negative poles, while several vice dimensions remain highly sparse in large-scale Twitter discourse. To reduce sparsity and improve robustness, we therefore collapse virtue and vice polarity scores into five foundation-level moral representations when constructing user-level moral profiles for endorsement analysis.

## B.2 Manual Validation of the COVID-19 Case Study

To assess the validity of the moral measurements underlying the downstream case study, we evaluate CHARM’s predictions on a stratified sample of 100 COVID-19 tweets. Tweets are stratified by predicted moral foundation and moral intensity to ensure coverage of different predicted moral profiles. Three annotators independently assign polarity-level moral labels while blinded to CHARM’s predictions. A label is retained in the human reference when selected by at least two annotators. Positive labels account for 17.6% of all tweet–label decisions. Given this class imbalance, we report observed agreement together with Fleiss’ κ, prevalence-adjusted bias-adjusted kappa (PABAK), and Gwet’s AC1. We additionally evaluate the alignment between CHARM’s continuous predictions and the majority-voted human reference using AUC and average precision (AP).

<table><tr><td>Variant</td><td>Avg.</td><td>MFTC</td><td>MFRC</td><td>News</td><td>ARG</td><td>VIG</td><td>SC</td><td>MIC</td><td>MFTCXpl.</td><td>HateBR</td></tr><tr><td>CHARM</td><td>.57</td><td>.72</td><td>.72</td><td>.55</td><td>.54</td><td>.67</td><td>.46</td><td>.50</td><td>.59</td><td>.38</td></tr><tr><td>w/o MFRC</td><td>.42</td><td>.58</td><td>.38</td><td>.49</td><td>.37</td><td>.34</td><td>.33</td><td>.40</td><td>.53</td><td>.37</td></tr><tr><td>w/o MFTC</td><td>.49</td><td>.51</td><td>.61</td><td>.53</td><td>.51</td><td>.44</td><td>.44</td><td>.43</td><td>.53</td><td>.38</td></tr><tr><td>w/o News</td><td>.51</td><td>.65</td><td>.61</td><td>.43</td><td>.52</td><td>.57</td><td>.43</td><td>.45</td><td>.58</td><td>.38</td></tr><tr><td>w/o MFTCXplain</td><td>.54</td><td>.77</td><td>.67</td><td>.61</td><td>.54</td><td>.59</td><td>.43</td><td>.45</td><td>.47</td><td>.34</td></tr></table>

Table 14: F1 results for the leave-one-corpus-out analysis. The average is computed across all nine evaluation datasets. Best results per dataset are shown in bold.
<table><tr><td>Case</td><td>Text</td><td>CHARM</td><td>w/o rationale</td></tr><tr><td>1</td><td>Minor or major, it&#x27;s all teamwork. You don&#x27;t want to be part of her team. It&#x27;s everyone pitching in with whatever, whenever, just because it makes everyone&#x27;s lives a bit better.</td><td>Care/Harm Fairness/Cheating Loyalty/Betrayal</td><td>Care/Harm Fairness/Cheating Authority/Subversion</td></tr><tr><td>2</td><td>Director James Comey was critical of Clinton&#x27;s use of the server but said he would not recommend pursuing criminal charges.</td><td>Authority/Subversion</td><td>Fairness/Cheating</td></tr><tr><td>3</td><td>The Katana is so sharp, they can simply cut the coronavirus if it infects someone, without any damage whatsoever to the infectee.</td><td>Fairness/Cheating Purity/Degradation</td><td>Fairness/Cheating</td></tr><tr><td>4</td><td>Bloomberg is another stupid Liberal politician; Sandy victims deserve attention and resources, not a marathon.</td><td>Care/Harm Fairness/Cheating Loyalty/Betrayal Authority/Subversion</td><td>Fairness/Cheating Loyalty/Betrayal Authority/Subversion</td></tr></table>

Table 15: Qualitative comparison of CHARM and the model without rationale supervision. CHARM matches the reference foundation set in each example, whereas the ablated model misses at least one reference foundation.

Annotation reliability. Across all tweet–label decisions, observed agreement is .830, while Fleiss κ is .415, PABAK is .660, and Gwet’s AC1 is .760. The discrepancy between raw agreement and Fleiss’ κ is most pronounced for low-prevalence labels. For example, only two tweets receive a majorityvoted Fairness label, such that its high observed agreement primarily reflects agreement on negative cases. The prevalence-adjusted measures therefore provide complementary evidence of annotation reliability under the sparse multilabel setting.

Human–model alignment. Against the majorityvoted human reference, CHARM achieves a polarity-level macro AUC of .842 with a 95% bootstrap confidence interval of [.801, .876], together with a macro average precision of .555. As shown in Table 16, performance varies across moral categories. The results indicate that CHARM’s continuous predictions generally align with human judgments in the COVID-19 domain, providing additional support for their use in the downstream endorsement analyses.

Limitations of label-level estimates. Label-level results for low-prevalence categories should be interpreted cautiously. In particular, Fairness, Loyalty, Purity, and Betrayal contain only 2, 5, 5, and 6 majority-voted positive examples, respectively. Consequently, high AUC or agreement values for these categories can be unstable and may be driven partly by the large number of negative instances. We therefore emphasize the aggregate validation results rather than drawing strong conclusions from individual rare-category estimates.

## B.3 Backbone Comparison Results

Table 17 compares CHARM with Qwen3-8B across both in-domain and out-of-domain datasets. CHARM achieves consistently stronger performance on most datasets, with particularly large gains on MFTC, MFRC, and News. The improvements are also stable under out-of-domain evaluation, especially on ARG and SC, suggesting that the combination of rationale supervision and MAC grounding provides more robust moral representations than direct prompting alone. While Qwen3- 8B remains competitive on VIG, overall results indicate that CHARM offers a stronger balance between efficiency and predictive performance.

<table><tr><td>Label</td><td>Pos.</td><td>Agreement</td><td>Fleiss&#x27;κ</td><td>PABAK</td><td>AUC</td><td>AP</td></tr><tr><td>Care</td><td>29</td><td>.79</td><td>.49</td><td>.57</td><td>.86</td><td>.73</td></tr><tr><td>Harm</td><td>37</td><td>.64</td><td>.25</td><td>.28</td><td>.71</td><td>.64</td></tr><tr><td>Fairness</td><td>2</td><td>.87</td><td>.03</td><td>.75</td><td>.95</td><td>.23</td></tr><tr><td>Cheating</td><td>20</td><td>.79</td><td>.30</td><td>.57</td><td>.90</td><td>.69</td></tr><tr><td>Loyalty</td><td>5</td><td>.89</td><td>.35</td><td>.79</td><td>.87</td><td>.44</td></tr><tr><td>Betrayal</td><td>6</td><td>.85</td><td>.19</td><td>.71</td><td>.71</td><td>.14</td></tr><tr><td>Authority</td><td>11</td><td>.81</td><td>.27</td><td>.63</td><td>.77</td><td>.49</td></tr><tr><td>Subversion</td><td>27</td><td>.76</td><td>.43</td><td>.52</td><td>.83</td><td>.72</td></tr><tr><td>Purity</td><td>5</td><td>.98</td><td>.80</td><td>.96</td><td>1.00</td><td>1.00</td></tr><tr><td>Degradation</td><td>12</td><td>.91</td><td>.58</td><td>.83</td><td>.83</td><td>.48</td></tr><tr><td>Overall</td><td>一</td><td>.83</td><td>.42</td><td>.66</td><td>.84</td><td>.56</td></tr></table>

Table 16: Annotation reliability and alignment between CHARM predictions and majority-voted human labels on 100 COVID-19 tweets. Pos. denotes the number of majority-voted positive examples, and AP denotes average precision. Overall Gwet’s AC1 is .760.

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Qwen3-8B</td><td colspan="2">CHARM</td></tr><tr><td>F1</td><td>AUC</td><td>F1</td><td>AUC</td></tr><tr><td>MFTC</td><td>.57</td><td>.76</td><td>.72</td><td>.87</td></tr><tr><td>MFRC</td><td>.46</td><td>.76</td><td>.72</td><td>.91</td></tr><tr><td>News</td><td>.39</td><td>.66</td><td>.55</td><td>.83</td></tr><tr><td>ARG</td><td>.48</td><td>.79</td><td>.54</td><td>.87</td></tr><tr><td>SC</td><td>.38</td><td>.70</td><td>.46</td><td>.77</td></tr><tr><td>MIC</td><td>.40</td><td>.67</td><td>.50</td><td>.73</td></tr><tr><td>VIG</td><td>.63</td><td>.90</td><td>.67</td><td>.89</td></tr></table>

Table 17: Performance comparison between Qwen3-8B and CHARM across datasets in terms of Macro-F1 and AUC.

## B.4 User Moral Profile Construction

User-level moral profiles are constructed by aggregating tweet-level moral predictions inferred by CHARM over the five MFT foundations. Virtue and vice polarity scores are collapsed into unified foundation-level moral intensity scores. For each user u, moral scores are aggregated over highintensity moral tweets:

$$
m _ { u } ^ { ( k ) } = \frac { 1 } { | T _ { u } ^ { ( k ) } | } \sum _ { t \in T _ { u } ^ { ( k ) } } s _ { t } ^ { ( k ) } ,
$$

where $s _ { t } ^ { ( k ) }$ denotes the predicted moral intensity for foundation k, and $T _ { u } ^ { ( k ) }$ contains tweets whose intensity exceeds the user-specific 90th percentile threshold. Figure 5 visualizes the distribution of users across the ten polarity-aware moral dimensions.

## B.5 Feature Construction

Table 18 summarizes the features used for endorsement prediction.

Pairwise moral similarity is computed using cosine similarity:

$$
\mathrm { c o s \_ s i m } ( u , v ) = \frac { \mathbf { m } _ { u } ^ { \top } \mathbf { m } _ { v } } { \| \mathbf { m } _ { u } \| \| \mathbf { m } _ { v } \| } ,
$$

and Euclidean distance:

$$
\mathrm { e u c l i d e a n \_ d i s t } ( u , v ) = \| \mathbf { m } _ { u } - \mathbf { m } _ { v } \| _ { 2 } .
$$

Per-foundation moral differences are defined as:

$$
\mathrm { d i f f } ^ { ( k ) } ( u , v ) = m _ { u } ^ { ( k ) } - m _ { v } ^ { ( k ) } .
$$

Behavioral statistics are aggregated using:

$$
\mathrm { a c t i v i t y \_ m e a n } ( u ) = \frac { 1 } { \left| T _ { u } \right| } \sum _ { t \in T _ { u } } a _ { t } ,
$$

where $a _ { t }$ denotes tweet-level engagement statistics such as replies, favorites, quotes, and retweets.

## B.6 Endorsement Prediction Model Training Details

We train a binary LightGBM classifier using moralprofile, interaction, behavioral, and semantic features. The model is trained with a binary logistic objective and evaluated across five random seeds.

We use a learning rate of .05, maximum tree depth of 6, 200 boosting rounds, and 64 leaves per tree. Early stopping is applied based on validationset AUC with a patience of 20 rounds. To reduce overfitting, we apply both feature and row subsampling with rates of .8 together with L2 regularization coefficient 1.0.

<table><tr><td>Feature Category</td><td>Level</td><td>Feature</td><td>Description</td></tr><tr><td>Moral Features</td><td>Author / Retweeter</td><td>Care, Fairness, Loyalty, Authority, Purity</td><td>User-level moral intensity scores aggregated over the five moral foundations using polarity-collapsed representations.</td></tr><tr><td>Moral Interaction</td><td>User Pair</td><td>Moral Similarity</td><td>Cosine similarity between author and retweeter moral representations.</td></tr><tr><td>Features</td><td>User Pair</td><td>Moral Distance</td><td>Euclidean distance between author and retweeter moral representations.</td></tr><tr><td></td><td>User Pair</td><td>Foundation-Level Moral Difference</td><td>Per-foundation differences between retweeter and author moral scores.</td></tr><tr><td>Behavioral</td><td>User-Level</td><td>Social Connectivity</td><td>User engagement statistics including follower count, friend count, and favourites count.</td></tr><tr><td>Features</td><td>User-Level</td><td>Interaction Activity</td><td>Average frequencies of replies, quotes, and retweets across authored tweets.</td></tr><tr><td></td><td>User Pair</td><td>Relative Influence Gap</td><td>Differences in user influence and activity between author and retweeter.</td></tr><tr><td>Semantic Features User Pair</td><td></td><td>Biography Similarity</td><td>Cosine similarity and Euclidean distance between user biography embeddings.</td></tr></table>

Table 18: Feature definitions for endorsement prediction.

All continuous features are z-score normalized before training. Semantic similarity features are computed using cosine similarity between userlevel sentence embeddings, while moral similarity features are computed using cosine similarity between aggregated user moral profiles inferred by CHARM. Reported results are averaged across all runs.

## B.7 Per-Foundation Moral Alignment Distributions

Figure 8 shows per-foundation distributions of moral divergence (|retweeter − author|) and overall cosine similarity for endorsed and non-endorsed pairs. Across all five foundations, endorsed pairs exhibit tighter moral alignment — smaller author– retweeter divergence — than non-endorsed pairs, confirming that homophily holds at the individualfoundation level. The separation is most pronounced for loyalty and fairness, consistent with the assortativity coefficients in Figure 3c.

## B.8 Moral Assortativity Computation

We measure moral homophily using Newman assortativity for continuous attributes. Given a directed network with edges $( i \to j ) \in E$ , each edge is associated with source and target moral scores for a specific foundation. The assortativity coefficient is computed as the Pearson correlation across edges:

$$
r \ = \ { \frac { \sum _ { ( i , j ) \in E } ( x _ { i } - { \bar { x } } _ { s } ) ( y _ { j } - { \bar { y } } _ { t } ) } { \sqrt { \sum _ { ( i , j ) \in E } ( x _ { i } - { \bar { x } } _ { s } ) ^ { 2 } } { \sqrt { \sum _ { ( i , j ) \in E } ( y _ { j } - { \bar { y } } _ { t } ) ^ { 2 } } } } }
$$

where $\bar { x } _ { s }$ and $\bar { y } _ { t }$ denote the edge-weighted means of source and target attributes. We compute assortativity separately for each moral foundation. Statistical significance is assessed using a degreepreserving null model in which edges are randomly rewired while preserving each node’s in-degree and out-degree distributions.

![](images/72bc0f66b242d4fc07c0afa846f807fe17441c915df28e2dfe9c3c9e76a4c3be.jpg)  
Figure 6: In-domain evaluation: F1 and AUC across datasets.

![](images/0c8320aa06ec38424eab6693ede37cdd8254da88180b34f067c2174948b6fca5.jpg)  
Figure 7: Out-of-domain evaluation: F1 and AUC across datasets.

![](images/08e6aa3eec11f106996b3984028ee98523d8c5dbcd7fa1b7eb11885f66e8e0a1.jpg)  
Figure 8: Per-foundation KDE distributions of |retweeter moral score − author moral score| for endorsed (positive) and non-endorsed (negative) pairs, across the five MFT foundations and overall cosine similarity (bottom right). Endorsed pairs consistently exhibit smaller per-foundation divergence and higher cosine similarity.