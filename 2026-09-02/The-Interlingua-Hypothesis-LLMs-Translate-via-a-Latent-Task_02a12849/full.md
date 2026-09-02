# The Interlingua Hypothesis: LLMs Translate via a Latent Task-agnostic Feature Space

Jacob Brinton<sup>1</sup> Jannik Brinkmann<sup>2</sup> Mark Crovella<sup>1</sup> Aaron Mueller<sup>1</sup>

<sup>1</sup>Boston University <sup>2</sup>Technische Universität Clausthal jbrin@bu.edu, jannik.brinkmann@uni-mannheim.de, crovella@bu.edu, amueller@bu.edu

## Abstract

Large language models (LLMs) have recently demonstrated improved machine translation performance over strong supervised baselines. This raises questions as to what mechanisms underlie how LLMs perform machine translation between languages. Motivated by recent interpretability findings—namely, that LLMs use massively multilingual latent feature representations to perform language modeling—we propose the interlingua hypothesis. The hypothesis holds that language models translate by reading a source sentence into a latent feature space, and generate a target sentence by reading from the latent feature space. We show three lines of evidence in support of this hypothesis: (1) variance in BLEU across language pairs is largely predictable from language-specific competences with no language pair–specific interaction terms; (2) many model components are causally influential in both monolingual tasks and translation tasks; and (3) fine-tuning on monolingual data recovers a large proportion of translation improvements relative to fine-tuning on aligned documents. Together, these provide convergent evidence in support of the interlingua hypothesis, and suggest new ways of understanding and improving how LLMs can be leveraged to perform translation tasks.

## 1 Introduction

Machine translation has long been an influential area of natural language processing. The attention mechanism was first proposed to improve machine translation performance (Bahdanau et al., 2015; Luong et al., 2015; Vaswani et al., 2017); one consequence of attention was the emergence of language models that could be efficiently trained on massive corpora (Devlin et al., 2019; Radford et al., 2018). Recently, large language models (LLMs) have revolutionized many areas of natural language processing, but have been relatively slowly adopted in machine translation (MT).

A reason for the slow adoption of LLMs in MT has been a focus on low-resource languages, where LLMs face significant challenges (Hendy et al., 2023; Robinson et al., 2023). Because LLMs require large training corpora, they are difficult to train effectively on low-resource languages. However, they also hold significant promise: even when not trained on a given language, LLMs can be prompted to translate from or into a language they have seen very little of in their training data (Cahyawijaya et al., 2024)—e.g., by prompting with a grammar book (Tanzer et al., 2024). Moreover, their representations of high-resource languages could enable cross-lingual transfer (Conneau et al., 2020).

Whether LLMs can deliver on this promise depends in part on the mechanisms underlying how they translate. Specifically, do LLMs rely on task- and language-agnostic mechanisms to translate, or do they rely on more specialized translation mechanisms or language pair–specific mechanisms? If the former, this suggests clear paths toward improving MT performance, potentially without large parallel corpora. Much recent work provides mechanistic evidence supporting the existence of massively multilingual representations in language modeling contexts (Wendler et al., 2024; Brinkmann et al., 2025; Wu et al., 2025).

We therefore hypothesize that LLMs perform MT in large part by reusing computational mechanisms for monolingual language modeling. The “stages of inference” hypothesis (Lad et al., 2024) holds that LLMs devote the first half of their layers to reading an input and composing increasingly abstract concept representations. Then, the latter half of the model reads these concept representations, and uses them to decide which token should be predicted given prior context. If language models perform translation this way (what we call the interlingua hypothesis), then translation would not necessarily require language pair–specific mechanisms; instead, it only requires that a model be capable of reading abstract features from the source language, and generating text for the target language conditioned on those features.

If this hypothesis is true, it would entail the following predictions: (1) Given language pair (S, T), translation performance should be predictable from monolingual capabilities in S and T in isolation without cross-linguistic interaction terms. (2) There should exist task-agnostic representations that are causally relevant for predicting correct outputs in both machine translation and monolingual task settings. Finally, (3) adding translation capabilities for a new language should primarily require improvements to monolingual capabilities for that language—i.e., fine-tuning on monolingual corpora should recover a substantial proportion of the improvement in translation performance as finetuning on parallel corpora, assuming the model can already effectively handle the other language in the language pair.<sup>1</sup>

We investigate each of these three implications, and find positive evidence for each. While no single experiment definitively confirms the interlingua hypothesis, each provides a different type of evidence in support of it. These findings could provide a preliminary explanation for the value of exposure to (largely monolingual) documents in producing models more effective at translation.

## 2 Related Work

Massively multilingual feature representations. Our work is partially motivated by the observation that grammatical concept representations are highly multilingual in large language models, even across typologically distinct languages (Brinkmann et al., 2025). Similar evidence has been observed in Wendler et al. (2024); Dumas et al. (2025). Notably, intervening on grammatical concept representations has predictable effects on model behavior in both monolingual and machine translation settings (Brinkmann et al., 2025). These findings do not in themselves confirm the interlingua hypothesis, but they do suggest the existence of abstract feature representations not tied to particular tasks nor languages.

Interlingua in multilingual NMT. A line of work in neural machine translation (NMT) seeks to engineer interlingua to improve cross-lingual transfer via architectural and training improvements, largely to encoder-decoder architectures (e.g., XLM-R, Conneau et al., 2020; M2M, Fan et al., 2020). Prior approaches introduce explicit shared representations via interlingua layers or networks (Lu et al., 2018; Zhu et al., 2020), or bottlenecks that naturally encourage languageindependent representations (Vázquez et al., 2019; Mao et al., 2023). These studies stipulate or engineer an interlingua and validate it behaviorally, whereas our study asks whether an interlingua emerges naturally and is causally relevant to MT performance in a decoder-only LM trained on general language data with no such directly implemented incentives.

Precedents to our mechanistic investigation include Vázquez et al. (2020), who investigate whether sentence representations in NMT models with attention bottlenecks are languageindependent. Others have investigated whether shared or language-specific components are needed at all (Escolano et al., 2021; Purason and Tättar, 2022); these studies investigate by controlling the degree of parameter sharing across languages, and find that fully shared representations underperformed representations with at least some languagespecific components. Thus, in smaller-scale settings using primarily parallel data, interlingua typically need to be imposed via the training objective, and this costs some performance. We investigate whether this also holds in contemporary decoderonly models trained on a larger quantity of taskand domain-general language data.

LLMs for machine translation. Machine translation is difficult when parallel data is limited (Koehn and Knowles, 2017), and large language models do not solve this problem (Court and Elsner, 2024). Some hope that LLMs could enable cross-lingual transfer. For example, Tanzer et al. (2024) showed that long-context LLMs can translate a previously unseen low-resource language when prompted with a grammar book containing linguistic descriptions and translation examples. However, Aycock et al. (2025) subsequently found that most of the improvement in this setting came from the book’s parallel examples rather than its grammatical explanations, highlighting the importance of parallel data for translation. One of our experiments asks a complementary question: once an LLM has already acquired multilingual representations during pretraining, to what extent can improving its monolingual competence in a low-resource language improve translation without additional parallel data? We provide preliminary evidence for cross-lingual transfer in §5.

Causal mediation analysis. Some of our evidence relies on estimates of the causal influence (Lewis, 1973) of specific model components. This relies on causal mediation analysis (Pearl, 2001; Vig et al., 2020), a common technique in the mechanistic interpretability literature (e.g., Finlayson et al., 2021; Geiger et al., 2021; Heimersheim and Nanda, 2024; Marks et al., 2025; Mueller et al., 2026). While the purpose of this study is not to understand the exact functional role of particular model components, we wish to characterize whether the same representations influence a model’s behavior in multiple task settings and language pairs.

## 3 Modeling Translation Performance as a Function of Monolingual Capabilities

How much of a model’s ability to translate can be explained by its monolingual language modeling capabilities? If a model uses an interlingua to perform translation, then much of its translation capabilities should be explainable as a function of its ability to parse latent features from an input, and produce coherent text in that language. In other words, one should not need language pair– specific terms to explain translation performance, unless a model deploys some separate translation mechanism that does not involve going through an interlingua. To investigate, we define a linear model that predicts translation performance from monolingual competencies.

We use several measures of monolingual competence. First, to measure a model’s ability to distinguish grammatical from ungrammatical sentences, we use MultiBLiMP (Jumelet et al., 2026), a multilingual version of the BLiMP (Warstadt et al., 2020) benchmark. For a given language, Multi-BLiMP accuracy is the fraction of minimal pairs for which the model assigns higher full-sentence log-probability to the grammatical completion over its ungrammatical counterpart. We average this over 200 randomly-selected samples per language. Because Llama and Aya saturate this benchmark for many of the languages in our analysis, we also compute the log-probability margin, which is the mean log p(grammatical)−log p(nongrammatical) per minimal pair. To overcome issues related to benchmark saturation, we also include accuracy on GlobalMMLU (Singh et al., 2025), a multilingual multiple-choice question answering dataset based on MMLU (Hendrycks et al., 2021). For each task, we filter the set of languages to those in common between the monolingual dataset and FLORES (Table 1).<sup>2</sup>

For translation, we have language pair (S, T), where S is the source and $T$ is the target language. Translation competence $t _ { S T }$ is quantified as a BLEU score for (S, T). To perform translation, we use a 2-shot prompt containing two randomly sampled sentences from FLORES.<sup>3</sup> We use sacrebleu (Post, 2018) to compute BLEUs.

Let l be a monolingual competence estimate using MultiBLiMP or GlobalMMLU. We have $l _ { S }$ and $l _ { T }$ for each language pair. We then learn coefficients $\beta _ { S }$ and $\beta _ { T }$ as well as bias term $\beta _ { 0 }$ to maximize predictive accuracy on BLEU scores $t _ { S T }$ in the following function:

$$
\beta _ { S } l _ { S } + \beta _ { T } l _ { T } + \beta _ { 0 } = t _ { S T }\tag{1}
$$

We also train a bilinear model that is nearly identical, but also contains a multiplicative interaction term:

$$
\beta _ { S } l _ { S } + \beta _ { T } l _ { T } + \beta _ { S T } ( l _ { S } \cdot l _ { T } ) + \beta _ { 0 } = t _ { S T }\tag{2}
$$

If the interlingua hypothesis holds, then the bilinear model should not have significantly greater predictive power than the linear model. This would imply that machine translation performance is better explained by monolingual terms, rather than by the existence of a translation mechanism (which should use terms specific to particular language pairs).

Monolingual task performance predicts translation competence. Fitting the linear model on the common 18-language subset, monolingual behavioral competence proxies are generally strong predictors of translation performance. MultiBLiMP grammatical accuracy reaches $R ^ { 2 } = 0 . 2 9 4 / 0 . 2 3 5$ for Llama/Aya. While significant, this is relatively low; we find that this is largely because Multi-BLiMP scores saturate at relatively low BLEU scores, such that it is a good predictor, but only up to middling BLEU scores. In contrast, GlobalMMLU accuracy is a very strong predictor at $R ^ { 2 } = 0 . 7 3 9 / 0 . 5 1 0$ for Llama/Aya. Figure 1 plots monolingual competencies against translation performance for each language shared across each evaluation dataset. Qualitatively, the grammatical and MMLU proxies trend closely with translation quality.

<table><tr><td>Proxy</td><td>Measures</td><td>N</td></tr><tr><td>FLORES perplexity</td><td>fluency / compression (↓)</td><td>24</td></tr><tr><td>MultiBLiMP accuracy</td><td>grammatical acceptability (↑)</td><td>18</td></tr><tr><td>MultiBLiMP margin</td><td>grammatical confidence (↑)</td><td>18</td></tr><tr><td>GlobalMMLU accuracy</td><td>world knowledge (↑)</td><td>23</td></tr></table>

Table 1: Monolingual competence proxies. N is the number of the 24 FLORES languages on which the proxy is defined; ↑/↓ indicates the direction of higher monolingual competence.
<table><tr><td rowspan="2">Proxy</td><td colspan="2">Llama-3.1-8B</td><td rowspan="2"></td><td colspan="3">Aya-23-8B</td></tr><tr><td> $R _ { \mathrm { { l i n } } } ^ { 2 }$ </td><td> $R _ { \mathrm { b i l } } ^ { 2 }$  MAE</td><td> $R _ { \mathrm { { l i n } } } ^ { 2 }$ </td><td> $R _ { \mathrm { b i l } } ^ { 2 }$ </td><td>MAE</td></tr><tr><td>Per-token perplexity</td><td>0.046</td><td>0.046</td><td>5.48</td><td>0.002</td><td>0.003</td><td>4.89</td></tr><tr><td>MultiBLiMP accuracy</td><td>0.294</td><td>0.294</td><td>4.50</td><td>0.235</td><td>0.236</td><td>4.22</td></tr><tr><td>MultiBLiMP margin</td><td>0.324</td><td>0.324</td><td>4.82</td><td>0.233</td><td>0.234</td><td>4.27</td></tr><tr><td>GlobalMMLU accuracy</td><td>0.739</td><td>0.741</td><td>2.91</td><td>0.510</td><td>0.510</td><td>3.56</td></tr></table>

Table 2: Comparison of the power of monolingual tasks in predicting BLEU scores using the linear $( R _ { \mathrm { l i n } } ^ { 2 } )$ and bilinear $( R _ { \mathrm { b i l } } ^ { 2 } )$ models with several monolingual competence proxies on 18 languages. GlobalMMLU is the strongest predictor. The bilinear interaction term does not add significant predictive power over the linear model.

Adding language pair interaction terms does not increase predictive power. Across each proxy and both models, the bilinear model has virtually the same predictive power as the linear model (Table 2; see also App. A and App. A.1).

The same trend holds when we predict BLEU scores from the language-specific marginal BLEU scores, computed by averaging BLEU scores across all language pairs for a given source or target language. Decomposing the full BLEU matrix into per-language source and target main effects, BLEU<sub>ST</sub> ≈ $\mu + \alpha _ { S } + \beta _ { T }$ , explains $R ^ { 2 } = 0 . 9 3 2$ (Llama) / 0.879 (Aya) of the centered variance, and a rank-1 multiplicative reconstruction recovers ≈90%/83% of the matrix energy (App. A.2). Hence, pairwise translation quality is largely captured by per-language competence, with pairwise interactions adding little predictive power over the simpler model.

![](images/5adcbf1bf446add5fd758b985c5daa3cd6a403636a7b3a5dd0f733d6c8fda3fb.jpg)

![](images/dcd1f928c036e43150ad3c1c445060823b03c27444870027a70cdfba685c3737.jpg)  
Figure 1: Per-language relationship between two monolingual proxies and the BLEU score for a given target language. Both monolingual terms correlate significantly with BLEU scores, but GlobalMMLU accuracies are a stronger predictor than MultiBLiMP accuracies.

Target language competence matters more than source language competence. Source and target competence contribute asymmetrically: changing the target language moves BLEU far more than changing the source. The variance across target languages of mean BLEU exceeds the variance across source languages by 9.3× for Llama and 3.9× for Aya, and the target coefficient is 1.7– 5× the source coefficient (MMLU target/source = 2.98 for Llama, 1.67 for Aya; MultiBLiMPmargin 4.88/2.30). This may be because it is easier for models to extract meaning from the source language than produce fluent outputs in the target language; alternatively, it may be an artifact of relying on an n-gram matching metric such as BLEU score.

## 4 Many Translation-relevant Components Also Perform Monolingual Tasks

Our previous results show that monolingual capabilities are predictors of translation performance, but also that language pair interactions are not strong predictors. This provides correlational evidence in support of our hypothesis, but not causal evidence. To obtain causal evidence, we now perform an analysis based on causal mediation analysis (Pearl, 2001; Vig et al., 2020).

Past work has found that language models use the first half of their layers to form progressively more abstract representations of the latent features in a given input (Lad et al., 2024). We hypothesize that language models could repurpose this language modeling machinery to perform translation by reading the source language into a latent feature space, and then using the latter half of its layers to read from this feature space and produce the translation in the output language. If this is true, then we should be able to find model representations or components that are causally influential in both machine translation and language modeling settings. To obtain causal evidence in support of this view, we now perform mechanistic experiments by patching model components. We show that the most influential attention heads for producing correct translations also strongly mediate a language model’s ability to produce grammatical outputs in monolingual settings.

## 4.1 Using GCM to Identify Translation-relevant Components

We first search for model components that are causally relevant in promoting the valid translation over an invalid translation. We do so using generative causal mediation (GCM; Sankaranarayanan et al., 2026). Intuitively, GCM measures how intervening on the activation of a model component (e.g., an attention head) influences the model’s relative preference for one continuation over another.

We measure the model’s preference for a single gold translation by swapping the source sentence that the model reads. Let r be a gold translation of a source sentence, held fixed throughout. Let $p _ { \mathrm { o r i g } }$ be a prompt whose final source sentence is the one that r actually translates, and let $p _ { \mathrm { c f } }$ be the same prompt with a distinct final source sentence. We define the preference metric M:

$$
M = \log \pi ( r \mid p _ { \mathrm { o r i g } } ) - \log \pi ( r \mid p _ { \mathrm { c f } } ) ,\tag{3}
$$

where log $\begin{array} { r } { \pi ( r \mid p ) = \sum _ { t } \log p ( r _ { t } \mid p , r _ { < t } ) } \end{array}$ is the teacher-forced log-probability of a continuation, summed over its tokens. M is the degree to which the model prefers to generate the gold translation r when it has read the matching source, rather than a mismatched one; it isolates how much of the production of r depends on having read the source content.

We wish to know how strongly each attention head influences this preference. To measure this, we take the activation of an attention head; we call this $z ,$ and let $z _ { \mathrm { o r i g } }$ and $z _ { \mathrm { c f } }$ be the values it takes when the model reads $p _ { \mathrm { o r i g } }$ and $p _ { \mathrm { c f } }$ respectively. Let $\operatorname { I E } ( z )$ be the causal contribution of $z { \mathrm { ~ t o ~ } } M$ measured as the change in M when we move the head from its mismatched-source state $z _ { \mathrm { c f } }$ to its matched-source state $z _ { \mathrm { o r i g } }$ while holding the rest of the computation fixed. We source the counterfactual activation $z _ { \mathrm { c f } }$ by running a forward pass on $p _ { \mathrm { c f } }$ and caching z at the final source-token position, then patch it into the corresponding position of the run we score:

$$
\mathrm { I E } ( z ) = M ( z _ { \mathrm { o r i g } } ) - M ( z _ { \mathrm { c f } } ) .\tag{4}
$$

Patching a single head and rerunning the model tells us that head’s exact contribution, but computing $\operatorname { I E } ( z )$ this way for every z is intractable: it would require $O ( Z \cdot n )$ forward passes, where Z is the number of mediators and n the number of examples. We instead use attribution patching (Syed et al., 2023), a first-order linear approximation of the IE based on gradient attributions (Simonyan et al., 2013):

$$
\widehat { \mathrm { I E } } ( z ) = \nabla _ { z } M \big | _ { z = z _ { \mathrm { o r i g } } } \cdot ( z _ { \mathrm { o r i g } } - z _ { \mathrm { c f } } ) .\tag{5}
$$

The gradient $\nabla _ { z } M$ for each component can be computed in a single backward pass, and the deltas $\left( z _ { \mathrm { o r i g } } - z _ { \mathrm { c f } } \right)$ for each component can be computed in two forward passes by caching each component’s activation on the two source prompts.

The magnitude of the indirect effect IEc is how much of a model’s output behavior (as quantified by M) flows through a component when all else is kept constant. Because M rewards producing r under the correct source, components with a positive indirect effect are those that cause the probability of the correct translation $r _ { \mathrm { o r i g } }$ to increase relative to the counterfactual translation $r _ { \mathrm { c f } } ,$ and those with a negative indirect effect cause the probability of the correct translation to decrease relative to the counterfactual translation.

We instantiate r as a gold translation from FLO-RES (Goyal et al., 2022), a massively parallel dataset that supports 101 languages. Given a source and target language, we construct a 2-shot prompt as follows:

$$
{ \begin{array} { r l } & { \{ { \mathrm { S o u r c e } } \} : \ \{ { \mathrm { s h o t ~ 1 } } \ { \mathrm { s o u r c e } } \} } \\ & { \{ { \mathrm { 7 a r g e t } } \} : \ \{ { \mathrm { s h o t ~ 1 } } \ { \mathrm { t a r g e t } } \} } \\ & { \{ { \mathrm { S o u r c e } } \} : \ \{ { \mathrm { s h o t ~ 2 } } \ { \mathrm { s o u r c e } } \} } \\ & { \{ { \mathrm { 7 a r g e t } } \} : \ \{ { \mathrm { s h o t ~ 2 } } \ { \mathrm { t a r g e t } } \} } \\ & { \{ { \mathrm { S o u r c e } } \} : \ \{ { \mathrm { q u e r y ~ s o u r c e } } \} } \\ & { \{ { \mathrm { 7 a r g e t } } \} : } \end{array} }
$$

and patch at the last source-token position (the final $\mathbf { \ " } \left( \mathbf { \ " } \cdot \mathbf { \ " } \right)$ . For each of the 56 ordered pairs where the source and target can be any language in {English, Spanish, German, French, Turkish, Arabic, Hindi, Hebrew} (excluding pairs where the source and target are the same), we compute the IEc for all attention heads over $n = 1 0 0$ uniformly sampled pairs.

![](images/d05da71a3deb159396124bdb6234a63284c9b4234194bbf33f83558e27cef9c2.jpg)  
Figure 2: Llama-3.1-8B mean $| \widehat { \mathrm { I E } } |$ for top attention heads under translation and control prompts, averaged over 56 translation directions. The leading heads have much larger effects in the translation setting than in the controls, indicating that these heads are selective for matching translations and do not directly increase the probability of the given target sentence.

Naïvely, we may find heads that respond at least in part to variations in the source samples that are unrelated to the translation task. To verify that the components we find are selective for correct source–target pairings, we compare indirect effects with three controls. First, the same-language control refers to cases where the source language is the same as the target, such that the correct “translation” is the same sentence copied from the source, and the counterfactual translation is a randomly sampled sentence in the same language as the source. Higher indirect effects for the translation setting than the same-language control indicate that the component specifically causes correct translations to be more probable, and not copies of the source sequence. The null cross-language control refers to a setup similar to the machine translation setup, but where neither the original nor counterfactual completion are the correct translation. We expect indirect effects here to be very small relative to the translation task. Finally, we have the null same-language control, where both the original and counterfactual completions are in the same language as the source, but neither are the same as the source sentence.

The heads we find have larger effects in the translation setting by far than in any control setting (Figure 2). The mean |IEc| in the translation task exceeds that of the null cross-language control by 3.0× over all heads, and 5.2× over the top-10 heads for Llama (and 2.5× for all heads/4.3× for the top-10 heads for Aya). We observe a similar magnitude of increase relative to both samelanguage controls. This suggests that the heads we have found are responsible for performing machine translation, and that their effects cannot be explained by their general utility in generating fluent target sequences regardless of the source.

The translation-specific heads are localized to layers 13–14 in Llama and 15–20 in Aya. With respect to the stages of inference hypothesis (Lad et al., 2024), this would correspond to the layers that refine concept representations into increasingly abstract representations.

## 4.2 Translation Heads Have Similar Effects Across Language Pairs

An interlingua should have multilingual feature representations; this would allow a model to reuse the same representations when reading a source sequence and generating the target sequence. Prior work has established the existence of massively multilingual representations (Wendler et al., 2024), including grammatical concept representations (Brinkmann et al., 2025); here, we confirm these findings for the model we study by investigating their indirect effects across language pairs.

We follow the procedure of §4.1 to compute IEc for each language pair in the translation test dataset. We show the indirect effects for the top heads by absolute IEc across language pairs; if there is feature reuse across pairs, then the sign of the effect should be the same for many language pairs. Note that the magnitude of IEc is not directly comparable across languages, as the initial probability of the target sequence and its translation capabilities are language pair–dependent.

The top attention heads by IEc are largely shared across language pairs. Figure 3 shows the 15 heads that appear most often in a single direction’s top-

![](images/c4843808641f19c73c447bf62ec0cd9b5778c507abec1c336610c30cd3ab30be.jpg)  
Figure 3: Signed mean IE for Llama-3.1-8B on machine translation for selected language pairs. Each column is 1 of the 15 top attention heads by IEc across all languages. The top axis shows for how many language pairs the head was in the top-20 set. Several heads have stable effect signs across language pairs, which suggests that the same heads are being reused across many language pairs.

20 by $| \widehat { \mathrm { I E } } |$ : the most universal of these are in the top-20 for all 56 directions. The sign of their effect is also stable—a given head keeps the same sign (favoring either correct or incorrect translations) across virtually every direction, regardless of source and target languages (For the full results from both models, see Figures 10 and 11). As predicted by our hypothesis, many of the translation mechanisms employed by the model do not depend on the choice of language pair; in fact, most of the top heads have the same directionality and general magnitude of effect on model performance for all language pairs.

## 4.3 Ablating Translation Heads Degrades Performance on Translation and Monolingual Tasks

Another mechanistic prediction of our hypothesis is that the heads most responsible for performing machine translation should reuse computational machinery used for monolingual tasks. To test this, we first verify that ablating the top heads by indirect effect harms machine translation performance. Then, we show that ablating the same heads also harms performance on acceptability judgments and multiple-choice question answering, suggesting that these heads are not selective for translation; rather, they may be reusing computational machinery from more general language modeling mechanisms.

For each target language $T _ { \cdot }$ , we aggregate signed IE over the 7 language pairs in which $T$ is the target, then select the top 10 positive-IE heads (POS-10, the correct translation–favoring set) by signed magnitude. We mean-ablate a head by replacing its output with its average activation over the translation prompts at all token positions. As a control, we ablate the same number of randomly sampled heads from the same layers. We then regenerate FLORES translations and measure the change in BLEU relative to the original model before ablations.

![](images/acef3cb9d5d78d1d36b7824c08995276fc4c2536af1b8dd5255f556b1b0b2d5c.jpg)  
Figure 4: Ablation experiments overview. Ablating the top heads by causal influence on MT (POS-10, in red) performance drives a significant decrease in BLEU, and more than ablating random heads (Control, in grey). For GlobalMMLU, ablating those top heads reduces the model’s ability to distinguish between the correct and incorrect answer more than ablating random heads does. The Control involves ablating the same number of heads in the same layers.

Ablating these heads causes significant reductions in BLEU. Ablating POS-10 degrades BLEU in all 8 target languages for both models (Figure 4, bottom) by roughly twice the amount as the random control. This suggests that the heads we have found are causally relevant to translation performance.

Having shown the importance of these heads for translation quality, we now ask whether these same heads are responsible for monolingual capabilities. For this, we again use the GlobalMMLU dataset. We uniformly sample 400 examples per language. Given prompt p with correct token completion $r _ { \mathrm { c o r r e c t } }$ and a randomly chosen incorrect completion $r _ { \mathrm { i n c o r r e c t } }$ , each minimal pair is scored by the acceptability margin

$$
\Delta = \log p ( r _ { \mathrm { c o r r e c t } } \mid p ) - \log p ( r _ { \mathrm { i n c o r r e c t } } \mid p )\tag{6}
$$

We report the change in $\Delta$ after ablating the same heads ablated in the translation experiments:

$$
\Delta _ { \mathrm { c h a n g e } } = \Delta _ { \mathrm { a b l a t i o n } } - \Delta _ { \mathrm { o r i g i n a l } } ,\tag{7}
$$

<table><tr><td></td><td colspan="3">Llama-3.1-8B</td><td colspan="3">TinyAya-3B</td></tr><tr><td>Direction</td><td>Base</td><td>Mono</td><td>Parallel</td><td>Base</td><td>Mono</td><td>Parallel</td></tr><tr><td>Xho → Eng</td><td>16.84</td><td>24.81</td><td>24.88</td><td>23.46</td><td>26.64</td><td>27.80</td></tr><tr><td> $\mathrm { F r a }  \mathrm { E n g }$ </td><td>43.27</td><td>42.28</td><td>38.87</td><td>40.70</td><td>40.26</td><td>38.11</td></tr><tr><td> $\mathrm { D e u \to E n g }$ </td><td>42.69</td><td>42.76</td><td>39.80</td><td>40.20</td><td>41.21</td><td>39.58</td></tr></table>

Table 3: Monolingual fine-tuning recovers most of the Xhosa translation gains obtained with parallel fine-tuning. Both conditions also retain most performance on French and German translation (despite their not appearing in the fine-tuning data), although monolingual fine-tuning retrains slightly more performance.

where a negative $\Delta _ { \mathrm { c h a n g e } }$ means the ablation weakened the model’s ability to distinguish the correct from the incorrect answer.

Ablating the POS-10 heads from the MT task results in a negative $\Delta _ { \mathrm { c h a n g e } } .$ , whereas ablating random heads has a smaller effect (Figure 4, top). That the same heads affect performance in both tasks provides preliminary evidence that these heads implement computations that are reused across tasks. Thus, these heads appear to implement computations that are not selective for translation alone. This provides further support for our hypothesis.<sup>4</sup>

## 5 Monolingual Fine-tuning Recovers Most Gains from Parallel Data

If LLMs translate via task-agnostic internal representations, translation quality should depend in part on the model’s monolingual competence in the source and target languages. For example, if we wish to translate between a low-resource and high-resource language, then we should be able to improve translation performance given access only to data that improves language modeling quality on the low-resource language, assuming that we preserve capabilities in the high-resource language. Under this view, low translation performance may reflect a failure to encode the source sentence into an adequate internal representation, or to decode the target sentence fluently from it. Recent work has shown that bilingual or mixed-language signals during pretraining can be important for acquiring translation capabilities in LLMs (Briakou et al., 2023; Qorib et al., 2025; Shao et al., 2026). We ask the complementary question of whether, given a fully pretrained model, improvements in monolingual language modeling can (at least in part) transfer to translation even without parallel data.

We test this idea using Llama-3.1-8B and

TinyAya-3B.<sup>5</sup> For each model, we compare how fine-tuning on an underrepresented language (Xhosa) affects translation performance when the training data consist of either parallel data or monolingual text. Specifically, in the monolingual setting, we train on unaligned text consisting of 80% Xhosa and 20% English, French, and German data to avoid catastrophic forgetting of the model’s existing language abilities. In the parallel setting, we use paired Xhosa and English sentences from OPUS MT560 (Gowda et al., 2021), presented in both translation directions.

Both settings use a next-token prediction loss and rank-16 LoRA adapters (Hu et al., 2022). The reported runs use a learning rate of $3 \times 1 0 ^ { - 5 } .$ , a maximum sequence length of 512 tokens, and one epoch over 100 million tokens. We evaluate translation with few-shot prompting on the FLORES devtest split (Goyal et al., 2022), using examples from the dev split as in-context demonstrations. Thus, the models are prompted to translate at evaluation time even though the monolingual condition contains no translation examples during training.

Table 3 shows that monolingual fine-tuning recovers most of the improvement obtained with parallel data. For Llama, monolingual fine-tuning raises BLEU from 16.84 to 24.81, compared with 24.88 under parallel fine-tuning. This recovers 99% of the improvement over the base model. For Aya, monolingual fine-tuning raises BLEU from 23.46 to 26.64, compared with 27.80 under parallel finetuning, recovering 73% of the improvement. For both models, both fine-tuning conditions largely preserve translation performance from French and German into English, although monolingual finetuning remains closer to the base performance.

These results are consistent with the view that once an LLM has been pretrained and acquired multilingual representations, improving its ability to model an underrepresented language can produce most of the available translation gains. Parallel data may still perform better because it can improve language modeling while also strengthening translation-specific mechanisms. Our results therefore do not imply that parallel data are unimportant, particularly during pretraining. They show that, after pretraining, a large share of the gains from parallel fine-tuning can be achieved using only unaligned data.

We provide results for additional languages (German and Thai) and translation directions in Appendix C. In short, changes in performance are smaller for higher-resource languages, but parallel and monolingual fine-tuning still achieve largely comparable results.

## 6 Discussion and Conclusions

We have provided evidence that machine translation capabilities in LLMs are mediated in significant part by multilingual representations that are also relevant for some monolingual tasks. Across three complementary analyses, we find evidence consistent with this view: Translation performance is largely explained by source- and target-language competence, translation-relevant components overlap with components used for monolingual grammatical behavior, and monolingual Xhosa finetuning recovers most of the gains from parallel fine-tuning across both models. Together, these results provide support for the hypothesis that LLMs translate by mapping source-language input into a latent feature space (that can also in theory be used for monolingual next-token predictions), and then generating target-language text from that shared representation.

This interpretation helps explain why monolingual competence is strongly associated with translation performance. If a model translates through a partially language-agnostic feature space, then improving its ability to read or write a language should improve translation involving that language. The observation that target-language competence explains more variance than source-language competence may suggest that fluent and grammatical generation is often the bottleneck. Under this view, poor translation into a language need not imply the absence of a dedicated translation circuit for that language pair; instead, it may reflect weak targetlanguage decoding capabilities given otherwise usable latent representations of the source sentence.

We emphasize that translation-specific mechanisms not based on interlingua are also likely to exist; indeed, Todd et al. (2024) find that there are components selective for word translation. We do not claim that an interlingua is the only means by which LLMs translate, but rather, that it is a significant mechanism that controls a large proportion of model performance on MT tasks. It is likely that LLMs use a mixture of mechanisms (Gur-Arieh et al., 2026) to achieve their machine translation capabilities.

## Limitations

The interlingua hypothesis is one plausible explanation for what we have observed, but our experiments do not rule out the existence of other mechanisms. The finding that the target language yields more predictive power than the source language in predicting BLEU score may be an artifact of the ngram-based computation of BLEU scores. It is also possible that the features underlying high-quality translations are not well represented in the BLEU score.

Our experiments were limited to two 8Bparameter models and one 3B-parameter model, and may not generalize to smaller or larger LLMs. Future work should investigate whether similar trends hold for a wider variety of language models, and whether recent developments in thinking models affect these findings.

For our GCM experiments, we focus on components that mediate at the last token, missing computations that happen at earlier positions.

Finally, our translation experiments are limited to a relatively small number of language pairs. Future work could scale up this experiment to investigate whether these trends hold across a large number of language pairs, and to what degree monolingual fine-tuning can recover the performance of parallel fine-tuning across many language pairs (and what other factors explain when this works well and when it does not).

## Acknowledgments

We are grateful to the members of the BAAIGL lab at Boston University for helpful comments on an earlier iteration of this work. The computational work reported on in this paper was performed largely on the Shared Computing Cluster, which is administered by Boston University’s Research Computing Services. Jannik Brinkmann is supported by the German Federal Ministry for Economic Affairs and the German Federal Ministry of Research, Technology and Space.

## References

Seth Aycock, David Stap, Di Wu, Christof Monz, and Khalil Sima’an. 2025. Can LLMs really learn to translate a low-resource language from one grammar book? In The Thirteenth International Conference on Learning Representations.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate.

Eleftheria Briakou, Colin Cherry, and George Foster. 2023. Searching for needles in a haystack: On the role of incidental bilingualism in PaLM’s translation capability. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9432–9452, Toronto, Canada. Association for Computational Linguistics.

Jannik Brinkmann, Chris Wendler, Christian Bartelt, and Aaron Mueller. 2025. Large language models share representations of latent grammatical concepts across typologically diverse languages. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6131–6150, Albuquerque, New Mexico. Association for Computational Linguistics.

Samuel Cahyawijaya, Holy Lovenia, and Pascale Fung. 2024. LLMs are few-shot in-context low-resource language learners. In Proceedings of the 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 405–433, Mexico City, Mexico. Association for Computational Linguistics.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics.

Sara Court and Micha Elsner. 2024. Shortcomings of LLMs for low-resource translation: Retrieval and understanding are both the problem. In Proceedings of the Ninth Conference on Machine Translation, pages 1332–1354, Miami, Florida, USA. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Clément Dumas, Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. 2025. Separating tongue from thought: Activation patching reveals language-agnostic concept representations in transformers. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 31822–31841, Vienna, Austria. Association for Computational Linguistics.

Carlos Escolano, Marta R. Costa-jussà, José A. R. Fonollosa, and Mikel Artetxe. 2021. Multilingual machine translation: Closing the gap between shared and language-specific encoder-decoders. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics, pages 944–948.

Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, and Armand Joulin. 2020. Beyond english-centric multilingual machine translation. Preprint, arXiv:2010.11125.

Matthew Finlayson, Aaron Mueller, Sebastian Gehrmann, Stuart Shieber, Tal Linzen, and Yonatan Belinkov. 2021. Causal analysis of syntactic agreement mechanisms in neural language models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1828–1843, Online. Association for Computational Linguistics.

Jaden Fried Fiotto-Kaufman, Alexander Russell Loftus, Eric Todd, Jannik Brinkmann, Koyena Pal, Dmitrii Troitskii, Michael Ripa, Adam Belfki, Can Rager, Caden Juang, Aaron Mueller, Samuel Marks, Arnab Sen Sharma, Francesca Lucchetti, Nikhil Prakash, Carla E. Brodley, Arjun Guha, Jonathan Bell, Byron C Wallace, and David Bau. 2025. NNsight and NDIF: Democratizing access to openweight foundation model internals. In The Thirteenth International Conference on Learning Representations.

Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. 2021. Causal abstractions of neural networks. In Advances in Neural Information Processing Systems, volume 34, pages 9574–9586. Curran Associates, Inc.

Thamme Gowda, Zhao Zhang, Chris Mattmann, and Jonathan May. 2021. Many-to-english machine translation tools, data, and pretrained models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations, page 306–316. Association for Computational Linguistics.

Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2022. The Flores-101 evaluation benchmark for low-resource and multilingual machine translation. Transactions ofthe Associationfor Computational Linguistics, 10:522–538.

Yoav Gur-Arieh, Mor Geva, and Atticus Geiger. 2026. Mixing mechanisms: How language models retrieve bound entities in-context. In The Fourteenth International Conference on Learning Representations.

Stefan Heimersheim and Neel Nanda. 2024. How to use and interpret activation patching. Preprint, arXiv:2404.15255.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Amr Hendy, Mohamed Abdelrehim, Amr Sharaf, Vikas Raunak, Mohamed Gabr, Hitokazu Matsushita, Young Jin Kim, Mohamed Afify, and Hany Hassan Awadalla. 2023. How good are gpt models at machine translation? a comprehensive evaluation. Preprint, arXiv:2302.09210.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Jaap Jumelet, Leonie Weissweiler, Joakim Nivre, and Arianna Bisazza. 2026. MultiBLiMP 1.0: A massively multilingual benchmark of linguistic minimal pairs. Transactions ofthe Associationfor Computational Linguistics, 14:193–216.

Philipp Koehn and Rebecca Knowles. 2017. Six challenges for neural machine translation. In Proceedings ofthe First Workshop on Neural Machine Translation, pages 28–39, Vancouver. Association for Computational Linguistics.

Vedang Lad, Wes Gurnee, and Max Tegmark. 2024. The remarkable robustness of LLMs: Stages of inference? In ICML 2024 Workshop on Mechanistic Interpretability.

David Lewis. 1973. Causation. The journal of philosophy, 70(17):556–567.

Yichao Lu, Phillip Keung, Faisal Ladhak, Vikas Bhardwaj, Shaonan Zhang, and Jason Sun. 2018. A neural interlingua for multilingual machine translation. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 84–92.

Thang Luong, Hieu Pham, and Christopher D. Manning. 2015. Effective approaches to attention-based neural machine translation. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1412–1421, Lisbon, Portugal. Association for Computational Linguistics.

Zhuoyuan Mao, Haiyue Song, Raj Dabre, Chenhui Chu, and Sadao Kurohashi. 2023. Variable-length neural interlingua representations for zero-shot neural machine translation. In Proceedings of the 1st International Workshop on Multilingual, Multimodal and Multitask Language Generation, pages 16–25, Tampere, Finland. European Association for Machine Translation.

Samuel Marks, Can Rager, Eric J Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller. 2025. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. In The Thirteenth International Conference on Learning Representations.

Aaron Mueller, Jannik Brinkmann, Millicent Li, Samuel Marks, Koyena Pal, Nikhil Prakash, Can Rager, Aruna Sankaranarayanan, Arnab Sen Sharma, Jiuding Sun, Eric Todd, David Bau, and Yonatan Belinkov. 2026. The quest for the right mediator: Surveying mechanistic interpretability for nlp through the lens of causal mediation analysis. Computational Linguistics, 52(1):331–378.

Judea Pearl. 2001. Direct and indirect effects. In Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence, UAI’01, page 411–420, San Francisco, CA, USA. Morgan Kaufmann Publishers Inc.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186– 191, Brussels, Belgium. Association for Computational Linguistics.

Taido Purason and Andre Tättar. 2022. Multilingual neural machine translation with the right amount of sharing. In Proceedings of the 23rd Annual Conference of the European Association for Machine Translation, pages 91–100.

Muhammad Reza Qorib, Junyi Li, and Hwee Tou Ng. 2025. Just go parallel: Improving the multilingual capabilities of large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 33411–33424, Vienna, Austria. Association for Computational Linguistics.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, and 1 others. 2018. Improving language understanding by generative pre-training.

Nathaniel R. Robinson, Perez Ogayo, David R. Mortensen, and Graham Neubig. 2023. Chatgpt mt: Competitive for high- (but not low-) resource languages. Preprint, arXiv:2309.07423.

Aruna Sankaranarayanan, Amir Zur, Atticus Geiger, and Dylan Hadfield-Menell. 2026. Activation steering via generative causal mediation. Preprint, arXiv:2602.16080.

Jiandong Shao, Raphael Tang, Crystina Zhang, Karin Sevegnani, Pontus Stenetorp, Jianfei Yang, and Yao Lu. 2026. The role of mixed-language documents for multilingual large language model pretraining. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 36807–36818, San Diego, California, United States. Association for Computational Linguistics.

Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. 2013. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034.

Shivalika Singh, Angelika Romanou, Clémentine Fourrier, David Ifeoluwa Adelani, Jian Gang Ngui, Daniel Vila-Suero, Peerat Limkonchotiwat, Kelly Marchisio, Wei Qi Leong, Yosephine Susanto, Raymond Ng, Shayne Longpre, Sebastian Ruder, Wei-Yin Ko, Antoine Bosselut, Alice Oh, Andre Martins, Leshem Choshen, Daphne Ippolito, and 4 others. 2025. Global MMLU: Understanding and addressing cultural and linguistic biases in multilingual evaluation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18761–18799, Vienna, Austria. Association for Computational Linguistics.

Aaquib Syed, Can Rager, and Arthur Conmy. 2023. Attribution patching outperforms automated circuit discovery. Preprint, arXiv:2310.10348.

Garrett Tanzer, Mirac Suzgun, Eline Visser, Dan Jurafsky, and Luke Melas-Kyriazi. 2024. A benchmark for learning to translate a new language from one grammar book. In The Twelfth International Conference on Learning Representations.

Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau. 2024. Function vectors in large language models. In The Twelfth International Conference on Learning Representations. ArXiv:2310.15213.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Raúl Vázquez, Alessandro Raganato, Mathias Creutz, and Jörg Tiedemann. 2020. A systematic study of inner-attention-based sentence representations in multilingual neural machine translation. Computational Linguistics, 46(2):387–424.

Raúl Vázquez, Alessandro Raganato, Jörg Tiedemann, and Mathias Creutz. 2019. Multilingual NMT with a language-independent attention bridge. In Proceedings of the 4th Workshop on Representation Learning for NLP (RepL4NLP-2019), pages 33–39, Florence, Italy. Association for Computational Linguistics.

Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. 2020. Investigating gender bias in language models using causal mediation analysis. In Advances in Neural Information Processing Systems, volume 33, pages 12388–12401. Curran Associates, Inc.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R. Bowman. 2020. BLiMP: The benchmark of linguistic minimal pairs for English. Transactions of the Association for Computational Linguistics, 8:377– 392.

Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. 2024. Do llamas work in English? on the latent language of multilingual transformers. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15366–15394, Bangkok, Thailand. Association for Computational Linguistics.

Zhaofeng Wu, Xinyan Velocity Yu, Dani Yogatama, Jiasen Lu, and Yoon Kim. 2025. The semantic hub hypothesis: Language models share semantic representations across languages and modalities. In The Thirteenth International Conference on Learning Representations.

Changfeng Zhu, Heng Yu, Shanbo Cheng, and Weihua Luo. 2020. Language-aware interlingua for multilingual neural machine translation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1650–1655, Online. Association for Computational Linguistics.

## A Additional Results for Modeling Translation Performance

Figure 5 shows predicted vs. actual BLEU scores for the linear and bilinear models. In general, linear and bilinear models produce visually indistinguishable predictions.

## A.1 Language-pair interaction tests

Table 4 reports the nested-model F-test on the l ·l interaction coefficient, on the common 18-language subset and the 17-language no-English subset. The interaction is non-significant in every cell.

## A.2 Predicting BLEU from Language-specific Translation Capabilities

Here, we show BLEU scores for all language pairs (Figure 6, left). We take each source or target language’s average BLEU across language pairs, and fit linear models based on these terms to predict each language pair’s BLEU score (see §3 for details). We observe that the error of a rank-1 linear predictor (Figure 6, right) is generally low at around 10–15%. This provides further evidence that BLEU scores are predictable as a function of language-specific capabilities.

<table><tr><td>Subset</td><td>Proxy (model)</td><td> $F ( 1 , \cdot )$ </td><td>p</td><td> $\Delta R ^ { 2 }$ </td></tr><tr><td>18-lang</td><td>MMLU, Aya</td><td>0.001</td><td>0.980</td><td> $< 1 0 ^ { - 5 }$ </td></tr><tr><td>18-lang</td><td>MMLU, Liama</td><td>1.74</td><td>0.188</td><td>0.0015</td></tr><tr><td>18-lang</td><td>MultiBLiMP acc., Aya</td><td>0.262</td><td>0.609</td><td>0.0007</td></tr><tr><td>18-lang</td><td>MultiBLiMP acc., Llama</td><td>0.000</td><td>0.990</td><td> $< 1 0 ^ { - 5 }$ </td></tr><tr><td>18-lang</td><td>MultiBLiMP margin, Aya</td><td>0.400</td><td>0.528</td><td>0.0010</td></tr><tr><td>18-lang</td><td>MultiBLiMP margin, Liama</td><td>0.028</td><td>0.867</td><td>0.0001</td></tr><tr><td>18-lang</td><td>Perplexity, Aya</td><td>0.439</td><td>0.508</td><td>0.0015</td></tr><tr><td>18-lang</td><td>Perplexity, Llama</td><td>0.009</td><td>0.924</td><td> $< 1 0 ^ { - 4 }$ </td></tr><tr><td>17-lang</td><td>MMLU, Aya</td><td>0.014</td><td>0.906</td><td> $< 1 0 ^ { - 4 }$ </td></tr><tr><td>17-lang</td><td>MMLU, Liama</td><td>0.955</td><td>0.329</td><td>0.0013</td></tr><tr><td>17-lang</td><td>MultiBLiMP acc., Aya</td><td>0.653</td><td>0.420</td><td>0.0019</td></tr><tr><td>17-lang</td><td>MultiBLiMP acc., Llama</td><td>0.001</td><td>0.978</td><td> $< 1 0 ^ { - 5 }$ </td></tr><tr><td>17-lang</td><td>MultiBLiMP margin, Aya</td><td>1.16</td><td>0.283</td><td>0.0038</td></tr><tr><td>17-lang</td><td>MultiBLiMP margin, Liama</td><td>0.414</td><td>0.520</td><td>0.0014</td></tr><tr><td>17-lang</td><td>Perplexity, Aya</td><td>0.497</td><td>0.482</td><td>0.0018</td></tr><tr><td>17-lang</td><td>Perplexity, Llama</td><td>0.170</td><td>0.681</td><td>0.0006</td></tr></table>

Table 4: Nested F-test for the language-pair interaction term. Non-significant everywhere; largest $\Delta R ^ { 2 } = 0 . 0 0 3 8$

![](images/b65a836448c4a4bc88a559db9349e2f2b7ba8ebe9192c55347ae928f731b86c3.jpg)  
Figure 5: Predicted (x) vs. actual (y) BLEU on the common 18-language subset, for each proxy (columns) under the linear (top row) and bilinear (bottom row) models, Llama (top) and Aya (bottom). The two rows are visually indistinguishable for every proxy—the graphical form of the null interaction in Table 2.

![](images/c24428339f99ae8ba3941c3403637f1d3f063154b1497c26941ba1aa7e41c546.jpg)  
Figure 6: Left: the observed $2 4 \times 2 4$ Llama BLEU matrix (self-translation diagonal masked, never imputed). Right: its rank-1 reconstruction $u _ { S } v _ { T } ,$ fit by masked alternating least squares over the 552 observed offdiagonal cells. Faithfulness = 89.8% (Aya: 83.4%); a single source-competence vector outer-producted with a single target-competence vector reconstructs most of the matrix.

## A.3 Where the grammatical signal concentrates

Restricting the MultiBLiMP margin to subject– verb agreement phenomena improves BLEU prediction (Llama, 18 languages: all-phenomena margin $R ^ { 2 } = 0 . 3 2 4  0 . 3 6 3$ for the SV-agreement subset). Per phenomenon, SV-Person reaches $R ^ { 2 } =$ 0.688, SV-Gender 0.519, SV-Number 0.412, versus subject–predicate agreement at 0.151 / 0.037. The production-side signal (agreement on the generated verb) is where translation predictiveness lives. We caution that per-phenomenon coverage is confounded with which languages each phenomenon is annotated for, so we report this as a strengthening analysis rather than a headline.

## A.4 MMLU subject subsets

Unlike the MultiBLiMP phenomenon breakdown, the BLEU-predictive signal in MMLU is not concentrated in any single subject category: aggregate MMLU $\displaystyle \begin{array} { l l l } { \displaystyle ( R ^ { 2 } } & { = } & { 0 . 5 9 8 } \end{array}$ Llama / 0.499 Aya, 23 languages) is stronger than every individual category (best single subset: Humanities 0.560 for Llama, Social Sciences 0.413 for Aya; STEM 0.360/0.190). As with phenomena, subjectcategory accuracies are highly collinear across languages and largely track per-language resource level.

## A.5 Functional form

BLEU is bounded and right-skewed, and grammatical accuracy saturates near 1. Replacing the raw fit with log BLEU $\sim \log ( 1 - a _ { S } ) + \log ( 1 - a _ { T } )$ (a logit-like transform that un-saturates accuracy) improves $R ^ { 2 }$ from 0.250 to 0.339 (Llama) and 0.394 to 0.444 (Aya); the analogous log–log fit for perplexity improves $0 . 0 7 1  0 . 1 2 7$ (Llama). We read these gains as variance-stabilization addressing the same ceiling effect that motivates the log-probability margin, rather than evidence of a specific (e.g. exponential) functional form, and so report the raw-scale linear models in the main text.

## A.6 Does grammatical competence add signal beyond Global MMLU?

Adding MultiBLiMP source and target terms on top of the Global MMLU-only regression gives a small but significant gain (Table 5). Grammatical competence carries a little translation-relevant signal that general task accuracy doesn’t capture.

## B Additional GCM Ablation Results

## B.1 Translation-specific head decompositions

Figure 7 shows the attention head decompositions as in Fig. 2 in the main text. The top heads concentrate in slightly later layers than Llama.

## B.2 Top translation-specific heads

Tables 6 and 7 show the details of the top 10 attention heads by translation-specific effect. A majority of the effect is concentrated in the top heads.

## B.3 SAE-feature decomposition analyses

Top features show the same REAL\_CROSSdominant pattern as attention heads but with

![](images/89ca6b46d4ed65eb413bbc89f9ddc2a1c7f43782cc67cfdc8c8a4ca691dea970.jpg)  
Figure 7: Prompt-swap GCM head IE under the four control tasks, top 20 heads sorted by REAL\_CROSS − NULL\_CROSS, for Aya-23-8B (cf. Fig. 2). The top heads concentrate in layers 15–20. Support and bar definitions as in Fig. 2.

weaker separation. A few top-ranked features are same-language features (Fig. 8 and 9).

## B.4 Universality of translation-relevant heads

Translation-relevant heads are universal across translation language pairs; Hebrew and Hindi show more divergence from the rest of the languages (Fig. 10 and 11).

## B.5 Head-IE concentration and sparsity

As further illustration of the head-IE sparsity, Fig. 12 and 13 show that the ratio of the largest mean |IE| to the median ranges around 2 orders of magnitude.

The sign of the translation task selection carries over to Multi-BLiMP: ablating the POS-10 heads lowers the grammaticality margin in all 8 target languages for both models, and ablating the NEG-10 heads raises it in all 8 (Fig. 14; Table 8). The magnitude of the POS-10 effect, however, is comparable to that of ablating random heads from the same layers, so it is the consistent direction of the effects, rather than their size, that suggests these heads are reused across tasks.

The choice of the number of heads to ablate is unimportant to the overall effect; ablating different numbers of top heads has monotonic effect on performance (Fig. 15).

<table><tr><td>Model</td><td>MMLU-only  $R ^ { 2 }$ </td><td>Full  $R ^ { 2 }$ </td><td>Partial  $R ^ { 2 }$ </td><td>F-test p</td></tr><tr><td>Llama</td><td>0.739</td><td>0.752</td><td>0.048</td><td> $5 . 8 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Aya</td><td>0.510</td><td>0.556</td><td>0.094</td><td> $3 . 7 \times 1 0 ^ { - 7 }$ </td></tr></table>

Table 5: Does grammatical competence add predictive value beyond general competence? We compare BLEU ∼ $\mathbf { M M L U } _ { S } + \mathbf { M M L U } _ { T }$ to the same model alongside MultiBLiMP source and target terms. Adding MultiBLiMP accuracy adds some small signal for Llama and Aya.

<table><tr><td colspan="7">mean  $| \widehat { \mathrm { I E } } |$ </td></tr><tr><td>Layer</td><td>Head</td><td>real_cr</td><td>null_cr</td><td>real_sm</td><td>null_sm</td><td>Δ</td><td>#dir</td></tr><tr><td>13</td><td>18</td><td>1.076</td><td>0.204</td><td>0.137</td><td>0.045</td><td>0.871</td><td>56</td></tr><tr><td>13</td><td>0</td><td>0.843</td><td>0.162</td><td>0.102</td><td>0.035</td><td>0.681</td><td>56</td></tr><tr><td>14</td><td>29</td><td>0.751</td><td>0.130</td><td>0.084</td><td>0.017</td><td>0.621</td><td>56</td></tr><tr><td>14</td><td>31</td><td>0.685</td><td>0.126</td><td>0.091</td><td>0.023</td><td>0.558</td><td>56</td></tr><tr><td>13</td><td>13</td><td>0.690</td><td>0.140</td><td>0.064</td><td>0.024</td><td>0.550</td><td>56</td></tr><tr><td>13</td><td>4</td><td>0.575</td><td>0.121</td><td>0.095</td><td>0.029</td><td>0.454</td><td>56</td></tr><tr><td>13</td><td>17</td><td>0.549</td><td>0.110</td><td>0.056</td><td>0.020</td><td>0.438</td><td>56</td></tr><tr><td>14</td><td>13</td><td>0.513</td><td>0.090</td><td>0.104</td><td>0.027</td><td>0.423</td><td>56</td></tr><tr><td>14</td><td>30</td><td>0.519</td><td>0.098</td><td>0.044</td><td>0.012</td><td>0.421</td><td>56</td></tr><tr><td>13</td><td>16</td><td>0.499</td><td>0.099</td><td>0.116</td><td>0.039</td><td>0.400</td><td>55</td></tr></table>

Table 6: Top 10 attention heads by translation-specific effect $\Delta = { \tt R E A L } .$ \_CROSS − NULL\_CROSS under promptswap GCM (Llama), with mean $| \widehat { \mathrm { I E } } |$ in each of the four control tasks. #dir is the number of the 56 cross-language directions in which the head ranks in that direction’s top-30 by REAL\_CROSS effect; all but one head appear in every direction.

<table><tr><td></td><td></td><td colspan="4"> $| \widehat { \mathrm { I E } } |$  mean</td><td></td><td></td></tr><tr><td>Layer</td><td>Head</td><td>real_cr</td><td>null_cr</td><td>real_sm</td><td>null_sm</td><td> $\Delta$ </td><td>#dir</td></tr><tr><td>15</td><td>19</td><td>0.867</td><td>0.224</td><td>0.179</td><td>0.057</td><td>0.644</td><td>56</td></tr><tr><td>18</td><td>25</td><td>0.436</td><td>0.089</td><td>0.072</td><td>0.020</td><td>0.347</td><td>56</td></tr><tr><td>16</td><td>1</td><td>0.417</td><td>0.090</td><td>0.044</td><td>0.013</td><td>0.327</td><td>56</td></tr><tr><td>17</td><td>18</td><td>0.379</td><td>0.074</td><td>0.067</td><td>0.014</td><td>0.305</td><td>56</td></tr><tr><td>16</td><td>10</td><td>0.375</td><td>0.093</td><td>0.111</td><td>0.042</td><td>0.282</td><td>56</td></tr><tr><td>17</td><td>6</td><td>0.318</td><td>0.069</td><td>0.038</td><td>0.011</td><td>0.248</td><td>55</td></tr><tr><td>20</td><td>29</td><td>0.319</td><td>0.076</td><td>0.054</td><td>0.012</td><td>0.243</td><td>56</td></tr><tr><td>18</td><td>24</td><td>0.305</td><td>0.063</td><td>0.058</td><td>0.016</td><td>0.242</td><td>55</td></tr><tr><td>16</td><td>29</td><td>0.317</td><td>0.085</td><td>0.065</td><td>0.018</td><td>0.232</td><td>56</td></tr><tr><td>20</td><td>20</td><td>0.311</td><td>0.082</td><td>0.105</td><td>0.014</td><td>0.229</td><td>50</td></tr></table>

Table 7: Top 10 Aya attention heads by translation-specific effect $\Delta = \mathrm { R E A L \_ C R O S S } - \mathrm { N U L L }$ CROSS under prompt-swap GCM (cf. Table 6). #dir is the number of the 56 cross-language directions in which the head ranks top-30 by REAL\_CROSS effect.

![](images/9a73ac07af9a8344847b807d097892d56678c0ffd570e59d148164710138c566.jpg)  
Figure 8: Llama-3.1-8B mean |IEc| for top SAE features under translation and control prompts.

![](images/bfe7a4f28fef307d95e83416b046c0a81fb97df5270cb5a583a641fa11227350.jpg)  
Figure 9: Aya-23-8B mean |IEc| for top SAE features under translation and control prompts.

<table><tr><td></td><td colspan="5">Llama-3.1-8B</td><td colspan="5">Aya-23-8B</td></tr><tr><td>target</td><td>base</td><td>POS-10</td><td>ctrlp</td><td>NEG-10</td><td> $\mathrm { c t r l _ { N } }$ </td><td>base</td><td>POS-10</td><td>ctrlp</td><td>NEG-10</td><td>ctrlN</td></tr><tr><td>ara</td><td>7.44</td><td>-0.055</td><td>-0.066</td><td>+0.077</td><td>+0.050</td><td>6.92</td><td>-0.021</td><td>-0.026</td><td>+0.035</td><td>+0.009</td></tr><tr><td>deu</td><td>10.18</td><td>-0.111</td><td>-0.019</td><td>+0.263</td><td>-0.076</td><td>7.88</td><td>-0.037</td><td>-0.026</td><td>+0.089</td><td>-0.023</td></tr><tr><td>eng</td><td>5.23</td><td>-0.025</td><td>-0.095</td><td>+0.050</td><td>-0.002</td><td>4.09</td><td>-0.009</td><td>-0.006</td><td>+0.023</td><td>0.000</td></tr><tr><td>fra</td><td>9.16</td><td>-0.072</td><td>-0.087</td><td>+0.172</td><td>+0.051</td><td>6.33</td><td>-0.008</td><td>+0.026</td><td>+0.036</td><td>0.000</td></tr><tr><td>heb</td><td>11.49</td><td>-0.075</td><td>+0.018</td><td>+0.113</td><td>-0.108</td><td>7.75</td><td>-0.015</td><td>-0.050</td><td>+0.049</td><td>-0.013</td></tr><tr><td>hin</td><td>6.47</td><td>-0.155</td><td>-0.188</td><td>+0.232</td><td>+0.012</td><td>7.61</td><td>-0.094</td><td>-0.162</td><td>+0.175</td><td>-0.077</td></tr><tr><td>spa</td><td>9.61</td><td>-0.105</td><td>-0.093</td><td>+0.224</td><td>-0.069</td><td>6.58</td><td>-0.017</td><td>-0.031</td><td>+0.056</td><td>-0.015</td></tr><tr><td>tur</td><td>9.56</td><td>-0.138</td><td>-0.027</td><td>+0.304</td><td>+0.093</td><td>9.45</td><td>-0.005</td><td>-0.085</td><td>+0.109</td><td>-0.071</td></tr><tr><td>mean</td><td>8.64</td><td>-0.092</td><td>-0.069</td><td>+0.179</td><td>-0.006</td><td>7.08</td><td>-0.026</td><td>-0.045</td><td>+0.072</td><td>-0.024</td></tr></table>

Table 8: Multi-BLiMP margin $\Delta _ { \mathrm { c h a n g e } }$ relative to baseline from mean-ablating the POS-10 and NEG-10 GCM head sets and their size-matched controls (ctrl and ctrl : 10 random heads drawn from the same layers as the corresponding set). Negative means ablation weakened the grammaticality preference. Ablating POS-10 lowers the margin in all 8 targets and ablating NEG-10 raises it in all 8, for both models; the size of the POS-10 effect, however, is comparable to that of its random same-layer control. n = 400 pairs per language except heb (n = 200) and hin (n = 100).

![](images/a487c34c841b9d3820d6777c59f7569efaa932052601e0e44295aee14f251772.jpg)  
Figure 10: Full Llama-3.1-8B signed-IE heatmap for the 15 most universal prompt-swap heads across all 56 translation directions.

![](images/92e687b80a4e37d5cd84801deb1a1a45f6d74072dde5cd111e9adbda2c2dd8d0.jpg)  
Figure 11: Full Aya-23-8B signed-IE heatmap for the 15 most universal prompt-swap heads across all 56 translation directions.

![](images/25f580d56876b5768fc3ae39a6cda1857b5b758e6f8ccb80f5505d3303d4444a.jpg)  
Figure 12: Per-direction head-IE sparsity under promptswap GCM: the ratio of the largest head’s mean |IEc| to the median head’s, for each source→target direction (Llama-3.1-8B; diagonal omitted). The ratio ranges from 60 to 162 (median 116). The top head is roughly two orders of magnitude above the median head in every direction, and is largest for directions into Spanish, German, and French and smallest into Hindi and Hebrew.

![](images/d56ae2fb848c0e2ff60fd59402757d9faece60acc4ef7827140e899309c481cb.jpg)  
Figure 13: Per-direction head-IE sparsity for Aya-23- 8B under prompt-swap GCM (cf. Fig. 12). The tophead/median-head ratio ranges from 51 to 104 (median 83).

![](images/27738db310c687abf6af4b34f69d4330aa329d7b05cf0cceb6bbf43167cb0320.jpg)

Aya-23-8B: Multi-BLiMP  
![](images/2f459726ceb5e6be940e3f1848eb376f3d1a2c7bff97abd8212b7a8f868be9e0.jpg)  
Figure 14: Change in the Multi-BLiMP grammaticality margin from mean-ablating the POS-10 and NEG-10 head sets and a random control. Dotted lines mark crosslanguage means.

![](images/045f145b85f42008f7074f3da8a8129c2f8aad8d37f203af07e8b9fc98cf47a8.jpg)  
Figure 15: Head-count dose response (deu/eng/fra) on Llama. The effect is monotonic with respect to the number of heads ablated.

## C Additional Fine-Tuning Results

This section reports the remaining fine-tuning results. We use the same models, training conditions, and evaluation procedure described in the main text.

## C.1 Additional directions involving Xhosa

Table 9 reports translation into Xhosa from English, French, and German. Both fine-tuning conditions improve performance across all three directions and both models. For Llama, parallel fine-tuning performs best when translating from English and French, while monolingual fine-tuning performs best when translating from German. For Aya, monolingual fine-tuning performs slightly better in all three directions. Thus, neither condition performs best in every setting.

<table><tr><td>Model</td><td>Method</td><td>Eng.</td><td>Fra.</td><td>Deu.</td></tr><tr><td>Llama</td><td>Base</td><td>1.39 2.49</td><td>0.57</td><td>0.91</td></tr><tr><td>Llama Llama</td><td>Monolingual Parallel</td><td>4.25</td><td>1.24 1.60</td><td>1.95 1.85</td></tr><tr><td>Aya</td><td>Base</td><td>2.47</td><td>1.42</td><td>0.96</td></tr><tr><td>Aya</td><td>Monolingual</td><td>4.00</td><td>2.00</td><td>2.13</td></tr><tr><td>Aya</td><td>Parallel</td><td>3.79</td><td>1.97</td><td>1.95</td></tr></table>

Table 9: BLEU for translation into Xhosa. Column labels identify the source language. Bold marks the best fine-tuned result for each model and direction.

## C.2 Fine-tuning on higher-resource languages

We also apply the same general pipeline to German and Thai using Llama-3.1-8B. Unlike Xhosa, both languages already have strong translation performance in the base model. As shown in Table 10, neither monolingual nor parallel fine-tuning produces a consistent improvement. Monolingual fine-tuning leaves performance largely unchanged, while parallel fine-tuning reduces performance in several directions.

One possible explanation is that German and Thai were already well represented during pretraining, leaving less room for improvement from continued training. Some of the fine-tuning documents may also have appeared in the pretraining corpus.

## C.3 Preservation of monolingual abilities

We also measure MultiBLiMP accuracy after finetuning Llama on Xhosa. As shown in Table 11, performance is largely preserved. However, the base model is already near the maximum score on all three languages. These results are therefore not strong enough to determine whether fine-tuning causes cross-lingual transfer of linguistic abilities.

<table><tr><td></td><td colspan="2">Adapted language</td><td colspan="2">Other directions</td></tr><tr><td>Method</td><td>Into Eng.</td><td>From Eng.</td><td>Fra. into Eng. Xho. into Eng.</td><td></td></tr><tr><td colspan="5">German adaptation</td></tr><tr><td>Base</td><td>42.69</td><td>34.94</td><td>43.27</td><td>16.84</td></tr><tr><td>Monolingual</td><td>43.48</td><td>34.55</td><td>43.16</td><td>19.13</td></tr><tr><td>Parallel</td><td>42.95</td><td>33.26</td><td>42.53</td><td>14.90</td></tr><tr><td colspan="5">Thai adaptation</td></tr><tr><td>Base</td><td>32.60</td><td>15.21</td><td>43.27</td><td>16.84</td></tr><tr><td>Monolingual</td><td>32.19</td><td>15.19</td><td>43.34</td><td>16.75</td></tr><tr><td>Parallel</td><td>30.98</td><td>10.31</td><td>42.46</td><td>14.95</td></tr></table>

Table 10: BLEU after fine-tuning Llama-3.1-8B on German or Thai. The first two result columns involve the adapted language; the final two measure other translation directions.
<table><tr><td>Model</td><td>English</td><td>French</td><td>German</td></tr><tr><td>Base</td><td>99.35</td><td>98.70</td><td>97.91</td></tr><tr><td>Xhosa monolingual</td><td>99.48</td><td>98.43</td><td>98.17</td></tr><tr><td>Xhosa parallel</td><td>99.22</td><td>96.15</td><td>95.65</td></tr></table>

Table 11: MultiBLiMP accuracy after fine-tuning Llama-3.1-8B. Scores are percentages. The near-ceiling base scores make small differences difficult to interpret.

Only the monolingual condition includes replay data from high-resource languages. Differences in retained performance therefore cannot be attributed only to the use of monolingual rather than parallel data. A parallel condition with similar replay data could also reduce forgetting.

## D Artifact Licenses

We use several existing scientific artifacts in our experiments. FLORES-101 (Goyal et al., 2022) is released under a CC BY-SA 4.0 license. Multi-BLiMP (Jumelet et al., 2026) is released under a CC BY 4.0 license. GlobalMMLU (Singh et al., 2025) is released under an Apache 2.0 license. For the Xhosa–English fine-tuning experiments, we use the English–Xhosa MT560 sentence-pair dataset (Gowda et al., 2021); because OPUS MT560 aggregates data from multiple sources, downstream users should consult the original OPUS MT560 provenance information before redistributing or extending the dataset.

We also use pretrained model checkpoints and evaluation software. Llama-3.1-8B is released under the Llama 3.1 Community License, and Aya-23-8B is released under CC BY-NC 4.0 with Cohere’s acceptable-use addendum. SacreBLEU (Post, 2018), which we use to compute BLEU scores, is released under an Apache 2.0 license.

In accordance with their licensing terms, all artifacts are used solely for research purposes.

## E Computational Budget

All experiments use 8B-parameter language models: Llama-3.1-8B and Aya-23-8B. We do not train any models from scratch. The compute cost is dominated by translation generation and scoring, activation caching, gradient-based attribution patching, ablation experiments, and LoRA fine-tuning for the Xhosa adaptation experiments.

The total computational budget was approximately 200 GPU hours on NVIDIA A100/H100 GPUs. The largest individual runs were the causal mediation experiments, which require forward passes with activation caching and backward passes for attribution estimates across attention heads, and the LoRA fine-tuning sweeps over learning rate, LoRA rank, and number of training examples.

For activation caching, we used NNSIGHT (Fiotto-Kaufman et al., 2025).