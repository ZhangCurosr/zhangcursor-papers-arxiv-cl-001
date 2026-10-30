# Prompts and probabilities in the linguistic assessment of language models: Evidence from semantic focus-sensitivity

Robert Pasternak

Leibniz-Centre General Linguistics (ZAS)

mail@robertpasternak.com

## Abstract

As language models (LMs) have improved, an increasingly common means of evaluating their linguistic competence is through explicit prompting (e.g., “Which sentence is more acceptable?”). Prior work indicates that such prompting underperforms compared to standard token-probability-based methods in smaller models, but improves in larger models. We test this in a new domain: semantic focus-sensitivity, responsible for contrasts like Lucy only SWAM yesterday vs. Lucy only swam YESTERDAY. Testing models of various sizes using text-based focus markers like boldface and italics, we find universal above-chance performance for token probabilities. Prompting results mostly align with previous findings, but in some larger models they outperform probabilities by a wide margin. We propose that this is because LMs are not prone to produce outof-the-blue focus marking, which muddies results from token probabilities in a way that does not affect prompting. We test this by replacing standard focus markers with randomly generated ones; token probabilities still beat chance but show significantly greater performance decrease compared to prompting. These results suggest that metalinguistic prompting, while non-trivial for LMs, can also be more robust when testing stylistically atypical data. We argue that this has important consequences for the assessment of LMs’ language competence, even if LMs do not perform metalinguistic judgments through human-like introspection.<sup>1</sup>

## 1 Introduction

The linguistic competence of language models (LMs) is often assessed by means of their output token probabilities: roughly, well-formed sentences are ceteris paribus expected to have higher token or string probabilities than ill-formed ones. While some prior work has used token probabilities to directly calculate acceptability scores on a per-sentence basis (Clark et al. 2013; Lau et al. 2015), a more common technique is to compare the probabilities assigned to minimal pairs, nearidentical pairs of sentences that differ in a single, targeted way, such as Mary is/\*are sad (e.g., Linzen et al. 2016; Gulordava et al. 2018; Marvin and Linzen 2018; Futrell et al. 2019; Warstadt et al. 2020). Comparison of minimal pairs is common in theoretical and experimental research on human language (see e.g. Sprouse et al. 2013), and is useful in NLP research because it helps control for irrelevant factors that affect token or string probabilities, such as sentence length and the general frequency of individual tokens.

Thanks to major improvements in LM capabilities, an increasingly common alternative means of evaluating LMs’ linguistic competence is to simply present the relevant sentence(s) as part of a prompt to the model, along with a request for a judgment pertaining to well-formedness (Dentella et al. 2023; Katzir 2023; Mahowald 2023; Murphy et al. 2025) or semantic/pragmatic truth or acceptability (e.g. Kamath et al. 2024; Tsvilodub et al. 2024). Such metalinguistic prompting is similar to judgment tasks commonly used in experimental research on humans, making it easy to construct studies comparing human and LM responses to more or less the same input.

However, research directly comparing metalinguistic prompting to token probabilities shows very different performance between the two methods. Weaker LMs generally perform much worse when assessed by prompting than they do when assessed through string probabilities (Hu and Levy 2023; Hu et al. 2024; Kauf et al. 2024), leading to significant underestimation of these models’ language competence. Hu and Frank (2024) observe that this gap in accuracy shrinks as models get larger and more capable; they attribute this to the auxiliary task demands of metalinguistic judgments, which stronger models are better able to cope with. Song et al. (2025), meanwhile, find that with sufficiently large models, metalinguistic judgments can slightly but significantly outperform string probabilities.

In this paper we report two studies extending this foundational methodological research to a new empirical domain. Our results suggest a novel generalization about the accuracy of prompting versus string probabilities: while metalinguistic prompting is indeed only accessible to sufficiently large models, it can also be more robust than string probabilities when evaluating sentences that deviate from a model’s stylistically typical outputs.

Our empirical domain is semantic focussensitivity, a linguistic phenomenon in which the truth or pragmatic acceptability of a sentence depends on the placement of stress. By way of illustration, consider an adverb like only. If Mary gave one book to her teacher and one to her friend (and nothing to anyone else), then (1) is true and (2) false, despite differing only in focus placement (Jackendoff 1972; Rooth 1985):

(1) Mary only gave her teacher a BOOK.

(2) Mary only gave her TEACHER a book.

As discussed in greater detail in Section 2, semantic focus-sensitivity arises in a variety of environments in human language. Notably, in many cases the appropriateness of focus placement is dependent on context: (1) and (2) are both fine sentences in and of themselves, and it is only the context that makes (1) acceptable and (2) not.

Our goals here are twofold. First, we wish to determine whether LMs have learned focussensitivity. And second, we compare and contrast string probabilities and prompting as methods of linguistically evaluating LMs. We test instruction-tuned models of various sizes from three model families—Gemma 4 (Gemma Team 2026), Qwen3 (Yang et al. 2025), and Qwen3.5 (Qwen Team 2026)—on a manually created 282- item minimal pair dataset featuring various kinds of semantic focus-sensitivity, comparing the results using string probabilities to those using two types of prompts: forced choice, in which the LM is asked to choose which of the two sentences in a pair is more acceptable; and yes/no, in which the LM is presented with one sentence at a time

and asked for a yes-or-no acceptability judgment.   
These methods are illustrated in Table 1.

In Experiment 1, discussed in Section 3, we test how models perform using eight relatively standard methods for text-based focus-marking: all caps; HTML and Markdown boldface, italics, and emphasis; and HTML underlining. We find that for all models, string probabilities perform significantly above chance, suggesting substantial LM acquisition of semantic focus-sensitivity.

Regarding the methodological comparison, in certain respects the results accord nicely with prior work: prompting methods perform poorly for small models and gradually improve as models get bigger, with forced choice outperforming yes/no prompting (cf. Panickssery et al. 2024). However, in the case of the larger Gemma 4 models forced choice prompting not only beats string probabilities, but does so by a wide margin. We hypothesize that this is because the models are not prone to produce out-of-the-blue focus marking, which lowers the general probability of unsolicited focus marking and muddies string probability results. But for models that are strong enough to perform forced choice, such stylistic unnaturalness is less of a hindrance, since (I) the focus marking is in the input, rather than the output; and (II) the side-by-side comparison of paired sentences clarifies the criteria of evaluation, hence why forced choice also outperforms yes/no prompting.

To test this, in Experiment 2 (Section 4) we replace standard focus markers with randomly generated focus bracketing (e.g., Mary only gave her teacher a \$:#book#:\$), on the assumption that these deviate even further from the models’ expected outputs. While we unsurprisingly find general accuracy decreases, string probabilities continue to beat chance for all models, suggesting that what the models have learned is more general and not confined to a closed class of canonical textbased focus markers. However, string probabilities also show a proportionally greater decline in accuracy than forced choice prompting, with the number of instances of forced choice significantly outperforming string probabilities more than doubling. This supports the hypothesis that while forced choice prompting is more difficult for a model to acquire, it can also be more robust than string probabilities when evaluating sentences that deviate from a model’s stylistic inclinations.

In Section 5 we discuss the consequences of these results for the linguistic assessment of LMs. This is especially pertinent because recent work suggests that the language knowledge LMs use to make metalinguistic judgments may be distinct from the implicit knowledge they use to generate outputs, inviting questions about which is the true measure of an LM’s linguistic competence, regardless of comparative accuracy (Song et al. 2025). However, we argue that even if we accept these results at face value, there needn’t be one “true” indicator of an LM’s language abilities: it is equally plausible that these methods test distinct but coequal facets of the core language modeling task, loosely analogous to the contrast between production (string probabilities) and comprehension (prompting). From this perspective, prompting and probabilities complement each other, and studying both promises to give a more holistic picture of LM competence than can be achieved through one methodology alone.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>String Prob.</td><td rowspan=1 colspan=1>Forced Choice</td><td rowspan=1 colspan=1>Yes/No</td></tr><tr><td rowspan=1 colspan=1>SYSTEM</td><td rowspan=1 colspan=3>You are a helpful assistant.</td></tr><tr><td rowspan=1 colspan=1>USER</td><td rowspan=1 colspan=1>Who did Phil give</td><td rowspan=2 colspan=1>You will be presented with a contextfollowed by two sentences, A and B.Your job is to respond with which sen-tence is better in the context provided.CONTEXT:Who did Phil give a book to?SENTENCE A: Phil gave a book tothe SENATOR.SENTENCE B: Phil gave a BOOK tothe senator.</td><td rowspan=2 colspan=1>You will be presented with a contextfollowed by a sentence. Your job is torespond with &quot;Yes&quot; or &quot;No&quot;, indicat-ing whether the sentence is acceptablein the context provided.CONTEXT:Who did Phil give a book to?SENTENCE: Phil gave a book to theSENATOR.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>ASSISTANT</td><td rowspan=1 colspan=1>Phil gave a bookto the SENATOR./Phil gave a BOOKto the senator.</td><td rowspan=1 colspan=1>My answer is A / B</td><td rowspan=1 colspan=1>My answer is Yes / No</td></tr></table>

Table 1: Illustration of string probability, forced choice, and yes/no evaluation methods.

## 2 Focus-sensitive environments

As was mentioned in the introduction, focussensitivity is common across a wide range of environments. We have already seen one example: association with only, as exemplified by the contrast between (1) and (2). Another example is questionanswer congruence, which ties the use of focus in an answer to its semantic relationship with the question (von Stechow 1991; Rooth 1992). Thus, in response to Who did Phil give a book to?, (3) is fine, while (4) is odd:

(3) Phil gave a book to the SENATOR.

(4) Phil gave a BOOK to the senator.

Yet another example is the focus-sensitivity of additive too: as a continuation of The doctor greeted the nurse, (5) is acceptable, while (6) is not (Kadmon 2001; Kripke 2009):

(5) and the PATIENT greeted the nurse, too.

(6) and the patient greeted the NURSE, too.

A fourth example of focus-sensitivity can be seen in what Lewis (1975) calls adverbs of quantification, such as always and usually (Rooth 1985, Partee 1991, Krifka 1992, von Fintel 1994). If Sarah rarely runs, but when she does she opts for a morning run, then (7) is true, while (8) is not:

(7) Sarah usually runs in the MORNING.

(8) Sarah usually RUNS in the morning.

Finally there is contrastive focus, such as socalled contrastive negation: if someone claims that Sarah bought a green shirt when she actually bought a red one, then (9) is a good response, while (10) is not (Jackendoff 1972):

(9) Sarah didn’t buy the GREEN shirt.

(10) Sarah didn’t buy the green SHIRT.

Contrastive focus often comes in pairs. Thus, (9) is better followed by (11) than (12):

(11) . . . she bought a RED shirt.

(12) . . . she bought a red SHIRT.

These do not constitute the full range of focussensitive environments (see Kadmon 2001; Beaver and Clark 2008 for more). However, we consider the environments discussed above to be both diverse and commonplace enough to constitute a reasonable test domain for LM competence. With this in mind, we now turn to our experiments.

## 3 Experiment 1: Standard focus markers

## 3.1 Methodology

We tested LMs’ semantic focus-sensitivity using a manually created dataset consisting of 282 minimal pairs, representing each kind of focussensitivity discussed above: association with only (ONLY, 52 items), question-answer congruence (QA, 50 items), association with too (TOO, 50 items), adverbs of quantification (ADV\_QT, 40 items), contrastive negation (CONTR1, 40 items), and contrastive focus following negation as in (11) (CONTR2, 50 items). In order to be able to test a variety of different text-based focus markers, the dataset was created as a “seed” with focus indicated by bracketing; bracketed material could then be replaced with any of a variety of focus markers.

Every item came with either a background, i.e., information that is required to assess the acceptability of a sentence in the minimal pair; or a prefix, the beginning of a statement for which the minimal pair sentences would serve as continuations. ONLY and ADV\_QT items came with informational backgrounds that entailed the good sentence, but not the bad sentence.<sup>2</sup> QA items’ backgrounds were simply the question to be answered, while CONTR1 items had backgrounds that made only the good sentence pragmatically felicitous. TOO and CONTR2 items had prefixes. Importantly, none of the backgrounds or prefixes had any focus marking themselves, so in all items focus marking was “out-of-the-blue”. Examples of the six categories of item can be seen in Table 2.

For Experiment 1, we tested eight anecdotally common ways of textually marking focus: all caps (standard in theoretical linguistics); HTML underlining; and HTML and Markdown boldface, italicized, and emphasized fonts. Thus, the full dataset consisted of $8 \times 2 8 2 = 2 { , } 2 5 6$ pairs.

We tested seventeen instruction-tuned models from three model families: Gemma 4 (Gemma

Team 2026), Qwen3 (Yang et al. 2025), and Qwen3.5 (Qwen Team 2026). These model families were selected because each has a relatively wide range of model sizes. The tested models ranged in size from 600 million to 35 billion parameters. The Gemma 4 models tested were E2B, E4B, 12B, 26B-A4B, and 31B; the Qwen3 models were 0.6B, 1.7B, 4B, 8B, 14B, and 32B; and the Qwen3.5 models were 0.8B, 2B, 4B, 9B, 27B, and 35B-A3B. Notably, for Gemma 4 and Qwen3.5, some of the models are architecturally distinct enough from their peers to preclude a true apples-to-apples size comparison, whether that be due to a mixture-of-experts architecture (Gemma 4 26B-A4B, Qwen3.5 35B-A3B) or per-layer embeddings (Gemma 4 E2B, E4B). Thus, discussion of the impact of model size will be framed exclusively in descriptive terms.

Each model was tested on the full dataset using three evaluation methods: string probabilities (SP), forced choice (FC), and yes/no (YN). Each method involved evaluating the models based on various output token probabilities, whether they be for the sentences themselves (SP) or for potential metalinguistic judgments (FC/YN); these token probabilities were obtained using the minicons package (Misra 2022).

String probabilities (SP) For the first method, whether a model got a minimal pair correct was determined by whether it assigned a higher log probability to the good sentence than to the bad sentence in a specific context �:

$$
\log P ( s _ { \mathrm { g o o d } } \mid C ) > \log P ( s _ { \mathrm { b a d } } \mid C )
$$

The context for a given item was determined as follows. All message histories started with a system message of “You are a helpful assistant.” If there was a background, it was encoded as a user message; if not, the user message was empty. If there was a prefix, then this was encoded as the beginning of an assistant message that the LM was to continue. For ONLY and ADV\_QT items, in addition to providing the background as a user message, the context included “Therefore, ” as an assistant message prefix; this was done to encourage LM outputs that served as deductions based on the information in the background.

Forced choice (FC) The first prompting method we tested was a forced choice paradigm. The system prompt (“You are a helpful assistant.”) was followed by a user message with a brief introduction to the task, the relevant background or prefix, and both the good and bad follow-up sentences (Sentences A and B). The model was then tasked with responding with “A” or “B”, based on which sentence was preferred. Two prompts were tested (FC1/FC2), with slight variations in both how the instructions were presented and whether the tested LM response was simply “A”/“B” or included an assistant message prefix of “My answer is ”.

<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Background/Prefix</td><td rowspan=1 colspan=1>Sentences</td></tr><tr><td rowspan=1 colspan=1>ONLY</td><td rowspan=1 colspan=1>Background: Larry&#x27;s Pub serves drinks at the bar.They do not serve drinks at other places.</td><td rowspan=1 colspan=1>Larry&#x27;s Pub only serves drinks at the BAR.Larry&#x27;s Pub only serves DRINKS at the bar.</td></tr><tr><td rowspan=1 colspan=1>QA</td><td rowspan=1 colspan=1>Background: Who did Phil give a book to?</td><td rowspan=1 colspan=1>Phil gave a book to the SENATOR.Phil gave a BOOK to the senator.</td></tr><tr><td rowspan=1 colspan=1>TOO</td><td rowspan=1 colspan=1>Prefix: A bonus was given to the senior engineer,</td><td rowspan=1 colspan=1>and the JUNIOR engineer got a bonus, too.and the junior engineer got a BONUS, too.</td></tr><tr><td rowspan=1 colspan=1>ADV_QT</td><td rowspan=1 colspan=1>Background: When Dolly buys rugs, it&#x27;s with cashmost of the time.</td><td rowspan=1 colspan=1>Dolly mostly uses CASH to buy rugs.Dolly mostly uses cash to buy RUGS.</td></tr><tr><td rowspan=1 colspan=1>CONTR1</td><td rowspan=1 colspan=1>Background: Sarah was considering buying agreen shirt. She ended up buying a red one instead.</td><td rowspan=1 colspan=1>Sarah didn&#x27;t buy the GREEN shirt.Sarah didn&#x27;t buy the green SHIRT.</td></tr><tr><td rowspan=1 colspan=1>CONTR2</td><td rowspan=1 colspan=1>Prefix: Sandy didn&#x27;t unlock the door.</td><td rowspan=1 colspan=1>The door is still LOCKED.The DOOR is still locked.</td></tr></table>

Table 2: Examples of the six categories of item in the dataset. For these examples, focus is marked with all caps.

To control for a potential ordering bias (Pezeshkpour and Hruschka 2024), each item was run twice: once with the good sentence as Sentence A, and once as Sentence B. A model was considered to have gotten a given item correct if the summed log probabilities of the correct answers exceeded the summed log probabilities of the incorrect answers:

$$
\begin{array} { r l } & { \log P ( ^ { \mathrm { * * } } \mathrm { A } ^ { \mathrm { * * } } \mid C _ { \mathrm { A } } ) + \log P ( ^ { \mathrm { * * } } \mathrm { B } ^ { \mathrm { * * } } \mid C _ { \mathrm { B } } ) > } \\ & { \log P ( ^ { \mathrm { * * } } \mathrm { A } ^ { \mathrm { * } } \mid C _ { \mathrm { B } } ) + \log P ( ^ { \mathrm { * * } } \mathrm { B } ^ { \mathrm { * * } } \mid C _ { \mathrm { A } } ) } \end{array}
$$

Yes/No (YN) The other metalinguistic prompting method we used was a simple yes-or-no prompt, where the model was presented with one sentence at a time along with the background/prefix, and was asked to answer “Yes” or “No” depending on if the sentence was acceptable. We again tested this paradigm with two prompts (YN1/YN2), which varied similarly to FC’s.

To decide the accuracy for a given item, we adopted a similar method as for FC: a model got an item correct if the summed log probabilities of accurate answers exceeded the summed log probabilities of inaccurate answers:

$$
\begin{array} { r } { \log P ( ^ { \mathrm { * * } } \mathrm { Y e s } ^ { \mathrm { , * } } \mid C _ { \mathrm { g o o d } } ) + \log P ( ^ { \mathrm { * } } \mathrm { N o } ^ { \mathrm { , * } } \mid C _ { \mathrm { b a d } } ) > } \\ { \log P ( ^ { \mathrm { * } } \mathrm { Y e s } ^ { \mathrm { , * } } \mid C _ { \mathrm { b a d } } ) + \log P ( ^ { \mathrm { * } } \mathrm { N o } ^ { \mathrm { , * } } \mid C _ { \mathrm { g o o d } } ) } \end{array}
$$

The items in the dataset are designed so that the good and bad sentences are both acceptable in and of themselves, and it is only the background or prefix that determines which sentence is the better one. But there is a potential for bias in the dataset. Note that for a sentence like Mary greeted the vicar, backgrounds could be devised that make the focusing of either the or vicar felicitous; however, the latter is more natural out of the blue. Thus, if a background is provided that encourages the focusing of vicar, and a model gets that right, then there is a risk that the model is relying on a simpler syntactic heuristic dictating that nouns like vicar are more likely to be focused than determiners like the, rather than using the background information as desired (cf. Gururangan et al. 2018 for similar observations on NLI datasets). To test for such bias, a control version of each item was run with the background/prefix removed, for all models and evaluation methods. Whether or not there was a bias in the dataset could then be inferred based on whether models’ performance significantly deviated from chance.

## 3.2 Results

The results of Experiment 1 can be seen in Figure 1. Starting with the control items, larger models tended to perform slightly but significantly above chance, suggesting that there is in fact a non-trivial degree of bias in the dataset. Nonetheless, the results for the target items outpace the controls by a wide enough margin that we can safely conclude that positive target results are not attributable to mere bias, and the LMs’ success is significantly dependent on the backgrounds and prefixes.

![](images/08fd9a7f3a995c0d27f00ef2023a4da56faad68b01f1d7e4a65cbb1608bea27e.jpg)  
Figure 1: Results of Experiment 1 (standard focus markers). Error bars are bootstrapped 95% confidence intervals. X-axes are logarithmically spaced.

Moving on to the target items, many of the results for focus look similar to observations from prior work. To test for differences between methods, we fitted a mixed-effect logistic regression for each LM, predicting accuracy with a fixed effect of method (baseline = SP) and random effects for focus marker and item; statistical tests were performed using the lme4 R package (Bates et al. 2015). SP performed above chance for all models (� < .001 across the board). For the smaller models, SP also significantly outperformed the prompting-based methods, in line with previous observations (Hu and Levy 2023; Hu et al. 2024; Kauf et al. 2024). FC and YN generally improved as models got bigger, which fits neatly with prior results from Hu and Frank (2024).

Among prompting methods, FC generally outperformed YN, especially in larger models. This is unsurprising, since FC has the advantage of directly comparing the paired sentences, though we also cannot rule out the possibility that YN could improve with prompt engineering. FC outperforming YN also fits well with prior work on LMs (e.g., Panickssery et al. 2024), as well as with research on humans, where forced choice results appear to be more powerful than single-item evaluations for both linguistic acceptability judgments (Sprouse et al. 2013) and tasks in other cognitive domains (Gigerenzer and Richter 1990).

However, a notable wrinkle in our results comes from those cases where prompting significantly outperformed string probabilities. This was observed for Qwen3.5 27B, where FC1 outperformed SP (� < .001), but the gap was much wider for Gemma 4 12B FC2, 26B-A4B FC2, and 31B FC1 and FC2 (all � < .001), each of which outperformed their respective SP scores by a considerable margin. In fact, among all model-method pairs, each of the top three performers used FC: Gemma 4 31B FC2 (95.3%) and FC1 (91.7%), and Qwen3.5 27B FC1 (91.3%).

Note also that the two mixture-of-experts models, Gemma 4 26B-A4B and Qwen3.5 35B-A3B, saw a dip in performance compared to smaller non-MoE models in the same family. This suggests that MoE’s reduction in active parameters has a negative impact on performance that is especially felt by prompting-based methods.

![](images/1de723501047470ac536971d2884e93bc9829e98adfff061b17f179e66951b57.jpg)  
Figure 2: SP/FC accuracy by type of focus-sensitivity for the top-performing model in each family. Error bars are bootstrapped 95% confidence intervals.

Figure 2 zooms in on the best-performing model in each model family, showing SP and FC accuracy for each focus-sensitive environment. On a descriptive level, ONLY items mostly saw high accuracy across models and methods, while TOO items showed lower accuracy. However, no definitive conclusions can be drawn from such observations, since manually created data could contain biases that make some items more difficult for independent reasons (Ettinger et al. 2018). More importantly for our purposes, note that Gemma 4 31B FC1 and FC2 outperformed SP across all categories, suggesting that the advantage of FC was general and not confined to a subclass of environments. The picture is more mixed for Qwen3.5 27B FC1 (which also beat SP), partly due to SP’s near-ceiling performance in multiple categories.

As was discussed above, the fact that SP outperforms prompting in smaller models is relatively unsurprising in light of previous work: SP is a direct reflection of the language modeling task, whereas metalinguistic judgments are a more demanding task that smaller models will tend to struggle with (Hu and Frank 2024). The puzzle, then, is how FC can come to significantly outperform SP, especially by the wide margins seen for Gemma 4. The answer, we propose, is that once FC is acquired, it is better equipped than SP to evaluate sentences that diverge substantially from normal model outputs. Recall that in the dataset focus marking is always out of the blue, as it is absent from the backgrounds and prefixes. It seems at least plausible that a result of the models’ training would be a disinclination to produce, say, HTML <b>. . . </b> tags without solicitation. This could in turn muddy the waters when it comes to comparing the string probabilities of sentences that do contain such out-of-the-blue tags, which would exert downward pressure on SP accuracy.

FC, meanwhile, has two advantages over SP in this regard. First, the evaluated sentences (and their focus marking) are part of the input rather than the output, rendering irrelevant the low likelihood that the model would produce them as outputs. And second, FC is unlike SP and YN in that the model is presented with both sentences simultaneously, which introduces additional clarity in terms of both the criteria of evaluation and the bounds of comparison. The poor performance of YN—which enjoys the first advantage but not the second—suggests that the latter may be more impactful, though YN’s poor performance could well be chalked up to independent factors.

The proposed picture, then, is as follows. First, the increased task demands of metalinguistic judgments entail that the success of prompting methods will be particularly dependent on model size. Hence, performance is poor for small models and improves for larger models. Second, when comparing between prompting methods, FC has a significant advantage over YN, since for FC the model sees both sentences simultaneously, while for YN the model must make each decision in isolation. And third, for those models that achieve strong prompting-based performance, FC has an advantage over SP when evaluating sentences that diverge substantially from normal model outputs.

This proposal makes a noteworthy prediction: if we modify our items so that they diverge even further from normal model outputs, then any concomitant negative impact on accuracy should more strongly affect SP than FC. In other words, once FC judgments are successfully learned, they should be more robust to stylistic atypicality than SP. We test this prediction next.

## 4 Experiment 2: Ad hoc focus markers

In Experiment 2, we replace common focus markers like boldface and italics with randomlygenerated ad hoc focus markers. This serves two purposes. The first is to determine whether the models’ knowledge of focus is specific to a closed class of canonical focus markers, or whether they have learned something more general. The second is to determine whether these more artificial focus markers lead to a greater decrease in accuracy for SP than for FC, as is predicted by our proposal.

## 4.1 Methodology

We generated ten ad hoc focus markers. Each consisted of a randomly generated sequence of three to five characters sampled with replacement from the set $\{ ^ { \ast \ast } @ ^ { \prime \prime } , \ ^ { \ast \ast } \# ^ { \prime \prime } , \ ^ { \ast \ast } \Phi ^ { \prime \prime } , \ ^ { \ast \ast } \varnothing ^ { \prime \prime } , \ ^ { \ast \ast } ; ^ { \prime \prime } , \ ^ { \ast \ast } ; ^ { \prime \prime } \}$ . This sequence was placed immediately before the focused constituent, with the reversed version of the sequence being placed immediately after. This is illustrated by the minimal pair in (13):

(13) a. Lucy only \$:#swam#:\$ yesterday.

b. Lucy only swam \$:#yesterday#:\$.

The total dataset therefore consisted of $1 0 \times 2 8 2$ = 2,820 minimal pairs. Other than the change in focus marking, the experiment was run identically to Experiment 1.

## 4.2 Results

The target item results for Experiment 2 can be seen in Figure 3. The control results, which we do not show for reasons of space, look similar to the control results from Experiment 1, but with lower peak accuracy. These continue to be outpaced by the target results, which again suggests that the backgrounds and prefixes play a crucial role. Put succinctly, the LMs are not just relying on a simple syntactic heuristic.

Turning to the target items, to compare methods we again fitted a mixed-effect logistic regression for each LM, predicting accuracy with a fixed effect of method (baseline = SP) and random effects of focus marker and item, using lme4. SP continued to perform significantly above chance for every tested model (all $p < . 0 0 1 )$ . This strongly suggests that what these models have learned about focus has a certain degree of generality, extending beyond canonical focus markers. The broad trend for FC also carried over from Experiment 1 to 2, with weak performance in smaller models that gradually improved to well beyond chance for larger, more capable models. The results for YN, which were the weakest in Experiment 1, more or less hovered around chance in Experiment 2.

![](images/e1099bbd86f5eced2d9f8527c936242317245c202fc6dec8fa25f65719bfb997.jpg)  
Figure 3: Target results of Experiment 2 (ad hoc focusmarkers). Error bars are bootstrapped 95% confidence intervals. X-axes are logarithmically spaced.

Importantly, the results from Experiment 2 do show the behavior predicted in Section 3: FC performance degraded less than SP performance when moving from canonical to randomly generated focus markers. The change in percentagepoint accuracy from Experiment 1 to Experiment 2 for each model and method can be seen in Figure 4; note that for nearly every model, both FC1 and FC2 had a less negative change than SP did.

This observation holds even when accounting for the fact that SP had “farther to fall”, in the sense that SP often outperformed FC in Experiment 1. For example, when looking at the five pairs of model and prompting method that outperformed SP in Experiment 1, in four out of five cases FC still showed a smaller decrease in accuracy than SP did. Moreover, there are many more instances of FC significantly outperforming SP in Experiment 2 (twelve) than in Experiment 1 (five). Neither of these observations is to be expected if the sole reason for the greater decline in accuracy for SP was its greater accuracy in Experiment 1.

![](images/eba7d8c082a961be709f44ffade869831fddf9c8c028a2e71aa5f9cad49f399c.jpg)  
Figure 4: Percentage-point change in accuracy from Experiment 1 to Experiment 2 by model and method.

As a statistical test, for each experiment we took the mean accuracy across all focus markers for each combination of model, method, and item. We then fitted a mixed-effect linear regression model predicting ad hoc focus marker accuracy, with fixed effects of standard focus marker accuracy and evaluation method (baseline = SP) and random effects for item and LM, using the lmerTest R package (Kuznetsova et al. 2017). Since one of the fixed effects is standard focus marker accuracy, this test controls for the fact that some modelmethod combinations had farther to fall than others. There was a significant positive effect of standard focus marker accuracy $( p < . 0 0 1 )$ , as well as positive effects for both FC1 and FC2 compared to the baseline of SP (both $p \ < \ . 0 0 1 )$ , meaning that going from standard to ad hoc focus markers had a less negative impact on FC than on SP. Finally, there was a significant negative effect for both YN1 and YN2 (� < .001 for both), though this is difficult to interpret due to the general poor performance of YN and the possibility that poor results could be attributed to faulty prompting.

## 5 Discussion and conclusion

Our findings support the proposal put forward above. SP outperforming prompting methods in smaller models (Hu and Levy 2023; Hu et al. 2024; Kauf et al. 2024), with prompting performance improving for larger models (Hu and Frank 2024), is predicted if metalinguistic judgments constitute a non-trivial auxiliary task. Meanwhile, FC can outperform SP in larger models— including by a wide margin for Gemma 4—at least in part because FC, once learned, is more robust in its evaluation of sentences that are stylistically unlikely outputs. Hence, when switching from standard to randomly generated focus markers, FC saw a smaller decrease in accuracy.

The strength of the conclusions we can draw from these results is somewhat limited by our focus on a singular empirical domain. A natural way to test the generality of these results would be to evaluate using other kinds of linguistic data that models might be similarly disinclined to generate, such as syntactically well-formed but semantically nonsensical sentences (cf. Gulordava et al. 2018); non-standard or -prestige dialects; socially unacceptable language use; and statements that are unambiguously false. Even beyond the particular hypotheses put forth in this paper, our results show that methodological comparisons can benefit from the investigation of linguistic phenomena beyond those covered by standard benchmarks like BLiMP (Warstadt et al. 2020).

Another promising path for future research would be to evaluate LMs’ linguistic competence using prompting-based methods that are not expressly metalinguistic in nature. Tasks like code generation, for example, are likely better represented in LMs’ training than metalinguistic judgments, so it is possible that the former will outperform the latter, particularly for smaller models. Not only might this lead to cleaner results, but it can also help to clarify which pros and cons of metalinguistic prompting are due specifically to the metalinguistic nature of the task.

Finally, it bears noting that the relative merits of prompting and string probabilities depend on more than just accuracy: they also depend on what these methods actually test, as it is not obvious that they test the same fundamental language abilities for a given model. A growing body of research is dedicated to discerning whether LMs are capable of introspection, defined as a model’s ability to predict or detect its own outputs via prompting. On the one hand, positive evidence exists suggesting that some models can non-trivially recognize their own outputs (Panickssery et al. 2024), describe implicit behavioral patterns acquired via finetuning (Betley et al. 2025), and learn with disproportionate success to predict their own hypothetical outputs (Binder et al. 2025). On the other hand, Song et al. (2025), who specifically test for introspection in metalinguistic judgments, fail to find it, though it is worth emphasizing that they base their conclusions of non-introspection on null results.

But suppose that Song et al. (2025) are right and LMs do not introspect when performing metalinguistic judgments, even when they judge with high accuracy. That is, the language knowledge LMs use to form judgments is distinct from the knowledge they use to generate tokens. One possible interpretation of such a result is that the explicit knowledge used by LMs to make judgments is essentially factual in nature, akin to knowing a bread recipe or a chess opening, whereas the implicit knowledge assessed via string probabilities is a more meaningful indicator of an LM’s linguistic competence. If this is true, then the accuracy of prompting is irrelevant, since it is simply not testing the relevant kind of language ability.

But there is an alternate interpretation that does not relegate prompting methods to such secondclass status. The core language modeling task could be (somewhat simplistically) broken down into the subtasks of (I) interpreting the model’s input and (II) generating a corresponding output. Importantly, there is no inherent reason to believe that the knowledge an LM uses to interpret a sentence � as an input is identical to the knowledge it uses to produce � as an output (West et al. 2024).

From this perspective, there are really two ways in which metalinguistic prompting and string probabilities differ from each other: explicit vs. implicit and input vs. output. Explicit judgments are still difficult, hence the poor performance in smaller models. But for models that can overcome that, the judgments they provide are informed by a facet of their language knowledge that is distinct from what is used for generation—hence, not introspective in the technical sense—but no less important in terms of the fundamental language modeling task. Or, indulging in some anthropomorphizing terminology, prompting tests comprehension, while probabilities test production. If this is the case, then prompting and probabilities are not so much competitors as they are complements, and both are needed for a complete understanding of LMs’ language competence.

All this being said, it remains to be seen whether introspection is in fact involved in LMs’ metalinguistic judgments, and if so, where and to what extent. But it is important to bear in mind as work on introspection progresses that the potential value of prompting-based linguistic assessment does not hinge entirely on whether LMs’ judgments are introspective in the particular sense defined above.

## Acknowledgments

This work is supported by the German Research Foundation, project #579297085 (Propositional Attitudes in Large Language Models), funded as part of the LaSTing Priority Program. Many thanks to members of the LaSTing community and to colleagues at ZAS for fruitful discussion.

## References

Douglas Bates, Martin Mächler, Ben Bolker, and Steve Walker. 2015. Fitting linear mixedeffects models using lme4. Journal of Statistical Software, 67(1):1–48.

David I. Beaver and Brady Z. Clark. 2008. Sense and Sensitivity: How Focus Determines Meaning. Wiley-Blackwell.

Jan Betley, Xuchan Bao, Martín Soto, Anna Sztyber-Betley, James Chua, and Owain Evans. 2025. Tell me about yourself: LLMs are aware

of their learned behaviors. In The Thirteenth International Conference on Learning Representations.

Felix J Binder, James Chua, Tomek Korbak, Henry Sleight, John Hughes, Robert Long, Ethan Perez, Miles Turpin, and Owain Evans. 2025. Looking inward: Language models can learn about themselves by introspection. In International Conference on Learning Representations, volume 2025, pages 3710–3756.

Alexander Clark, Gianluca Giorgolo, and Shalom Lappin. 2013. Statistical representation of grammaticality judgements: the limits of ngram models. In Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics, pages 28–36, Sofia, Bulgaria. Association for Computational Linguistics.

Vittoria Dentella, Fritz Günther, and Evelina Leivada. 2023. Systematic testing of three language models reveals low language accuracy, absence of response stability, and a yesresponse bias. Proceedings of the National Academy ofSciences, 120(51):e2309583120.

Allyson Ettinger, Ahmed Elgohary, Colin Phillips, and Philip Resnik. 2018. Assessing composition in sentence vector representations. In Proceedings of the 27th International Conference on Computational Linguistics, pages 1790–1801.

Kai von Fintel. 1994. Restrictions on Quantifier Domains. Ph.D. thesis, University of Massachusetts, Amherst.

Richard Futrell, Ethan Wilcox, Takashi Morita, Peng Qian, Miguel Ballesteros, and Roger Levy. 2019. Neural language models as psycholinguistic subjects: Representations of syntactic state. In Proceedings of the 2019 conference of the North American chapter of the Associationfor Computational Linguistics: Human language technologies, Volume 1 (Long and Short Papers), pages 32–42.

Gemma Team. 2026. Gemma 4 technical report.

Gerd Gigerenzer and Hans R Richter. 1990. Context effects and their interaction with development: Area judgments. Cognitive Development, 5(3):235–264.

Kristina Gulordava, Piotr Bojanowski, Edouard Grave, Tal Linzen, and Marco Baroni. 2018. Colorless green recurrent networks dream hierarchically. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1195–1205.

Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel R Bowman, and Noah A Smith. 2018. Annotation artifacts in natural language inference data. In Proceedings ofthe 2018 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 107–112.

Jennifer Hu and Michael C. Frank. 2024. Auxiliary task demands mask the capabilities of smaller language models. In First Conference on Language Modeling.

Jennifer Hu and Roger Levy. 2023. Prompting is not a substitute for probability measurements in large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5040–5060.

Jennifer Hu, Kyle Mahowald, Gary Lupyan, Anna Ivanova, and Roger Levy. 2024. Language models align with human judgments on key grammatical constructions. Proceedings of the National Academy of Sciences, 121(36):e2400917121.

Ray Jackendoff. 1972. Semantic Interpretation in Generative Grammar. MIT Press, Cambridge, MA.

Nirit Kadmon. 2001. Formal Pragmatics: Semantics, Pragmatics, Presupposition, and Focus. Blackwell, Oxford.

Gaurav Kamath, Sebastian Schuster, Sowmya Vajjala, and Siva Reddy. 2024. Scope ambiguities in large language models. Transactions of the Association for Computational Linguistics, 12:738–754.

Roni Katzir. 2023. Why large language models are poor theories of human linguistic cognition: A reply to Piantadosi. Biolinguistics, 17(e13153).

Carina Kauf, Emmanuele Chersoni, Alessandro Lenci, Evelina Fedorenko, and Anna A Ivanova. 2024. Log probabilities are a reliable estimate of semantic plausibility in base and instructiontuned language models. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 263–277, Miami, Florida, US. Association for Computational Linguistics.

Manfred Krifka. 1992. A framework for focussensitive quantification. In Semantics and Linguistic Theory (SALT), volume 1, pages 215– 236. Linguistic Society of America.

Saul Kripke. 2009. Presupposition and anaphora: Remarks on the formulation of the projection problem. Linguistic Inquiry, 40(3):367–386.

Alexandra Kuznetsova, Per B Brockhoff, and Rune HB Christensen. 2017. lmertest package: tests in linear mixed effects models. Journal of statistical software, 82:1–26.

Jey Han Lau, Alexander Clark, and Shalom Lappin. 2015. Unsupervised prediction of acceptability judgements. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing, pages 1618–1628. Association for Computational Linguistics.

David Lewis. 1975. Adverbs of quantification. In Edward Keenan, editor, Formal Semantics of Natural Languages, pages 3–15. Cambridge University Press, Cambridge.

Tal Linzen, Emmanuel Dupoux, and Yoav Goldberg. 2016. Assessing the ability of LSTMs to learn syntax-sensitive dependencies. Transactions of the Association for Computational Linguistics, 4:521–535.

Kyle Mahowald. 2023. A discerning several thousand judgments: GPT-3 rates the Article + Adjective + Numeral + Noun construction. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 265–273. ACL.

Rebecca Marvin and Tal Linzen. 2018. Targeted syntactic evaluation of language models. In

Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1192–1202.

Kanishka Misra. 2022. minicons: Enabling flexible behavioral and representational analyses of transformer language models.

Elliot Murphy, Evelina Leivada, Vittoria Dentella, Raquel Montero, Fritz Günther, and Gary Marcus. 2025. Fundamental principles of linguistic structure are not represented by ChatGPT. Biolinguistics, 19(e19021).

Arjun Panickssery, Samuel R Bowman, and Shi Feng. 2024. LLM evaluators recognize and favor their own generations. Advances in Neural Information Processing Systems, 37:68772– 68802.

Barbara Partee. 1991. Topic, focus and quantification. In Semantics and Linguistic Theory (SALT), volume 1, pages 159–187. Linguistic Society of America.

Pouya Pezeshkpour and Estevam Hruschka. 2024. Large language models sensitivity to the order of options in multiple-choice questions. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 2006–2017.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Mats Rooth. 1985. Association with Focus. Ph.D. thesis, University of Massachusetts Amherst, Amherst.

Mats Rooth. 1992. A theory of focus interpretation. Natural Language Semantics, 1(1):75– 116.

Siyuan Song, Jennifer Hu, and Kyle Mahowald. 2025. Language models fail to introspect about their knowledge of language. In Second Conference on Language Modeling.

Arnim von Stechow. 1991. Current issues in the theory of focus. In Arnim von Stechow and Dieter Wunderlich, editors, Semantics: An International Handbook of Contemporary Research, pages 804–825. Walter de Gruyter, Berlin.

Jon Sprouse, Carson T. Schütze, and Diogo Almeida. 2013. A comparison of informal

and formal acceptability judgments using a random sample from Linguistic Inquiry 2001– 2010. Lingua, 134:219–248.

Polina Tsvilodub, Paul Marty, Sonia Ramotowska, Jacopo Romoli, and Michael Franke. 2024. Experimental pragmatics with machines: Testing LLM predictions for the inferences of plain and embedded disjunctions. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 46, pages 3960–3967.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R Bowman. 2020. BLiMP: The benchmark of linguistic minimal pairs for English. Transactions ofthe Associationfor Computational Linguistics, 8:377–392.

Peter West, Ximing Lu, Nouha Dziri, Faeze Brahman, Linjie Li, Jena Hwang, Liwei Jiang, Jillian Fisher, Abhilasha Ravichander, Khyathi Chandu, Benjamin Newman, Pang Wei Koh, Allyson Ettinger, and Yejin Choi. 2024. The Generative AI Paradox: “What it can create, it may not understand”. In International Conference on Learning Representations, volume 2024, pages 47159–47187.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report.