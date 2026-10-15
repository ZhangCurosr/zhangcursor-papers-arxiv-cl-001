# Likelihood Is Not Authorship: Can We Reliably Measure Human Contribution in AI-Assisted Writing?

Hasan Amin Zhuoran Lu Ming Yin Department of Computer Science, Purdue University hasanamin@purdue.edu, zhuoranlu33@gmail.com, mingyin@purdue.edu

## Abstract

AI-text detection generally asks whether a machine wrote a text. As AI becomes an increas ingly integral and accepted part of writing, the more relevant question is no longer whether AI was used, but how much of the text reflects human contribution. A prominent line of work estimates this by measuring how predictive a person’s prompt is of the final text. We show that this measure can be easily manipulated. We introduce SPECIFICITY PADDING that simply asks the model to copy a few random IDs from the prompt, and that alone makes a titleonly request look more human than one that supplies the article’s key ideas. A context-swap audit then holds each document fixed, and only shows or hides the IDs to the evaluator. The score of the unchanged text swings by a third to half of its scale, and it rises for every document we test, on news and scientific abstracts, across varied LLMs from 3.8B to 753B parameters. The same articles can average 85% human or 15% human depending only on which prompt is on record, and even with no prompt at all, the output-only variant rewards the random strings themselves. An exact identity, the credit ledger, explains why. The score is a surprisal-weighted average over the text, so rare strings that the context makes copyable collect the credit, whatever they mean. A two-point bound marks a deeper limit. If the prompt could have been machine-written, every transcript-only method must, in the worst case, miss by half the contribution at stake. Likelihood-based scores alone cannot reliably establish human contribution without evidence of what the person added and how the text was produced.

## 1 Introduction

People who write with language models rarely submit raw model output, and they rarely write every word themselves. They supply a draft, an outline, a title, or a few edits, and the model does the rest (Lee et al., 2022; Yuan et al., 2022). For instructors, publishers, and copyright offices, the pressing question is therefore no longer the one that launched a decade of detection research, was this written by AI? (Gehrmann et al., 2019; Mitchell et al., 2023; Sadasivan et al., 2025), but how much of it is the person’s? (Cheng et al., 2025; Guo et al., 2025; U.S. Copyright Office, 2025).

![](images/86985844998645a3c29d19502132bd6c9f59bcaa938c0782547ff24f3dae0b3f.jpg)  
Figure 1: Copying a few random IDs outranks supplying the key ideas. Left: the likelihood-based humancontribution score (Xie et al., 2026) falls as the prompt supplies less of the article. Right: articles generated from a title-only request with SPECIFICITY PADDING, which asks the model to copy eight random IDs. Scored with the IDs in the evaluator’s context (MATCHED), they rank above articles written from the key ideas. Scored without them (HIDDEN), the same articles fall below a subject-only request. The swing on unchanged text is larger than the whole validation ladder. Results shown for Llama-3.1-8B on same news items.

Information theory offers an elegant answer. If the person’s input makes the final text easy for a language model to predict, the person supplied much of its information. If the text stays surprising, the model supplied it. Xie et al. (2026) turn this intuition into a score, the fraction of the output’s surprisal that the human input explains. The score needs no training data, falls in the expected order from polishing a draft to expanding a summary, a title, or a subject, agrees with 96% of human pairwise judgments, and resists adaptive prompts that ask the model to use rare words or mimic human style. Its core, a pointwise log-likelihood ratio, also underlies V-usable information, Shannongame summary metrics, and likelihood-based evaluation (Ethayarajh et al., 2022; Egan et al., 2022; Yuan et al., 2021), so its failures are theirs whenever they are read as contribution.

These successes rest on an unstated premise, that the information a prompt supplies is the contribution a person made. The premise fails in the simplest way imaginable. Ask a model to include the string PX-7A9-A11. Without the prompt, the string is nearly impossible to predict. With it, the string is nearly certain. Its tokens therefore earn almost full credit, although they add nothing to the article. We call the resulting attack SPECIFICITY PADDING. Put a few random IDs into the prompt and let the model copy them.

The effect is large enough to reorder the very ladder on which the score was validated (Figure 1). A title plus a handful of copied IDs scores above a request that hands the model the article’s key ideas, two-thirds of the way to polishing a complete human draft. Hide the IDs from the evaluator, and the same articles fall below a request that names only the subject. That swing, on text that has not changed by a byte, is wider than the entire ladder from a full human draft down to a bare subject.

To isolate the cause, we introduce a context-swap audit. We hold a document fixed and vary only what the evaluator conditions on: the matching IDs (MATCHED), no IDs (HIDDEN), or different IDs of the same form (WRONG). Nothing about the text can explain a difference, because the text does not change. Showing the matching IDs moves the score by a third to half of its scale (32–51 points), and it raises the score of every document, for every model we test, from 3.8B to 753B parameters. Wrong IDs of the same form recover only part of the gain. The remaining 16–29 points pay for exact copying. In ordinary generation, the attack inflates scores by 13–27 points. The same articles can average 85% human or 15% human depending only on whether the evaluator sees the full prompt or just what the user typed. Even with no prompt at all, the outputonly variant of the score rewards the random strings themselves. Scientific abstracts fail alike.

Why does a validated score fail so badly? Because credit follows surprisal. An exact identity, which we call the credit ledger, writes the score of any text as a surprisal-weighted average of the scores of its parts. Rare strings carry disproportionate weight. Once the context makes them predictable, they carry disproportionate credit, whatever they mean. The same identity predicts when padding inflates a score and when it cannot (a polished human draft gains nothing), and it explains why the output-only variant rewards unusual text.

Padding exploits a meaning gap, but it also exposes a provenance gap. The score credits whatever context is on record, and the record does not say who wrote it. A prompt a person typed and one pasted from an earlier model call look identical. If the recorded prompt could have been machinewritten, any transcript-only estimator must, in the worst case, miss by at least half the contribution at stake. A better model cannot close this gap. Only evidence about how the text was produced can.

To our knowledge, this is the first adversarial test of the construct validity of graded humancontribution scores, and the first analysis of what transcript-only contribution claims can identify at all. The parallel with detection is direct. Binary detectors fell to paraphrasing (Krishna et al., 2023; Sadasivan et al., 2025). Graded scores fall to copying. None of this makes likelihood useless. It makes precise what likelihood measures, which is how much a context helps a model predict a text. That is a relation among three objects, and authorship is not among them. Our contributions are:

• A trivial attack and a reusable audit. SPECI-FICITY PADDING needs no gradients, weights, or evaluator queries, and the context-swap audit applies to any context-conditioned likelihood metric (§3–4).

• One identity that explains the failure. The credit ledger accounts for the attack, its dose response and headroom, and the output-only failure (§5).

• A boundary on transcript evidence. Certifying human contribution requires provenance records rather than better scoring (§6–7).

## 2 Contribution as Predictability

Let x be the recorded input, $y = ( y _ { 1 } , \dotsc , y _ { N } )$ the output, and $q _ { \theta }$ an autoregressive evaluator. Write the token surprisals without and with the input as

$$
\begin{array} { l } { s _ { i } = - \log q _ { \theta } ( y _ { i } \mid y _ { < i } ) , } \\ { t _ { i } = - \log q _ { \theta } ( y _ { i } \mid x , y _ { < i } ) , } \end{array}
$$

so that $\begin{array} { r } { \mathcal { I } ( y ) = \sum _ { i } s _ { i } } \end{array}$ and $\textstyle { \mathcal { T } } ( y \mid x ) = \sum _ { i } t _ { i }$ . The score of Xie et al. (2026) is

$$
\phi ( x , y ) = 1 - \frac { \mathcal { T } ( y \mid x ) } { \mathcal { T } ( y ) } = \frac { \sum _ { i } ( s _ { i } - t _ { i } ) } { \sum _ { i } s _ { i } } .\tag{1}
$$

If an output carries 100 nats of surprisal without the prompt and 40 with it, then $\phi = 0 . 6$ and the prompt explains 60% of the model’s surprise at the exact tokens. The numerator is a pointwise log-likelihood ratio, the quantity behind pointwise V-information (Xu et al., 2020; Ethayarajh et al., 2022), so ϕ can be negative for an individual pair. Xie et al. (2026) describe it as a ratio of mutual information to self-information. We call it what it computes, normalized contextual predictability.

Output-only estimate. When the input is unavailable, Xie et al. (2026) estimate the minimal human contribution needed to produce y by replacing each $t _ { i }$ with a model-specific plausibility threshold $b = - \log \tau > 0 \mathrm { { : } }$

$$
{ \widehat { \phi } } ( y ) = { \frac { \sum _ { i } ( s _ { i } - b ) } { \sum _ { i } s _ { i } } } = 1 - { \frac { b } { { \bar { \mathcal { T } } } ( y ) } } ,\tag{2}
$$

where $\bar { \mathcal { I } } ( y ) = \mathcal { I } ( y ) / N$ . Both scores credit each token with its surprisal to the unconditioned model, minus what the input (or the constant b) explains. Section 5 shows that this shared structure is the root of both failures.

What validation can and cannot show. The central evidence for Equation 1 is an ordering. The score falls as the prompt supplies less of the article, from polishing to summary, title, and subject prompts (Figure 1; Appendix A.1 confirms it for our implementation). Along that ladder, however, the information a person supplies and the substance of what the person contributes move together. Agreement with the ladder, or with human judgments of ladder pairs, therefore cannot distinguish a score that tracks contribution from one that tracks copyable information. Separating the two requires varying information while holding contribution fixed. That is an invariance test in the sense of measurement theory (Cronbach and Meehl, 1955; Jacobs and Wallach, 2021) and behavioral testing (Ribeiro et al., 2020).

Definition 1 (Task-null transformation). Let $c \tau ( h )$ be the contribution that task T targets for a production history h. A transformation Π is T-null if the task specification declares $c _ { \mathcal { T } } ( \Pi ( h ) ) = c _ { \mathcal { T } } ( h )$ A score that moves substantially under a T-null transformation is not a valid measure of $c \tau$

![](images/2ef06bf30d5c1ab2c2795e5bf2a4eef7205bb6c5e348352750522c88cbf3b4a8.jpg)  
Figure 2: The context-swap audit: one text, three scores. A single padded output is scored under three evaluator contexts. The text is byte-identical, so any difference in score comes from the context alone. Llama-3.1-8B, eight IDs, mean over 300 news items (the clean article scores 0.55).

Nullness is task-relative. An identifier is irrelevant to the substance of a news story but may be the point of a lab protocol. For the news articles we study, inserting random IDs (codes, nonce dataset names, made-up measurements) changes no event, claim, or argument, so it is task-null. The adaptive attacks tested by Xie et al. (2026) add instructions without information. SPECIFICITY PADDING adds information without contribution. One might reply that the IDs are information the person supplied. They are, and that is the point. Under this score, the most valuable thing a person can contribute is text that is hard to guess and easy to copy, whether or not it matters to the task.

## 3 Specificity Padding and the Context-Swap Audit

SPECIFICITY PADDING has a single step. Append random IDs to a prompt and ask the model to use them verbatim. It needs no gradients, weights, or evaluator queries. We write $a _ { K }$ for a list of K IDs and call K the dose. The IDs cycle through nonce dataset names, protocol codes, decimal measurements, condition labels, and alphanumeric tags. Their values are hashed from the item index and a seed, so they carry nothing about the article: Virelia-C22D8, PX-7A9-A11, 244.6 ms, phase-6CDB-8C6, and so on.

The context-swap audit. For a clean pair $( x , y )$ let $z ( a _ { K } )$ render the IDs as a one-line metadata suffix. We score the single padded output $\tilde { y } = y \oplus$ $z ( a _ { K } )$ under three contexts (Figure 2): MATCHED $( x \oplus a _ { K } )$ , HIDDEN (x), and WRONG $( \boldsymbol { x } \oplus \boldsymbol { a } _ { K } ^ { \prime } )$ where $a _ { K } ^ { \prime }$ matches $a _ { K }$ in ID types and count but has disjoint values. Since y˜ is byte-identical across conditions, so are its length, wording, quality, meaning, and unconditional surprisal $\mathcal { T } ( \tilde { y } )$ . The conditions differ only in what the evaluator is told. MATCHED−HIDDEN is the credit earned by showing the IDs, and MATCHED−WRONG asks whether generic specificity suffices or exact values are needed. The audit requires only scoring passes and applies unchanged to any context-conditioned likelihood metric. The clean score $\phi ( x , y )$ serves as a reference, but any comparison with it involves a different text.

Generation-time padding. To test the attack in ordinary use, we append eight IDs to the title request and let the model weave them into its article. We compare the attacked article with a separately generated clean one (attack−clean), which is realistic but not isolated. We then swap contexts on the same attacked article, scoring it under the full prompt (MATCHED), the visible request alone (HIDDEN), or the request with wrong IDs (WRONG). MATCHED−HIDDEN here models a common evidence gap. A provider logs the complete invocation, whereas an instructor or auditor sees only what the user typed.

Setup. We use all 300 items of the news benchmark released by Xie et al. (2026), generating from titles with the source article’s length as the word limit. Seven instruction-tuned models from six families each generate and evaluate their own outputs, the provider-side setting that Xie et al. (2026) envision. Four run locally from open weights, Llama-3.1-8B (Grattafiori et al., 2024), Qwen3-8B (Yang et al., 2025), Gemma-3-4B (Gemma Team et al., 2025), and Phi-4-mini (Abouelenin et al., 2025). Three run on served endpoints, DeepSeek-V4- Flash (DeepSeek-AI, 2026), Qwen3.8-27B (Qwen Team, 2026), and GLM-5.2 (Z.ai, 2026), for 2,100 model–document pairs in all. We replicate on 200 arXiv abstracts (Clement et al., 2019) generated from their titles by the three served models (600 pairs). Decoding uses temperature 0.7, topp 0.9, and at most 384 new tokens. Likelihoods are teacher-forced over response tokens plus EOS under each model’s native chat template, without clipping. Fixed-output doses are $K \in \{ 1 , 2 , 4 , 8 \}$ and we report $K = 8$ unless noted. Effects are paired means with 95% percentile bootstrap intervals over items (10,000 resamples), and no example is discarded. Before any attack, our implementation reproduces the expected assistance ordering. Appendix A gives prompts, checkpoints, and scoring conventions.

<table><tr><td></td><td colspan="2">News</td><td colspan="2">arXiv</td></tr><tr><td>Contrast</td><td>Points</td><td>十</td><td>Points</td><td>十</td></tr><tr><td colspan="5">Fixed output, different evaluator context</td></tr><tr><td>MATCHED-HIDDEN</td><td>32-51</td><td>2100</td><td>43-47</td><td>600</td></tr><tr><td>MATCHED-WRONG Generated article, different evaluator context</td><td>16-29</td><td>2100</td><td>25-28</td><td>600</td></tr><tr><td>MATCHED-HIDDEN</td><td>51-84</td><td>2097</td><td>66-78</td><td>600</td></tr><tr><td colspan="5">Padded vs. clean text</td></tr><tr><td>Generated attack—clean</td><td>13-27</td><td>2040</td><td>13-20</td><td>599</td></tr><tr><td>Output-only estimate*</td><td>+32</td><td>300</td><td>一</td><td>一</td></tr></table>

Table 1: Every document moves. Paired mean effects in points (range over models) at eight IDs, and the number of model–document pairs with a positive effect (+), out of 2,100 for news (seven models, 300 items each) and 600 for arXiv (three models, 200 each). Every 95% bootstrap interval excludes zero, with half-widths of at most 3.1 points (Appendix B.1). <sup>∗</sup>Llama-3.1-8B only, out of 300; this estimate uses no prompt.

## 4 Copied IDs Earn Human Credit

## 4.1 A few copied IDs outrank the article’s key ideas

Figure 1 places the attack on the ladder that validated the score. On the 30 items where we reproduce the ladder, the eight IDs are worth 20 points over the clean title request, more than the entire step from a title to a summary (17 points). Nothing the person contributed changed between these conditions except eight strings that the task declares meaningless. The 300-item run agrees (0.75 versus 0.55).

## 4.2 One text, many authors

Table 1 and Figure 3a report the audit. Showing the matching IDs raises the fixed-output score by 32–51 points on average, and it raises the score of every one of the 2,100 model–document pairs. For Llama-3.1-8B, it turns a document the score calls mostly machine-written (0.17) into one it calls mostly human (0.68) without changing a byte. Because the text is identical, this gain is exactly the log-likelihood the IDs add, as a fraction of the document’s unconditional surprisal (§5).

Exact values carry the credit. Wrong IDs of identical form recover part of the gain, because the suffix’s field names, prefixes, and layout become predictable. For Llama-3.1-8B they account for 25 of the 51 points. Matched IDs still beat wrong ones by 16–29 points, again in all 2,100 pairs. What the score rewards beyond form is value-level correspondence between prompt and output, which is to

![](images/19e921da487d1adb83cbb2ccdf797335ed7ce105e1f39add06f22e8017bc27ef.jpg)  
Figure 3: One text, many scores. Positions are changes from each model’s clean score, and the number beside each row is the MATCHED−HIDDEN gap in points on the same text. (a) A fixed padded output scored under MATCHED, WRONG, and HIDDEN contexts. (b) Articles generated with eight requested IDs, scored under the full prompt (MATCHED) or the visible request alone (HIDDEN). The gap is positive for all 2,700 fixed outputs and for 2,697 of 2,700 generated articles.

say copying.

Whether a fixed suffix also lifts the score above the clean article depends on headroom (§5). At eight IDs it does for four models, but not for Qwen3-8B, Gemma-3-4B, and DeepSeek-V4- Flash, whose MATCHED scores dip at one ID and are still climbing at eight (Appendix Figure 5). Neither pattern affects the audit, which compares identical text.

## 4.3 Whoever holds the prompt holds the score

Models asked to use eight IDs reproduce 65–99% of them verbatim, and their articles score 13–27 points above separately generated clean articles, with gains in 2,040 of 2,100 pairs. Because these are different documents, we again swap contexts on the same attacked articles (Figure 3b). Removing only the IDs from the evaluator’s context lowers their scores by 51–84 points, in 2,097 of 2,100 pairs. For DeepSeek-V4-Flash, the same articles average 0.85 with the full prompt and 0.15 with the visible request. A provider that logs the full invocation and an auditor who sees what the user typed would reach opposite verdicts about one unchanged document. The score is a relation among the text, the evaluator, and whichever context is taken to be the human’s.

## 4.4 Without any prompt, rarity itself earns credit

The output-only estimate (Equation 2) fails in the opposite direction and needs no matching prompt at all. Appending eight IDs raises Llama-3.1-8B’s estimate of minimal human contribution from 0.36 to 0.68, and raises it for all 300 articles (Appendix Figure 6).

## 4.5 Findings extend to other datasets

On 200 arXiv abstracts, showing matching IDs raises fixed-output scores by 43–47 points and beats wrong IDs by 25–28 points, in all 600 pairs. Generation-time padding inflates scores by 13–20 points, and hiding the IDs lowers the same abstracts by 66–78 points (Figure 3; Appendix B.4). Fabricated measurements can read as scientific claims, so here the generation-time comparison tests the score, not content preservation.

## 5 Why It Works: Credit Follows Surprisal

Three facts need explaining. Exact values earn credit that their form does not, the same suffix lifts some scores and lowers others, and rarity pays without a prompt. A single identity explains all three. It uses only the chain rule and assumes nothing about token independence, ID format, or architecture.

The credit ledger. Partition an output into consecutive nonempty spans, $y = u _ { 1 } u _ { 2 } \cdot \cdot \cdot u _ { m } \colon$ tokens, sentences, or an article body followed by a suffix. For each span, let $S _ { k } = \mathcal { T } ( u _ { k } \ \vert \ u _ { < k } )$ and $T _ { k } = \mathcal { T } ( u _ { k } \mid x , u _ { < k } )$ be its surprisal given the preceding text, without and with the context. Define the span’s weight $w _ { k } = S _ { k } / \mathcal { T } ( y )$ , its local score $\phi _ { k } = 1 - T _ { k } / S _ { k }$ , and its credit $\kappa _ { k } = w _ { k } \phi _ { k } =$ $( S _ { k } - T _ { k } ) / \mathcal { T } ( y )$

Proposition 1 (Credit follows surprisal). For any context x and any partition of $y$ into nonempty spans, the weights sum to one and

$$
\phi ( x , y ) = \sum _ { k } \kappa _ { k } = \sum _ { k } w _ { k } \phi _ { k } .\tag{3}
$$

The output-only estimate is the same ledger with each $T _ { k }$ replaced by $b n _ { k } ,$ , where $n _ { k }$ is the span’s length in tokens: $\begin{array} { r } { \widehat { \phi } ( y ) = \sum _ { k } ( S _ { k } - b n _ { k } ) / \mathcal { T } ( y ) } \end{array}$

The proof utilizes the chain rule (Appendix C.1). The score is thus a weighted average of local scores whose weights are surprisal shares. Length, position, and importance play no role. A decisive argument and an arbitrary ID are treated alike whenever they induce the same likelihoods. Surprisal, in turn, concentrates in rare strings. A random alphanumeric tag costs a language model several nats per token, while fluent model-written prose costs about one. Whatever the context makes predictable among such strings dominates the ledger. Suppose an article body carries 400 nats and scores 0.40, and a 100-nat suffix of IDs is nearly determined by the prompt, with a local score of 0.95. The document scores $0 . 8 \cdot 0 . 4 0 + 0 . 2 \cdot 0 . 9 5 = 0 . 5 1$ . A fifth of the surprisal moves the score by 11 points.

Padding pulls scores toward its own score. The two-span case of Proposition 1, a body y followed by a suffix $z ,$ describes every padding experiment.

Corollary 1 (Padding). For any context x<sup>′</sup>, let $w =$ $\mathcal { T } ( z \mid y ) / \mathcal { T } ( y z )$ be the suffix’s surprisal share and $\phi _ { z } ( x ^ { \prime } ) = 1 - \mathcal { T } ( z \mid x ^ { \prime } , y ) / \mathcal { T } ( z \mid y )$ its local score. Then

$$
\phi ( x ^ { \prime } , y z ) = ( 1 - w ) \phi ( x ^ { \prime } , y ) + w \phi _ { z } ( x ^ { \prime } ) .\tag{4}
$$

Hence the padded score lies between the body’s score and the suffix’s local score, and exceeds the body’s score if and only $i f \phi _ { z } ( x ^ { \prime } ) > \phi ( x ^ { \prime } , y )$ . If IDs accumulate so that $\mathcal { I } ( z \mid y ) ~  ~ \infty$ while $\mathcal { T } ( z \mid x ^ { \prime } , y ) + \mathcal { T } ( y \mid x ^ { \prime } )$ grows more slowly, then $\phi ( x ^ { \prime } , y z ) \to 1$ , whatever the article says.

For the audit, Equation 4 splits the matched– hidden gap $\Delta$ on a fixed output, with superscripts M and H for the two contexts, into

$$
\begin{array} { r l } & { \Delta = ( 1 - w ) \left[ \phi ^ { \mathrm { M } } ( y ) - \phi ^ { \mathrm { H } } ( y ) \right] } \\ & { \quad \quad + w \left[ \phi _ { z } ^ { \mathrm { M } } - \phi _ { z } ^ { \mathrm { H } } \right] . } \end{array}\tag{5}
$$

![](images/e97f1beab5151ee6f7c47520b808669322c9f0aab541a69d35907fd169bc6ebf.jpg)  
Figure 4: Padding pulls scores toward its own score. The same eight-ID suffix appended to clean Llama-3.1-8B articles at each assistance level, scored under MATCHED or HIDDEN. MATCHED padding inflates weak contributions most and cannot inflate a polished human draft. The gain falls linearly with the clean score $( R ^ { 2 } = 0 . 9 9$ over four levels) and vanishes near 0.84.

The second term is the $\mathrm { I D s } '$ surprisal share times how much the context helps predict them. The first is any change in how well the context predicts the body. When showing the IDs makes them near-certain and hiding them leaves them unpredictable, the gap approaches the suffix’s surprisal share. Llama-3.1-8B’s 51-point gap thus says that eight short strings carry about half of what the model finds surprising in a 200-word article. Its output-only estimates independently point to the same share (Appendix C.2).

Headroom. Equation 4 also says that padding does not simply push a score up. It pulls the score toward the suffix’s local score. A fixed suffix should therefore inflate documents that score below that value, deflate documents above it, and inflate most where the clean score is lowest. Figure 4 confirms this across Llama-3.1-8B’s four assistance levels. The gain from MATCHED padding falls from 14 points for subject and title prompts to zero for polishing, roughly linearly in the clean score, and vanishes near 0.84. A complete human draft already sits where the padding would pull it. Hiding the IDs lowers every level by 35–40 points. Without the IDs in context, the suffix gains little from the prompt, its local score falls far below the clean score, and Equation 4 drags every document down.

The output-only estimate rewards rarity itself. Proposition 1 also covers Equation 2, whose ledger subtracts a constant b per token.

Corollary 2 (Output-only estimate). For fixed $0 < \tau < 1 , \widehat { { \phi } } ( y )$ is strictly increasing in the mean surprisal $\bar { \mathcal { T } } ( y )$ and tends to 1 as $\bar { \mathcal { I } } ( y )  \infty$ . Appending any span whose mean surprisal exceeds $\bar { \mathcal { T } } ( y )$ strictly raises it.

Under Llama-3.1-8B’s threshold, any text averaging six nats per token, typical of random characters, receives an estimated minimal human contribution above 0.9. IDs are far more surprising per token than fluent prose, so they raise the estimate with no prompt at all (§4). The two estimators fail from opposite sides of one design. With a context, rare IDs earn credit because the prompt predicts them. Without one, they earn it because they are rare. Neither estimator can see who supplied the IDs or whether they matter.

## 6 The Provenance Gap: What No Transcript Can Certify

The ledger explains one score. A broader limit applies to any method that infers causal contribution from the visible transcript alone, however well it models text.

Consider two workflows with the same recorded prompt and final document. In one, a person developed the claims and asked a model to polish them. In the other, an earlier model call produced both the claims and the prompt, which the person pasted into the interface. A transcript-only evaluator receives identical evidence, although the causal contribution differs. Let $h \in \mathcal H$ be a full production history (drafts, edits, system messages, retrieval results, tool calls, prior model outputs, and the final document), $g : { \mathcal { H } }  { \mathcal { O } }$ the map to what an evaluator observes, and $c ( h ) \in \mathbb { R }$ a specified causal contribution target. The identification set of an observation o is ${ \mathcal { C } } ( o ) = \{ c ( h ) : g ( h ) = o \}$ , with diameter $D ( o ) = \operatorname* { s u p } { \mathcal { C } } ( o ) - \operatorname* { i n f } { \mathcal { C } } ( o )$

Proposition 2 (Identification boundary). For any transcript-only estimator $f : \mathcal { O } $ R and any observation o with bounded, nonempty ${ \mathcal { C } } ( o )$ ,

$$
\operatorname* { s u p } _ { h : g ( h ) = o } | f ( o ) - c ( h ) | \ \geq \ { \frac { D ( o ) } { 2 } } ,\tag{6}
$$

with equality at the midpoint $o f { \mathcal { C } } ( o )$ , and randomizing f does not help. In particular, ifo is compatible with a history in which the person authored the recorded input (contribution $c _ { 1 } )$ and with a laundered history in which that input was machinegenerated (contribution 0), then every transcriptonly estimator errs by at least $c _ { 1 } / 2$ on one ofthem.

The bound is elementary. Its significance lies in the second statement. The more contribution a transcript appears to certify, the larger the worst-case error of any method that sees only the transcript. A stronger model cannot help. It learns which history is typical, not which one occurred.

This is not a claim that detection is impossible. Distributional detection works when human and model text differ (Chakraborty et al., 2024), and average-case contribution estimates can be accurate under a stable workflow prior. What the proposition rules out is a universal, instance-level certificate from an observation that remains compatible with materially different histories. SPECI-FICITY PADDING makes this concrete. The score can correctly report that the visible IDs predict the output, but not whether a person typed them, copied them, retrieved them, or received them from a hidden model call. Nor does the gap require an adversary. System prompts, retrieved passages, templates, and text pasted from another model all enter the recorded context of ordinary workflows, and a likelihood score credits them to the person exactly as it credits the IDs. Provenance is a property of the process, not of the string.

## 7 What Contribution Scores Should Report

Our findings suggest five cheap practices.

State the relation. A likelihood score is defined by a text, a context, and an evaluator. Reports should name all three and the estimand, whether token predictability, semantic contribution, or causal provenance, rather than “human contribution” in general. Copyright practice already draws this line. The U.S. Copyright Office concluded that the mere provision of prompts does not make a user the author of AI output (U.S. Copyright Office, 2025).

Audit with context swaps. Before a contribution score is deployed, it should be run on fixed outputs under MATCHED, HIDDEN, and WRONG contexts with task-null insertions (§3). The audit needs no generation and exposes a sensitivity that ladderstyle validation cannot detect (§2). It transfers unchanged to any score built on a context-conditioned likelihood ratio, including pointwise V-information and Shannon-game summary metrics.

Report the ledger. Proposition 1 turns any score into an exact account of which spans earn the credit. Credit concentrated in identifiers, names, or numbers is a visible warning that a reviewer can check. Whether such spans matter remains a task-specific semantic judgment that no likelihood can make.

Do not patch the symptom. Masking IDs, numbers, or rare strings would blunt this attack but not the mechanism. Any copyable high-surprisal span, such as a name, a quotation, or a technical term, earns credit the same way, and the same mask would erase legitimate contributions such as real measurements. Deciding which spans matter is precisely the semantic judgment a likelihood score cannot make (§5), and no mask closes the provenance gap (§6).

Tie certificates to provenance. Claims about who contributed what to a specific document require evidence about the process: authenticated interaction logs, revision histories, or content credentials (Lee et al., 2022; Hoque et al., 2024; Coalition for Content Provenance and Authenticity, 2026). Such records shrink the identification set in a way no transcript-only score can, although they do not by themselves assign intellectual value.

## 8 Related Work

Measuring human contribution. Recent work classifies human, machine, and collaborative text (Zhang et al., 2024b; Ta et al., 2026), locates machine-written segments (Richburg et al., 2024; Su et al., 2025), and estimates graded involvement (Cheng et al., 2025; Guo et al., 2025). Xie et al. (2026) propose the likelihood score we study and caution that informational contribution is not creativity or authorship. We show that this caveat hides a large, exploitable failure. Their assistance ladder and pairwise human judgments vary information and contribution together, so neither can reveal the failure. Our audit supplies the missing invariance test, and our identification result bounds what any transcript-only score can certify.

Information-theoretic evaluation. Pointwise likelihood ratios underlie V-usable information (Xu et al., 2020; Ethayarajh et al., 2022; Lu et al., 2023), summary evaluation by how much a summary helps reconstruct its source (Vasilyev et al., 2020; Egan et al., 2022), and metrics that score text by model likelihood (Thompson and Post, 2020; Yuan et al., 2021; Fu et al., 2024). Such metrics have blind spots and can be gamed (He et al., 2023; Pang et al., 2023). The ledger names one systematic blind spot shared by the whole family. A normalized likelihood gain weights content by surprisal, so any member of the family that is read as importance or contribution will overpay for copyable rare strings.

Detection and its limits. Zero-shot detectors use likelihood, rank, or curvature statistics (Gehrmann et al., 2019; Mitchell et al., 2023; Bao et al., 2024; Hans et al., 2024), alongside trained detectors and watermarks (Verma et al., 2024; Kirchenbauer et al., 2023; Wu et al., 2025). Paraphrasing degrades detectors (Krishna et al., 2023; Sadasivan et al., 2025), detection remains possible when distributions separate (Chakraborty et al., 2024), and strong watermarking faces impossibility results (Zhang et al., 2024a), despite cryptographically undetectable constructions (Christ et al., 2024). Graded contribution inherits these dynamics and adds a new one.

Construct validity and provenance. Construct validity asks which interpretations a measurement supports (Cronbach and Meehl, 1955). NLP work makes its assumptions explicit for fairness and generation metrics (Jacobs and Wallach, 2021; Xiao et al., 2023; Subramonian et al., 2023), and invariance tests expose failures that aggregate accuracy hides (Ribeiro et al., 2020). Attribution and factuality ask whether claims are supported by sources (Rashkin et al., 2023; Gao et al., 2023; Min et al., 2023), which is distinct from who created them. Process records observe exactly the variables that transcript-only scores lack (Lee et al., 2022; Hoque et al., 2024; Coalition for Content Provenance and Authenticity, 2026). Our identification result applies the standard logic of causal identification (Pearl, 2009) to authorship.

## 9 Conclusion

Moving from binary detection to graded contribution does not remove the adversarial problem. It moves the attack surface into the evaluator’s context. A likelihood score is an exact measure of how much a context helps a model predict a text, and our results show how far that is from authorship. A few meaningless strings outrank the article’s key ideas, and hiding them drops the same articles below a subject-only request. The credit ledger shows why. Credit follows surprisal, and meaning never enters the account. And no transcript, however well modeled, reveals who wrote the prompt. Can we reliably measure human contribution in AI-assisted writing? Not from likelihood alone, and not from the transcript alone. Reliable claims need a declared target, a context-swap audit, a reported ledger, and evidence of how the text was made. Likelihood measures what a context lets a model predict, not who authored the text.

## Limitations

Scope. We study English news with seven models and scientific abstracts with three, using one clean and one attacked generation per item. The assistance ladder and the headroom analysis use a fixed 30-item Llama-3.1-8B subset. Effect sizes may differ for other languages, genres, decoding settings, and generation seeds.

Evaluator. Each model scores its own outputs, the provider-side setting of Xie et al. (2026). This may maximize sensitivity to exact strings, and we do not test surrogate evaluators. Absolute scores are model-relative and not calibrated across families. The served endpoints do not expose weight revisions or numerical precision (Appendix A.3).

Stealth and meaning. The fixed suffix is conspicuous, and generation-time padding can introduce unsupported details into news and abstracts. A construct-validity test does not require stealth, but we do not claim a meaning-preserving or undetectable attack. In pilots with one model, subtler lexical variants (required rare words, synonym edits) produced small or no gains (Appendix B.5).

Task relativity. Declaring synthetic metadata null suits the substance of news, not every task. In a lab protocol, an identifier may be the contribution. If contribution is defined as every bit a prompt supplies, exact identifiers count toward it by definition.

Theory. The headroom analysis rests on four assistance levels of one model and treats the body’s predictability as unaffected by the IDs. Span-level measurements would sharpen it. The identification bound is worst-case, not an estimate of error under a particular workflow distribution.

Reproduction. We use the 300-item news set released by Xie et al. (2026) and reproduce their assistance ordering, not the numbers of their original four-domain benchmark (Appendix A.1).

## Ethical Considerations

Contribution scores can shape judgments of originality, academic integrity, ownership, and compensation, and false confidence harms authors in both directions: inflation conceals extensive model generation, while deflation erases genuine human work. Our attack uses deterministic synthetic strings and no personal data. We release it as a measurement stress test for developers and auditors of contribution scores, not as a method for evading institutional policy. The defenses we recommend (context-swap audits, credit ledgers, provenance records) are cheap to adopt. Transcript-only scores should not serve as dispositive evidence about an individual. Deployments need a stated estimand, informed consent, auditable evidence, an appeal process, and error analysis across languages and communities.

## References

Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, et al. 2025. Phi-4-Mini technical report: Compact yet powerful multimodal language models via mixture-of-LoRAs. arXiv preprint arXiv:2503.01743.

Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. 2024. Fast-DetectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature. In The Twelfth International Conference on Learning Representations.

Souradip Chakraborty, Amrit Singh Bedi, Sicheng Zhu, Bang An, Dinesh Manocha, and Furong Huang. 2024. Position: On the possibilities of AI-generated text detection. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 6093–6115. PMLR.

Zihao Cheng, Li Zhou, Feng Jiang, Benyou Wang, and Haizhou Li. 2025. Beyond binary: Towards finegrained LLM-generated text detection via role recognition and involvement measurement. In Proceedings of the ACM on Web Conference 2025. Association for Computing Machinery.

Miranda Christ, Sam Gunn, and Or Zamir. 2024. Undetectable watermarks for language models. In Proceedings of the Thirty-Seventh Conference on Learning Theory, volume 247 of Proceedings ofMachine Learning Research, pages 1125–1139. PMLR.

Colin B. Clement, Matthew Bierbaum, Kevin P. O’Keeffe, and Alexander A. Alemi. 2019. On the use of ArXiv as a dataset. Preprint, arXiv:1905.00075.

Coalition for Content Provenance and Authenticity. 2026. C2PA technical specification, version 2.4. Technical specification.

Lee J. Cronbach and Paul E. Meehl. 1955. Construct validity in psychological tests. Psychological Bulletin, 52(4):281–302.

DeepSeek-AI. 2026. DeepSeek-V4-Flash-0731. Hugging Face model card.

Nicholas Egan, Oleg Vasilyev, and John Bohannon. 2022. Play the shannon game with language models: A human-free approach to summary evaluation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 10599–10607.

Kawin Ethayarajh, Yejin Choi, and Swabha Swayamdipta. 2022. Understanding dataset difficulty with V-usable information. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 5988–6008. PMLR.

Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. 2024. GPTScore: Evaluate as you desire. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6556–6576. Association for Computational Linguistics.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling large language models to generate text with citations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6465–6488. Association for Computational Linguistics.

Sebastian Gehrmann, Hendrik Strobelt, and Alexander Rush. 2019. GLTR: Statistical detection and visualization of generated text. In Proceedings ofthe 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 111–116. Association for Computational Linguistics.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, et al. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Yuchen Guo, Zhicheng Dou, Huy H. Nguyen, Ching-Chun Chang, Saku Sugawara, and Isao Echizen. 2025. Measuring human involvement in AI-generated text: A case study on academic writing. arXiv preprint arXiv:2506.03501.

Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024. Spotting LLMs with binoculars: Zero-shot detection of machine-generated text. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 17519–17537. PMLR.

Tianxing He, Jingyu Zhang, Tianle Wang, Sachin Kumar, Kyunghyun Cho, James Glass, and Yulia Tsvetkov. 2023. On the blind spots of model-based evaluation metrics for text generation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12067–12097. Association for Computational Linguistics.

Md Naimul Hoque, Tasfia Mashiat, Bhavya Ghai, Cecilia D. Shelton, Fanny Chevalier, Kari Kraus, and Niklas Elmqvist. 2024. The HaLLMark effect: Supporting provenance and transparent use of large language models in writing with interactive visualization. In Proceedings ofthe 2024 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery.

Abigail Z. Jacobs and Hanna Wallach. 2021. Measurement and fairness. In Proceedings ofthe 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 375–385. Association for Computing Machinery.

John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. 2023. A watermark for large language models. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 17061–17084. PMLR.

Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, and Mohit Iyyer. 2023. Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense. In Advances in Neural Information Processing Systems, volume 36.

Mina Lee, Percy Liang, and Qian Yang. 2022. CoAuthor: Designing a human–AI collaborative writing dataset for exploring language model capabilities. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems, pages 1–19. Association for Computing Machinery.

Sheng Lu, Shan Chen, Yingya Li, Danielle Bitterman, Guergana Savova, and Iryna Gurevych. 2023. Measuring pointwise V-usable information in-context-ly. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 15739–15756. Association for Computational Linguistics.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long-form text generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076–12100. Association for Computational Linguistics.

Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and Chelsea Finn. 2023. DetectGPT: Zero-shot machine-generated text detection using probability curvature. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 24950–24962. PMLR.

Richard Yuanzhe Pang, Vishakh Padmakumar, Thibault Sellam, Ankur Parikh, and He He. 2023. Reward gaming in conditional text generation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4746–4763. Association for Computational Linguistics.

Judea Pearl. 2009. Causal inference in statistics: An overview. Statistics Surveys, 3:96–146.

Qwen Team. 2026. Qwen3.8-27B. Hugging Face model card.

Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and David Reitter. 2023. Measuring attribution in natural language generation models. Computational Linguistics, 49(4):777–840.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4902– 4912. Association for Computational Linguistics.

Aquia Richburg, Calvin Bao, and Marine Carpuat. 2024. Automatic authorship analysis in human–AI collaborative writing. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 1845–1855. ELRA and ICCL.

Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, and Soheil Feizi. 2025. Can AI-generated text be reliably detected? stress testing AI text detectors under various attacks. Transactions on Machine Learning Research.

Zhixiong Su, Yichen Wang, Herun Wan, Zhaohan Zhang, and Minnan Luo. 2025. HACo-Det: A study towards fine-grained machine-generated text detection under human–AI coauthoring. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 22015–22036. Association for Computational Linguistics.

Arjun Subramonian, Xingdi Yuan, Hal Daumé III, and Su Lin Blodgett. 2023. It takes two to tango: Navigating conceptualizations of NLP tasks and measurements of performance. In Findings of the Associationfor Computational Linguistics: ACL 2023, pages 3234–3279. Association for Computational Linguistics.

Minh Ngoc Ta, Dong Cao Van, Duc-Anh Hoang, Minh Le-Anh, Truong Nguyen, My Anh Tran Nguyen, Yuxia Wang, Preslav Nakov, and Dinh Viet Sang. 2026. FAID: Fine-grained AI-generated text detection using multi-task auxiliary and multi-level contrastive learning. In Proceedings ofthe 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3275–3296. Association for Computational Linguistics.

Brian Thompson and Matt Post. 2020. Automatic machine translation evaluation in many languages via zero-shot paraphrasing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 90–121. Association for Computational Linguistics.

U.S. Copyright Office. 2025. Copyright and artificial intelligence, part 2: Copyrightability. Technical report, U.S. Copyright Office.

Oleg Vasilyev, Vedant Dharnidharka, and John Bohannon. 2020. Fill in the BLANC: Human-free quality estimation of document summaries. In Proceedings of the First Workshop on Evaluation and Comparison of NLP Systems, pages 11–20. Association for Computational Linguistics.

Vivek Verma, Eve Fleisig, Nicholas Tomlin, and Dan Klein. 2024. Ghostbuster: Detecting text ghostwritten by large language models. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1702–1717. Association for Computational Linguistics.

Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Lidia Sam Chao, and Derek Fai Wong. 2025. A survey on LLM-generated text detection: Necessity, methods, and future directions. Computational Linguistics, 51(1):275–338.

Ziang Xiao, Susu Zhang, Vivian Lai, and Q. Vera Liao. 2023. Evaluating evaluation metrics: A framework for analyzing NLG evaluation metrics using measurement theory. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 10967–10982. Association for Computational Linguistics.

Yueqi Xie, Tao Qi, Jingwei Yi, Xiyuan Yang, Ryan Whalen, Junming Huang, Qian Ding, Yu Xie, Xing Xie, and Fangzhao Wu. 2026. Measuring human contribution in AI-assisted content generation. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6168–6190. Association for Computational Linguistics.

Yilun Xu, Shengjia Zhao, Jiaming Song, Russell Stewart, and Stefano Ermon. 2020. A theory of usable information under computational constraints. In International Conference on Learning Representations.

An Yang, Anfeng Li, Baosong Yang, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Ann Yuan, Andy Coenen, Emily Reif, and Daphne Ippolito. 2022. Wordcraft: Story writing with large language models. In Proceedings of the 27th International Conference on Intelligent User Interfaces, pages 841–852. Association for Computing Machinery.

Weizhe Yuan, Graham Neubig, and Pengfei Liu. 2021. BARTScore: Evaluating generated text as text generation. In Advances in Neural Information Processing Systems, volume 34.

Z.ai. 2026. GLM-5.2. Hugging Face model card.

Hanlin Zhang, Benjamin L. Edelman, Danilo Francati, Daniele Venturi, Giuseppe Ateniese, and Boaz Barak. 2024a. Watermarks in the sand: Impossibility of strong watermarking for language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 58851–58880. PMLR.

Qihui Zhang, Chujie Gao, Dongping Chen, Yue Huang, Yixin Huang, Zhenyang Sun, Shilin Zhang, Weiye Li, Zhengyan Fu, Yao Wan, and Lichao Sun. 2024b. LLM-as-a-coauthor: Can mixed human-written and machine-generated text be detected? In Findings of the Association for Computational Linguistics: NAACL 2024, pages 409–436. Association for Computational Linguistics.

<table><tr><td>Condition</td><td>n</td><td>Mean</td><td>Median</td><td>95% CI</td></tr><tr><td>Polish</td><td>30</td><td>0.846</td><td>0.840</td><td>[0.836, 0.856]</td></tr><tr><td>Summary</td><td>30</td><td>0.708</td><td>0.714</td><td>[0.687, 0.728]</td></tr><tr><td>Title</td><td>30</td><td>0.536</td><td>0.529</td><td>[0.524, 0.549]</td></tr><tr><td>Subject</td><td>30</td><td>0.488</td><td>0.505</td><td>[0.466, 0.509]</td></tr></table>

Table 2: Our implementation reproduces the assistance ordering. Clean Llama-3.1-8B scores on the fixed 30-item news subset. Intervals bootstrap over items.

## A Experimental Details

This appendix collects what is needed to reproduce Sections 3 and 4: data, prompts and IDs, models, scoring conventions, and statistics. The settings apply to all ten runs (seven news, three arXiv) unless we say otherwise.

## A.1 Data

News. We use the 300-item news set released by Xie et al. (2026), and our implementation reproduces their assistance ordering (Table 2). On a seeded 30-item subset with Llama-3.1-8B, mean scores fall from polishing (0.846) to summary (0.708), title (0.536), and subject (0.488), and the strict item-level ordering holds for 23 of 30 items. The same subset serves the ladder in Figure 1 and the headroom analysis. Every other news result uses all 300 items per model.

arXiv. A seed-17 uniform reservoir sample selects 200 of the 1,999,486 records with nonempty identifiers, titles, and abstracts in the public gfissore/arxiv-abstracts-2021 snapshot<sup>1</sup>, pinned at revision e4c5fbd4. The name denotes the snapshot, not a restriction to papers from 2021. We normalize whitespace and apply no subject or length filter. Reference abstracts average 143.98 words (range 12–316). The prompt asks for an abstract from the paper’s title, and the source abstract sets the word limit but is never shown to the model. All three models see the same items in the same order.

## A.2 Prompts and IDs

The clean title prompt is

Generate a news article with the title: {TITLE}.

Limit your response to {N} words.   
Start with "News:".

where N is the source article’s whitespace-token count. Polish, summary, and subject prompts replace the first sentence with the corresponding released prompt forms. For generation-time padding, the IDs are appended to the prompt as

Project-specific context:

Incorporate every detail below

verbatim and naturally. Do not

describe these details as constraints.

\- dataset: Virelia-C22D8

\- protocol: PX-7A9-A11

\- measurement: 244.6 ms

\- condition: phase-6CDB-8C6

\- label: AX-DBE86C

and for fixed-output padding, the same IDs are appended to the clean output as

Implementation metadata recorded

dataset Virelia-C22D8;

protocol PX-7A9-A11;

measurement 244.6 ms;

condition phase-6CDB-8C6;

label AX-DBE86C.

Larger doses continue the same deterministic type cycle. WRONG IDs keep the types and count but draw their values from a disjoint seed.

## A.3 Models and Computations

Table 3 lists the checkpoints. Every model generates and evaluates its own outputs. The four local models run in bfloat16 without quantization on one NVIDIA A10 each, with Python 3.12, PyTorch 2.6.0, CUDA 12.6, and Transformers 4.57.6. The 30-item Llama-3.1-8B ladder and headroom runs use Transformers 4.48.3. The three served models are queried with at most four concurrent requests.

In total, we used approximately 15 NVIDIA A10 GPU-hours for the local-model experiments and 16 million input and output tokens across the three served models. These estimates cover generation and likelihood scoring for the main experiments, excluding pilot experiments and reruns.

## A.4 Scoring conventions

Each model uses its native chat template with the system message You are a helpful assistant., and optional thinking modes are disabled. Conditional scoring concatenates the formatted prefix, the response, and EOS, masks every prefix label, and sums the negative log-likelihood over the response-plus-EOS positions. Unconditional scoring evaluates the same response-plus-EOS tokens after one boundary token, which is BOS where the tokenizer defines one. Qwen3-8B, Qwen3.8- 27B, and GLM-5.2 define no BOS and use EOS as the left boundary. Because a fixed output has the same unconditional denominator under every context, same-output contrasts do not depend on this convention. Absolute scores remain model-relative and are not calibrated across families. Scores are never clipped.

<table><tr><td>Name</td><td>Checkpoint</td><td>Developer</td><td>Params</td><td>Access</td><td>Precision</td><td>Revision</td></tr><tr><td>Llama-3.1-8B</td><td>meta-1lama/Llama-3.1-8B-Instruct</td><td>Meta</td><td>8.0B</td><td>local weights</td><td>bfloat16</td><td>0e9e39f2</td></tr><tr><td>Qwen3-8B</td><td>Qwen/Qwen3-8B</td><td>Alibaba</td><td>8.2B</td><td>local weights</td><td>bfloat16</td><td>b968826d</td></tr><tr><td>Gemma-3-4B</td><td>google/gemma-3-4b-it</td><td>Google</td><td>4.3Ba</td><td>local weights</td><td>bfloat16</td><td>093f9f38</td></tr><tr><td>Phi-4-mini</td><td> $\mathsf { m i c r o s o f t / P h i - 4 - m i n i - i n s t r u c t }$ </td><td>Microsoft</td><td>3.8B</td><td>local weights</td><td>bfloat16</td><td>cfbefacb</td></tr><tr><td>DeepSeek-V4-Flash</td><td>deepseek-ai/DeepSeek-V4-Flash-0731</td><td>DeepSeek</td><td>284B MoEb</td><td>served endpoint</td><td>unreported</td><td>7872f01bc</td></tr><tr><td>Qwen3.8-27B</td><td>Qwen/Qwen3.8-27B</td><td>Alibaba</td><td>27Bd</td><td>served endpoint</td><td>unreported</td><td>1d4bf0f2c</td></tr><tr><td>GLM-5.2</td><td>zai-org/GLM-5.2-FP8</td><td>Z.ai</td><td>753B MoE</td><td>served endpoint</td><td>FP8e</td><td>f33c6dc5c</td></tr></table>

Table 3: One panel of seven models from six families. We use the short names in the first column throughout. Revisions are Hugging Face commits. <sup>a</sup>Includes the vision wrapper; only text inputs are used. <sup>b</sup>Per the DeepSeek-V4 report; the Hugging Face listing for the 0731 checkpoint gives 304B because it includes an attached speculativedecoding module. <sup>c</sup>Revision of the tokenizer and chat template we pin; the endpoint does not expose its weight revision. <sup>d</sup>Dense; 28B with its vision encoder. <sup>e</sup>Named by the served model alias and not independently verified. MoE: mixture of experts; we do not report active parameters.

On served endpoints we obtain teacher-forced prompt log-probabilities and check every returned token ID against the pinned tokenizer. The one completion token the endpoint must generate is excluded. For GLM-5.2 we disable automatic specialtoken insertion because the native template already supplies its control tokens. For fixed suffixes, character offsets are mapped to token offsets so that body and suffix log-likelihoods can be separated. The output-only estimate uses the released Llama-3.1-8B threshold − log τ = 0.5496. No thresholds were released for other models, so we report the estimate for Llama-3.1-8B only.

## A.5 Decoding, seeds, and statistics

Generation uses temperature 0.7, top-p 0.9, and at most 384 new tokens, with one clean and one attacked generation per item. Served runs seed generation with 17 + i for clean and 10017 + i for attacked outputs at manifest index i. Local runs use a seeded batched sampling stream, and run metadata record every seed. The 384-token limit is reached by 16/300 clean and 25/300 attacked DeepSeek-V4- Flash news generations, 50/300 and 148/300 for Qwen3.8-27B, and 2/300 and 0/300 for GLM-5.2 (arXiv counts are in Appendix B.4). All outputs are retained, including truncated ones. Every interval is a 95% percentile interval from 10,000 bootstrap resamples of items, with all paired conditions kept together, and no example is discarded after generation.

## B Additional Results

This appendix shows the estimates behind each claim of Section 4. We give per-model intervals, the dose response for every run shown, the Llama-3.1-8B diagnostics, the arXiv replication, and three subtler lexical variants that the ledger predicts should barely work (they barely do).

## B.1 Per-model effects

No single model drives the averages (Table 4). Showing the matching IDs raises the fixed-output score by 0.324–0.505 and does so in all 2,100 pairs, and MATCHED beats WRONG in every pair as well. Generation-time attack−clean effects are positive for 2,040 of 2,100 articles, and generationtime MATCHED−HIDDEN for 2,097 of 2,100. Exact ID recall in attacked articles is 0.65 (Llama-3.1-8B), 0.98 (Qwen3-8B), 0.98 (Gemma-3-4B), 0.85 (Phi-4-mini), 0.94 (DeepSeek-V4-Flash), 0.93 (Qwen3.8-27B), and 0.99 (GLM-5.2). Mean clean/attacked word counts, in the same order, are 216/198, 232/195, 220/204, 263/273, 229/222, 281/265, and 227/209. The generation-time effect therefore does not track length, and the fixed-output contrasts remove length altogether.

## B.2 Dose response

Corollary 1 predicts the shape of these curves. As IDs accumulate, the suffix’s surprisal share w grows, so MATCHED scores move toward the suffix’s local score and HIDDEN scores fall. Figure 5 shows both trends in every run. Where MATCHED first dips below clean at one ID (every run shown except Phi-4-mini), the pattern is consistent with the first IDs perturbing the body while the short suffix is still dominated by its unpredictable metadata header. Added IDs then outweigh both, and the curves climb.

<table><tr><td></td><td colspan="3">Fixed output</td><td colspan="2">Generation-time</td></tr><tr><td>Model</td><td>MATCHED—clean</td><td>MATCHED-HIDDEN</td><td>MATCHED—WRONG</td><td>Attack—clean</td><td>MATCHED—HIDDEN</td></tr><tr><td>Llama-3.1-8B</td><td>+0.130 [+0.125, +0.135]</td><td>+0.505 [+0.500, +0.510]</td><td> $+ 0 . 2 5 2 \ [ + 0 . 2 5 0 , + 0 . 2 5 4 ]$ </td><td> $+ 0 . 1 9 9 \ [ + 0 . 1 8 5 , + 0 . 2 1 3 ]$ </td><td>+0.521 [+0.490, +0.551]</td></tr><tr><td>Qwen3-8B</td><td>-0.144 [-0.150, -0.138]</td><td>+0.333 [+0.327, +0.339]</td><td>+0.293 [+0.290, +0.297]</td><td>+0.146 [+0.142, +0.150]</td><td>+0.793 [+0.783, +0.802]</td></tr><tr><td>Gemma-3-4B</td><td>-0.062 [-0.073, -0.052]</td><td>+0.339 [+0.330, +0.347]</td><td>+0.231 [+0.228, +0.234]</td><td>+0.235 [+0.224, +0.245]</td><td>+0.843 [+0.831, +0.855]</td></tr><tr><td>Phi-4-mini</td><td>+0.211 [+0.206, +0.215]</td><td>+0.388 [+0.383, +0.392]</td><td> $+ 0 . 2 2 4 \stackrel { \cdot } { [ } + 0 . 2 2 1 , + 0 . 2 2 6 \stackrel { \cdot } { ] }$ </td><td> $+ 0 . 2 7 1 \ \bar { [ + 0 . 2 6 1 , + 0 . 2 8 2 ] }$ </td><td>+0.505 [+0.494, +0.516]</td></tr><tr><td>DeepSeek-V4-Flash</td><td>-0.047 [-0.051, -0.042]</td><td>+0.408 [+0.402, +0.415]</td><td> $+ 0 . 2 4 1 \ [ + 0 . 2 3 8 , + 0 . 2 4 4 ]$ </td><td> $+ 0 . 1 3 3 \ [ + 0 . 1 2 8 , + 0 . 1 3 9 ]$ </td><td>+0.701 [+0.690, +0.711]</td></tr><tr><td>Qwen3.8-27B</td><td>+0.095 [+0.091, +0.099]</td><td>+0.324 [+0.318, +0.331]</td><td> $+ 0 . 1 6 3 \ [ + 0 . 1 6 0 , + 0 . 1 6 7 ]$ </td><td> $+ 0 . 1 6 1 \ [ + 0 . 1 5 0 , + 0 . 1 7 1 ]$ </td><td>+0.506 [+0.491, +0.521]</td></tr><tr><td>GLM-5.2</td><td>+0.072 [+0.065, +0.080]</td><td>+0.426 [+0.420, +0.432]</td><td> $+ 0 . 2 4 8 \ [ + 0 . 2 4 5 , + 0 . 2 5 2 ]$ </td><td> $+ 0 . 2 1 5 \ [ + 0 . 2 0 6 , + 0 . 2 2 4 ]$ </td><td>+0.621 [+0.613, +0.629]</td></tr></table>

Table 4: Every model moves, and every interval excludes zero. Paired mean effects at eight IDs with 95% bootstrap intervals, 300 news items per model. Fixed-output contrasts score one padded text under two contexts, except MATCHED−clean, which also changes the text. Attack−clean compares separately generated articles. Generation-time MATCHED−HIDDEN rescores the same attacked article with the full prompt and with the visible request.

<table><tr><td>Evaluator context</td><td>Mean score</td><td>∆</td><td>95% CI</td><td>Recall</td></tr><tr><td>MATCHED (full prompt)</td><td>0.747</td><td>+0.199</td><td>[+0.185, +0.213]</td><td>0.65</td></tr><tr><td>HIDDEN (visible request)</td><td>0.226</td><td>-0.322</td><td>[-0.340, -0.304]</td><td>0.65</td></tr><tr><td>WRONG (wrong IDs)</td><td>0.526</td><td>-0.022</td><td>[-0.028, -0.016]</td><td>0.65</td></tr></table>

Table 5: Hiding the IDs erases the attack and more. Llama-3.1-8B generation-time articles with eight requested IDs, 300 items. ∆ is relative to the separately generated clean article, and recall is exact ID recall in the output.
<table><tr><td>Assistance level</td><td>Clean score</td><td>MATCHED ∆</td><td>HIDDEN ∆</td><td>WRONG∆</td></tr><tr><td>Polish (full draft)</td><td>0.846</td><td>-0.004</td><td>-0.364</td><td>-0.172</td></tr><tr><td>Summary (key ideas)</td><td>0.708</td><td>+0.054</td><td>-0.397</td><td>-0.159</td></tr><tr><td>Title</td><td>0.536</td><td>+0.137</td><td>-0.366</td><td>-0.112</td></tr><tr><td>Subject</td><td>0.488</td><td>+0.143</td><td>-0.353</td><td>-0.120</td></tr></table>

Table 6: MATCHED inflation vanishes as the clean score nears the suffix’s local score. Eight-ID fixedoutput shifts from the clean score across assistance levels (Llama-3.1-8B, 30-item subset; Figure 4).

## B.3 Llama-3.1-8B diagnostics

Llama-3.1-8B is the one model with a released output-only threshold and the model behind the ladder, so we report two further views of it. In generation, attacked articles score 0.747 under MATCHED. Removing only the IDs from the context drops the same articles to 0.226, a paired gap of 0.521 (95% CI [0.490, 0.551]), and WRONG yields 0.526 (Table 5, Figure 6, left). With no prompt at all, the output-only estimate rises from 0.36 for clean outputs to 0.68 at eight IDs (Figure 6, right). Table 6 gives the fixed-output shifts behind Figure 4.

## B.4 Replication on arXiv abstracts

The arXiv runs repeat both interventions with the three served models (600 model–abstract pairs), using the settings of Appendix A. Showing the matching IDs raises fixed-output scores by 0.431–0.471, and MATCHED beats WRONG by 0.246–0.279, in all 600 pairs (Table 7). MATCHED padding stays below the clean reference for DeepSeek-V4-Flash (−0.120), whose clean abstracts already score 0.79, and exceeds it for Qwen3.8-27B (+0.035) and GLM-5.2 (+0.021), as the headroom account of §5 predicts. Generation-time attack−clean effects are +0.134, +0.196, and +0.196, positive for 200, 199, and 200 of 200 abstracts. Generation-time MATCHED−HIDDEN effects range from 0.663 to 0.778 and MATCHED−WRONG effects from 0.287 to 0.364, positive in all 600 pairs. ID recall exceeds 0.99. Mean clean/attacked word counts are 133.66/124.74 (DeepSeek-V4-Flash), 161.37/153.00 (Qwen3.8-27B), and 135.75/118.71 (GLM-5.2). The 384-token limit is reached by 6/6, 18/31, and 3/3 of 200 clean/attacked abstracts, and all are retained. Differences from the news effects are descriptive, since items and output lengths differ.

## B.5 Subtler lexical variants

We also piloted three less conspicuous manipulations with DeepSeek-V4-Flash on 60 predetermined news items (Table 8). Two required words in the prompt, or one required three-token phrase, raise the score by about 3 points over the clean prompt, and rarity-based selection does not beat random or common choices. Replacing up to two words near the end of a fixed article with rarer synonyms, with the prompt unchanged, lowers the score. Proposition 1 predicts both outcomes. A few copyable words carry little surprisal, so they earn little credit. Rare words that the prompt does not predict have local scores near zero, so they pull the ledger down. The same rare edits raise the outputonly estimate in 58 and 59 of 60 articles for the random and rarity-greedy strategies, as Corollary 2 predicts. Repeated scoring of identical texts on this endpoint had a mean within-text standard deviation of 0.65 points (maximum 1.29), so differences of a point or less are not interpretable.

![](images/bbd6cd588f7628d91af656910248198b68ae5688dd523e8c97c3b6c9f57be8e2.jpg)  
Figure 5: MATCHED stays above HIDDEN and WRONG at every dose, for every run shown. Fixed-output change from the clean score against the number of IDs K (K = 0 is the clean output). At each positive dose the three contexts score the same padded text. Top: news, 300 items per model. The GLM-5.2 news dose curve is not shown, and its eight-ID estimates are in Table 4. Bottom: arXiv, 200 items per model. Bands are paired 95% bootstrap intervals. The embedded labels predate our naming scheme. “Details hidden” is HIDDEN, “Wrong details” is WRONG, “Specificity anchors” counts IDs, and panel titles abbreviate the names in Table 3.

## C Theory

This appendix proves the results of Sections 5 and 6, then puts the ledger’s quantitative predictions to the Llama-3.1-8B estimates. The checks use aggregate scores only, so they bound the suffix’s surprisal share rather than measure it span by span.

## C.1 Proofs

Throughout, the scored sequence is the response followed by EOS, and the terminal token belongs to the last span.

Proposition 1. By the chain rule, $\begin{array} { r } { \boldsymbol { \mathcal { I } } ( \boldsymbol { y } ) = \sum _ { k } \boldsymbol { S } _ { k } } \end{array}$ and $\begin{array} { r } { \mathcal { I } ( y \mid x ) = \sum _ { k } T _ { k } } \end{array}$ . Hence

$$
\begin{array} { c } { \displaystyle \phi ( x , y ) = \frac { \sum _ { k } S _ { k } - \sum _ { k } T _ { k } } { \mathcal { T } ( y ) } } \\ { \displaystyle = \sum _ { k } \frac { S _ { k } - T _ { k } } { \mathcal { T } ( y ) } = \sum _ { k } \kappa _ { k } . } \end{array}
$$

Under a softmax model every nonempty span has $S _ { k } > 0 ;$ , so $\kappa _ { k } = ( S _ { k } / \mathcal { T } ( y ) ) ( 1 - T _ { k } / S _ { k } ) = w _ { k } \phi _ { k } .$ and $\textstyle \sum _ { k } w _ { k } = 1$ . For the output-only estimate, $\begin{array} { r } { \widehat { \phi } ( y ) \ = \ 1 - \ b N / \mathbb { Z } ( y ) \ = \ \sum _ { k } ( S _ { k } - b n _ { k } ) / \mathbb { Z } ( y ) } \end{array}$ because $\textstyle \sum _ { k } n _ { k } = N$

Corollary 1. Apply Proposition 1 to the two spans y and z under $x ^ { \prime } .$ . The first span’s local score is $1 - { \mathcal { T } } ( y \mid x ^ { \prime } ) / { \mathcal { T } } ( y ) = \phi ( x ^ { \prime } , y )$ and its weight is 1 − w. The second span has weight w and local score $\phi _ { z } ( x ^ { \prime } )$ , which gives Equation 4. Since $0 < w < 1$ , the padded score lies between the two local scores, and $\phi ( x ^ { \prime } , y z ) - \phi ( x ^ { \prime } , y )$ = w $[ \phi _ { z } ( x ^ { \prime } ) \ : - \ : \phi ( x ^ { \prime } , y ) ]$ is positive exactly when $\phi _ { z } ( x ^ { \prime } ) > \phi ( x ^ { \prime } , y )$ . Finally,

![](images/5b749814393f062bdc6cec46b673543a7cd6d50b1fbb7856541b7be4e6e48725.jpg)

![](images/5ce0f08de301b528061285d570e582a1df896271b7ca8d1b353c98ca699754df.jpg)

Figure 6: Two ways to inflate one model’s score. Llama-3.1-8B, all 300 news items. Left: generation-time articles scored under each evaluator context. The same attacked articles average 0.75 under MATCHED and 0.23 under HIDDEN; diamonds are means. Axis labels “Baseline”, “Full context”, “Visible only”, and “Mismatched” denote the clean article, MATCHED, HIDDEN, and WRONG. Right: the output-only estimate of fixed padded outputs rises with the number of IDs (axis “Anchors”) although no prompt is used. Bands are paired 95% bootstrap intervals.
<table><tr><td>Model</td><td colspan="2">MATCHED—clean</td><td colspan="2">MATCHED-HIDDEN</td><td colspan="2">MATCHED—WRONG</td></tr><tr><td>DeepSeek-V4-Flash</td><td></td><td> $- 0 . 1 2 0 \left[ - 0 . 1 2 7 , - 0 . 1 1 4 \right]$ </td><td>+0.431</td><td> $[ + 0 . 4 1 7 , + 0 . 4 4 5 ]$ </td><td> $+ 0 . 2 7 9 \ [ + 0 . 2 7 4 , + 0 . 2 8 4 ]$ </td><td></td></tr><tr><td>Qwen3.8-27B GLM-5.2</td><td>+0.035 +0.021</td><td> $\left[ + 0 . 0 2 7 , + 0 . 0 4 3 \right]$   $\mathbf { \bar { \rho } } _ { [ + 0 . 0 1 3 , + 0 . 0 2 8 ] }$ </td><td>+0.471 +0.465</td><td> $\dot { [ + 0 . 4 5 9 , + 0 . 4 8 3 ] }$ </td><td>+0.246 [+0.241, +0.252] +0.270  $[ + 0 . 2 6 5 , + 0 . 2 7 5 ]$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td> $\dot { [ } + 0 . 4 5 4 , + 0 . 4 7 5 \dot { ] }$ </td><td></td><td></td></tr><tr><td>Model</td><td></td><td>Attack—clean</td><td>MATCHED—HIDDEN</td><td></td><td>MATCHED—WRONG</td><td>Recall</td></tr><tr><td>DeepSeek-V4-Flash</td><td></td><td> $+ 0 . 1 3 4 \left[ + 0 . 1 2 8 , + 0 . 1 4 0 \right]$ </td><td> $+ 0 . 7 7 8 \left[ + 0 . 7 6 9 , + 0 . 7 8 8 \right]$ </td><td></td><td>+0.364 [+0.357, +0.370]</td><td>0.996</td></tr><tr><td>Qwen3.8-27B</td><td></td><td> $+ 0 . 1 9 6 \left[ + 0 . 1 8 8 , + 0 . 2 0 4 \right]$ </td><td>+0.663</td><td> $[ + 0 . 6 4 9 , + 0 . 6 7 6 ]$ </td><td>+0.287[+0.279, +0.296]</td><td>0.991</td></tr><tr><td>GLM-5.2</td><td></td><td> $+ 0 . 1 9 6 \ \dot { [ + 0 . 1 8 8 , + 0 . 2 0 4 ] }$ </td><td>+0.678</td><td> $\left[ + 0 . 6 6 9 , + 0 . 6 8 6 \right]$ </td><td>+0.300 [+0.295, +0.306]</td><td>0.995</td></tr></table>

Table 7: The audit replicates on scientific abstracts. Eight IDs, $n = 2 0 0$ per model, paired mean differences with 95% bootstrap intervals. Top, fixed output: MATCHED−HIDDEN and MATCHED−WRONG compare identical padded outputs, while MATCHED−clean also changes the output. Bottom, generation time: attack−clean compares separately generated outputs, the other two contrasts rescore the same attacked output, and recall is exact ID recall.

<table><tr><td>Variant (vs. clean, points)</td><td>Mean [95% CI]</td></tr><tr><td>Two required words in the prompt</td><td></td></tr><tr><td>Common words</td><td> $+ 3 . 0 \ [ + 1 . 9 , + 4 . 1 ]$ </td></tr><tr><td>Random rare words</td><td>+3.2 [+2.2, +4.2]</td></tr><tr><td>Score-selected rare words</td><td>+3.5 [+2.5, +4.4]</td></tr><tr><td>One required three-token phrase</td><td></td></tr><tr><td>Surprisal-selected</td><td> $+ 3 . 4 \ [ + 2 . 4 , + 4 . 4 ]$ </td></tr><tr><td>Surprisal-selected − random phrase +0.6 [−0.4, +1.6]</td><td></td></tr><tr><td>Up to two rarer synonyms near the end, prompt fixed</td><td></td></tr><tr><td>Random</td><td> $- 5 . 1 \left[ - 5 . 8 , - 4 . 4 \right]$ </td></tr><tr><td>Rarity-greedy</td><td> $- 6 . 2 \ : [ - 7 . 0 , - 5 . 5 ]$ </td></tr><tr><td>Score-greedy</td><td> $- 0 . 2 \ : [ - 0 . 4 , - 0 . 0 ]$ </td></tr></table>

Table 8: A few rare words buy almost nothing. Lexical pilots with DeepSeek-V4-Flash (n = 60 news items, paired bootstrap intervals). Score-greedy editing keeps the original text when no candidate helps, which happened for 47 items.

$$
\phi ( x ^ { \prime } , y z ) = 1 - { \frac { { \mathcal { T } } ( y \mid x ^ { \prime } ) + { \mathcal { T } } ( z \mid x ^ { \prime } , y ) } { { \mathcal { T } } ( y ) + { \mathcal { T } } ( z \mid y ) } } ,
$$

and the fraction tends to zero when $\mathcal { T } ( z \mid y )  \infty$ with $\textstyle { \mathcal { T } } ( y )$ fixed and the numerator $o ( \mathcal { I } ( z \mid y ) )$

Context swaps. For a fixed output $\tilde { y }$ and contexts $x _ { 1 } , x _ { 0 }$ , the denominators of Equation 1 coincide, so $\phi ( x _ { 1 } , \tilde { y } ) - \phi ( x _ { 0 } , \tilde { y } ) = [ \mathcal { T } ( \tilde { y } \mid x _ { 0 } ) - \mathcal { T } ( \tilde { y } \mid$ $x _ { 1 } ) ] /  { \mathbb { T } } ( \tilde { y } )$ . Applying Equation 4 under both contexts and subtracting gives Equation 5. Both terms share the weight w because unconditional surprisals do not depend on the context.

Corollary 2. With $b ~ > ~ 0 , ~ \frac { d } { d \bar { \mathcal { Z } } } \big ( 1 ~ - ~ b / \bar { \mathcal { Z } } \big ) ~ =$ $b / \bar { \mathcal { Z } } ^ { 2 } > 0$ , and $1 - b / \bar { \mathcal { Z } } \to 1$ as $\bar { \mathcal { T } } \to \infty$ . If an ntoken output has mean surprisal L<sup>¯</sup> and an appended m-token span has mean surprisal $\bar { S } > \bar { L }$ given the output, the new mean $( n \bar { L } + m \bar { S } ) / ( n + m )$ exceeds ${ \bar { L } } .$

Proposition 2. Let $\begin{array} { r c l c r } { c _ { - } } & { = } & { \operatorname* { i n f } \mathcal { C } ( o ) , \ c _ { + } } & { = } \end{array}$ sup C(o), and $a = f ( o )$ . For any $\epsilon > 0 ,$ choose compatible values within ϵ of c<sub>−</sub> and $c _ { + }$ . By the triangle inequality, the two absolute errors sum to at least $c _ { + } - c _ { - } - 2 \epsilon ,$ , so the larger is at least $D ( o ) / 2 - \epsilon .$ . Let $\epsilon \downarrow 0$ . The midpoint $a ^ { * } = ( c _ { - } + c _ { + } ) / 2$ satisfies $| a ^ { * } - c | \leq D ( o ) / 2$ for every $c \in { \mathcal { C } } ( o )$ . For a randomized estimator A, choose $c _ { 1 } , c _ { 2 } \in \mathcal { C } ( o )$ with $c _ { 2 } - c _ { 1 } \geq D ( o ) - 2 \epsilon$

Then $\mathbb { E } | A - c _ { 1 } | + \mathbb { E } | A - c _ { 2 } | \geq c _ { 2 } - c _ { 1 }$ , so one expected error is at least $D ( o ) / 2 - \epsilon .$ . For the laundering statement, both 0 and $c _ { 1 }$ lie in ${ \mathcal { C } } ( o )$ so $D ( o ) \geq c _ { 1 }$

Scope. The target $c ,$ the observation map $^ { g , }$ and the admissible history class jointly determine ${ \mathcal { C } } ( o )$ Authenticated logs or workflow constraints shrink this set, and selectively exported transcripts enlarge it. The result permits prior-based prediction, population estimation, and distributional detection. It states only that transcript-level point identification requires c to be constant on each observationalequivalence class.

## C.2 Mechanism checks

How much surprisal do eight IDs carry? By Equation 5, the MATCHED−HIDDEN gap approximates the suffix weight w when showing the IDs makes them near-certain $( \phi _ { z } ^ { \mathrm { M } }$ near one), hiding them leaves them unpredictable $( \phi _ { z } ^ { \mathrm { H } }$ near zero), and the body term is small. Llama-3.1-8B’s outputonly estimates give an independent handle on w. Inverting Equation 2 with $b = 0 . 5 4 9 6$ , the mean estimates for clean outputs (0.356) and eight-ID padded outputs (0.679) correspond to mean surprisals of about 0.85 and 1.71 nats per token (harmonic means across items). Appending m suffix tokens to an N-token body gives

$$
w = 1 - \frac { N } { N + m } \cdot \frac { \bar { \mathcal { Z } } ( y ) } { \bar { \mathcal { Z } } ( y z ) } \ge 1 - \frac { 0 . 8 5 } { 1 . 7 1 } \approx 0 . 5 0 ,
$$

ignoring item heterogeneity and the terminal token. Both routes indicate that the eight IDs carry about half of the padded document’s surprisal, although they occupy a minority of its tokens.

Headroom. Regressing the MATCHED shift in Table 6 on the clean score gives slope $- 0 . 4 3$ , intercept 0.36, a zero crossing at 0.84, and $R ^ { 2 } = 0 . 9 9$ over the four levels. Under Corollary 1, if the IDs leave the body’s own score unchanged, the MATCHED shift is $w ( \phi _ { z } ^ { \mathrm { M } } - \phi _ { 0 } )$ , so inflation vanishes when the clean score $\phi _ { 0 }$ reaches the suffix’s local score. A change $\delta$ in the body $\mathrm { ^ { , } s }$ score adds $( 1 - w ) \delta$ and moves the crossing accordingly. Because the HIDDEN context equals the clean context, the HIDDEN shift is exactly $w ( \phi _ { z } ^ { \mathrm { H } } - \phi _ { 0 } )$ , up to the terminal token. Its near-constancy across levels (−35 to −40 points) implies that w shrinks as the clean score rises, as expected when polished outputs follow a human-written article and therefore carry more surprisal in the body. Span-level measurements would separate w from the local scores.