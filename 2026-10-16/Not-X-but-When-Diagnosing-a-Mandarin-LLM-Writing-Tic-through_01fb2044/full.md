# Not X, but When: Diagnosing a Mandarin LLM Writing Tic through Contextual Appropriateness

Yihang Peng<sup>1∗</sup> Yi Jing<sup>1∗</sup> Dewen Liu<sup>2</sup> Weihan Meng<sup>1</sup> Ying Liu<sup>1†</sup> <sup>1</sup>Tsinghua University <sup>2</sup>Fudan University yingliu@tsinghua.edu.cn

## Abstract

Recurring expressions in model-written text are often treated as signatures of machine authorship, yet most of them serve legitimate functions. Whether such an expression is a tic depends on whether the model knows when it is called for. This judgment of appropriateness, central to communicative competence, remains largely unexamined for language models. We study it through the Mandarin corrective construction bushi X, ershi Y (‘not X, but Y’). Drawing on Gricean pragmatics and corrective negation, we introduce a decision tree that classifies uses as substantive, optional, or defective by testing whether the context makes X a live alternative and whether rejecting X is informative. Five commercial models use the construction 2.5 to 9.2 times as densely as a human reference. Of 307 annotated uses, only 1.30% are defective while 52.8% are optional, that is, correct but unneeded, so models fall short in appropriateness far more often than in correctness. Several reward models favor the construction even over quality-matched variants, an LLM judge favors it even where the context does not license it, and both supervised fine-tuning and preference optimization tend to raise its frequency in open-weight models. We hope this work encourages the development of models attentive to the appropriateness of their expressions.

## 1 Introduction

Large language models can now resolve real-world issues and complete multi-step tasks with tools (Jimenez et al., 2024; Yao et al., 2024), and a growing share of the text people read is machinegenerated. In Grice’s (1975) account of cooperative conversation, a contribution should be as informative as the exchange requires and no more, and it should be relevant to the purpose at hand. An entirely accurate response can still fall short of these maxims by elaborating beyond what was asked or by addressing a question that was never at issue. Conformity to these maxims is therefore part of model quality alongside factual accuracy.

![](images/3a7c478c1d6b4943d27799103c7de3fa58a8ac164bee05b8e5938afaaf6c1805.jpg)  
Figure 1: One form, four discourse functions: substantive (S), unlicensed $( O _ { L } )$ , redundant $( O _ { U } )$ , and defective (D) uses of not X, but Y. Translated excerpts retain the target sentence; ellipses mark omitted context.

Model-written text also contains expressions that recur often enough for readers to treat them as signatures of machine authorship (Juzek and Ward, 2025; Shaib et al., 2025). Most of these expressions are ordinary resources of the language, and human writers use the same devices where the discourse calls for them. Whether a recurring expression amounts to a tic thus depends on whether the model knows when it is called for. Hymes (1972) placed this knowledge at the core of communicative competence, as a judgment of whether an utterance is appropriate to its context, separate from whether it is grammatically possible and from how often it is in fact produced.

Research on language models, however, has mostly approached pragmatics from the listener’s side, testing whether models recover implicatures and presuppositions (Ruis et al., 2023; Hu et al., 2023; Sravanthi et al., 2024), and recent comparisons show that competence in comprehension does not transfer to appropriate production (Sieker and Zarrieß, 2026). When model production is studied, the usual measure is frequency, with words and phrases that are over-represented relative to human text treated as targets for suppression (Juzek and Ward, 2025; Paech et al., 2025). Frequency answers the last of Hymes’s questions, how often an expression is in fact produced. For expressions with a legitimate function, it conflates appropriate and inappropriate uses, and suppression removes both. Whether models command appropriateness itself remains to be examined.

We examine this question through corrective negation, which denies one alternative and asserts another in its place. Its English form, it’s not X, it’s Y, is regularly cited as a hallmark of chatbot writing (Oremus, 2026), and Chinese online commentary singles out the Mandarin bushi X, ershi Y (‘not X, but Y’; 不是 X，而是 Y) in the same way, chiefly because the rejected X often appears to be a position that no one has taken.<sup>1</sup> Corrective negation is also the ordinary way to correct a misconception. Each use of the construction thus raises the two questions behind Grice’s maxims (Figure 1): whether the context made X a live alternative (relevance), and whether rejecting X contributes information not already established (quantity). We ask whether models, like competent speakers, know when the construction is called for.

Linguistic theory makes these questions precise as conditions on licensing. As a construction, bushi X, ershi Y conventionally pairs its form with a corrective function (Goldberg, 2006; Shao and Wang, 2010; Zuo, 2025). The correction presupposes alternatives that contrast along a shared dimension (Rooth, 1992), and X need not have been asserted by anyone as long as it is accessible in the common ground (Toosarvandani, 2013; Stalnaker, 2002). What is rejected may be a proposition or the manner in which it is expressed, as in metalinguistic negation (Horn, 1985), and whether the correction is informative depends on the question under discussion (Roberts, 2012). These accounts separate three cases that are formally identical: a correction that resolves a live alternative, one that is well formed but unneeded, and one that is defective because it rejects something true or misattributes a view.

We introduce a diagnostic framework built around an ordered decision tree (Figure 2) that sorts each use of the construction into these three cases, first screening out defective uses and then testing the relevance and quantity conditions; a use that passes both is substantive, and one that fails either is optional. We apply the tree to answers from five commercial models on Chinese prompts and compare their usage with a human reference. We then investigate the sources of this preference, scoring each answer against minimally edited counterparts with reward models and an LLM judge, and tracing the construction’s frequency across training stages of open-weight models and in public preference data.

For a substantial share of uses, models do not. They use the construction 2.5 to 9.2 times as densely as the human reference, and the shortfall lies in appropriateness more than in correctness, with only 1.30% of annotated uses defective and 52.8% classified as optional. Several reward models favor the construction even when variants match in general quality, and an LLM judge favors it even where the context does not license it. Along the training pipelines of open-weight models, both supervised fine-tuning and preference optimization tend to raise its frequency.

Our contributions are as follows:

• We formalize stylistic tics in LLM-generated text as a failure of appropriateness in the sense of communicative competence.

• We distinguish uses that are correct from uses that are needed, and operationalize this distinction as a decision tree derived from linguistic theory.

• We provide a method for diagnosing how reward models and training stages shape the preference for such expressions.

## 2 Related Work

Our work draws on three lines of research. Studies of language models as speakers identify which expressions models overproduce; linguistic accounts of corrective negation specify when the construction is appropriate; and work on evaluator bias suggests where a preference for it may originate.

## 2.1 Language Models as Speakers

Research on language models has examined their pragmatic competence and their characteristic output largely in isolation. Pragmatic evaluations have mostly taken the listener’s side, testing whether models recover implicatures and presuppositions (Ruis et al., 2023; Hu et al., 2023; Sravanthi et al., 2024) or recognize constructional meaning (Weissweiler et al., 2022). The speaker’s side has a longer history in generation, where Gricean maxims constrain referring expressions (Dale and Reiter, 1995) and pragmatic speakers reason about a listener in reference games (Frank and Goodman, 2012; Andreas and Klein, 2016), though in tasks whose communicative goal is fixed in advance. Studies of open-ended model writing instead focus on its characteristic expressions, including over-represented words (Juzek and Ward, 2025; Kobak et al., 2025), recurring syntactic templates (Shaib et al., 2024), and broader measures of low-quality “slop” (Shaib et al., 2025), with suppression proposed as a remedy (Paech et al., 2025). These studies establish which expressions models produce too often, which is a matter of frequency.

## 2.2 Corrective Negation

Corrective negation is well described in both English and Mandarin, and these descriptions specify when a speaker uses it appropriately. The construction denies one alternative and asserts another in its place. Work on English shows that it presupposes alternatives contrasting along a shared dimension and can target the form of an utterance as well as its content (Horn, 1985; Toosarvandani, 2013). Studies of bushi…ershi describe the same corrective function in Mandarin. Shao and Wang (2010) identify a pseudo-denial use, in which the negation does not retract any claim held in the context; Tan (2016) compares the construction with English not A but B and finds both semantic and metalinguistic correction in each; and Zuo (2025) analyzes ershi as the connective that lexicalizes correction, distinct from those for contrast and adversative relations. We turn these descriptions into conditions for assessing each use in model output.

## 2.3 Evaluator Bias

The evaluators used to train and assess language models are biased toward surface features of responses, and training can carry these biases into model output. Reward models and LLM judges favor longer responses (Singhal et al., 2024), list formatting (Zhang et al., 2025), and style more broadly (Wu and Aji, 2025), and LLMjudges show position and verbosity biases (Zheng et al., 2023). Preference optimization also narrows the range of model outputs (Kirk et al., 2024), and open posttraining pipelines that release intermediate checkpoints make it possible to trace such changes stage by stage (Lambert et al., 2024; OLMo et al., 2025). Unlike length or formatting, a corrective construction’s appropriateness depends on context; the question is whether evaluators and training favor it only where it is called for.

## 3 Methods

We propose a diagnostic framework that assesses each use of the construction by whether its context calls for it, rather than by how often it occurs. Its core is a decision tree, derived from corrective negation and Gricean pragmatics, that labels each occurrence as substantive, optional, or defective. We pair these labels with frequency measures, evaluator comparisons, and checkpoint analyses to trace the sources of the preference.

## 3.1 Analytical Criteria

We operationalize contextual appropriateness through corrective meaning and discourse contribution. A descriptive reading approximates p<sub>X</sub> p<sub>Y</sub>; metalinguistic negation rejects wording or degree (Horn, 1985), while reframing replaces an explanatory perspective. Alternatives share a recoverable contrast dimension (Rooth, 1992; Toosarvandani, 2013), and their availability difers from a participant’s endorsement (Stalnaker, 2002). Grice’s Relation and Quantity maxims motivate separate tests of contextual licensing and informative contribution; Manner motivates checking interpretability (Grice, 1975). We assess contribution against the current question under discussion (QUD), which tracks the distinctions still open in the discourse (Roberts, 2012). For instance, the bottleneck is not the CPU, but disk I/O is licensed and informative when the user reports that upgrading the CPU did not help; it is unlicensed when nothing has raised the CPU, and redundant when the answer has already ruled the CPU out.

## 3.2 An Evidence-Based Diagnostic Tree

The unit is one occurrence in its complete answer and available context. Annotators recover X/Y, reading, scope, commitment holder, and QUD, then follow Figure 2, citing evidence at each visited node.

![](images/138610e621e1530ff6b355f7f57270e6c76af7cc8904d9397126a114051b4a7e.jpg)  
Figure 2: Diagnostic tree and experimental framework. (a) Recover the contextual reading, then test defects (T/A/P) and corrective function (R/L/U). (b) Connect generation patterns to evaluator preferences, checkpoint trajectories, and preference-data composition.

Detect established defects: $\mathbf { T }  \mathbf { A }  \mathbf { P } .$ T tests for factual conflict, contradiction, or invalid exclusion under the recovered reading. A tests whether the correction misattributes a position. P tests for inappropriate rejection or disrupted interpretation, requiring a target signal, an independent defect basis, and a specific consequence, such as obstructed reference or unjustified issue replacement. Missing motivation alone is assessed at L. The first positive finding assigns D (defective).

Assess corrective function: R L U. For remaining occurrences, R asks whether X/Y express a recoverable rejection and replacement. L requires a concrete contextual cue motivating the exclusion of X, independently of the contrast itself. U asks whether the correction advances an open distinction, supplies a missing replacement, or repairs a commitment under the QUD. Restating a completed correction fails U. The first negative answer assigns O (optional), subtyped $O _ { R }$ $O _ { L } .$ , or $O _ { U } ;$ passing all three assigns S (substantive).

Annotators receive the original response without edited variants, evaluator scores, source identity, or prior labels. Factual status and confidence are recorded separately.

## 3.3 Frequency and Function Measurement

For valid answer $i , c _ { i }$ counts detected occurrences and $\ell _ { i } ^ { ( u ) }$ counts Han characters $( u = H )$ or all Unicode characters, including punctuation and whitespace $( u = C )$ . Activation, intensity, and log frequency are

$$
A = N ^ { - 1 } \sum _ { i } \mathbf { 1 } [ c _ { i } > 0 ] ,\tag{1}
$$

$$
I _ { u } = 1 0 ^ { 4 } \frac { \sum _ { i } c _ { i } } { \sum _ { i } \ell _ { i } ^ { ( u ) } } , \quad L F _ { u } = \log _ { 1 0 } ( 1 + I _ { u } ) .\tag{2}
$$

Human–model comparisons average $I _ { H }$ equally over four shared broad dimensions before computing $L F _ { H }$ . Full generation and checkpoint analyses average $A _ { b } , I _ { C }$ equally over four benchmarks; there I abbreviates $I _ { C }$ . Empty or missing answers are excluded. Within a corpus, $I \ = \ 1 0 ^ { 4 } A B / \bar { \ell } ,$ where B is occurrences per activated answer and $\bar { \ell }$ is mean answer length. This decomposition distinguishes activation from repetition and exposure. A fixed literal detector identifies candidates (Appendix A); functional proportions describe selected construction-bearing responses.

## 3.4 Experimental Framework

Generation analysis measures frequency and applies the tree. Evaluator experiments compare four variants: A retains not X, but Y; B presents Y, not X; C separates Y and the rejection across sentences; D retains only Y. A–B probes order, B–C sentence realization, and A–D deletes the rejected alternative and its corrective content.

For reward model m, $\begin{array} { r c l } { { d _ { i m } ^ { A B } } } & { { = } } & { { ( R _ { i m A } ~ - } } \end{array}$ $R _ { i m B } ) / s _ { m A }$ standardizes the contrast by originalarm reward variation. We also report A wins conditional on non-ties. Each pairwise judge evaluates two independent presentation orders. Restoredarm verdicts map to 1, .75, .5, .25, 0 in favor of A; their mean $p _ { i }$ is an ordinal preference score. Functional sensitivity, $F S _ { G } = \bar { p } _ { S } - \bar { p } _ { G }$ , compares substantive uses with unlicensed uses $( G = O _ { L } )$ or with all optional uses $( G = O ) { \mathrm { ; } }$ ; for reward models, p is replaced by the standardized contrast d. A positive value means that the evaluator favors A more when the correction is needed. All intervals for FS are 95% item-bootstrap intervals.

Quality matching uses general-quality scores from HY3 (Tencent Hy Team, 2026), retaining unchanged A–B/A–D texts scored equally on every applicable dimension except construction clarity. Checkpoint comparisons follow documented parent–child relations; preference-data audits compare context-matched chosen/rejected answers and examine length, language composition, and repetition.

## 4 Experiments

Our experiments address two questions: whether models use the construction where it is called for, and where their preference for it comes from. We answer the first by comparing model and human usage in frequency and function, and the second by examining evaluator preferences, training stages, and public preference data.

## 4.1 Frequency and Function

Comparison. We examine 5,285 answers from five API models—GPT, DeepSeek, Gemini, GLM, and Qwen—to 1,061 Chinese prompts drawn from AlignBench (Liu et al., 2024), WritingBench (Wu et al., 2025), COIG-CQIA (Bai et al., 2024), and Arena-Hard-v2 (LMArena, 2026). A human reference comprises community answers, examination responses, student essays, and role-play dialogue. We compare construction density per 10,000 Han characters, averaging equally over four shared task dimensions: professional ability, general question answering, Chinese understanding, and writing.

(a) Frequency across four shared tasks  
![](images/ef8355939966ea8a00b8e67fb7ebea0e942d411828bbb389cd8f8e76d5026fba.jpg)

(b) Optional uses by task and source
<table><tr><td rowspan="2"></td><td rowspan="2">Human</td><td rowspan="2">GLM</td><td rowspan="2">Gemini</td><td rowspan="2">GPT</td><td rowspan="2">Qwen</td><td colspan="2">DeepSeek All LLMs</td></tr><tr><td>46.7%</td><td></td></tr><tr><td>Professional</td><td>50.0%</td><td>66.7%</td><td>43.5%</td><td>54.5%</td><td>70.4%</td><td></td><td>55.3%</td></tr><tr><td>General QA</td><td>30.8%</td><td>85.7%</td><td>10.0%</td><td>56.3%</td><td>0%</td><td>37.5%</td><td>43.2%</td></tr><tr><td>Chinese</td><td>33.3%</td><td>100%</td><td>50.0% 47.1%</td><td>66.7% 37.5%</td><td>100%</td><td>50.0%</td><td>73.3%</td></tr><tr><td>Writing</td><td>33.3%</td><td>57.1%</td><td></td><td></td><td>100%</td><td>60.0%</td><td>60.7%</td></tr><tr><td>Basic Mathematics</td><td>NA NA</td><td>50.0% 0%</td><td>0% NA</td><td>0% 0%</td><td>NA NA</td><td>66.7%</td><td>42.9%</td></tr><tr><td>Role play</td><td>NA</td><td></td><td>27.3%</td><td>62.5%</td><td></td><td>0%</td><td>0%</td></tr><tr><td></td><td>NA</td><td>53.8%</td><td></td><td></td><td>36.4% 0%</td><td>66.7%</td><td>46.9%</td></tr><tr><td>Reasoning</td><td></td><td>40.0%</td><td>66.7%</td><td>100%</td><td></td><td>100%</td><td>42.9%</td></tr><tr><td>All tasks</td><td>33.3%</td><td>61.4%</td><td>37.3%</td><td>53.3%</td><td>63.3%</td><td>50.8%</td><td>52.8%</td></tr></table>

Optional share (%)

(c) Functional composition by source  
![](images/e915a2360c202945ea4c062e91abc10da7b96424bac41e7b3c5c6ac16f85a361.jpg)  
Figure 3: Frequency and corrective function. (a) Occurrences per 1,000 punctuation-delimited sentences, per 10,000 Han characters $\left( I _ { H } \right)$ , and $\log _ { 1 0 } ( 1 + I _ { H } )$ Rates average equally over four shared tasks; the log is taken after averaging. Human sentence counts are unavailable (NA). (b) Optional-use percentages by task and source; NA denotes no classified cases. All LLMs pools model cases. (c) Functional shares: substantive (S), unlicensed $( O _ { L } )$ , redundant $( O _ { U } )$ , relation failure $( O _ { R } ) _ { : \ l }$ , and defective (D). Panels (b,c) include 307 selected model occurrences across four benchmarks and 33 eligible human-authored occurrences annotated with the same tree. Their denominators include every classified case; these descriptive samples difer from the fullanswer frequency panel in (a).

Tables 2 and 3 summarize the human corpus and model generation panel; Appendix A provides the detection details.

Models repeatedly select the construction. Across the five models, density is 2.5–9.2 times the human reference (Figure 3a). The human reference answers diferent prompts, and its detector also admits comma-free and cross-sentence forms that the model detector excludes; this detection diference, if anything, understates the gap. The construction is thus a recurrent choice across models, rather than a peculiarity of one system. Its local contribution is a separate question: frequent corrections may resolve an outstanding alternative, but they may also introduce a contrast the answer does not need.

Many corrections lack a substantive corrective function. Of 308 selected occurrences, one cybersecurity-related item triggered a refusal from the Claude annotation service and yielded no valid label, leaving 307 occurrences for functional analysis in their full context. Labels were generated with claude-sonnet-5-5 and subsequently audited. Optional uses account for 52.8% of the 307 annotated cases (Figure 3b,c; Table 4). Contextual licensing or issue advancement accounts for 96.4% of non-substantive decisions. The recurring problem is thus a correction introduced without a contextual reason, or repeated after its work has already been done. Only four occurrences are classified as defective.

Human-authored uses are more often substantive. We apply the same tree to the human reference. Of its 38 detected occurrences, five are quoted or metalinguistic and are excluded before tree application. Among the remaining 33, 63.6% are substantive and 33.3% optional, compared with 45.9% and 52.8% in model answers (Figure 3c). The diference is concentrated in unlicensed uses (12.1% vs. 24.1%); redundant uses difer less (21.2% vs. 28.0%). Models therefore not only use the construction more densely, but a larger share of their uses introduces an alternative that the context gives no reason to reject (Appendix A.1).

## 4.2 Evaluator Preference

Two comparisons test evaluative preference. For each selected answer, we compare the original not X, but Y with a reordered version, Y, not X, and an afirmative-only version retaining Y. Reordering preserves the rejected alternative; afirmativeonly editing removes it. Five reward models score full question–answer pairs. Claude Sonnet 5.5 and GPT-6.1-sol apply the same pairwise rubric in both presentation orders, yielding mean ordinal preference scores from 0 to 1, with .5 indicating a tie. Three of GPT’s 1,204 planned judgments were blocked by the provider’s content filter on one item. General-quality scores provide a further comparison among variants receiving identical scores on every applicable dimension (Appendix B).

![](images/ad818ae55348773c0f5403a3f497ce8dead98a8e7a8caf6fb4f7c2ab353b558e.jpg)

(c) Judge preference  
![](images/363dc1f29ebe44ff02d75748a1f4107da9eb8d198c6750a4855fa5d8f6016fc3.jpg)

(d) Judge sensitivity  
![](images/bce98d8e7efb5bee8bd2ad9d3d542cd49cee5386a4b5175931cb357d7cc2006f.jpg)  
Figure 4: Evaluator preference and functional sensitivity. A is the original, B reorders the correction, and D retains only Y. (a) RM non-tie A-win rates for optional uses. (b) Substantive-minus-optional reward contrasts, standardized by A-arm score SD. (c) Mean ordinal preference by function, separately for Claude Sonnet 5.5 and GPT-6.1-sol. (d) Their $F S _ { G } = \bar { p } _ { S } - \bar { p } _ { G }$ with O pooling $O _ { L }$ and $O _ { U }$ . Dashed lines mark neutrality (a,c) or zero sensitivity (b,d). Panels (c,d) use each judge’s complete pairs and shared scales across judges. Error bars are pointwise 95% Wilson intervals (a) or 10,000 item-bootstrap intervals (b–d).

Reward preference extends to optional corrections. FsfairX (Dong et al., 2024), Skywork (Liu et al., 2025), and OfsetBias (Park et al., 2024) prefer the original order in 68.3–76.8% of non-tied optional-use comparisons (Figure 4a). ArmoRM (Wang et al., 2024) shows a weaker preference, whereas InternLM2 (Cai et al., 2024) favors the reordered version. The result identifies a recurring preference in three evaluators: they reward the original realization even where the tree classifies the correction as optional.

Judges favor retaining unlicensed corrections. For corrections without contextual licensing, Claude Sonnet 5.5 prefers the original over the afirmative-only answer (mean score 0.630) and over reordering (0.563; Figure 4c; Table 7). GPT-6.1-sol shows weaker but directionally consistent preferences (0.519 and 0.517, respectively). Critically, neither judge’s preference varies with corrective function: substantive-minus-unlicensed diferences are 0.002 and 0.005 for Claude, and 0.002 and 0.001 for GPT; all four 95% intervals include zero (Figure 4d). This insensitivity indicates that both judges reward the construction’s surface form regardless of whether the context licenses it.

Ordering preference persists under qualityscore matching. To examine whether reward models’preference for the original order is associated with general answer quality, we retain original–reordered pairs with identical scores on every applicable quality dimension. Excluding rewardmodel ties, FsfairX, Skywork, and OfsetBias still prefer “not X, but Y”over “Y, not X”in 73.1– 78.3% of comparisons (Table 6). Thus, these three reward models exhibit an ordering preference even when the quality scores do not distinguish the two versions.

## 4.3 Training Stages and Data Attribution

We examine SFT and DPO through 12 approximately 7B–8B checkpoints from DeepSeekMath (Shao et al., 2024), Llama-3.1/Tülu-3 (Grattafiori et al., 2024; Lambert et al., 2024), OLMo-2 (OLMo et al., 2025), MiMo (Xiaomi LLM-Core Team, 2025), and InternLM2 (Cai et al., 2024). All five families provide Base and SFT checkpoints; Tülu and OLMo also provide DPO checkpoints. The panel yields 12,518 valid answers to the same 1,061 prompts used for the API models. Appendix C provides the generation protocol.

Open checkpoints use the construction less frequently than API models. Across the four shared task dimensions, open-checkpoint density ranges from 0 to 1.41 occurrences per 10,000 Han characters, compared with 0.942 for the human reference and 2.38–8.68 for the API models (Figure 5a). Ten of the twelve open checkpoints fall below 0.500. The high-frequency pattern is therefore concentrated in the API cohort rather than shared by the selected open checkpoints.

Density rises across the displayed SFT and DPO stages. Under the four-benchmark Unicode measure, all five families have higher construction density at SFT than at Base. Both available DPO checkpoints also exceed their SFT predecessors, reaching the highest density within the displayed trajectories (Figure 5b). Paired density changes on common successful prompts are also positive for all seven transitions (Table 8). These descriptive patterns associate SFT and DPO with increased use of the construction in this checkpoint panel.

Preferred training answers more often contain the construction. In public preference data, chosen answers activate the construction more often than rejected answers in both datasets: +14.7% relative activation in Tülu 3 and +26.5% in OLMo 2 (Table 9). The direction is consistent across datasets and the gap is larger in OLMo 2. Both Han-content subsets in each corpus also show higher chosen-answer density. This enrichment is directionally consistent with the displayed DPO increases. Its magnitude depends on corpus composition and repetition.

<table><tr><td>Target 此刻我脑海里闪过的，</td><td>Swimmer&#x27;s victory speech</td><td>Commentary on Messi&#x27;s title</td></tr><tr><td></td><td>不是终点触壁的那一 瞬间，而是过去这几 年的每一个清晨…… ‘not the moment I touched the wall, but every morning of these past years&#x27;</td><td>梅西的夺冠，不是因 为他终于赢了，而是 因为他用二十多年…… 换来了这一次最完整 的救赎。‘Messi&#x27;s title is not because he finally won, but because he traded twenty-odd years ... for this most complete redemption&#x27;</td></tr><tr><td>Tree</td><td>S: winning makes the finish accessible; injury motivates rejecting it.</td><td>OL at L: no one reduces the title to winning; the target introduces X itself.</td></tr><tr><td>Scores</td><td>RMs 3/3, 3/3; judge 0.75, 0.75</td><td>RMs 2/3, 1/3; judge 0.75, 0.625</td></tr><tr><td>Reason</td><td>‘rejects first, then asserts; more natural, more emotional impact&#x27;</td><td>&#x27;slightly odd logically but reads smoothly and passionately&#x27;</td></tr></table>

Table 1: Same evaluation, diferent function. Scores list A–B, then A–D. RMs counts how many of FsfairX, Skywork, and OfsetBias score the original higher; judge scores average two presentation orders (0.5 is a tie). Reasons are quoted from the judge’s A–B output.

![](images/0d2c8ecf22fb97deebe08d454781be9235b5a02bbf1ee630d77657349e5ed06c.jpg)  
Figure 5: Cohort gap and SFT/DPO trajectories. (a) Twelve open checkpoints from five families, the human reference, and five API models on four shared task dimensions. Points show $L F _ { H }$ , with $I _ { H }$ on the upper axis; open points are ofset vertically for visibility. (b) Descriptive Base–SFT–DPO trajectories using four-benchmark macro $L F _ { C }$ . Points use valid answers at each checkpoint; lines connect available stages. Nonempty truncated answers are retained.

## 4.4 Case Study

Table 1 presents two sports answers difering mainly in whether context supplies X. In a victory speech by a swimmer long troubled by injury, the finish is what a new champion would naturally recall, and the injury motivates setting it aside. In commentary on Messi’s World Cup title, no one suggests that the title amounts merely to winning; the target raises this view only to reject it, although winning the final is precisely what the title is.

Yet the LLM judge favors the original in every judgment for both answers except one tie. For Messi, it concedes that the sentence is “slightly odd logically” and still prefers it because it “reads smoothly and passionately.” FsfairX and Skywork also score the original above the reordered version in both cases. The evaluators reward the construction’s rhetorical efect, not whether the context raises the rejected alternative.

## 5 Discussion and Conclusion

Frequency and appropriateness are separate properties of model usage. Models produce the construction far more often than people do, and many of these uses are correct but not called for. Evaluators do not supply the missing judgment: several reward models prefer the construction after general-quality matching, and both judges prefer it within unlicensed uses. In the available checkpoints, its density also rises through SFT and DPO. Suppressing the form would remove its appropriate uses along with the rest; reducing tics of this kind requires evaluation and training signals that register whether an expression is called for. The diagnostic is not specific to Mandarin: screening for defects and then testing contextual licensing and issue advancement can be applied to English it’s not X, it’s Y and to other expressions that are legitimate in some contexts and overused in others.

A controlled follow-up can test this attribution by holding SFT initialization, source composition, update budget, and prompts fixed, intervening on construction-bearing preference pairs with matched edits, and repeating training across seeds. Measuring activation, repetition, and local function would then separate changes in how often the construction is selected from changes in what it contributes.

## Limitations

Data coverage. Our analysis covers five commercial models, a limited set of task domains, and 307 annotated uses. Broader models, domains, and human data would strengthen the estimates.

Observational training evidence. The checkpoint and preference-data comparisons are observational, so they show that frequency rises along training pipelines but do not establish which stage or data causes the rise. Controlled retraining is needed to test this attribution.

Reader perception. We assess appropriateness through linguistic criteria and do not measure how readers perceive the construction. Whether unneeded uses cause annoyance or afect comprehension requires a separate user study.

Languages and constructions. We study a single Mandarin construction. Although the decision tree is designed to transfer to English it’s not X, it’s Y and other recurring expressions, we have not applied it beyond bushi X, ershi Y.

## Ethical Considerations

This section discusses the ethical considerations and broader impact of this work.

Potential Risks. The decision tree describes when a corrective construction is called for, and this description could be used to make modelwritten text resemble human writing more closely and thus harder to identify. Conversely, the frequency measures could be misread as evidence of machine authorship and applied to judge human writers, who also use the construction. Our analyses concern aggregate usage across models and corpora; they are not designed or validated for attributing individual texts to humans or machines.

Intellectual Property. The benchmarks, openweight checkpoints, reward models, and preference mixtures used in this work are publicly released for research, and we use them in accordance with their respective licenses. Commercial models are accessed through their oficial APIs under the providers’ terms of service. All materials are used solely for scientific research.

Data Privacy. The human reference is drawn from publicly available datasets, and personally identifiable information was removed before analysis. Model answers are generated from public benchmark prompts. The human annotators are students with a background in linguistics who participated voluntarily; no personal information about them is collected beyond their annotations.

Intended Use. The diagnostic framework is intended for analyzing contextual appropriateness in model-generated text and for examining how evaluators and training stages shape such usage. It is not intended for authorship detection or for evaluating individual writers.

Documentation of Artifacts. We will release our code and the annotation guidelines for the decision tree. Both are documented with respect to their scope: the Mandarin construction bushi X, ershi Y, the task domains covered, and the procedures described in Appendices A–D.

AI Assistants in Research or Writing. We use Cursor for code development assistance and a large language model for refining and polishing the language of the paper. The use of language models as annotators and judges is part of the method and is described in Section 3 and Appendices A.1 and B.

## References

Jacob Andreas and Dan Klein. 2016. Reasoning about pragmatics with neural listeners and speakers. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1173–1182, Austin, Texas. Association for Computational Linguistics.

Yuelin Bai, Xinrun Du, Yiming Liang, Yonggang Jin, Junting Zhou, Ziqiang Liu, Feiteng Fang, Mingshan Chang, Tianyu Zheng, Xincheng Zhang, Nuo Ma, Zekun Wang, Ruibin Yuan, Haihong Wu, Hongquan Lin, Wenhao Huang, Jiajun Zhang, Chenghua Lin, Jie Fu, and 3 others. 2024. Coig-cqia: Quality is all you need for chinese instruction fine-tuning. Preprint, arXiv:2403.18058.

Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, et al. 2024. InternLM2 technical report. Preprint, arXiv:2403.17297.

Robert Dale and Ehud Reiter. 1995. Computational interpretations of the Gricean maxims in the generation of referring expressions. Cognitive Science, 19(2):233–263.

Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. 2024. RLHF workflow: From reward modeling to online RLHF. Preprint, arXiv:2405.07863.

Michael C. Frank and Noah D. Goodman. 2012. Predicting pragmatic reasoning in language games. Science, 336(6084):998.

Adele E. Goldberg. 2006. Constructions at Work: The Nature ofGeneralization in Language. Oxford University Press.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, et al. 2024. The Llama 3 herd of models. Preprint, arXiv:2407.21783.

H. Paul Grice. 1975. Logic and conversation. In Peter Cole and Jerry L. Morgan, editors, Syntax and Semantics, Volume 3: Speech Acts, pages 41–58. Academic Press, New York.

Laurence R. Horn. 1985. Metalinguistic negation and pragmatic ambiguity. Language, 61(1):121–174.

Jennifer Hu, Sammy Floyd, Olessia Jouravlev, Evelina Fedorenko, and Edward Gibson. 2023. A finegrained comparison of pragmatic language understanding in humans and language models. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4194–4213, Toronto, Canada. Association for Computational Linguistics.

Dell Hymes. 1972. On communicative competence. In J. B. Pride and Janet Holmes, editors, Sociolinguistics: Selected Readings, pages 269–293. Penguin, Harmondsworth.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. SWE-bench: Can language models resolve real-world GitHub issues? In The Twelfth International Conference on Learning Representations.

Tom S. Juzek and Zina B. Ward. 2025. Why does ChatGPT “delve” so much? exploring the sources of lexical overrepresentation in large language models. In Proceedings of the 31st International Conference on Computational Linguistics, pages 6397– 6411, Abu Dhabi, UAE. Association for Computational Linguistics.

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. 2024. Understanding the effects of RLHF on LLM generalisation and diversity. In The Twelfth International Conference on Learning Representations.

Dmitry Kobak, Rita González-Márquez, Emőke-Ágnes Horvát, and Jan Lause. 2025. Delving into LLM-assisted writing in biomedical publications through excess vocabulary. Science Advances, 11(27):eadt3813.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Eficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, and 4 others. 2024. Tulu 3: Pushing frontiers in open language model post-training. Preprint, arXiv:2411.15124.

J. Richard Landis and Gary G. Koch. 1977. The measurement of observer agreement for categorical data. Biometrics, 33(1):159–174.

Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang, Jiacheng Xu, Yang Liu, and Yahui Zhou. 2025. Skywork-Reward-V2: Scaling preference data curation via human-AI synergy. Preprint, arXiv:2507.01352.

Xiao Liu, Xuanyu Lei, Shengyuan Wang, Yue Huang, Andrew Feng, Bosi Wen, Jiale Cheng, Pei Ke, Yifan Xu, Weng Lam Tam, Xiaohan Zhang, Lichao Sun, Xiaotao Gu, Hongning Wang, Jing Zhang, Minlie Huang, Yuxiao Dong, and Jie Tang. 2024. AlignBench: Benchmarking Chinese alignment of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11621–11640, Bangkok, Thailand. Association for Computational Linguistics.

LMArena. 2026. Arena-Hard-Auto: Evaluation data and judge prompts. Repository accessed October 2026. Pairwise judge prompt frozen at commit 196f6b826783b3da7310e361a805fa36f0be83f3.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, and 24 others. 2025. 2 olmo 2 furious. Preprint, arXiv:2501.00656.

Will Oremus. 2026. The most famous AI writing tic is also the most mysterious. The Atlantic. Published July 12, 2026.

Samuel Paech, Allen Roush, Judah Goldfeder, and Ravid Shwartz-Ziv. 2025. Antislop: A comprehensive framework for identifying and eliminating repetitive patterns in language models. Preprint, arXiv:2510.15061.

Junsoo Park, Seungyeon Jwa, Ren Meiying, Daeyoung Kim, and Sanghyuk Choi. 2024. OfsetBias: Leveraging debiased data for tuning evaluators. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 1043–1067, Miami, Florida, USA. Association for Computational Linguistics.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, and 2 others. 2019. PyTorch: An imperative style, highperformance deep learning library. In Advances in Neural Information Processing Systems, volume 32.

Craige Roberts. 2012. Information structure: Towards an integrated formal theory of pragmatics. Semantics and Pragmatics, 5(6):1–69.

Mats Rooth. 1992. A theory of focus interpretation. Natural Language Semantics, 1:75–116.

Laura Ruis, Akbir Khan, Stella Biderman, Sara Hooker, Tim Rocktäschel, and Edward Grefenstette. 2023. The goldilocks of pragmatic understanding: Finetuning strategy matters for implicature resolution by LLMs. In Advances in Neural Information Processing Systems, volume 36.

Chantal Shaib, Tuhin Chakrabarty, Diego Garcia-Olano, and Byron C. Wallace. 2025. Measuring AI “slop” in text. Preprint, arXiv:2509.19163.

Chantal Shaib, Yanai Elazar, Junyi Jessy Li, and Byron C. Wallace. 2024. Detection and measurement of syntactic templates in generated text. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 6416– 6431, Miami, Florida, USA. Association for Computational Linguistics.

Jingmin Shao and Yiguang Wang. 2010. The functional value of pseudo-denial in the pattern “bushi A, ershi B”. Chinese Teaching in the World, 24(3):325–333. In Chinese.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Judith Sieker and Sina Zarrieß. 2026. How hypocritical is your LLM judge? listener–speaker asymmetries in the pragmatic competence of large language models. Preprint, arXiv:2604.15873.

Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. 2024. A long way to go: Investigating length correlations in RLHF. In Proceedings of the First Conference on Language Modeling.

Settaluri Sravanthi, Meet Doshi, Pavan Tankala, Rudra Murthy, Raj Dabre, and Pushpak Bhattacharyya. 2024. PUB: A pragmatics understanding benchmark for assessing LLMs’ pragmatics capabilities. In Findings of the Association for Computational Linguistics: ACL 2024, pages 12075–12097, Bangkok, Thailand. Association for Computational Linguistics.

Robert Stalnaker. 2002. Common ground. Linguistics and Philosophy, 25:701–721.

Fangfang Tan. 2016. A multidimensional comparison of “bushi A ershi B” and “not A but B”. Foreign Language Education, 37(1):43–48. In Chinese.

Tencent Hy Team. 2026. Hy3. Hugging Face model card. Accessed October 2026.

Maziar Toosarvandani. 2013. Corrective but coordinates clauses not always but sometimes. Natural Language & Linguistic Theory, 31:827–863.

Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. 2024. Interpretable preferences via multi-objective reward modeling and mixtureof-experts. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 10582–10592, Miami, Florida, USA. Association for Computational Linguistics.

Leonie Weissweiler, Valentin Hofmann, Abdullatif Köksal, and Hinrich Schütze. 2022. The better your syntax, the better your semantics? probing pretrained language models for the English comparative correlative. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10859–10882, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Minghao Wu and Alham Fikri Aji. 2025. Style over substance: Evaluation biases for large language models. In Proceedings of the 31st International Conference on Computational Linguistics, pages 297– 312, Abu Dhabi, UAE. Association for Computational Linguistics.

Yuning Wu, Jiahao Mei, Ming Yan, Chenliang Li, Shaopeng Lai, Yuran Ren, Zijia Wang, Ji Zhang, Mengyue Wu, Qin Jin, and Fei Huang. 2025. Writingbench: A comprehensive benchmark for generative writing. Preprint, arXiv:2503.05244.

Xiaomi LLM-Core Team. 2025. MiMo: Unlocking the reasoning potential of language model – from pretraining to posttraining. Preprint, arXiv:2505.07608.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. Preprint, arXiv:2406.12045.

Xuanchang Zhang, Wei Xiong, Lichang Chen, Tianyi Zhou, Heng Huang, and Tong Zhang. 2025. From lists to emojis: How format bias afects model alignment. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 26940–26961, Vienna, Austria. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging

LLM-as-a-judge with MT-bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36.

Baiyao Zuo. 2025. A pragmatics-cognitive approach to the opposition relations lexicalized in Chinese. Journal ofLinguistics, 61(2):439–462.

## A Measurement and Annotation

Human-reference sources. The human baseline draws on cMedQA2 for medical community question–answer pairs; HC3 for Chinese human answers on finance, law, psychology, and everyday topics; and COIG for Chinese-language examination answers and explanations. The Chinese Essay Dataset supplies student essays in six genres, and CrossWOZ supplies travel-assistant dialogues role-played by human participants. We also use worked solutions for the 2016–2018 management entrance examinations hosted by China University of Petroleum (East China), solutions for the 2020 examination hosted by Shanghai Normal University, and reference answers for the 2020 independent-admission mathematics examination hosted by the High School Afiliated to Anhui Normal University.

Corpora and generation. The human reference contains 572,412 Han characters and 38 occurrences across eight dimensions. The main comparison gives equal weight to the four dimensions represented by these occurrences: professional ability, general question answering, Chinese understanding, and writing. Han exposure counts code points in U+4E00--U+9FFF. The four-dimension mean is .942 occurrences per 10,000 Han characters; pooling all eight human dimensions gives .664. Human materials allow comma-free and cross-sentence realizations, whereas the model detector requires a comma.

The model panel contains 683 Align-Bench, 100 WritingBench, 200 COIG-CQIA, and 78 Arena-Hard-v2 prompts. The five logged model identifiers are gpt-5.6-luna, deepseek-v4.1-flash, gemini-3.8-flash, glm-5.3-flash, and qwen3.8-flash. Each prompt is a single user message; endpoints retain default reasoning behavior and receive scenedependent temperatures when supported. Of 5,305 expected answers, 5,285 are valid.

Detection and frequency measures. The fixed detector finds non-overlapping literal matches, permits an optional preceding bing, requires a Chinese or ASCII comma before ershi, and limits the captured alternatives to 160 and 220 characters. Slots exclude sentence-final punctuation, semicolons, and newlines; intervening whitespace may span a newline. Nested starts and comma-free forms can be missed, and interrogative strings can produce spurious detections.

For valid answer i, let $c _ { i }$ count occurrences and $\ell _ { i } ^ { ( u ) }$ count Han or all Unicode characters. Activation and density are

$$
A = N ^ { - 1 } \sum _ { i } \mathbf { 1 } [ c _ { i } > 0 ] , \qquad I _ { u } = 1 0 ^ { 4 } \frac { \sum _ { i } c _ { i } } { \sum _ { i } \ell _ { i } ^ { ( u ) } } .
$$

For Figure 3a, sentence density is $\textstyle { 1 0 ^ { 3 } \sum _ { i } c _ { i } } { \big / } \sum _ { i } s _ { i }$ , where $s _ { i }$ counts punctuationdelimited units. We strip line-initial numbered-list prefixes, split at Chinese full stops and Chinese/ASCII question and exclamation marks, and split at ASCII periods unless flanked by digits. Segments must contain a Unicode letter; a final unterminated segment is retained. Newlines and semicolons do not split units. Sentence and Han densities are computed within each of the four shared tasks and then averaged equally. Human sentence counts are unavailable. The log-frequency panel transforms the macro Han density after averaging.

The archived log scale is $L F _ { u } = \log _ { 1 0 } ( 1 + I _ { u } )$ Human comparisons average Han density over four task dimensions. Benchmark analyses average activation and Unicode density equally over four benchmarks. Within a corpus, density decomposes into activation, occurrences per activated answer, and mean answer length.

Full-panel uncertainty uses 5,000 promptcluster bootstraps stratified by benchmark. Between-model tests use common successful prompts and within-prompt exchanges, with Holm correction over ten activation and ten density contrasts separately.

## A.1 Functional Annotation and Human Comparison

The sample comprises 308 selected response groups. One cybersecurity-related item triggered a refusal from the Claude annotation service and yielded no valid label; the functional analysis therefore covers 307 annotated targets. Labels were generated with claude-sonnet-5-5 and subsequently audited. Annotation excludes templates, purely metalinguistic quoted strings, and known incomplete materials. Annotators assess the original response without edited variants or evaluator scores, recover the supported reading and question under discussion, and record factual uncertainty separately. An implicit question under discussion requires contextual evidence independent of the target contrast.

<table><tr><td>Task dimension</td><td>Han chars.</td><td>Count</td><td> $I _ { H }$ </td><td> $L F _ { H }$ </td></tr><tr><td colspan="5">Four shared task dimensions (main comparison)</td></tr><tr><td>Professional tasks</td><td>98,450</td><td>2</td><td>0.203</td><td>0.080</td></tr><tr><td>General question answering</td><td>100,020</td><td>13</td><td>1.30</td><td>0.362</td></tr><tr><td>Chinese understanding</td><td>100,146</td><td>11</td><td>1.10</td><td>0.322</td></tr><tr><td>Writing</td><td>102,823</td><td>12</td><td>1.17</td><td>0.336</td></tr><tr><td>Macro average of four shared dimensions</td><td></td><td></td><td>0.942</td><td></td></tr><tr><td colspan="5">Other human dimensions</td></tr><tr><td>Basic tasks</td><td>44,867</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Logical reasoning</td><td>17,683</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Mathematical calculation</td><td>8,249</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Role play</td><td>100,174</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Corpus pooled (all 8 dimensions)</td><td>572,412</td><td>38</td><td>0.664</td><td>0.221</td></tr></table>

Table 2: Human corpus reference. $I _ { H }$ counts occurrences per 10,000 Han characters; $L F _ { H } = \log _ { 1 0 } ( 1 + I _ { H } )$ . The main comparison in the paper uses the macro average of the four shared dimensions $( I _ { H } = 0 . 9 4 2 )$ . The corpuspooled estimate over all eight dimensions is $I _ { H } = 0 . 6 6 4$ . Dimensions with zero counts are shown for completeness; they are not used to compute the main comparison baseline.

<table><tr><td rowspan="2">Model</td><td colspan="5">Four-benchmark generation panel</td><td colspan="2">Four shared task dimensions</td></tr><tr><td>N</td><td>A (%)</td><td>I</td><td>95% CI for I</td><td>LF</td><td> $I _ { H }$ </td><td>Model/Human</td></tr><tr><td>GPT</td><td>1061</td><td>31.6</td><td>2.54</td><td>[2.15, 2.94]</td><td>0.549</td><td>4.59</td><td>4.87×</td></tr><tr><td>DeepSeek</td><td>1058</td><td>23.5</td><td>2.97</td><td>[2.49, 3.51]</td><td>0.599</td><td>8.68</td><td>9.21×</td></tr><tr><td>Gemini</td><td>1061</td><td>17.0</td><td>1.20</td><td>[0.987, 1.41]</td><td>0.343</td><td>3.41</td><td>3.62×</td></tr><tr><td>GLM</td><td>1046</td><td>11.8</td><td>1.10</td><td>[0.840, 1.38]</td><td>0.321</td><td>2.38</td><td>2.52×</td></tr><tr><td>Qwen</td><td>1059</td><td>34.2</td><td>3.13</td><td>[2.60, 3.75]</td><td>0.616</td><td>6.75</td><td>7.17×</td></tr><tr><td>Human</td><td></td><td></td><td></td><td></td><td></td><td>0.942</td><td>1.00×</td></tr></table>

Table 3: Five-model generation panel. N is the number of valid responses. Within each benchmark, A is the percentage of valid answers containing at least one occurrence, and I is occurrences per 10,000 Unicode characters; the reported A and I average equally over four benchmarks. $L F = \log _ { 1 0 } ( 1 + I )$ . The two rightmost columns use four shared task dimensions: $I _ { H }$ is macro-average density per 10,000 Han characters, and Model/Human divides model $I _ { H }$ by the unrounded human baseline (0.942 when rounded; Table 2). These ratios correspond to Figure 3a. Model names abbreviate the logged identifiers in the text.

<table><tr><td>Outcome / nested subtype</td><td>Count</td><td>Share (%)</td></tr><tr><td>1. Defective (D) First positive check:  $T  A  P$ </td><td>4</td><td>1.30</td></tr><tr><td>T: Truth conflict</td><td>2</td><td>0.651</td></tr><tr><td>A: Misattribution</td><td>2</td><td>0.651</td></tr><tr><td>P: Pragmatic defect</td><td>0</td><td>0</td></tr><tr><td>2. Optional (O) If no defect, first failure:</td><td>162  $R \to L \to U$ </td><td>52.8</td></tr><tr><td>R: Relation failure</td><td>2</td><td>0.651</td></tr><tr><td>L: No contextual licensing</td><td>74</td><td>24.1</td></tr><tr><td>U: No issue advancement</td><td>86</td><td>28.0</td></tr><tr><td>3. Substantive (S) No defect; all  $R \dot { / } L / U$ </td><td>141</td><td>45.9</td></tr><tr><td></td><td>checks passed</td><td></td></tr><tr><td>Total</td><td>307</td><td>100</td></tr></table>

Table 4: Functional composition of the 307 annotated cases. S/O/D are top-level outcomes; R/L/U (optional) and T/A/P (defective) are mutually exclusive subcategories assigned in priority order $( \mathbf { R }  \mathbf { L }  \mathbf { U } ; \mathbf { T } $ $\mathbf { A }  \mathbf { P } )$ . One cybersecurity-related item was excluded due to an annotation refusal. Labels were generated by an LLM and reviewed by humans; inter-rater agreement is reported in Table 5. This table describes the current sample $( N = 3 0 7 )$ and should not be interpreted as a precise estimate of population proportions.

The annotated sample contains 141 substantive, 162 optional, and four defective uses (Table 4). Optional uses comprise 52.8% of all 307 annotated cases and 53.5% of the 303 non-defective cases. Figure 3b,c includes all classified cases. Of the 166 non-substantive uses, 160 (96.4%) terminate at contextual licensing or issue advancement.

Agreement with the human annotations. We compare annotations from two independent human annotators, both with a background in linguistics, on the same 100 cases (Table 5). We report exact agreement and unweighted Cohen’s κ for the top-level outcomes (S/O/D) and the seven terminal labels (S, R, L, U, T, A, P). Terminal-label agreement requires the full classification, including the subtype when applicable, to match. Both annotators labeled the cases independently without consensus discussion. The κ values indicate moderate agreement (Landis and Koch, 1977). The expected drop at the terminal level reflects the inherent dificulty of distinguishing fine-grained pragmatic subcategories (e.g., O<sub>L</sub> vs. O<sub>U</sub>), even for trained linguists.

Human-authored functional baseline. The human corpus contains 38 detected occurrences. After excluding five quoted or metalinguistic instances before tree application, Codex classifies 33 eligible occurrences under the same protocol: 21 substantive, four unlicensed, seven redundant, and one defective. These labels did not undergo the review applied to the model sample. Section 4.1 compares them with the model sample. They are automated annotations of human-authored text and are distinct from the human raters described above.

<table><tr><td>Label level</td><td></td><td>N Matches</td><td>s Agreement</td><td>κ</td></tr><tr><td>Top-level  $( S / O / D )$ </td><td>100</td><td>77</td><td>77.0%</td><td>0.577</td></tr><tr><td>Terminal (7 iabels)</td><td>100</td><td>66</td><td>66.0%</td><td>0.521</td></tr></table>

Table 5: Inter-annotator agreement on 100 cases. Two independent annotators with linguistics backgrounds labeled all cases. Exact Matches (n) and Exact Agreement (%) report raw agreement. κ is unweighted Cohen’s κ. The terminal level includes the seven labels (S, R, L, U, T, A, P). The moderate κ values reflect the inherent dificulty of fine-grained pragmatic classification, particularly distinguishing lack of contextual licensing $( O _ { L } )$ from no discourse advancement $( O _ { U } )$

Discourse position. Target position is its first start index divided by answer length. Median positions are 0.341 for substantive corrections, 0.544 for unlicensed corrections, and 0.862 for redundant corrections. The substantive–unlicensed difference has $p < . 0 0 1$ . Summary cues occur before 7.80%, 14.9%, and 32.6% of these targets. Both optional subtypes occur later than substantive corrections. These associations are descriptive and do not enter the evaluator tests.

## B Evaluator Design and Supplementary Results

Variants and reward models. Each selected response yields four variants. For compatibility with the released data, A denotes the original, B the reordered version, C a cross-sentence version, and D the afirmative-only version. The main text uses the first, second, and fourth. The cross-sentence variant is not used in the reported comparisons.

ArmoRM-8B, FsfairX-8B, InternLM2-7B-Reward, OfsetBias-8B, and Skywork-Reward-V2-Qwen3-8B receive the question and full answer without functional or variant labels. Overlength inputs are excluded. The first, second, and fourth scorers cover 252 complete groups; InternLM2 and Skywork cover 305. Their common intersection contains 252 groups. The scoring release preserves all originals and repairs 67 edited variants in 35 groups.

<table><tr><td rowspan=1 colspan=2>Pair RM       N  W/L/T Rate</td></tr><tr><td rowspan=1 colspan=2>A-BFsfairX    5736/10/110.783</td></tr><tr><td rowspan=1 colspan=2>A-BSkywork  6730/11/260.732</td></tr><tr><td rowspan=1 colspan=2>A-BOffsetBias57 38/14/50.731</td></tr><tr><td rowspan=1 colspan=2>A-BArmoRM 5724/20/130.545</td></tr><tr><td rowspan=1 colspan=2>A-BInternLM26716/28/230.364</td></tr><tr><td rowspan=1 colspan=2>A-DFsfairX    59 39/18/20.684</td></tr><tr><td rowspan=1 colspan=1>A-D</td><td rowspan=1 colspan=1>Skywork  7136/16/190.692</td></tr><tr><td rowspan=1 colspan=1>A-D</td><td rowspan=1 colspan=1>OffsetBias59 29/26/40.527</td></tr><tr><td rowspan=1 colspan=1>A-D</td><td rowspan=1 colspan=1>ArmoRM 59 38/17/40.691</td></tr><tr><td rowspan=1 colspan=2>A-DInternLM27128/24/190.538</td></tr></table>

Table 6: General-quality score-matched subset on unchanged scored texts. W/L/T are A wins, losses, and ties; rate is W/(W+L), not W/N. Matching requires equality on every applicable general-quality dimension.

Mean reward contrasts are standardized by the original-answer score standard deviation. Functional-sensitivity intervals use 10,000 item bootstraps.

General-quality matching. The numericalquality rubric contains 24 general dimensions. Matching requires unchanged scored texts and equal scores on every applicable dimension, excluding the construction-clarity manipulation check. Repaired variants whose scores refer to earlier texts are excluded. Repeated scores have 78.5% exact agreement and 96.2% agreement within one point; weighted $\kappa ~ = ~ . 8 2 2 .$ Table 6 reports the retained groups, wins, losses, and ties. Matching does not condition on functional category.

Pairwise judgment. The Arena-Hard rubric assesses correctness, helpfulness, relevance, concision, completeness, and applicable creativity. The logged judge aliases are claude-sonnet-5-5 and gpt-6.1-sol. Each judge generates a reference answer and independently evaluates both presentation orders. After restoring variant identity, strong/weak original preference, tie, and weak/strong counterpart preference map to 1, .75, .5, .25, 0. The two-order average is an ordinal preference score. Twelve GPT judgments were recovered with the original 8,192-token client-side output cap omitted; prompts, reasoning efort, and scoring were unchanged.

There are 301 eligible groups: 141 substantive, 74 unlicensed, and 86 redundant. Defective, relation-failure, and unresolved cases are excluded. Claude has 297 complete two-order pairs per comparison: 139/138 substantive, 74/73 unlicensed, and 84/86 redundant for A–B/A–D. GPT has 300 per comparison, comprising 141 substantive, 74 unlicensed, and 85 redundant pairs for each edit. Figure 4 reports non-tie rewardmodel win rates with Wilson 95% intervals. Judge preference and functional-sensitivity intervals use $1 0 { , } 0 0 0$ item bootstraps; sensitivity resamples substantive and optional groups independently.

Functional sensitivity of the judges. Mean preference for retaining unlicensed corrections over afirmative-only answers is 0.630 (95% CI [0.603, 0.656]) for Claude and 0.519 ([0.505, 0.535]) for GPT. Their substantive-minusunlicensed and substantive-minus-pooled-optional diferences range from 0.002 to 0.006 across the two edits; all eight 95% intervals include zero (Figure 4d).

Functional diferences across reward models. The pooled reward-model $F S _ { O }$ , based on standardized contrasts, is .00194 for reordering, with a 95% interval of [ .00924, .0125]. For afirmativeonly editing it is .0154, with an interval of [.000439, .0301]. The latter comparison removes corrective content as well as its realization. The main claim rests on positive preference within optional and unlicensed uses, rather than on an equivalence claim across functional groups.

## C Checkpoint and Preference-Data Protocols

Checkpoint panel. The panel contains Base and SFT checkpoints from DeepSeekMath, Llama-3.1/Tülu-3, OLMo-2, MiMo, and InternLM2, plus DPO checkpoints from Tülu and OLMo: 12 checkpoints in total. All receive the same 1,061 prompts as the API panel, yielding 12,518 valid answers. Base checkpoints use a neutral question–answer frame; chat checkpoints use native templates without an added system message.

Decoding uses temperature .1 for 725 prompts and .7 for 336 according to scene, with top\_p = 1. The context cap is the smaller of the native model limit and 32,768 tokens; the initial generation budget is the smaller of 16,384 tokens and remaining context. Empty answers and overlengthinput records are excluded. Nonempty truncated responses are retained. Each plotted point uses the valid answers at that checkpoint and averages density equally over the four benchmarks. Table 8 gives the corresponding descriptive paired changes on common successful prompts.

<table><tr><td>Contrast</td><td>Group</td><td> $n / N$ </td><td>Mean score</td><td>95% CI</td></tr><tr><td>A-D</td><td>Overall</td><td>297/301</td><td>0.633</td><td>[0.621, 0.646]</td></tr><tr><td>A-D</td><td> $\mathrm { \bf S }$ </td><td>138/141</td><td>0.632</td><td>[0.613, 0.650]</td></tr><tr><td>A-D</td><td> $O _ { L }$ </td><td>73/74</td><td>0.630</td><td>[0.603, 0.656]</td></tr><tr><td>A-D</td><td> $O _ { U }$ </td><td>86/86</td><td>0.638</td><td>[0.615, 0.661]</td></tr><tr><td>A-B</td><td>Overall</td><td>297/301</td><td>0.566</td><td>[0.554, 0.577]</td></tr><tr><td>A-B</td><td> $\mathrm { \bf S }$ </td><td>139/141</td><td>0.567</td><td>[0.551, 0.585]</td></tr><tr><td>A-B</td><td> $O _ { L }$ </td><td>74/74</td><td>0.563</td><td>[0.541, 0.586]</td></tr><tr><td>A-B</td><td> $O _ { U }$ </td><td>84/86</td><td>0.565</td><td>[0.545, 0.588]</td></tr></table>

Table 7: Claude Sonnet 5.5 rubric-guided pairwise judgments by functional group. Scores average two presentation orders after restoring variant identity; values above 0.5 favor A. Intervals use 10,000 item bootstraps. $n / N$ denotes complete pairs/eligible groups. Overall covers the 301 substantive, unlicensed, and redundant cases; the two relation failures and four defective cases are excluded.

<table><tr><td>Family</td><td>Transition</td><td>Paired N</td><td> $\Delta I _ { C }$ </td></tr><tr><td>DeepSeekMath</td><td>Base→SFT</td><td>1,028</td><td>+0.0154</td></tr><tr><td>Llama/Tülu 3</td><td>Base→SFT</td><td>1,055</td><td>+0.0187</td></tr><tr><td>Llama/Tülu 3</td><td>SFT→DPO</td><td>1,061</td><td>+0.154</td></tr><tr><td>OLMo 2</td><td>Base→SFT</td><td>1,006</td><td>+0.0767</td></tr><tr><td>OLMo 2</td><td>SFT→DPO</td><td>1,018</td><td>+0.124</td></tr><tr><td>MiMo</td><td>Base→SFT</td><td>1,053</td><td>+0.131</td></tr><tr><td>InternLM2</td><td>Base→SFT</td><td>1,056</td><td>+0.165</td></tr></table>

Table 8: Descriptive parent–child density changes for SFT and DPO. $\Delta I _ { C }$ is child minus parent density per 10,000 Unicode characters, averaged equally over four benchmarks. Each transition uses common successful prompts and retains nonempty truncated answers. These paired changes need not equal diferences between the marginal checkpoint points in Figure 5b.

![](images/161827c3e39d9f747da01876c1f0f673c3e6854bbc36a573f6e513249f9f29a0.jpg)  
Figure 6: Descriptive Base–SFT–DPO trajectories by family. Points show benchmark-macro log-frequency, $L F _ { C } =$ $\log _ { 1 0 } ( 1 + I _ { C } )$ , using valid answers at each checkpoint and retaining nonempty truncated answers. Lines connect available stages; vertical scales difer across panels.

Density reflects completion length and repeated use as well as construction selection. Excluding truncated answers reverses the OLMo Base– SFT direction, while concentrated repetition contributes substantially to MiMo’s density. The displayed trajectories describe the retained-output protocol.

Preference-data audit. We analyze the fixed Tülu-3-8B and OLMo-2-1124-7B preference mixtures. Of 651,239 released rows, 648,868 have valid context-matched chosen/rejected final assistant answers: 271,958 Tülu pairs and 376,910 OLMo pairs. The datasets share 264,594 complete prompts, covering 99.74% and 72.34% of their unique valid prompts. The audit uses only the final assistant answer with identical prior context. Table 9 reports construction activation in each answer arm.

Preference-data intervals use a prompt-clusterrobust delta method.

For the Han-prompt proxy, the released prompt field (serialized when structured) must contain at least 20 Han characters and a Han-to-nonwhitespace-character ratio of at least .2. For the both-Han-answers proxy, each chosen and rejected final answer must separately meet these thresholds. Subset intervals are unadjusted; density exposure remains all Unicode characters, including whitespace.

Paired conditional-logit models adjusting log length ratio and Han proportion give construction odds ratios of .947 for Tülu (95% CI [.673, 1.332]) and 1.040 for OLMo ([.839, 1.289]). One Tülu chosen answer supplies 41 of 145 occurrences; removing its prompt cluster reduces the intensity diference by 73.8%. OLMo chosen answers repeat the construction less within activated answers: 1.213 versus 1.360 instances.

## D Implementation Details

Local inference for the open-weight checkpoints and reward models used NVIDIA A800 and H800 GPUs, totaling 8.35 A800 GPU hours and 17.07 H800 GPU hours. Checkpoint generation used vLLM 0.11.0 (Kwon et al., 2023). Rewardmodel scoring used the Hugging Face Transformers 4.56.1 implementations (Wolf et al., 2020) with PyTorch 2.8.0 (Paszke et al., 2019) and CUDA 12.8. The five commercial models, the annotation model, and both pairwise judges were accessed through their providers’ APIs. Decoding settings for the checkpoint panel are given in Appendix C, and those for the API panel in Appendix A.

<table><tr><td>Data</td><td>Arm</td><td>Pairs</td><td>Hits</td><td>Instances</td><td>A (%)</td><td>A per 10k pairs</td><td> $\Delta _ { \mathrm { r e l } }$  vs. rejected</td></tr><tr><td>Tülu 3</td><td>Chosen</td><td>271,958</td><td>86</td><td>145</td><td>0.03162</td><td>3.162</td><td> $+ 1 4 . 7 \%$ </td></tr><tr><td>Tülu 3</td><td>Rejected</td><td>271,958</td><td>75</td><td>91</td><td>0.02758</td><td>2.758</td><td></td></tr><tr><td>OLMo 2</td><td>Chosen</td><td>376,910</td><td>253</td><td>307</td><td>0.06712</td><td>6.712</td><td> $+ 2 6 . 5 \%$ </td></tr><tr><td>OLMo 2</td><td>Rejected</td><td>376,910</td><td>200</td><td>272</td><td>0.05306</td><td>5.306</td><td></td></tr></table>

Table 9: Public preference data, restricted to complete context-matched chosen/rejected pairs and final assistant responses. Hits count activated answers. A per 10k pairs = 10<sup>4</sup> Hits/Pairs. $\Delta _ { \mathrm { r e l } }$ is the relative increase of chosen over rejected activation, computed from unrounded counts. Chosen answers consistently show higher activation than rejected answers in both datasets, with a larger gap in OLMo 2. These rates are corpus-pooled, not benchmarkmacro estimates.