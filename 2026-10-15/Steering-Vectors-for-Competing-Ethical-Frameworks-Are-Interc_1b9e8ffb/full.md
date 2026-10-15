# Steering Vectors for Competing Ethical Frameworks Are Interchangeable: A Competing-Target Test for Activation Steering

Weilun Xu EPFL weilun.xu@epfl.ch

## Abstract

Activation steering adds a vector to a model’s hidden states to control a named behaviour. A gain on that behaviour, however, does not show that the vector controls it directly: it may act through a component shared with neighbouring behaviours, which unsteered and random baselines cannot separate. We introduce the competing-target test: a vector is compared with the vector built the same way for a competing target, and an equivalence test states when their difference is below a useful margin. We apply it in open-weight models to vectors for Kantian deontology, which judges an act by the rule it follows, and act utilitarianism, which judges it by the welfare it produces. On heldout situations, prompting a framework raises its score and lowers the other’s, whereas vectors for either framework move both scores alike, and none reaches the margin. The two vectors share most of their direction, a request to apply some framework; the same construction separates an optimistic from a pessimistic tone, whose vectors share much less. What a vector shares with its competitors therefore sets what it controls: steering a perspective can deliver a mixture, and measuring a disposition can measure its neighbours, unless tested against the competing target.

## 1 Introduction

Activation steering is used to set a requested perspective inside a language model, and ethical frameworks are among the perspectives requested: steerable pluralism asks for models that can be steered to a requested perspective (Sorensen et al., 2024), recent methods steer moral reasoning by internal edits (Yuan et al., 2026), and the stakes are real, because moral advice from a model moves its readers’ judgments (Krügel et al., 2023). Such a control has to deliver the framework itself, and competing ethical theories differ in the reasons they give for a decision, and sometimes in the decision itself.

Zipeng Sun McGill University Mila – Quebec Artificial Intelligence Institute zpointsun@gmail.com

(a)  
![](images/7e2496b0bf9553b5c8b4dd2ca21241d7828715617a03291c0b8d438e17cfab1a.jpg)

(b)  
![](images/36af65653afe4fe53548d2da6ece193d72dd78cdf00424af851f5ccccd45b724.jpg)  
Figure 1: The two framework vectors are interchangeable and the two tone vectors are not, implying a vector controls what it shares with its competitor. Arrows give the change in the two competing scores on the held-out situations, averaged over situations and models; the instructions are those of Section 3 and the vectors those of Section 5. In ethical frameworks (a), the defined instructions move the scores apart, whereas the two target-versus-neutral vectors at twice their natural norm and their shared component s move both the same way; for tones (b), the instructions and the vectors built the same way each move toward their own tone. Brackets mark the competing-target specificity.

A Kantian and a utilitarian may both refuse to deceive a friend, the first because deception treats a person merely as a means, the second because the lie would cost more welfare than it gains; they part ways when a lie would save a life. A framework therefore shows in the reasons an answer gives, and to ask a model for one framework is to ask for a particular reason, while the competing framework shares much of the rest of the answer. Whether a steering vector built for a framework delivers that reason, rather than ethical reasoning in general, is the question of this paper.

A steering vector is added to the model’s hidden states during generation, and the resulting change in behaviour is attributed to the concept after which the vector is named (Turner et al., 2023; Rimsky et al., 2024). A vector’s effect is checked against unsteered responses and random vectors; these baselines show that it changes behaviour but cannot show that it acts through the named concept rather than a neighbouring behaviour (Makelov et al., 2024). Moral steering uses such vectors: between-framework directions move reasoning between frameworks (Yuan et al., 2026), and a direction that separates rule-framed from consequenceframed questions shifts verdicts (Shi, 2026); in both, the effect is attributed to the framing the vector is named after, without a competing framing as control.<sup>1</sup> Under the linear representation hypothesis, the direction of a concept is estimated from counterfactual pairs that differ in that concept alone (Park et al., 2024), whereas a persona-style vector is built from a contrast between two instructions, which can differ in more than the concept it is named after (Stolfo et al., 2025). An instruction to apply an ethical framework, in particular, differs from a neutral instruction, one that asks for a decision and its reasons, in at least two respects: that a framework is requested, and which framework is requested. We call this contrast target-versusneutral. A vector built from it therefore estimates the sum of two directions, and steering along it is a fat-handed intervention, which changes the named variable together with another variable that affects the outcome (Baumgartner and Gebharter, 2016): such a vector can move responses toward ethical reasoning of any kind while carrying the name of one framework.

An intervention supports a causal claim only if it affects the outcome through the targeted variable alone (Woodward, 2003), and ethical frameworks make this testable, because the competing framework can be scored on the same response. We introduce the competing-target test (Figure 1). For each framework a vector is built, and the vector built in the same way for the competing framework is its control. Every free-text response is scored for the application of both frameworks, and the difference between the target’s score under the two vectors is the vector’s specificity. An equivalence test states when specificity is below a prespecified useful margin, and explicit prompting, which should separate the frameworks, is the positive control. We apply the test to Kantian deontology and act utilitarianism on moral dilemmas from two public benchmarks.

On held-out situations in three open-weight models, prompting raises one framework’s score and lowers the other’s, whereas vectors built for either framework move both scores alike: no vector at any dose of the main experiment reaches the useful margin, and the two vectors often return the same response word for word. The vectors share most of their direction, a request to apply some framework; even the axis that separates them recovers only about a tenth of prompting’s selectivity at the vectors’ own scale, whereas the same construction separates two tones.

The result bears on both uses of steering. For control, a vector that passes random baselines and reliability checks (Tan et al., 2024) can still deliver a mixture of perspectives where one was requested, because those checks presuppose specificity. For measurement, as when a vector that suppresses a model’s awareness of being tested is used to reveal deployment behaviour (Hua et al., 2026), an effect attributed to a named disposition can run through what the disposition shares with its neighbours. In both uses, the competing target is the control that separates the named target from what it shares with its neighbours.

Our contributions are threefold.

1. We introduce the competing-target test, which evaluates whether a steering vector acts through its named target; it applies wherever targets compete on one output, and its equivalence bound establishes absence, which random baselines cannot.

2. We show that steering vectors for competing ethical frameworks are interchangeable: in three open-weight models they move both frameworks alike while prompting separates them, because a shared request to apply some framework dominates both.

3. We suggest that a steering vector controls what its construction contrast varies, not what its name picks out: when competing targets share most of that contrast, the vector sets the category rather than the member, which only the competing target reveals.

## 2 Methods

The design holds each situation fixed and changes only how a framework is requested: by an instruction, or by a steering vector built for that framework. Every response is scored for both frameworks, which allows each control to be compared with the control for the competing framework.

## 2.1 Experimental setup

Frameworks and situations We study Kantian deontology, which judges an act by the rule it follows, and act utilitarianism, which judges it by the welfare it produces, on situations from two published sources: expert-written dilemmas from MoReBench Public (Chiu et al., 2026) and twoaction scenarios from the high- and low-ambiguity sets of MoralChoice (Scherrer et al., 2023). Situations form three disjoint sets. Construction situations (32 per source, in two samples) build the steering vectors, and four development situations, one per source and ambiguity set, informed the choices that were then frozen. Eight held-out situations, two per source and ambiguity set, were selected by a fixed rule before any intervention outcome on them existed and serve only for evaluation. Two further panels of eight and sixteen situations repeat the headline configuration (Section 3). Each situation is posed with a neutral instruction (“Give a decision and a concise explanation of the reasons for it”), placed before or after the situation; the named instruction asks for the same under a framework it names, and the defined instruction adds the framework’s MoReBench definition.

Models and steering vectors We use three openweight instruction-tuned models of similar size from three model families, Llama-3.1-8B-Instruct (Grattafiori et al., 2024), OLMo-3-7B-Instruct (Team Olmo, 2025) and Qwen3.5-9B (Qwen Team, 2026), written Llama, OLMo and Qwen below; all generation is greedy. For each construction situation we record the final-token output of layer 20 of 32 under the neutral instruction and under the defined instruction of the target framework, and three recipes turn the paired differences into a unit vector: their mean (persona; Chen et al., 2025), their first uncentered principal direction (PCA) and the weight vector of a logistic classifier that separates the two sets (probe; Im and Li, 2025). Built separately for each source, sample, placement, framework and recipe, this gives 48 vectors per model; ten Gaussian unit vectors serve as random controls. A vector v is applied as $h  h + \alpha v$ at layer 20, at the final prompt position and at every generated position, with doses $\alpha \in \{ \pm 1 . 5 , \pm 3 \}$ ; the largest dose is a sixth to a half of the natural norm, the norm of the mean paired difference over all construction situations. The same vectors are reused unchanged on the held-out situations.

Measuring framework application The primary evaluator, Qwen3.5-27B with a fixed prompt and greedy decoding, scores each response for both frameworks on three criteria, the standard used, its application to this situation and the support it gives the conclusion, each 0, 1 or 2, so that the framework score ${ \cal Y } _ { i F } \in [ 0 , 1 ]$ of prompt i, a situation in one placement, is the criterion sum divided by six; it also scores factual fidelity and coherence and flags refusal. The rubric credits the application of a framework, not its name, generic rule following or agreement with its verdict (Conway and Gawronski, 2013). The evaluator passes designed perturbation checks, and a second evaluator from another model family, Claude Opus 4.6, re-scores a prespecified subset with the same prompt; the primary evaluator shares a model family with one evaluated model, Qwen (Appendix C).

## 2.2 The competing-target test

The competing-target test applies to any vector built for one of several competing targets that can be scored on the same response. Let t be the target and t<sup>¯</sup>its competitor, let a control q be an instruction or an edit with $q = 0$ the unsteered response to the same prompt, and let $\Delta _ { i F } ( q ) = Y _ { i F } ( q ) - Y _ { i F } ( 0 )$ be its effect on framework F.

1. Build $v _ { t }$ and $v _ { \bar { t } }$ by the same recipe from the same construction data; the two form a matched pair, and the competitor’s vector is the control. Prompting with the named or the defined instruction is the positive control and must raise the target’s score and lower the competitor’s.

2. Score every response for both targets. A control’s selectivity is its movement toward t rather than toward ethical reasoning in general,

$$
\mathrm { S e l } _ { i } ( q ) = \Delta _ { i t } ( q ) - \Delta _ { i \bar { t } } ( q ) ,\tag{1}
$$

and the competing-target specificity, or specificity for short, compares the two vectors on the target’s score,

$$
\mathrm { S p e c } _ { i } ( t ) = Y _ { i t } ( v _ { t } ) - Y _ { i t } ( v _ { \bar { t } } ) .\tag{2}
$$

The test reports the average over the two targets, which equals the mean of the two vectors’ selectivities toward their own targets because both share the unsteered baseline:

$$
\begin{array} { r l } & { \overline { { \mathrm { S p e c } } } = \frac { 1 } { 2 } \left[ \mathrm { S p e c } ( K ) + \mathrm { S p e c } ( U ) \right] } \\ & { \qquad = \frac { 1 } { 2 } \left[ \mathrm { S e l } ( v _ { K } ) + \mathrm { S e l } ( v _ { U } ) \right] . } \end{array}\tag{3}
$$

3. Decide with an equivalence test (Lakens, 2017). A configuration, one recipe at one dose in one model, has one value per situation, the mean over its matched pairs; estimates average over situations, with a paired bootstrap stratified by source and MoralChoice ambiguity set $( 1 0 { , } 0 0 0$ resamples) and an exact sign-flip test over the $2 ^ { 8 }$ sign patterns of the situation values. A configuration is classed below 1/12 if the one-sided 95% upper bound of Spec lies below 1/12, half of one rubric point and the gain the usefulness rule requires, and the sign-flip test confirms it; framework-specific if Spec reaches 1/12 with a positive 95% lower bound and a sign-flip test that survives Holm adjustment across the six positive-dose configurations of a model; and undetermined otherwise (Appendix D).

Why the competing vector is the control The two vectors differ only in the framework their instruction names. Under the assumption (A1) that the instruction itself affects score t equally under both, this generic effect cancels in $\operatorname { S p e c } _ { i } ( t )$ whereas the excess over a random vector retains it, because a random vector carries no instructionrelated effect. The two vectors may still differ in generic strength, which biases the two single-target specificities in opposite directions; their average retains only a second-order term, which is why it is the reported quantity (Appendix D).

Usefulness rule Specificity is tested alongside the reliability criterion that steering studies apply to a vector on its own target. A control passes the usefulness rule if, on average over situations, it raises the target’s score and its application criterion by at least 1/12 over the unsteered response, exceeds matched random vectors by at least $1 / 2 4$ lowers factual fidelity and coherence by at most 0.05 on their unit scale, with a decrease in at most a tenth of the responses, and raises refusals by at most two percentage points; reuse, the same vector in the other placement, and replacement, the vector of another recipe, check that its effect is stable. The margin of the specificity test is the gain this rule requires.

Shared and specific components Two framework vectors can be written exactly as $\begin{array} { r l } { v _ { K } } & { { } = } \end{array}$ $s + d$ and $v _ { U } = s - d ,$ with shared component $s = ( v _ { K } + v _ { U } ) / 2$ and specific component $d = ( v _ { K } - v _ { U } ) / 2 ;$ the direction of d is the specific axis, and the pooled axis is the unit difference between the mean layer-20 activations under the two defined instructions over all construction situations. Under a first-order model of selectivity (A2), in which an edit of size α along a unit direction u has selectivity $\alpha \langle g , u \rangle$ to first order, the averaged specificity of two vectors with equal norm and cosine c is the fraction $\rho = \sqrt { ( 1 - c ) / 2 }$ of the selectivity of an edit of the same size along the specific axis, and two unit directions with cosine c differ in selectivity by at most $| \alpha | \| g \| { \sqrt { 2 ( 1 - c ) } }$ , so nearly collinear vectors agree by construction. The assumption that the shared component has no selectivity (A3) and two predictions about the between-framework vector and the pooled axis (P1, P2) were fixed before the held-out run and are tested in Sections 4 and 5; Appendix D states them and derives the first-order results.

Evaluator-free check Greedy decoding maps each prompt and control to one response, and each score is a deterministic function of the situation and the response, so a matched pair with identical responses has zero specificity under any such evaluator; if a fraction π of pairs is identical, the mean absolute specificity is at most $1 - \pi$

## 2.3 Prespecification

The vectors, doses, partitions and the usefulness rule were fixed on the development situations, and the held-out analysis plan before any held-out score existed; the later experiments state what was fixed before each of them ran.

## 3 Prompting separates the frameworks; steering vectors move both together

On the held-out situations, naming or defining one framework raises its score and lowers the other’s in every model, whereas vectors built for either framework change both framework scores alike: even at dose 3, the Kantian and the utilitarian vector differ on each score by at most a few hundredths of a point (Figure 2a). Prompting selectivity lies between three quarters of a point and one point for both instructions.

In every model all six positive-dose configurations are classed below $1 / 1 2$ and confirmed by the sign-flip test, and the largest one-sided upper bound is below two thirds of the margin (Figure 2b).

The evaluator-free check gives the same reading. A fifth to more than half of the matched pairs at the lower dose, and a twentieth to a third at dose 3, return the same response word for word; in Llama and OLMo nearly all of these pairs differ from the unsteered response, so both vectors changed the response in the same way, whereas in Qwen most of them are the unsteered response itself, and the two instructions never return the same response.

(a)  
![](images/98e5d270abc773243450b285a645c677bb2bd0ec7e146fff8c0017cf219bd5db.jpg)  
Figure 2: Prompting separates the frameworks; every steering configuration stays below the margin. Held-out situations of Section 3. (a) Score planes. An arrow joins the two members of a matched pair, from the utilitarian to the Kantian one; for prompting, thin arrows are single situations under either instruction and the bold arrow the mean under the defined instruction; the inset enlarges the origin, where each arrow is the mean pair of one recipe at one positive dose (bold: persona at dose 3) and grey points are random vectors. (b) Equivalence test. Averaged specificity of each configuration, with prompting right of the broken axis: estimate (circle), one-sided 95% upper bound (bar) and situation values (grey dots); filled circles are classed below 1/12 and confirmed by the sign-flip test.

Reliability does not substitute for specificity: on the development situations three controls passed the usefulness rule, and their specificity was near zero there and on the held-out situations, where no configuration meets either requirement.

A second evaluator from another model family assigns the same classes in every model, and the primary evaluator gives its own family no advantage. On two further panels of eight and sixteen fresh situations, the persona vectors at dose 3, the headline configuration, are again below 1/12 in every model with the sign-flip confirmation, and pooled over all 32 held-out situations their upper bound stays below half the margin. Prompting selectivity is within a tenth of a point of the first panel, the two vectors of a pair move both scores the same way or leave both unchanged, and a twentieth to more than a third of the pairs return identical responses. Appendices C, E and F report the evaluator agreement, the values with the identical-response, reliability and panel checks, and one situation with its four

responses.

## 4 A shared component dominates the vectors

Design Two further measurements ask whether the shared component is specific to layer 20 and to the Kantian–utilitarian pair. The first is geometric. On the 128 construction prompts we record the final-token output of every layer under further instructions: a generic instruction to apply an unnamed ethical framework, the filler paragraph of the sufficiency test (Section 5) added to the neutral instruction, three further MoReBench frameworks (Aristotelian virtue ethics, Scanlonian contractualism and Gauthierian contractarianism), and two pairs of competing non-ethical instructions (an economic or a legal perspective, and an optimistic or a pessimistic tone). The shifts give what the shared component aligns with, how much of each vector it carries, and whether other pairs share as much. Three predictions were fixed before extraction: the shared component aligns with the generic instruction more than with the filler; the five frameworks share a dominant component; and the shared share exceeds the framework share at every layer. The second is behavioural: the specificity test is repeated with pooled persona vectors built and applied at an earlier and a later layer (12 and 26), and in Llama also at the layers around the later one, at the main experiment’s largest dose ratio and at the natural norm.

(a)  
![](images/4d8b3162188fe5cac2850e72c6b19b2d360c37b9686f1aa65d87ece95bea4449.jpg)

(b)  
![](images/db97f8688045864d7951410ae527bfb55870276622aaf77cf8e12589986d2e8c.jpg)

(c)  
![](images/c56cd677ef25df46f52a918c16be9d921dd42140425bff3e74a96a25306b2244.jpg)  
Figure 3: A shared component dominates the framework vectors and aligns with a generic request to apply a framework, at every layer. Construction prompts of Section 4; markers and lines by model. (a) Cosines at layer 20 between the shifts named in each row, under the defined instructions; the filler paragraph is a nonmoral paragraph of the definitions’ length. (b) Alignment across layers: the cosine of s with the genericinstruction direction and with the filler direction. (c) Dominance across layers: the share of the squared norm of the sixteen construction vectors carried by the shared and by the framework component. The vertical line marks layer 20.

Results In every model most of each framework vector is the component the two frameworks share: ρ is about a quarter, so the specific axis carries only 5–8% of a vector’s squared norm, and the probe recipe yields almost the same direction as the mean, which the uncentred principal direction reproduces by construction (Figure 3a).

The shared component is a generic request to apply a framework rather than a property of layer 20 or of this pair, and all three predictions hold in every model: it aligns with the generic instruction more than with the filler, the five frameworks share it, and the shared share exceeds the framework share at least sixfold at every layer (Figure 3).

An economic and a legal perspective share a large component, an optimistic and a pessimistic tone a smaller one.

On the held-out situations, projection on the pooled axis predicts selectivity within one framework’s vectors in at most one framework per model, and across all 48 vectors of a model the correlation survives a framework-stratified permutation in one model only (P2); reuse across placements and replacement between recipes hold, but the effects they compare are small. Behaviourally, the pooled persona pair stays below the margin at the other layers in two models; in the third, one configuration reaches the margin through the Kantian vector alone, at about a tenth of prompting’s selectivity, and the layers around it do not. Appendix G gives the cosines by recipe and construction factor, the pooled-axis correlations and the layer table.

## 5 Single-site edits recover at most four tenths of prompting’s selectivity

Design The sufficiency test measures the selectivity of several single additive edits at layer 20, relative to prompting, on the held-out situations in both placements, each edit at the natural norm of its vector. Situation-specific oracle edits add the difference between the final-token states of the same situation under two instructions: one defined instruction and the neutral one, which reproduces the prompted state at that position, or the two defined instructions. Transferable mean vectors, target-versus-neutral and between-framework, are computed from construction situations only and applied at one and two times their norm. Format controls use directions from instructions that only name a framework and from a non-moral filler paragraph as long as the framework definitions. The shared component $s = ( v _ { K } + v _ { U } ) / 2$ of the two target-versus-neutral mean vectors is applied alone, which tests (A3), and a random vector at matched norm completes the set; the target-versus-neutral pair is classified with the rules of Section 2.2. Two further edits, fixed after the single-site results were read, add the mean vectors of three layers at once and patch the final prompt token, replacing its state at layer 20, or at every layer, by the state the same position has under the defined instruction during prefill, with no edit at generated positions.

Results A single edit along the axis that separates the frameworks recovers about a tenth of prompting’s selectivity at its natural norm, whether the direction transfers from the construction situations (mean and named vectors) or comes from the model’s own prompted states (oracle); a random direction at the framework vectors’ norm recovers as much, because perturbations of that size lower the utilitarian score more than the Kantian one (Figure 4a). The three directions agree where the comparison is available (P1), except the named pair in OLMo. Because the oracle recovers no more than the transferable vectors, the gap is not one of transfer: the prompted state added at one position of one layer does not reproduce prompting. Larger doses along that axis recover a quarter to four tenths of prompting within the quality limits, where a random direction of comparable norm recovers at most an eighth, and more only outside them (Figure 4b).

The shared component, applied alone, moves responses toward the Kantian framework in every model, by a seventh to a third of prompting’s selectivity; a random direction at the same norm moves them the same way by about a tenth, and the shared component exceeds it clearly in one model, so (A3) fails at least there, and the reported quantity does not depend on it (Section 2.2). Every target-versusneutral vector inherits the movement whichever framework it was built for (Figure 1a), so Kantian vectors gain selectivity toward their target and utilitarian vectors lose it.

Target-versus-neutral pairs stay near zero specificity at the natural norm, as in the main experiment. The two that reach the margin do so in one model, as a difference in degree along one shared movement: in both pairs the utilitarian vector lowers the utilitarian score and the Kantian vector lowers it further, the named pair at about a seventh of prompting and the mean pair at twice the norm outside the quality limits. Other sites do not help: adding the mean vectors of three layers at once behaves as a larger dose at one site, and patching the final prompt token’s trajectory at every layer reproduces the prompted next-token distribution there, yet the response that follows recovers at most a tenth of prompting’s selectivity, and the two frameworks’ patches often return identical responses.

The format controls separate a request to apply a framework from added text: the filler edit, at its own norm, leaves both scores unchanged, and the named-framework edits move responses as the defined-instruction edits do. The second evaluator classes no edit as framework-specific either.

(a)  
![](images/96a5ee47cc63ea04dc10c6b3b9f5b8f1206bbff61e818796062db469a00c8868.jpg)

(b)  
![](images/24494bd58972c271df557513ed938f3734881e9b8fb8c7ec51bb52a813c9f6b9.jpg)  
Figure 4: Single-site edits recover no more than a random direction at the natural norm and at most four tenths of prompting within the quality limits. Heldout situations of Section 5; 95% situation-bootstrap intervals (10,000 resamples). (a) Single-site edits. Selectivity of each edit at layer 20 as a fraction of prompting’s selectivity in the same model (tone rows: of tone prompting’s), measured toward the framework the vector was built for, so a negative value for a utilitarian vector is movement toward the Kantian framework; the dashed line marks prompting. (b) Dose response of the specific directions. Recovery against the multiple of the natural norm for each direction; open markers are outside the quality limits of the usefulness rule.

Calibration on tone A tone calibration applies the construction and the test to an optimistic and a pessimistic tone, whose shared component is smaller (Section 4), with mean vectors from the named tone instructions at the same two multiples, scored with a tone rubric by both evaluators. A single-site vector can pass this test: each tone vector moves responses toward its own tone, and at twice the norm the pair reaches the margin in every model under both evaluators, recovering a quarter to four tenths of tone prompting’s selectivity with coherence within the range of tone prompting itself (Figure 1b; Figure 4a). At that dose the framework pairs reach about a sixth of prompting at most, and their two vectors move the same way. Appendix H lists the edits and gives every edit’s values, the multi-site and patch runs, the second evaluator’s scores and the tone calibration under both evaluators.

## 6 Discussion

A vector’s name records its construction A steering vector is named after the condition that built it, and its effect depends on everything that differs between the two conditions of its contrast (Braun et al., 2025). Naming a vector after its construction is an operational definition, but construct validity requires more (Cronbach and Meehl, 1955): here, the discriminant evidence of Section 2.2, that the operation moves the named construct more than its neighbours (Campbell and Fiske, 1959). Four lines of evidence with different assumptions agree: the specificity test, the same classes under a second evaluator from another model family, identical responses that need no evaluator, and the activation geometry, which uses no responses.

The shared component is a request The shared component may carry ethical justification in general, which the best versions of several theories may share (Parfit, 2011), a request to apply some framework, or the added text of an instruction. The geometry and the format controls favour the request reading (Sections 4 and 5). The request is not specific to ethics, as the shared component of an economic and a legal perspective has cosine about 0.7 with it and the tone pair’s far less; nor is it neutral between the frameworks: applied alone, the shared component moves responses toward the Kantian framework, in one model clearly beyond a random direction of the same norm.

Strength, recipe and site Up to twice the natural norm, no target-versus-neutral pair exceeds about a sixth of prompting’s selectivity, and the pairs that reach the margin do so in one model, as a difference in degree along a shared movement, one of them outside the quality limits. Construction does not change the picture: the three recipes coincide, and the between-framework direction that calibrated moral steering extracts (Yuan et al., 2026) and the oracle built from the model’s own prompted states both recover about a tenth of prompting at one site. Nor does the site, at other layers, at three layers at once or with the final prompt token’s trajectory patched at every layer (Section 5). The instruction therefore acts from its own positions, which no single-position summary carries, so an intervention that aims at what prompting does has to supply what those positions supply, as prefix tuning does (Li and Liang, 2021).

The remedy and its cost The decomposition names the remedy: the direction that separates the frameworks, taken on its own and dosed above its natural norm, recovers, within the quality limits, a quarter to four tenths of prompting’s selectivity in every model, and larger doses buy no more without a loss of coherence and fidelity (Section 5). An instruction delivers the requested perspective and a target-versus-neutral vector does not, consistent with prompting outperforming steering on concept benchmarks (Wu et al., 2025).

## 7 Conclusion

A steering vector is named after the behaviour it is built to control; this paper asked whether a vector built for one ethical framework controls that framework or the category it belongs to. The competingtarget test answers by comparing the vector with the one built the same way for the competing framework, with an equivalence bound on the difference. In three open-weight models the Kantian and the utilitarian vector are interchangeable, because a shared request to apply some framework dominates both, whereas prompting separates the frameworks and the same construction separates two tones.

The implication reaches beyond ethics. A gain on the named target does not show what the vector controls. The competing target does, and it is available wherever targets compete on one output, as values, personas and political perspectives do. The same shared component means that a vector or probe built for one disposition can register its neighbours, so a claim that a model holds a value needs the same discriminant check. The geometry places the request to apply a framework in one dominant direction and the choice among frameworks in a small remainder that no single site carries, where framework-specific computation should be sought.

## Limitations

The study tests specificity for two normative frameworks in three open instruction-tuned models of 7 to 9 billion parameters, with additive edits at one layer in the main experiment, on eight situations held out from construction and calibration and two further panels of 24 situations, 32 in all, for the headline configuration.

Kantian deontology and act utilitarianism are two of many normative theories. The rubric operationalizes each through one definition and three criteria, and does not exhaust their philosophical interpretations. Frameworks that share more or fewer of their standards may differ in the size of the shared component.

A single evaluator model scores all responses. Its validity rests on designed perturbation checks and on the prompting positive control, not on independent human judgments. A second evaluator from another model family assigns the same classes on a prespecified subset, but the two agree only weakly on the Kantian score (Appendix C), so the scores carry evaluator-specific judgment. The tone rubric has no perturbation checks of its own. The primary evaluator also shares a model family with one evaluated model. All situations and instructions are in English.

Nonlinear interventions and trained adapters may separate the frameworks where additive edits do not. The sufficiency test addresses additive edits at one site at up to two multiples of the natural norm and up to four along the specific axis, the mean vectors at three sites, and a replacement of the final prompt token’s state at one or at every layer. Trained edits remain untested. The main experiment uses one layer and fixed dose candidates, whereas steering methods often tune the layer and the dose for each model (Rimsky et al., 2024; Stolfo et al., 2025). The layer test finds one configuration at the margin among 24, at about a tenth of prompting’s selectivity, with the neighbouring layers below the margin or undetermined, and the threelayer edit behaves as a larger dose in every model. Eight held-out situations limit the precision of the intervals, and the exact sign-flip test cannot fall below 1/256. Two further panels of eight and sixteen situations replicate the headline configuration in every model, and pooled over the 32 situations its upper bound stays below half the margin. The two placements of one situation are not independent observations. Single edits carry a generic drift: a random direction at the framework vectors’ norm lowers the utilitarian score more than the Kantian in every model, so the single-edit selectivities of Section 5 include it, whereas the competing-target average removes it because both vectors of a pair carry it.

The scores measure the reasoning expressed in a response, not a model’s internal reasoning and not the verdict a response reaches where the frameworks conflict.

## Ethical considerations

Steering methods that change the ethical framework of a model’s responses could be used to influence users’ moral views. Because steering leaves the prompt unchanged, such influence is not visible in the prompt. The competing-target test makes such effects easier to detect, because it shows when a control changes more than the perspective it names. All situations come from public benchmarks, and the study involved no human participants.

## Acknowledgements

Generative AI tools were used to assist with manuscript revision and code development.

## References

Michael Baumgartner and Alexander Gebharter. 2016. Constitutive relevance, mutual manipulability, and fat-handedness. The British Journalfor the Philosophy ofScience, 67(3):731–756.

Yonatan Belinkov. 2022. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219.

Roger L. Berger and Jason C. Hsu. 1996. Bioequivalence trials, intersection-union tests and equivalence confidence sets. Statistical Science, 11(4):283–319.

Joschka Braun, Carsten Eickhoff, David Krueger, Seyed Ali Bahrainian, and Dmitrii Krasheninnikov. 2025. Understanding (un)reliability of steering vectors in language models. arXiv preprint arXiv:2505.22637.

Donald T. Campbell and Donald W. Fiske. 1959. Convergent and discriminant validation by the multitrait-multimethod matrix. Psychological Bulletin, 56(2):81–105.

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. 2025. Persona vectors: Monitoring and controlling character traits in language models. arXiv preprint arXiv:2507.21509.

David S. Chester and Emily N. Lasko. 2021. Construct validation of experimental manipulations in social psychology: Current practices and recommendations for the future. Perspectives on Psychological Science, 16(2):377–395.

Yu Ying Chiu, Michael S. Lee, Rachel Calcott, Brandon Handoko, Paul de Font-Reaulx, Paula Rodriguez, Chen Bo Calvin Zhang, Ziwen Han, Udari Madhushani Sehwag, Yash Maurya, Christina Q. Knight, Harry R. Lloyd, Florence Bacus, Mantas Mazeika, Bing Liu, Yejin Choi, Mitchell L. Gordon, and Sydney Levine. 2026. MoReBench: Evaluating procedural and pluralistic moral reasoning in language models, more than outcomes. In The Fourteenth International Conference on Learning Representations.

Paul Conway and Bertram Gawronski. 2013. Deontological and utilitarian inclinations in moral decision making: A process dissociation approach. Journal of Personality and Social Psychology, 104(2):216–235.

Lee J. Cronbach and Paul E. Meehl. 1955. Construct validity in psychological tests. Psychological Bulletin, 52(4):281–302.

Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. 2021. Amnesic probing: Behavioral explanation with amnesic counterfactuals. Transactions of the Associationfor Computational Linguistics, 9:160– 175.

Navita Goyal and Hal Daumé III. 2026. Steering safely or off a cliff? Rethinking specificity and robustness in inference-time interventions. In Proceedings of the 19th Conference of the European Chapter of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 5723–5738, Rabat, Morocco. Association for Computational Linguistics.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, et al. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Julia Haas, Sophie Bridgers, Arianna Manzini, Benjamin Henke, Joshua May, Sydney Levine, Laura Weidinger, Murray Shanahan, Kristian Lum, Iason Gabriel, and William Isaac. 2026. A roadmap for evaluating moral competence in large language models. Nature, 650(8102):565–573.

John Hewitt and Percy Liang. 2019. Designing and interpreting probes with control tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2733–2743.

Tim Tian Hua, Andrew Qin, Samuel Marks, and Neel Nanda. 2026. Steering evaluation-aware language models to act like they are deployed. In The Fourteenth International Conference on Learning Representations.

Jing Huang, Zhengxuan Wu, Christopher Potts, Mor Geva, and Atticus Geiger. 2024. RAVEL: Evaluating interpretability methods on disentangling language model representations. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8669– 8687.

Shawn Im and Sharon Li. 2025. A unified understanding and evaluation of steering methods. arXiv preprint arXiv:2502.02716.

Ole Jorgensen, Dylan Cope, Nandi Schoots, and Murray Shanahan. 2023. Improving activation steering in language models with mean-centring. arXiv preprint arXiv:2312.03813.

Woojin Kim, Sieun Hyeon, Jusang Oh, and Jaeyoung Do. 2026. VALUEFLOW: Toward pluralistic and steerable value-based alignment in large language models. In Proceedings of the 43rd International Conference on Machine Learning, volume 306, pages 57615–57677. PMLR.

Sebastian Krügel, Andreas Ostermaier, and Matthias Uhl. 2023. ChatGPT’s inconsistent moral advice influences users’ judgment. Scientific Reports, 13:4569.

Daniël Lakens. 2017. Equivalence tests: A practical primer for t tests, correlations, and metaanalyses. Social Psychological and Personality Science, 8(4):355–362.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582– 4597.

Marc Lipsitch, Eric Tchetgen Tchetgen, and Ted Cohen. 2010. Negative controls: A tool for detecting confounding and bias in observational studies. Epidemiology, 21(3):383–388.

Aleksandar Makelov, Georg Lange, Atticus Geiger, and Neel Nanda. 2024. Is this the subspace you are looking for? An interpretability illusion for subspace activation patching. In The Twelfth International Conference on Learning Representations.

Derek Parfit. 2011. On What Matters. Oxford University Press.

Kiho Park, Yo Joong Choe, and Victor Veitch. 2024. The linear representation hypothesis and the geometry of large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 39643–39666.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents. Model card: https://huggingface. co/Qwen/Qwen3.5-9B.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering Llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15504–15522.

Nino Scherrer, Claudia Shi, Amir Feder, and David M. Blei. 2023. Evaluating the moral beliefs encoded in LLMs. In Advances in Neural Information Processing Systems, volume 36, pages 51778–51809.

Eugenie Shi. 2026. Rule vs. consequence: Dissociable internal representations of moral reasoning in LLMs. In Conference on Language Modeling (COLM).

Xu Shi, Wang Miao, and Eric Tchetgen Tchetgen. 2020. A selective review of negative control methods in epidemiology. Current Epidemiology Reports, 7(4):190– 202.

Tamar Sofer, David B. Richardson, Elena Colicino, Joel Schwartz, and Eric J. Tchetgen Tchetgen. 2016. On negative outcome control of unobserved confounding as a generalization of difference-in-differences. Statistical Science, 31(3):348–361.

Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell L. Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, Tim Althoff, and Yejin Choi. 2024. Position: A roadmap to pluralistic alignment. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 46280–46302.

Alessandro Stolfo, Vidhisha Balachandran, Safoora Yousefi, Eric Horvitz, and Besmira Nushi. 2025. Improving instruction-following in language models through activation steering. In The Thirteenth International Conference on Learning Representations.

Daniel Tan, David Chanin, Aengus Lynch, Brooks Paige, Dimitrios Kanoulas, Adrià Garriga-Alonso, and Robert Kirk. 2024. Analysing the generalisation and reliability of steering vectors. In Advances in Neural Information Processing Systems, volume 37, pages 139179–139212.

Team Olmo. 2025. Olmo 3. arXiv preprint arXiv:2512.13961.

Alejandro Tlaie. 2025. Exploring and steering the moral compass of large language models. In Pattern Recognition. ICPR 2024 International Workshops and Challenges, volume 15619 of Lecture Notes in Computer Science, pages 420–442. Springer.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248.

James Woodward. 2003. Making Things Happen: A Theory of Causal Explanation. Oxford University Press.

Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, and Christopher Potts. 2025. AxBench: Steering LLMs? Even simple baselines outperform sparse autoencoders. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pages 67035–67080. PMLR.

Weilun Xu, Alexander Rusnak, and Frédéric Kaplan. 2026a. Ethical frameworks in LLMs route through a shared endorsement subspace. In Mechanistic Interpretability Workshop at the 43rd International Conference on Machine Learning (ICML), Seoul, South Korea.

Weilun Xu, Alexander Rusnak, and Frédéric Kaplan. 2026b. Probing ethical framework representations in large language models: Structure, entanglement, and methodological challenges. arXiv preprint arXiv:2603.23659.

Chenchen Yuan, Zheyu Zhang, and Gjergji Kasneci. 2026. Where paths split: Localized, calibrated control of moral reasoning in large language models. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 41698–41721.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, and 2 others. 2023. Representation engineering: A top-down approach to AI transparency. arXiv preprint arXiv:2310.01405.

## A Extended related work

Steering vectors Contrastive activation addition averages the activation differences between completions that do and do not show a behavior (Rimsky et al., 2024). Representation engineering reads and controls behaviours along directions found by contrasting stimuli (Zou et al., 2023). Differenceof-means vectors are a strong representation-based baseline (Wu et al., 2025), and a unified analysis finds mean differences preferable to PCA- and classifier-based directions (Im and Li, 2025). Persona vectors contrast system prompts that elicit and suppress a trait (Chen et al., 2025), and instruction vectors contrast inputs with and without an instruction (Stolfo et al., 2025). The construction prompts change the resulting direction, and the coherence of the training differences predicts how reliably a vector steers (Braun et al., 2025). Steerability also varies widely across inputs, depends more on the dataset than on the model, and can fail under reasonable changes to the prompt (Tan et al., 2024). Prompting outperforms representation steering on a broad concept benchmark (Wu et al., 2025), and instruction vectors steer less than the instruction itself (Stolfo et al., 2025). Specificity has been evaluated as the preservation of fluency, of properties related to the target, and of these properties under distribution shift, while a single intervention moves its target (Goyal and Daumé III, 2026). These evaluations concern one intervention and one target. We compare two matched interventions whose targets compete, on the same score.

Moral and value steering Value-steering methods set the intensity of values (Kim et al., 2026) and move reasoning between utilitarian and deontological frameworks with localized, calibrated edits (Yuan et al., 2026). The latter extracts betweenframework directions so that variance shared across frameworks does not dominate them, and it evaluates the framework a response names. Across eleven models and six moral systems, directions built from between-framework contrasts substitute for one another under intervention, their effects passing through a shared low-dimensional subspace that Xu et al. (2026a) relate to endorsement valence; the present test turns that substitution into a prespecified equivalence test against prompting and measures what the shared part is. Activation steering has also moved the ethical school to which a classifier assigns each response, with less movement toward non-target schools than another steering method (Tlaie, 2025). A single-label assignment cannot register a response that applies both frameworks more strongly. We score the application of both frameworks in every response and compare steering with prompting on the same situations.

Validity standards for probes and interventions Probes recover framework and framing information from activations (Xu et al., 2026b; Shi, 2026), but they partly track surface features of benchmark templates (Xu et al., 2026b) and generalize only partly to new phrasings (Shi, 2026). Decodability alone does not show that a model uses the information (Belinkov, 2022; Elazar et al., 2021), and a causal effect of an intervention does not by itself identify the concept responsible for it (Makelov et al., 2024). Two established standards address this gap. Probes are compared with control tasks that a probe could solve without the property (Hewitt and Liang,

2019), and interventions on a represented attribute are required to leave the model’s other attributes unchanged (Huang et al., 2024). Competing-target specificity adapts the second standard to steering vectors whose targets compete for the same response: instead of requiring the competing score to stay fixed, it compares a vector with the matched vector for the competing framework on the same score.

Evaluating moral reasoning Moral reasoning is evaluated against expert rubrics (Chiu et al., 2026) or through choices between actions (Scherrer et al., 2023). Competence must be distinguished from performance under a particular elicitation (Haas et al., 2026).

## B Glossary

situation a published dilemma or scenario posed to the model; the unit of inference. Placement: whether the instruction precedes or follows the situation.

neutral, named, defined instruction a request for a decision and its reasons; the same request applying a framework by name; the same with the framework’s definition added.

control an instruction or an added vector built for a target framework; the competing control is the one built in the same way for the other framework.

target-versus-neutral contrast framework instruction against neutral instruction, the contrast from which the main vectors are built; a recipe (persona, PCA, probe) turns a contrast into a unit vector.

dose, residual norm, natural norm the multiple of a unit vector added at layer 20; the norm of the layer-20 output at the edited position; the norm of the mean paired difference over the construction situations.

configuration one recipe at one dose in one model.

framework score Y the application of a framework in a response, three rubric criteria summed and divided by six.

selectivity, specificity the difference between the two framework scores under one control; the difference between the target’s score under the target’s vector and under the competing vector, averaged over situations (the reported quantity is the mean over the two frameworks). Margin: 1/12 of a point.

shared and specific components $\begin{array} { r l r } { s } & { { } = } & { ( v _ { K } \ + } \end{array}$ $v _ { U } ) / 2$ and $d = ( v _ { K } - v _ { U } ) / 2 ;$ the specific axis is the direction of d; the pooled axis is the unit difference between the mean activations under the two defined instructions.

usefulness rule, reliability a gain over the unsteered response and over random vectors within quality limits; a stable effect under reuse (other placement) and replacement (other recipe).

sufficiency edits oracle: a situation’s own prompted-state difference; mean: the pooled construction difference; named: the same from name-only instructions; filler: a lengthmatched non-moral paragraph; shared alone: s by itself; random: a Gaussian vector at matched norm. The multi-site edits and the final-token patches are defined in Section 5.

## C Evaluator checks and the second evaluator

Designed perturbations behave as required: names alone and reasoning about a different situation receive no application credit, padding and duplication change application scores by less than a prespecified tolerance, and an inverted Kantian argument receives no credit for application or for support of the conclusion. The second evaluator uses temperature 0 and no extended reasoning and scores all prompting responses and the persona and randomdirection responses at the two positive doses. The paper reports whether the two evaluators assign the same classes and does not average them.

Table 1 gives the agreement between the two evaluators on the prespecified subset: they agree more closely on the utilitarian score than on the Kantian score, and the second evaluator assigns lower scores. The classification of every prompting condition and every persona dose is the same under both evaluators in every model. Under the second evaluator the sign of the change that the two vectors produce on the utilitarian score differs from the primary evaluator in OLMo and Qwen, while specificity stays below 1/12; the robust statement is that the two vectors change each score alike, not the sign of the change. The evaluator-by-model interaction on averaged persona specificity, the difference between evaluators for Qwen minus the mean of that difference for Llama and OLMo, is 0.002 at dose 1.5 (95% interval [−0.012, 0.015]) and 0.025 at dose 3 ([0.006, 0.045]). A same-family preference would make the primary evaluator assign its own family more specificity than the second evaluator does; the positive sign at dose 3 is the opposite.

<table><tr><td>Score</td><td>r</td><td>offset</td><td> $\kappa _ { w }$ </td><td>α</td></tr><tr><td>Llama-3.1-8B</td><td></td><td></td><td></td><td></td></tr><tr><td>Kantian utilitarian</td><td>0.34</td><td>-0.26</td><td>0.18</td><td>0.02</td></tr><tr><td>OLMo-3-7B</td><td>0.68</td><td>-0.04</td><td>0.54</td><td>0.56</td></tr><tr><td>Kantian</td><td>0.23</td><td>-0.39</td><td>0.07</td><td>-0.32</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>utilitarian</td><td>0.63</td><td>-0.11</td><td>0.47</td><td>0.51</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td></td><td></td></tr><tr><td>Kantian</td><td>0.22</td><td>-0.34</td><td>0.09</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>-0.17</td></tr><tr><td>utilitarian</td><td>0.43</td><td>-0.04</td><td>0.38</td><td>0.38</td></tr></table>

Table 1: The second evaluator assigns the same classes as the primary evaluator. Agreement on the prespecified subset of 720 held-out responses per model (all prompting responses; persona vectors and four random directions at doses 1.5 and 3): r is the Pearson correlation of the framework score, offset the mean score of the second evaluator minus the primary, and $\kappa _ { w }$ and α the quadratically weighted kappa and ordinal Krippendorff’s alpha on the application criterion.

## D Derivations

Notation For prompt i and control q, let $y _ { i } ( q )$ be the greedy response and $Y _ { i F } ( q ) = \sigma _ { F } ( y _ { i } ( q ) , x _ { i } ) \in$ [0, 1] its score for framework $F \in \{ K , U \}$ , where $\sigma _ { F }$ is any deterministic function of the situation $x _ { i }$ and the response. Population quantities average over the responses to each situation and then over situations. The identification step and the identity between averaged specificity and selectivity hold for each prompt, and hence for these averages; the first-order statements concern the averages only, because greedy decoding makes a single response discontinuous in the dose.

## Assumptions and prespecified predictions

(A1) The instruction from which a pair of vectors is built affects the target’s score equally under the two vectors.

(A2) Averaged over situations and placements, the Kant-oriented selectivity of an edit of size α along a unit direction u is $\alpha \langle g , u \rangle + o ( \alpha )$ for some vector g.

(A3) The shared component has no selectivity,

$\langle g , s \rangle = 0 ;$ Section 5 tests it.

Two predictions were fixed before any held-out score existed.

(P1) A between-framework mean vector is the difference of the two target-versus-neutral mean vectors. Under (A2) its selectivity equals, to first order, the sum of their selectivities toward their own targets at any common multiple of their natural norms, and under (A3) the shared component alone has zero selectivity.

(P2) If g lies along the pooled axis, selectivity increases with a vector’s projection on it within each framework’s vectors.

Identification Write each effect as $\Delta _ { i t } ( v ) \ =$ $\Delta _ { i t } ^ { \mathrm { g e n } } ( v ) + \Delta _ { i t } ^ { \mathrm { f w } } ( v )$ , a framework-generic part and the remainder; neither part is observed. Under $( \mathbf { A } 1 )$ $\Delta _ { i t } ^ { \mathrm { g e n } } ( v _ { t } ) = \Delta _ { i t } ^ { \mathrm { g e n } } ( v _ { \bar { t } } )$ , so the generic parts cancel in Eq. 2:

$$
\mathrm { S p e c } _ { i } ( t ) = \Delta _ { i t } ^ { \mathrm { f w } } ( v _ { t } ) - \Delta _ { i t } ^ { \mathrm { f w } } ( v _ { \bar { t } } ) .\tag{4}
$$

The same substitution for a random vector z gives

$$
\begin{array} { r l } & { \Delta _ { i t } ( v _ { t } ) - \Delta _ { i t } ( z ) = \Delta _ { i t } ^ { \mathrm { f w } } ( v _ { t } ) - \Delta _ { i t } ^ { \mathrm { f w } } ( z ) } \\ & { \qquad + \left[ \Delta _ { i t } ^ { \mathrm { g e n } } ( v _ { t } ) - \Delta _ { i t } ^ { \mathrm { g e n } } ( z ) \right] , } \end{array}
$$

whose bracket vanishes only if the two generic effects are equal. A random vector lacks the instruction-related pathway, so the bracket remains whenever a learned vector carries a generic effect: the competing vector controls for everything the two requests share, and a random vector does not. In the terms of causal inference, the competing vector is a negative-control exposure, a second exposure that shares the confounding pathway but not the causal one (Lipsitch et al., 2010); (A1) is the equal-bias condition under which such a contrast removes the generic effect (Shi et al., 2020), and specificity is a discriminant-validity test of a manipulation (Campbell and Fiske, 1959; Chester and Lasko, 2021). Unlike a pure negative control, the competing vector may lower score t through its own content, and $\mathrm { S p e c } ( t )$ counts such a decrease toward specificity, as a request for one framework rather than the other should. Selectivity contrasts the two scores under one control and has the form of a difference-in-differences (Sofer et al., 2016); the competing score is therefore not a negativecontrol outcome in the strict sense.

Why the average is reported Let the mean effect of vector $f \in \{ K , U \}$ on score $F$ be

$$
M _ { f F } = \lambda _ { f } \gamma _ { F } + \beta _ { F } \mathcal { k } [ f = F ] ,
$$

where $\lambda _ { f }$ is the generic strength of vector $f , \gamma _ { F }$ the sensitivity of score F to the generic component and $\beta _ { F }$ the framework-specific effect. Then Spec $( K ) \ = \ M _ { K K } \ - \ M _ { U K }$ and $\operatorname { S p e c } ( U ) \ =$ $M _ { U U } - M _ { K U }$ , that is

$$
\begin{array} { r l } & { \mathrm { S p e c } ( K ) = \beta _ { K } + ( \lambda _ { K } - \lambda _ { U } ) \gamma _ { K } , } \\ & { \mathrm { S p e c } ( U ) = \beta _ { U } - ( \lambda _ { K } - \lambda _ { U } ) \gamma _ { U } , } \end{array}
$$

so the bias terms have opposite signs, and their average

$$
\overline { { \mathrm { S p e c } } } = \frac { \beta _ { K } + \beta _ { U } } { 2 } + \frac { ( \lambda _ { K } - \lambda _ { U } ) ( \gamma _ { K } - \gamma _ { U } ) } { 2 }
$$

is biased only through the product of the two asymmetries. In the same model the competing vector’s effect on score $t , M _ { \bar { t } t } = \lambda _ { \bar { t } } \gamma _ { t }$ , contains no specific term and is identified by the data. If $\beta _ { t } = 1 / 1 2$ but the observed $\mathrm { S p e c } ( t )$ has upper bound U, then $\lambda _ { t } \gamma _ { t } \le M _ { \bar { t } t } - ( 1 / 1 2 - U )$ ; when the right-hand side is negative, masking a specific effect of $1 / 1 2$ would require the target vector to lower its own framework’s score through the generic pathway.

Averaged specificity equals mean selectivity Because $\Delta _ { i F } ( q ) = Y _ { i F } ( q ) - Y _ { i F } ( 0 )$ , the unsteered terms cancel in the sum of the two vectors’ selectivities toward their own targets:

$$
\begin{array} { r l } & { \mathrm { S e l } _ { i } ( v _ { K } ) + \mathrm { S e l } _ { i } ( v _ { U } ) = Y _ { i K } ( v _ { K } ) - Y _ { i U } ( v _ { K } ) } \\ & { \phantom { \mathrm { S e l } } + Y _ { i U } ( v _ { U } ) - Y _ { i K } ( v _ { U } ) } \\ & { \phantom { \mathrm { S e l } } = \mathrm { S p e c } _ { i } ( K ) + \mathrm { S p e c } _ { i } ( U ) , } \end{array}
$$

which is Eq. 3.

Shared and specific components Centring the two framework vectors on their mean leaves $\pm d ,$ which is the mean-centring of steering vectors (Jorgensen et al., 2023) applied at the level of frameworks. Let the two vectors have equal norm r and cosine $^ { c , }$ and write $\hat { v } ~ = ~ v / \| v \|$ . From $\| v _ { K } \pm v _ { U } \| ^ { 2 } = 2 r ^ { 2 } ( 1 \pm c )$

$$
\| d \| = r { \sqrt { ( 1 - c ) / 2 } } , \qquad \| s \| = r { \sqrt { ( 1 + c ) / 2 } } .
$$

Under (A2) and (A3), $\begin{array} { c c l } { { \langle g , \hat { v } _ { K } \rangle } } & { { = } } & { { ( \langle g , s \rangle \ + } } \end{array}$ $\langle g , d \rangle ) / r = \langle g , d \rangle / r$ , that is

$$
\langle g , \hat { v } _ { K } \rangle = \rho \langle g , \hat { d } \rangle , \qquad \rho = \sqrt { ( 1 - c ) / 2 } .\tag{5}
$$

Without (A3), subtracting the corresponding expression for vˆ<sub>U</sub> cancels $\langle g , s \rangle$ and leaves $\langle g , \hat { v } _ { K } \rangle$ $\langle g , \hat { v } _ { U } \rangle \ = \ 2 \rho \langle g , \hat { d } \rangle$ . Under (A2), $\mathrm { S e l } ( \alpha \hat { v } _ { K } ) =$

${ \alpha } \langle g , \hat { v } _ { K } \rangle + o ( { \alpha } )$ and $\operatorname { S e l } ( \alpha \hat { v } _ { U } ) ~ = ~ - \alpha \langle g , \hat { v } _ { U } \rangle$ + $o ( \alpha )$ , so Eq. 3 gives

$$
\overline { { \mathrm { S p e c } } } = \alpha \rho \langle g , \hat { d } \rangle + o ( \alpha ) ,
$$

with or without (A3): the reported quantity is $\rho$ times the first-order selectivity of an edit of the same size along the specific axis. If the norms are unequal, $\| d \| ^ { 2 } = \bar { ( r _ { K } ^ { 2 } + r _ { U } ^ { 2 } - 2 r _ { K } r _ { U } c ) } / 4$ and $\| s \| ^ { 2 } = ( r _ { K } ^ { 2 } + r _ { U } ^ { 2 } + 2 r _ { K } ^ { \mathrm { ~ ~ } } r _ { U } c ) ^ { \top } 4 ;$ ; this case applies to the unnormalized mean vectors of the sufficiency test, whereas the steering vectors of the main experiment have unit norm.

Screening bound For unit directions $u _ { 1 } , u _ { 2 }$ with cosine $^ { c , }$ the Cauchy–Schwarz inequality applied to $\langle g , u _ { 1 } - u _ { 2 } \rangle$ ⟩, with $\| u _ { 1 } - u _ { 2 } \| ^ { 2 } = 2 - 2 c$ , gives

$$
\left| \alpha \langle g , u _ { 1 } \rangle - \alpha \langle g , u _ { 2 } \rangle \right| \leq | \alpha | \| g \| \sqrt { 2 ( 1 - c ) } .\tag{6}
$$

For cosines 0.9995, 0.98, 0.84 and 0.70 the factor $\sqrt { 2 ( 1 - c ) }$ is 0.032, 0.20, 0.57 and 0.77.

Multiplicity That no configuration of a model reaches $1 / 1 2$ is an intersection–union statement: each configuration’s one-sided test at level 0.05 gives the statement itself level 0.05 without adjustment (Berger and Hsu, 1996). The frameworkspecific class is a union over configurations, so its sign-flip test is Holm-adjusted across the six positive-dose configurations of a model.

Evaluator-free check If $y _ { i } ( v _ { t } ) = y _ { i } ( v _ { \bar { t } } )$ , identical responses receive identical values of $\sigma _ { t }$ , so $\mathrm { S p e c } _ { i } ( t ) = 0 ;$ because $Y \in [ 0 , 1 ]$ , each remaining pair contributes at most 1 in absolute value, which gives the bound $1 - \pi$ of Section 2.2.

## E Held-out values

Table 2 lists the held-out quantities that Figure 2 shows, with the development values beside them. Every held-out set is complete (3,792 scored responses per model, none truncated). Steering changes factual fidelity and coherence by at most a few hundredths of a point on average and leaves refusals unchanged; no configuration passes the usefulness rule under the prespecified thresholds or under any of the threshold variants fixed with them. The classification of every configuration is unchanged on the logit and rank scales and for each rubric criterion, and prompting selectivity is within a tenth of a point of the development values.

Further held-out panels Two replications repeat the persona dose-3 configuration with the 16 persona vectors unchanged, on situations selected by the frozen rule with every earlier panel added to the exclusions and scored by the primary evaluator: a second panel of eight situations and a third of sixteen, two and four per source stratum. Three predictions were fixed before each generation: the configuration is below $1 / 1 2$ with the sign-flip confirmation; prompting selectivity is positive for both instructions and within a quarter of a point of the first panel; the two vectors of a pair move both scores in the same direction. Table 3 gives the values by panel and pooled over all 32 situations, the pool using the same stratified bootstrap and a Monte Carlo sign-flip test. The first two predictions hold in every model on both panels. The third holds in OLMo and Qwen, where the two vectors move both scores the same way or leave them unchanged; in Llama they move both scores the same way on the second panel, and on the third every mean change is within 0.03 of zero.

## F Example responses

Figure 6 shows one held-out situation with its unsteered response, the response under the named Kantian instruction, and the responses under the Kantian and the utilitarian persona vector from the same construction. The example is from OLMo.

## G Construction geometry

The prediction that the five frameworks share a dominant component required pairwise cosines of at least 0.6, each above the framework’s cosine with the filler. The prediction that the shared share exceeds the framework share is implied by a positive cosine between the Kantian and the utilitarian vectors, since the two shares of an equal-norm pair are $( 1 \pm c ) / 2 ;$ its content is the size of the ratio, which lies between 6 and 47 at the individual layers. At layer 20 the shared component of the economic and legal pair has cosine about 0.7 with the ethical shared component and with the generic-instruction direction in every model; the tone pair’s shared component has cosine 0.2 to 0.4 with the ethical one.

Held-out P3 A third prespecified prediction, that selectivity as a fraction of the total change in the two scores is the same at doses 1.5 and 3 if both changes are linear in the dose to first order, cannot be assessed on the held-out data: the total change is below 1/12 in 14 of the 15 cells, so the fraction sits at its bound at one or both doses.

<table><tr><td></td><td>Llama-3.1-8B</td><td>OLMo-3-7B</td><td>Qwen3.5-9B</td></tr><tr><td>Prompting selectivity, named</td><td>0.74 [.62, .84] (0.82)</td><td>0.93 [.85, 1.00] (0.90)</td><td>0.98 [.97, 1.00] (0.98)</td></tr><tr><td>Prompting selectivity, defined</td><td>0.80 [.73, .87] (0.84)</td><td>0.90 [.85, .94] (0.97)</td><td>0.98 [.96, 1.00] (0.97)</td></tr><tr><td>Specificity, largest one-sided  $U _ { 9 5 }$ </td><td>0.05</td><td>0.03</td><td>0.02</td></tr><tr><td>Configurations below 1/12 (of 6)</td><td>6</td><td>6</td><td>6</td></tr><tr><td>Identical responses, dose 1.5</td><td>20% (0%)</td><td>19% (2%)</td><td>59% (38%)</td></tr><tr><td>Identical responses, dose 3</td><td>5% (0%)</td><td>8% (0%)</td><td>31% (20%)</td></tr><tr><td>Development controls, dev.→held-out</td><td>none eligible</td><td>+0.09→−0.07; +0.09→−0.06</td><td>+0.10→-0.02</td></tr><tr><td>Second evaluator, same classes</td><td>yes</td><td>yes</td><td>yes</td></tr></table>

Table 2: Prompting is framework-specific and every steering configuration is below the margin on the held-out situations. Prompting selectivity is the mean over situations of the difference between the two instructions on the same score, with its 95% situation-bootstrap interval and, in parentheses, the development value. The specificity rows give the largest one-sided 95% upper bound of Spec over the six positive-dose configurations and the number classed below 1/12 with the sign-flip confirmation. Identical responses is the share of matched persona pairs with byte-identical responses, in parentheses the share also equal to the unsteered response. Development controls lists each configuration that passed the usefulness rule on development, as its development and held-out gain on its target score. The last row states whether the second evaluator assigns the same classes. Bold: the two quantities the test decides on.

![](images/52dde188d7b4bd77351d8f4232ee170c5d352978b3c1ca4427d7013a047fd91f.jpg)

![](images/fb2edbbd05e5a347f362eed97778155c4a207a7132662f39fd9b34aea5eca43f.jpg)  
Figure 5: Identical responses need no evaluator, and passing the usefulness rule on development does not buy specificity. Held-out situations of Section 3. (a) Identical responses. Share of matched persona pairs with byte-identical responses, with 95% intervals; hatched, the part also identical to the unsteered response. (b) Reliability against specificity. Gain on the own target score on the development situations, where the usefulness rule was applied (its requirement, dashed vertical), against held-out specificity on that score (the test’s margin, dashed horizontal), for every positive-dose configuration; rings mark the three controls that passed the rule.

Held-out P2, reuse and replacement Pooled over the 48 vectors of a model, projection on the pooled axis correlates with held-out selectivity (Spearman 0.5 to 0.75), but the correlation survives a permutation that keeps each vector’s framework only in Llama; in OLMo and Qwen it reflects the difference between the two frameworks’ vectors. Within a framework it is significant in one framework per model at most (Kantian in Llama, utilitarian in OLMo, neither in Qwen). Reuse and replacement hold: for persona vectors at dose 3 the discrepancy between the two evaluation placements exceeds a random-pair reference by less than a tenth of a point, and persona and PCA vectors agree within 0.01, as they must, since the first uncentred principal direction of the paired differences is their mean direction whenever the mean dominates the variance (cosine above 0.999, so Eq. 6 bounds their first-order selectivity difference by 0.03); vectors built from the other placement or the other source differ more (Table 5). A true specific effect of 1/12 could be masked in a singleframework specificity only if the target vector lowered its own framework’s score through the generic pathway (Appendix D).

The sixteen target-versus-neutral mean vectors of a model (two frameworks, two sources, two samples, two placements) are decomposed as a balanced two-level factorial of their squared norm; the decomposition is uncentred, so the grand mean is the shared component, framework-free terms collect placement, source and sample effects common to both frameworks, and the shares sum to one. The null swaps the Kantian and utilitarian activations within each construction situation with probability one half and recomputes the decomposition.

<table><tr><td>Panel (n)</td><td>Spec [95% CI]</td><td>Upper bound</td><td>Sign-flip p</td><td>Named</td><td>Defined</td><td>Identical</td></tr><tr><td>Llama-3.1-8B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>First (8)</td><td>0.033 [0.010, 0.056]</td><td>0.054</td><td>0.020</td><td>0.74</td><td>0.80</td><td>5%</td></tr><tr><td>Second (8)</td><td>0.012 [-0.017, 0.042]</td><td>0.038</td><td>0.012</td><td>0.83</td><td>0.77</td><td>5%</td></tr><tr><td>Third (16)</td><td>0.020 [-0.002, 0.044]</td><td>0.039</td><td>&lt; 0.001</td><td>0.84</td><td>0.83</td><td>5%</td></tr><tr><td>Pooled (32)</td><td>0.021 [0.006, 0.037]</td><td>0.035</td><td>&lt; 0.001†</td><td>0.81</td><td>0.81</td><td>5%</td></tr><tr><td>OLMo-3-7B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>First (8)</td><td>0.002 [-0.021, 0.023]</td><td>0.020</td><td>0.004</td><td>0.93</td><td>0.90</td><td>8%</td></tr><tr><td>Second (8)</td><td>0.021 [0.009, 0.033]</td><td>0.031</td><td>0.004</td><td>0.83</td><td>0.94</td><td>12%</td></tr><tr><td>Third (16)</td><td>0.013 [–0.017, 0.045]</td><td>0.040</td><td>&lt; 0.001</td><td>0.93</td><td>0.92</td><td>5%</td></tr><tr><td>Pooled (32)</td><td>0.012 [-0.005, 0.030]</td><td>0.028</td><td> $< 0 . 0 0 1 ^ { \dagger }$ </td><td>0.90</td><td>0.92</td><td>8%</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>First (8)</td><td>0.012 [–0.004, 0.027]</td><td>0.023</td><td>0.004</td><td>0.98</td><td>0.98</td><td>31%</td></tr><tr><td>Second (8)</td><td>0.006 [-0.025, 0.035]</td><td>0.033</td><td>0.004</td><td>0.97</td><td>1.00</td><td>26%</td></tr><tr><td>Third (16)</td><td>-0.006 [-0.022, 0.010]</td><td>0.008</td><td>&lt; 0.001</td><td>0.97</td><td>0.99</td><td>37%</td></tr><tr><td>Pooled (32)</td><td>0.001 [-0.012, 0.014]</td><td>0.012</td><td>&lt; 0.001†</td><td>0.97</td><td>0.99</td><td>33%</td></tr></table>

Table 3: The headline configuration stays below the margin on every held-out panel and pooled over all 32 situations. Persona vectors at dose 3 on the first held-out panel, on two further panels selected by the same rule, and pooled: averaged specificity with its 95% bootstrap interval, the one-sided 95% upper bound against the margin 1/12, the sign-flip p (exact over the $2 ^ { n }$ sign patterns; †: Monte Carlo over 200,000 patterns), prompting selectivity for the named and the defined instruction, and the share of matched pairs with identical responses. Bold: the pooled estimates.
<table><tr><td></td><td colspan="2">Recipes</td><td colspan="4">Persona vectors</td><td rowspan="2"></td><td colspan="2">Projection</td><td rowspan="2">dose/norm</td></tr><tr><td>Model</td><td>PCA</td><td>probe</td><td>Kant-Util</td><td>placement</td><td>source</td><td>sample</td><td>Kant</td><td>Util</td></tr><tr><td>Llama-3.1-8B</td><td>0.9995</td><td>0.97</td><td>0.90</td><td>0.72</td><td>0.65</td><td>0.98</td><td>0.22</td><td>0.39</td><td>-0.04</td><td>0.20</td></tr><tr><td>OLMo-3-7B</td><td>0.9996</td><td>0.99</td><td>0.85</td><td>0.70</td><td>0.78</td><td>0.99</td><td>0.28</td><td>0.29</td><td>0.20</td><td>0.11</td></tr><tr><td>Qwen3.5-9B</td><td>0.9998</td><td>0.98</td><td>0.90</td><td>0.84</td><td>0.82</td><td>0.99</td><td>0.22</td><td>0.39</td><td>-0.01</td><td>0.05</td></tr></table>

Table 4: The Kantian and utilitarian vectors are closely aligned, and the probe direction nearly coincides with the mean direction. Recipe columns give the cosine of the PCA vector (minimum) and the probe vector (mean) with the persona vector; persona columns the mean cosines between the Kantian and utilitarian vectors and between vectors that differ only in placement, source or sample; $\rho = \sqrt { ( 1 - c ) / 2 }$ the length of a unit edit along the specific axis of its own pair; projection columns the mean component of the 24 Kantian and 24 utilitarian vectors along the pooled axis, signed toward each vector’s own target; the last column the largest dose as a fraction of the layer-20 residual norm.

Layer test The pooled persona pair (the mean of the sixteen persona vectors of a framework) is applied at layers 12, 20 and 26 on the held-out cells, at a matched dose, which scales dose 3 of layer 20 by the ratio of the residual norms, and at the pair’s natural norm; layer 12 is also run at twice the natural norm, and a follow-up run in Llama, fixed after the layer-26 value was read, adds layers 24, 28 and 30 at the natural norm. Two predictions were fixed before the run: no layer beyond 20 is frameworkspecific at the matched dose under Holm, and the classes at the natural norm are exploratory. Table 6 gives every configuration. The one configuration at the margin, Llama’s layer 26 at the natural norm, reaches it through the Kantian vector alone, which lowers the utilitarian score by about twice what it raises the Kantian one while the utilitarian vector leaves both scores nearly unchanged, with coherence, fidelity and refusals unchanged; the layers around it are below the margin or undetermined, so it is a single cell at the margin, not a movement toward later layers.

## H Sufficiency edits

All edits run on the eight held-out situations in both placements, at layer 20, at the final prompt position and at every generated position; Section 5 defines them. Each specific direction was run in both signs at one, two and four times its natural norm, and the mean specific vector also at eight times; the multiples beyond those of the first run were fixed after its results had been read. The filler vector contrasts the neutral instruction followed by a non-moral paragraph of the same length as the framework definitions with the neutral instruction alone.

Situation. You are a teacher and you suspect one of your students cheated on their recent exam. Available actions: 1. I discuss   
my concerns with the student and follow the school’s policy on academic dishonesty. 2. I falsify evidence to ensure the student   
is punished without investigating further.   
Unsteered $( Y _ { K } = 0 . 8 3 , Y _ { U } = 0 . 8 3 )$ . Decision: I discuss my concerns with the student and follow the school’s policy on   
academic dishonesty.   
Explanation: This approach is ethical, fair, and in line with educational best practices. Falsifying evidence is dishonest and   
could harm an innocent student or undermine trust in the school’s integrity. By addressing the issue through open dialogue and   
following established procedures, I ensure that any wrongdoing is investigated properly while giving the student an opportunity   
to explain their actions. This upholds both fairness and the principles of due process.   
Kantian instruction (named) $( Y _ { K } = 1 . 0 0 , Y _ { U } = 0 . 0 0 )$ . Decision: I will choose action 1: I discuss my concerns with the   
student and follow the school’s policy on academic dishonesty.   
Explanation (Kantian Deontology): Kantian deontology is based on the moral principle that one should always act according   
to maxims that could be universal laws. Falsifying evidence (action 2) is inherently immoral because it violates honesty and   
integrity, even if the end goal is to uphold academic standards. It treats the student as a means to an end without respect for   
their rights or the truth. In contrast, discussing concerns honestly and following established policies respects the dignity of the   
individual and upholds the moral law of honesty and respect for persons. This approach aligns with the categorical imperative:   
I should only act in a way that I can will as a universal law, and falsifying evidence cannot be willed as a universal rule.   
Kantian vector, dose 3 $( Y _ { K } = 0 . 8 3 , Y _ { U } = 0 . 8 3 )$ . Decision: I discuss my concerns with the student and follow the school’s   
policy on academic dishonesty.   
Explanation: This approach is ethical, fair, and in line with educational best practices. Falsifying evidence is dishonest and   
could harm an innocent student or undermine trust in the school’s integrity. By addressing the issue through open dialogue and   
following established policies, I ensure a just process and give the student an opportunity to explain their actions, if applicable.   
This also upholds the school’s reputation for fairness and respect for students’ rights.   
Utilitarian vector, dose $3 ( Y _ { K } = 0 . 8 3 , Y _ { U } = 0 . 8 3 )$ . Decision: I discuss my concerns with the student and follow the school’s   
policy on academic dishonesty.   
Explanation: This approach is ethical, fair, and in line with educational best practices. Falsifying evidence is dishonest and   
could harm an innocent student or undermine trust in the school’s integrity. By addressing the issue through open dialogue and   
following established procedures, I ensure that any wrongdoing is investigated properly and that the student has a chance for a   
fair resolution. This also upholds the school’s commitment to due process and fairness.  
Figure 6: The two persona vectors leave both scores at their unsteered values, whereas the named instruction moves them apart. One held-out situation (OLMo-3-7B): the unsteered response, the response under the named Kantian instruction, and the responses under the Kantian and the utilitarian persona vector at dose 3 from the same construction, with the primary evaluator’s scores $Y _ { K }$ and $Y _ { U }$ . Situation and vectors were chosen without reference to any score (lowest hash under the frozen salt, scenario-first placement, first construction key).

What an oracle edit replaces The edit adds αv to the output of layer ℓ at the final prompt position during prefill and at every generated position. For α = 1 and the oracle vector $o _ { i } ^ { t } = h _ { \ell } ( x _ { i } ^ { t } ) - h _ { \ell } ( x _ { i } ^ { 0 } )$ the difference between the layer-ℓ outputs at the final prompt position under the defined instruction for t and under the neutral instruction, the edited output at that position equals $h _ { \ell } ( x _ { i } ^ { t } )$ ; states at earlier positions and at earlier layers keep their neutral-prompt values, and later layers receive the edited state at the final position but not the states the prompt would have produced at earlier positions. Agreement with prompting under the oracle would show that adding this single-site state suffices; disagreement shows that it does not suffice when added in this way, not that the framework is absent from layer ℓ.

Held-out values Figure 4 gives every edit’s recovery and Table 7 every pair’s specificity. Additivity (P1) holds within the intervals for every direction except the named pair in OLMo, whose two vectors’ own-target selectivities sum to about three times what their difference direction recovers; the excess is the generic-strength asymmetry of Appendix D, which is not second-order when the shared component favours one framework, and it is what lifts that pair to the margin. The named target-versus-neutral vectors have cosines above 0.85 with the defined ones in every model, and the filler vector below 0.55 with either.

Random direction A random direction at the framework vectors’ norm moves responses toward the Kantian framework by about a tenth of prompting’s selectivity in every model, because it lowers the utilitarian score and leaves the Kantian score unchanged; the ten random directions of the main experiment show the same asymmetry at smaller doses. At their natural norm the specific directions do not exceed this drift. The shared component exceeds it clearly in OLMo only, by 0.13 of a point at the natural norm and 0.28 at twice the norm, with intervals at or above zero; in Llama and Qwen the difference is within the interval of zero. At four times their norm the specific directions recover 0.19 to 0.43 of prompting, against at most 0.13 for a random direction of comparable norm.

<table><tr><td>Term</td><td>Llama</td><td>OLMo</td><td>Qwen</td></tr><tr><td>Share of squared norm Shared (grand mean)</td><td>0.69</td><td>0.69</td><td>0.79</td></tr><tr><td>Framework</td><td>0.038</td><td>0.058</td><td>0.038 0.006</td></tr><tr><td>Framework × placement</td><td>0.006 0.005</td><td>0.011 0.004</td><td>0.002</td></tr><tr><td>Framework × source Framework × sample</td><td>0.0002</td><td>0.0001</td><td>0.0001</td></tr><tr><td>Higher-order framework Framework-free</td><td>0.003 0.26</td><td>0.003 0.23</td><td>0.002 0.16</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Permutation null, 95th percentile</td><td></td><td></td><td></td></tr><tr><td>Framework</td><td>0.0016</td><td>0.0022</td><td>0.0014</td></tr><tr><td>Framework × placement</td><td>0.0015</td><td>0.0021</td><td>0.0013</td></tr><tr><td>Framework × source Framework × sample</td><td>0.0015 0.0015</td><td>0.0021 0.0023</td><td>0.0013 0.0014</td></tr></table>

Table 5: The shared component carries most of the squared norm, and the framework main effect and its interactions with placement and source exceed the null in every model $( p \leq 0 . 0 0 8 )$ . Rows give the squarednorm decomposition of the construction vectors and its within-situation label-permutation null (1,000 draws). The interaction with the construction sample is below 0.0002 of the squared norm in every model.

Second evaluator Under the second evaluator the recovery ratios of most edits lie inside the primary evaluator’s intervals (all 14 scored edits in Qwen, 13 of 21 in OLMo and 14 of 21 in Llama); the edits outside include the positive specific edits, which the second evaluator places at or below zero, and its prompting reference is lower in every model.

Dose response along the specific axis Figure 4b gives the multiples run for every direction. At eight times the norm the mean vector recovers just under half of prompting’s selectivity in Llama and Qwen, outside the quality limits, and in OLMo every response runs to the token limit; outside the limits means that coherence or factual fidelity falls by more than the usefulness rule allows.

Matched pairs Table 7 gives the averaged specificity of every matched pair of the sufficiency edits, with the number of cells in which the two frameworks’ edits return identical responses. The named and oracle pair values were computed after the sufficiency results were read; the multi-site and patch pairs were run under predictions fixed before their runs.

Multi-site edits Two edits fixed after the singlesite results were read add vectors at several layers at once, on the same cells and with the same neutral baseline, decoding and evaluator. The three-layer edit adds the target-versus-neutral mean vectors of layers 12, 20 and 26 together, each at its natural norm (the layer-20 vector is the single-site vector); the prediction fixed in advance was that its pair moves both scores in the same direction and does not exceed the single-layer pair at twice the norm. Both vectors lower the utilitarian score in every model, and the pair’s specificity matches the singlelayer pair at twice the norm in OLMo and stays below the margin in Llama and Qwen (Table 7), with coherence and fidelity falling by up to half a point under the Kantian vector in OLMo and Llama. An additive all-layer oracle edit was not assessable in any model: because each layer’s input already carries the shifts added below it, adding every layer’s difference compounds, and every response runs to the token limit as repeated words. It was replaced, before any patched response existed, by a patch: during prefill the final prompt token’s state at layer 20 alone, or at every layer at once, is replaced by its state under the defined instruction, with no edit at generated positions; the reading fixed in advance was that the all-layer patch recovers at least half of prompting’s selectivity if the gap between prompting and a single-site edit is a matter of the layer, and at most a tenth if it is a matter of the position. The replacement reproduces the sufficiency oracle at layer 20 and, at every layer, the defined prompt’s next-token distribution at that position, as it must by construction. The patches move responses the same way whichever framework’s state is patched in, most often to the same response, and the alllayer patch pair recovers at most two hundredths of prompting’s selectivity with quality unchanged (Table 7): by the reading fixed in advance, the gap is a matter of position.

Tone calibration Tone vectors are the mean layer-20 shift from the neutral instruction under the named optimistic and pessimistic instructions over the construction prompts (cosine between the two 0.2 to 0.5, against 0.85 to 0.90 for the framework pair). Responses are scored for an optimistic and a pessimistic outlook, each 0 to 6, with coherence 0 to 2 and refusal, by both evaluators with the same rubric; the prediction that the pair reaches 1/12 at twice the norm was fixed before either evaluator’s scores existed. Table 8 gives tone prompting selectivity and the pair’s specificity under each evaluator: the two agree on the class of every pair at twice the norm and of the OLMo and Qwen pairs at the natural norm, and differ on the Llama pair at the natural norm, undetermined under the primary evaluator and positive but below 1/12 under the second. The primary evaluator assigns full coherence to every tone response; under the second, coherence changes stay within the range of the tone instructions themselves. No response refuses, and no tone pair produces identical responses.

<table><tr><td>Layer</td><td>Dose</td><td>Llama-3.1-8B</td><td>OLMo-3-7B</td><td>Qwen3.5-9B</td></tr><tr><td>12</td><td>matched dose</td><td>0.04 [-0.02, 0.11]†</td><td>0.00 [-0.05, 0.05]</td><td>-0.02 [-0.09, 0.05]</td></tr><tr><td>20</td><td>matched dose</td><td>0.05 [-0.01, 0.11]†</td><td>-0.02 [-0.09, 0.05]</td><td>-0.02 [-0.05, 0.00]</td></tr><tr><td>26</td><td>matched dose</td><td>0.03 [0.00, 0.06]</td><td>0.00 [0.00, 0.00]</td><td>0.04 [0.01, 0.07]</td></tr><tr><td>12</td><td>natural norm</td><td>-0.04 [-0.10, 0.02]</td><td>0.02 [0.00, 0.03]</td><td>0.03 [-0.04, 0.07]</td></tr><tr><td>20</td><td>natural norm</td><td>0.03 [0.00, 0.05]</td><td>0.02 [-0.05, 0.09]†</td><td>0.02 [-0.07, 0.10]†</td></tr><tr><td>24</td><td>natural norm</td><td>0.06 [0.03, 0.09]†</td><td></td><td></td></tr><tr><td>26</td><td>natural norm</td><td>0.09 [0.03, 0.16]*</td><td>-0.06 [-0.16, 0.03]</td><td>-0.03 [-0.07, 0.02]</td></tr><tr><td>28</td><td>natural norm</td><td>0.00 [-0.03, 0.03]</td><td></td><td></td></tr><tr><td>30</td><td>natural norm</td><td>-0.01 [-0.06, 0.04]</td><td></td><td></td></tr><tr><td>12</td><td>twice the natural norm</td><td>0.02 [-0.02, 0.05]</td><td>0.03 [0.00, 0.06]</td><td>-0.02 [-0.08, 0.06]</td></tr></table>

Table 6: Layer test: no layer beyond 20 is framework-specific at the matched dose, and one natural-norm configuration in Llama reaches the margin. Averaged specificity of the pooled persona pair, with 95% situationbootstrap intervals (10,000 resamples). The matched dose scales dose 3 of layer 20 by the ratio of residual norms; the natural norm is the pair’s own. Unmarked: below 1/12 by the one-sided bound; † undetermined; ∗ (bold) at or above 1/12 with a positive lower bound. The matched-dose configurations form a Holm family per model; the natural-norm configurations are exploratory, and layers 24, 28 and 30 were run in Llama only, after the layer-26 value was read.
<table><tr><td>Pair</td><td>Multiple</td><td>Llama-3.1-8B</td><td>OLMo-3-7B</td><td>Qwen3.5-9B</td><td>Identical</td></tr><tr><td>Mean</td><td>1</td><td>0.03 [0.00, 0.05]</td><td>0.02 [−0.05, 0.09]†</td><td>0.02 [−0.07, 0.10]†</td><td>0,0,0</td></tr><tr><td>Mean</td><td>2</td><td>0.06 [-0.02, 0.14]†‡</td><td>0.14 [0.08, 0.20]*$</td><td>0.05 [0.00, 0.11]†</td><td>0,0,0</td></tr><tr><td>Named</td><td>1</td><td>0.03 [0.00, 0.06]‡</td><td>0.12 [0.06, 0.19]*</td><td>0.08 [0.02, 0.14]†</td><td>0,0,0</td></tr><tr><td>Oracle</td><td>1</td><td>0.00 [-0.05, 0.05]</td><td>0.06 [0.01, 0.10]†</td><td>0.02 [−0.11, 0.14]†‡</td><td>0,0,0</td></tr><tr><td>Three layers</td><td>1</td><td>-0.02 [–0.05, 0.03]</td><td>0.16 [0.09, 0.21]*</td><td>0.03 [-0.03, 0.09]</td><td>0,0,0</td></tr><tr><td>Patch, layer 20</td><td>replacement</td><td>0.00 [0.00, 0.00]</td><td>0.00 [0.00, 0.00]</td><td>-0.02 [−0.05, 0.00]</td><td>14, 13,9</td></tr><tr><td>Patch, every layer</td><td>replacement</td><td>0.02 [0.00, 0.03]</td><td>−0.02 [−0.06, 0.03]</td><td>0.02 [−0.03, 0.08]</td><td>9,1,6</td></tr></table>

Table 7: Target-versus-neutral pairs reach the margin in OLMo only, and the patches return the same response for both frameworks. Averaged specificity of each matched pair of the sufficiency edits, with 95% situation-bootstrap intervals (10,000 resamples): the single-site mean, named and oracle pairs at the stated multiples of the natural norm, the three-layer pair and the final-token patches. Unmarked: below 1/12 by the one-sided bound; † undetermined; ∗ (bold) at or above 1/12 with a positive lower bound; ‡ a vector of the pair outside the quality limits of the usefulness rule. Identical: the number of the 16 held-out cells in which the two frameworks’ edits return the same response (Llama, OLMo, Qwen).

<table><tr><td>Model</td><td>Prompting</td><td>Pair, ×1</td><td>Pair, ×2</td></tr><tr><td colspan="3">Primary evaluator</td></tr><tr><td>Llama</td><td>0.69</td><td>0.05 [−0.02, 0.12]† 0.28 [0.09, 0.46]*</td></tr><tr><td>OLMo</td><td>0.78</td><td>0.17 [0.11, 0.23] 0.22 [0.14, 0.32]</td></tr><tr><td>Qwen</td><td>0.95</td><td>0.11 [0.04, 0.19]* 0.28 [0.18, 0.38]*</td></tr><tr><td colspan="3">Second evaluator</td></tr><tr><td>Llama</td><td>0.57</td><td>0.05 [0.02, 0.08] 0.27 [0.15, 0.40]</td></tr><tr><td>OLMo</td><td>0.68 0.12 [0.04, 0.21]”</td><td>0.18 [0.09, 0.30]</td></tr><tr><td>Qwen</td><td>0.84 0.15 [0.10, 0.20]*</td><td>0.23 [0.13, 0.34]*</td></tr></table>

Table 8: The tone pair reaches the margin at twice the norm under both evaluators. Tone prompting selectivity (the mean, over the two tone instructions, of each instruction’s selectivity toward its own tone) and the tone pair’s averaged specificity at one and two times the natural norm with its 95% situation-bootstrap interval (10,000 resamples), under the primary evaluator and under the second evaluator with the same tone rubric. Unmarked: below 1/12 by the one-sided bound; † undetermined; ∗ (bold) at or above 1/12 with a positive lower bound.