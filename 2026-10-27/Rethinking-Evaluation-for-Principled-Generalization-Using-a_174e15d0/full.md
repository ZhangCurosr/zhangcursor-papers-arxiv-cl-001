# Rethinking Evaluation for Principled Generalization Using a Kripkean Challenge to Meaning as Use

Niyar R Barman Université de Toulouse niyar-r.barman@irit.fr

Omar Naim\* Université de Toulouse omar.naim.docs@gmail.com

Nicholas Asher\* Université de Toulouse nicholas.asher@irit.fr

## Abstract

Large language models operationalize an intuitive, view of meaning as use. We argue that it faces a rule-following challenge. Using Kripke’s interpretation of Wittgenstein, we argue that finite histories of use do not determine a unique rule for future extension. This reframes the debate about OOD generalization but also about NLP evaluation; benchmarks that measure performance on fixed use cases or local robustness, we argue, do not by themselves establish semantic competence. To evaluate whether models preserve rules and concepts under principled domain extension, we propose to evaluate two capacities, central to this ability: abstraction and recursion; and we offer an experimental paradigm for testing both.

## 1 Introduction: LLMs and skepticism

Large language models (LLMs) have made operational a powerful version of Wittgenstein (1953)’s meaning-as-use hypothesis: the patterns of an expression’s use fixes its meaning, not an independently specified rule or denotation. In an LLM, words, sentences and texts contribute content through the contexts in which they occur.

Exploiting this paradigm, LLMs have achieved broad linguistic fluency, substantial world knowledge, and strong task performance. Yet this success raises a foundational question: does mastery of observed patterns of word usage justify extending those patterns to genuinely novel cases? This question has a direct and general analogue for language models in the following question about out of distribution (OOD) generalization: does the mastery of a learned pattern, as observed on a finite OOD dataset, justify claiming that the model has mastered the pattern? On a purely meaning-as-use account, no further fact determines the correct extension to the OOD domain.

Kripke’s interpretation of Wittgenstein’s rulefollowing argument (Kripke, 1982) offers a strong negative answer to our question. The rule following problem addresses the question of how a finite history of use could determine one rule rather than another for future application. Kripke’s skeptical approach, illustrated with addition, is that such a finite history never determines a fully general rule. The skeptical point, set out in §2, is not that addition is mathematically invalid, but that finite use alone does not justify one extension of that use over another.

This negative answer has two important consequences for LLMs. First, it changes how we should interpret OOD generalization. With regard to LLMs, the finite training distribution plays the role of the finite history of use in Kripke’s argument: success on observed examples, or even on a finite set of OOD extensions, does not by itself establish that the model has acquired the rule governing further cases. Empirical failures under increasing distribution shift reinforce this observation. Second, this distinction has consequences for evaluation. If finite behavioral success does not by itself establish rule acquisition, then held-out accuracy or robustness under selected distribution shifts should not be treated as sufficient evidence of principled generalization. For rule-governed phenomena, evaluation should also ask whether a model can identify and reliably apply the relevant abstraction as the domain is systematically extended. This motivates evaluations that separate success on particular instances from the ability to form and apply rules or concepts beyond those instances.

While use and statistical observation contribute to semantic competence, the transition from finite observations to generally applicable concepts requires inferential leaps beyond the data. Two capacities are essential for this transition: abstraction and recursion. Abstraction forms general functions from concrete instances by replacing particular argument values with variables; once formed, these functions can be applied to novel inputs and composed into more complex structures. Recursion allows finite procedures to extend systematically over domains not bounded by the examples encountered during training.<sup>1</sup> These principles offer a key to improve NLP evaluation: models should not be evaluated only through benchmark improvement or ordinary robustness metrics, but also through tests of whether they acquire abstractions and recursive procedures that support stable rule and concept extension. They also provide a more fine-grained basis for evaluation: once a task is represented as a composition of functions, its constituent operations and their composition can be evaluated separately rather than only through final task accuracy.

We evaluate abstraction and recursion in rulegoverned settings where the intended extension can be specified independently. For abstraction, ASDiv and GSM8K allow us to distinguish direct problem solving from forming an abstract function and applying it to new instances, complemented by rule-application and composition diagnostics. For recursion, we test whether a fixed procedure remains reliable as the required depth of application grows.

This position paper with empirical diagnostics makes three contributions. First, we apply Kripke’s rule-following argument to OOD evaluation, distinguishing successful performance on finite extensions from evidence that a model has acquired the rule governing further cases. Second, we argue that mastering abstraction and recursion yield a form of principled generalization beyond good ID performance. Third, we introduce an empirical paradigm for evaluating abstraction and recursion on models, distinguishing direct task performance, formation of an abstract function, its application to novel instances, the composition of abstract functions and finally the iteration of a simple function for addition. We test 10 models of various sizes on simple arithmetic functions, geometric concepts, and an NLP reasoning benchmark.

## 2 A Kripkean Challenge

Kripke’s interpretation of Wittgenstein’s private language argument (Kripkenstein) provides a strong argument against the distributional model of meaning that underlies LLM training. Here is the argument with some objections and replies.

## 2.1 The rule-following argument

Suppose that up until now you have never added numbers n, m such that n, m ≥ 57. You are now presented with the problem of 68 plus 57. Suppose you apply the function + you have gathered from prior experience to arrive at 125.

The skeptic replies:

There is no fact about your past usage of the addition function that determines 125 as the right answer. Nothing justifies you in giving this answer rather than another.

As you have never added numbers $\geq 5 7$ , it’s consistent with all your prior usage of plus that + has the meaning of quus:

$$
x { \mathrm { ~ q u u s ~ } } y = { \left\{ \begin{array} { l l } { x + y } & { { \mathrm { f o r ~ } } x , y < 5 7 } \\ { } \\ { 5 } & { { \mathrm { f o r ~ } } x \geq 5 7 { \mathrm { ~ o r ~ } } y \geq 5 7 } \end{array} \right. }
$$

Your past usage of the addition function is susceptible to an infinite number of different quus-like interpretations. Kripke doesn’t question the mathematical validity of the "+" function, but rather the meta-linguistic usage of "plus": what fact can we point to that shows that "plus" refers to the mathematical function "+"?

This argument applies to any linguistic expression. According to Kripkenstein, any semantic concept linked to a linguistic expression in the “meaning as use” theory is infinite because it is based on a notion of universal quantification over future uses. Therefore, it cannot be captured by a model based solely on finite, past uses.

With regard to OOD generalization, mastery of patterns over a finite sampling of a training distribution can capture task essential information. Our claim is that encoding such information is not enough to answer the Kripkean challenge. Meaning-as-use can explain why a model behaves correctly in contexts close to its training distribution, but it does not by itself justify why the same expression should retain the same meaning under unbounded out-of-distribution extension.

## 2.2 Objections and replies

Why addition and quantification? Kripke’s rulefollowing argument uses addition as a canonical example of a concept with a clear domain extension with a constant underlying operation. We use addition and universal quantification as controlled testbeds for our evaluation methodology. Universal quantification provides a linguistic counterpart in which the same condition must be maintained over increasingly long sequences. Because both admit clear reference procedures, they allow us to examine abstraction, application, and iterative execution without conflating these with ambiguity in the task itself. To apply these lessons in natural language settings, we apply these evaluation principles on the ASDiv, GSM8K data sets.

Types and dispositions. For a transformer, the training is always over types of contexts of use because of limits on context size. We can think of the objective function as determining a probability of use of token m over a finite set of types of contexts. Like a disposition (McGinn, 1984), the objective function would support counterfactual reasoning and apply to arbitrary new instances.

However, appealing to dispositions invites a familiar regress argument: how do you know that you have a plus disposition rather than a quus disposition? Similarly, we can estimate very well probabilities on an unbounded domain, if the "unbounded" test domain obeys the same distributional patterns as those revealed by sampling from the training domain. But the skeptic can reply: what justifies you in assuming that the unbounded test domain has the same distribution as the sampling of the training domain? Dispositions or probability distributions offer us no reason to believe they extend in the right way beyond the domain on which they have been formed.

Inductive biases. We can add inductive biases to have the model learn the correct rule. But where are these biases added? (Pérez et al., 2021) show that transformers are Turing complete, so models can be programmed with computable procedures. But this doesn’t give the models rule following ability in general; nor does it show they can learn new rule-following behavior.

Learning in the limit. While finite histories do not determine general rules, “learning in the limit” (Osherson and Weinstein, 1986), which studies learning in extending finite samples, may justify learning of general rules. We show (Appendix A)

that this does not answer Kripke’s critique.

Shared practice. Kripke’s own answer is that an extension is correct because it survives a community’s practice of correction, not an inner fact. This bears on our case asymmetrically: a speaker is held to a rule by an ongoing practice, while a model at test time faces only a benchmark, a frozen record of judgments about fixed cases. And the practice fixes which extension counts as correct without telling us what capacity produces it.

## 3 Related work

Whether models use causal language modeling or another training objective, they acquire linguistic competence from patterns in corpora. In this sense, contemporary NLP gives computational form to the Wittgensteinian meaning as use slogan. In contrast, turning meaning as use into a formal, linguistic theory of meaning has had difficulty, especially incapturing meaning’s intrinsic link to reference, truth and validity (Davidson, 1967). Dummett (1993) develops a semantics linking use and truth but only for the propositional connectives; going further becomes complicated (Brandom, 1998; Asher, 2011).

Kripkenstein is distinct from the more general Humean skeptical attack on induction. Kripke’s argument focuses specifically on the meaning as use theory. Many empiricists have also worried that finite patterns of use or stimulus meaning fail to ground the content of abstract rule governed expressions (Quine, 1960; Carnap, 1950). Kripkenstein also differs from critiques involving external world grounding (Bender and Koller, 2020; Bisk et al., 2020) and faithfulness tests (Chaturvedi et al., 2024), according to which the semantic content of an input should figure in a model’s inferences. Our issue is whether patterns learned from finite uses justify a stable extension of a rule or concept to cases outside the observed distribution. A model may encode useful semantic regularities and have in distribution expertise, while failing to possess the abstraction needed to preserve those regularities under principled domain extension.

Our use of Kripke’s argument also differs from older debates about rules and connectionism (Rumelhart and McClelland, 1986; Pinker and Prince, 1988). They advance a kind of faithfulness argument and claim that networks don’t represent the right sort of rules (with the target rules determined by linguistic research); the Kripkean argument concerns what determines the target. The question recurs for LLMs, which have been probed for abstract structural representations (Sinclair et al., 2022) and for the handling of recursively nested structure (Lampinen, 2024) (again the rule tested is fixed in advance). These are particular rules, but we are interested in general rulefollowing ability and the ability to generalize itself.

Concerning generalization principles, abstraction is an important Aristotelian, cognitive principle (See de Anima III.4, Metaphysics I.1, Physics I.1.), further developed by Avicenna, Averroes and Saint Thomas Aquinas.

Work on transformers under distribution shift has shown that performance deteriorates sharply when inference-time inputs differ from the training distribution (Xie et al., 2021; McCoy et al., 2024). Such failures are not always reducible to overfitting or memorization (Naim et al., 2025b) but are associated with autoregressive training and architectural mechanisms. Failure of generalization of arithmetic competence to larger digits is also well documented Nye et al. (2021); Muffo et al. (2022); Qian et al. (2022); Yan et al. (2025). Compositional benchmarks such as SCAN (Lake and Baroni, 2018), COGS (Kim and Linzen, 2020), and CFQ (Keysers et al., 2020) examine a model’s ability to generalize semantic structures and observe substantial performance drop on OOD inputs.

Other work characterizes such shifts systematically. Hupkes et al. (2023) propose a taxonomy of generalization research in NLP organized by the motivation, type, and locus of the shift,. Questions about the in-distribution/out-of-distribution boundary are raised in the GPT-2 and GPT-3 reports (Radford et al., 2019; Brown et al., 2020). Arithmetic experiments with digit length shifts (Cho et al., 2024)ny provide a structural shift that holds the rule fixed and varies only the size of its arguments.

Recent work shows that arithmetic length generalization depends strongly on positional representations, data formatting, and training conditions (McLeish et al., 2024; Zhou et al., 2026). Studies of structural recursion and shortest-path reasoning distinguish learning a rule from reliably applying it to more complex inputs (Zhang et al., 2024; Tong et al., 2026). Our experiments complement these approaches by separately evaluating whether models can produce an abstract expression and correctly apply it to new values. This provides a more finegrained diagnostic of rule application.

Prior work (Sinclair et al., 2022; Lampinen, 2024; Qiu et al., 2022) have looked at rule mastery of of particular general rules to unseen data that are fixed in advance. We argue that this isn’t sufficient; this just shows a mastery of another finite extension of uses. Building on work in neuroscience (Pouget et al., 2000) and interpretability(Olsson et al., 2022), Fel et al. (2026); Karkada et al. (2026); Bhalla et al. (2026) provide evidence that models might learn something close to a parameterized concept amenable to abstraction.

## 4 Lessons from Kripkenstein: OOD Generalization as Rule-Following Failure

We draw two lessons from the Kripkenstein argument. First, Kripkenstein provides a rational explanation for why we should expect simple testing on OOD experiments, augmented synthetic data such as that in (Qiu et al., 2022), to fail to isolate rule-following behavior; these are simply extensions of the finite training models have already had. Moreover, some NLP benchmarks labeled as OOD may not actually probe substantial distribution shifts. Experiments on controlled numerical settings (Naim et al., 2025a) show that small transformers can perform well in-distribution and remain robust to small OOD shifts, while exhibiting a sharp degradation as the distribution moves further away from the training distribution. If commonly used OOD benchmarks lie too close to the training distribution, then strong performance on them may be misleading: what appears to be OOD generalization may simply reflect robustness to a small distribution shift. This raises a fundamental question: how can we design benchmarks that genuinely test extrapolation beyond the training distribution, rather than robustness to near-ID shifts?

The second lesson of Kripkenstein is that we should not expect models to have learned rulefollowing behavior from finite usage. Rulefollowing behavior on a task should extend naturally to OOD data or at least have a graceful degradation due on it due to representation issues, memory, integer precision—in short performance issues. However, prior research has isolated a clifflike failure on OOD inputs (Naim et al., 2025a). Small GPT-style models trained on linear and polynomial regression fail under significant distribution shift (Naim et al., 2025a), as do pretrained and finetuned LLMs on quantification (Naim et al., 2026).

The cliff-like collapse on far OOD tests reoccurs with arithmetic. We use arithmetic as a controlled illustration of the rule-following problem for linguistic expressions with a clear, formally specifiable meaning. Addition has a clear rule, finite training instances, and has, intuitive, principled extensions to larger numbers. Increasing the number of digits changes the size of the input, but not the rule to be applied. Thus, if a model has acquired the addition rule, we should expect it to extend beyond the digit lengths observed during training. If the model has instead learned distributionbound regularities, performance may remain high in-distribution but will collapse on longer inputs as the plus/quus problem predicts.

![](images/a1ad5181e52cfc32c6f3e66877b6c605f6c0f46e8ef9374c910294c38bfd9b3d.jpg)  
Figure 1: Exact match on addition vs. answer length relative to each model’s maximum training operand length n. All models are near ceiling in distribution but collapse beyond it.

Given the collapse of performance (the performance boundary cliff), it is natural to attempt to induce some sort of bias into the model to improve OOD performance. In the Appendix E, we examine 6 ways given in the literature. Many extend OOD performance, some to a significant degree. But they do not eliminate the cliff-like collapse of accuracy.

## 5 An empiricist response: two principles

While the experiments on arithmetic in Appendix B go beyond pure inductive learning and instill various biases in the model, the results suggest that neither carefully targeted data for particular rules, favorable representations, explicit rule descriptions, nor ordinary benchmark success suffice to establish rule-like semantic generalization in models. Our approach is to provide models with the basic principles for forming general rules: abstraction, composition and recursion.

Abstraction and composition. Abstraction is the capacity to take a complex concrete structure and replace certain details with variables or simply get rid of them altogether In logic, abstraction is a formal operation on formulas ϕ(x) with x free that yields a term λxϕ denoting a function from individuals into formula denotations. If we take rules to be functions, then principles of combining concepts become simply the operation of mathematical function composition.

Recursion A second principle is recursion (Fibonacci, 1202), which gives us infinite sets. This allows us to extend generalization of some function’s values to hypotheses in the limit using sequences : $l i m _ { n  \infty } f ( n ) , l i m _ { n  \omega } f ( 1 / n )$ . For instance,Cusanus (1440) asks us to visualize a circle and determine what happens to the degree of curvature as the radius increases in size. Cusanus abstracts a single quality of a physical object, and then lets the value of that abstracted element tend toward infinity, at which point he says that the circle now "collapses" to a line and the curvature is 0. Lonergan (1958) provides other intuitive examples.

Quantification, with which models have difficulty (Appendix K), offers a simple example of how abstraction and application can work. Suppose we observe a concrete situation in which $a \subseteq b$ Let’s suppose in addition that a and b are parameters in the model’s representation for this concept. What does abstraction add? Abstraction replaces the particular sets a, b with variables P, Q and transforms a concrete fact representation into the function, λP.λQ. $P \subseteq Q$ . This gives a schema defining generalized-quantifier meaning of every: λP.λQ. every $P , Q ) \iff P \subseteq Q$ . Linking this schema to the semi-decision procedure of Section B.6 and Appendix Eprovides a content for universal claims that goes beyond that given by any finite set of observed instances. Similarly, abstraction over concrete facts about the successor function, $1 = s ( 0 ) , 2 = s ( 1 )$ , and so together with our quantification definition allow us to axiomatize the successor function: ∀x, $y ( s ( x ) \neq 0 \land ( s ( x ) =$ $s ( y ) \to x = y ) )$ . Recursion defines the natural numbers N as the smallest set containing 0 and closed under successor. We can now define algorithms such as column-wise or abacus-style addition over N.

In sum, abstraction and recursion provide these steps to general knowledge: (1) perception of concrete situations and statistical regularities; (2) selective attention to one or more properties or elements P; (3) abstraction over P, replacing P with variables to form a general schema or function f; and (4) investigation of how f behaves as its variables are systematically varied, including limiting cases. Going beyond mere induction over observed cases, abstraction and recursion provide concepts whose range of application can exceed the finite data that motivated them.

## 6 How to evaluate abstraction capability

To evaluate a model’s ability to do abstraction, we need to test whether a model can replace particular quantities with variables, preserve their relationships, and apply the resulting function to new values. We do this by supplying math problema expressed in natural language from GSM8K (Cobbe et al., 2021).We used 214 GSM8K problems paired with a variant that preserves the originals functional structure while changing the scenario and quantities. We ask the model both to abstract the general function from the text and to apply it to new numerical values in the variant.

In the first "abstraction" step, the model gets a source problem as in Table 1. We also provide a quantity-to-variable format—in the example a = 16, $b = 3 , c = 4 .$ , and $d = 2 .$ . This supplies the variables over which the model must identify the right function. We ask the model to produce the expression for the solution and check if it retrieves the function $f ( a , b , c , d ) = ( a - b - c ) d$

Source   
Janet produces 16 eggs per day, eats 3, uses 4 for baking, and   
sells the remainder for \$2 each. What is her daily revenue?   
Target Eval 1   
A laboratory produces 347 ceramic blanks per day, reserves   
89 for testing, uses 53 for calibration, and sells the remainder   
for \$17 each. What is its daily revenue?   
Abstraction Eval 2   
$f ( a , b , c , d ) = ( a - b - c ) d$   
Transfer Eval 3   
$f ( 3 4 7 , 5 3 , 8 9 , 1 7 ) = ( 3 4 7 - 5 3 - 8 9 ) \times 1 7 = 3 4 8 5$  
Table 1: Illustrative GSM8K transfer example. Eval 1 measures direct target solving, Eval 2 abstraction, and Eval 3 transfer through the induced abstraction.

The second step is to see if the model can apply the general pattern function to new data. We give the model its generated expression and ask it to use that to solve the paired problem. The setting and quantities have changed, but the relationship between them is the same. This tests whether the model understands that what it has produced is a function that can be applied to arbitrary inputs of the appropriate type. Our experimental design differs from prior diagnostics on compositional generalization (Lake and Baroni, 2018; Keysers et al., 2020; Kim and Linzen, 2020), who use abstraction to generate test data for the model. We are evaluating whether the model can learn how to use abstraction and application to find and use generalizations.

Prior to evaluating on full GSM8K problems. we evaluate abstraction on 190 ASDiv addition and subtraction problems (Miao et al., 2020), comparing numerical-answer accuracy with symbolicexpression accuracy.

In a second experiment, we test several geometrical mathematical concepts. We construct 500 zero-shot geometry problems in ten families (e.g., collinearity, coplanarity, concyclicity, unitsphere and convex-hull membership, and affine maps $y = M x + b )$ , with 50 problems per family balanced between yes and no answers. We isolate the application step by supplying the rule explicitly. Each prompt states the general criterion (e.g., $P _ { 1 } , P _ { 2 } , P _ { 3 }$ are collinear iff $P _ { 3 } - P _ { 1 } = t ( P _ { 2 } - P _ { 1 } ) )$ gives exact rational inputs, and asks the model to show its substitution and calculation before answering. In the two n-dimensional families, the ambient dimension d ranges from 4 to 4096 while the rule stays fixed.

To test recursion, we supply a rule and ask the model to apply it to its own output. Each prompt gives two n-digit integers and a tail-recursive addition rule over registers $( A , B , C , O )$ : each call adds the last digits of A and B and the carry C, prepends the result digit to the output string O, and recurses on A and B without their last digits until $A = B = 0$ (Appendix K). Only the number of calls, n, varies. We use 100 pairs for each $n \in \{ 5 , 1 0 , . . . , 3 5 \}$ , ask for every call state and then the sum, and score the sum (EM) and the state sequence (verified EM).

## 7 Results

Results for function abstraction from simple arithmetic statements Figure 2 focuses on the two 8B models. In the zero-shot condition, the originalto-abstracted score gap is 25.3 percentage points for Qwen3-8B and 31.1 points for Llama-3.1-8B-Instruct. Five-shot reduces these gaps to 15.3 and 25.3 points, respectively. Thus, even for these elementary operations, high numerical accuracy does not imply equally reliable symbolic answers.

![](images/0e77dcb525a5b58f98758ac87eea13262f2ce2dca4a61a273e75813177ebf3d6.jpg)  
Figure 2: Original and abstracted ASDiv problems restricted to addition and subtraction (n = 190 per condition). Original scores measure numerical-answer accuracy; abstracted scores measure formula exact match.

Results for function abstraction from 214 GSM8K problems and variants We evaluate direct target solving (Eval 1), source abstraction (Eval 2), and application of the model’s generated expression to the target quantities (Eval 3). The application stage uses the model’s own expression rather than the reference expression. Figure 3 reports the baseline condition. Qwen3-32B and Qwen3-8B achieve similar direct-solving accuracies of 91.6% and 90.2%, respectively. Their transfer accuracies, however, are 86.9% and 72.0%. The difference between the models therefore increases from 1.4 percentage points on direct solving to 15.0 points on transfer. On these GSM8K-derived problems, direct-answer accuracy conceals a substantial difference in the ability to produce and apply an abstraction.

![](images/a15ff55c70db624852bad8ae5db1455ebcbe35bdb7acc802a1a26333223b25e4.jpg)  
Figure 3: Direct solving, abstraction, and transfer in the baseline condition on 214 matched GSM8K problem and variant pairs. The two models obtain similar direct-solving scores but differ substantially in transfer accuracy.

Results for abstraction application under numerical scaling for GSM8K. We also examine whether an abstracted function can be applied reliably as its inputs change. Using the same 214 paired problems, we compare four conditions: baseline, scaled quantities,fractional perturbation, and extended scale. These conditions vary the magnitude and precision of target quantities while preserving the source problems and reference expressions. Appendix G specifies the numerical ranges and perturbations and reports results for all models.

Figure 4 shows declining transfer accuracy as numerical conditions change. From baseline to extended scale, Qwen3-32B falls from 86.9% to 43.5%, and Qwen3-8B from 72.0% to 30.4%. At extended scale, Llama-3.1-70B-Instruct achieves only 2.3% transfer accuracy and OLMo-2-0325- 32B-Instruct achieves 0.0%. Source-abstraction accuracy remains comparatively stable because source problems are unchanged

![](images/115978c7abe0b654fddc2ce4e188655d59a0be4f7d158499e54ae2c8226577a2.jpg)  
Figure 4: Direct-solving and transfer accuracy (%) for ten models on 214 matched GSM8K problem and variant pairs.

Correct abstraction does not ensure correct application. Qwen3-32B answers 181 of 191 correctly abstracted problems correctly at baseline, compared with 90 of 191 at extended scale. An abstraction seeks to specify a function beyond its observed instances; applying it to new values therefore tests rule following more directly than producing its symbolic form alone. Transfer makes this distinction empirically assessable. Complete results and evaluation details are in Appendix G.

![](images/c7a2b8f3f3563ec3fa779e4f97d8e6190e3366995a3207d8dc73d81a89068620.jpg)

Results evaluating geometrical concepts given an explicit rule In this experiment, we examined how models used a given formula to check for collinearity, coplanarity, and unit sphere membership. Given that final answers were binary (yes/no) in Figure 5a, most models performed poorly: Llama-3.1-70B-Instruct achieved 69.0% verified accuracy, OLMo-2-0325-32B-Instruct 17.6%, and Qwen3-0.6B 8.6%. Only Qwen3-8B and Qwen3- 32B produced sound derivations for nearly all their correct answers. Correct answers therefore underdetermine whether the stated rule was applied, even when the rule itself is given.

The gap widens as the rule is extended to higher dimensions (Figure 5b). The criterion does not change with d, yet from d = 4 to d = 4096 as EM falls from 95% to 55% for Llama-3.1- 70B-Instruct and from 55% to 5% for OLMo-2- 0325-32B-Instruct. For OLMo, EM stays at 50% throughout, so final-answer accuracy hides a collapse in rule application. In each case the specification of the rule is fixed, but its reliable execution does not extend with it.

![](images/614ba979609ef3592eda1adbef6dd9f9008aa45868a6d37967b4397285654381.jpg)

![](images/d43317540cb42befd3f80d90bea0fbf7276151949af0c68985cf467a7e98bdc2.jpg)

![](images/518a29a7d7a8dbbb3d3bcbea468807067c1d37d9d9044d2c57e044163aab5934.jpg)  
Figure 5: (a) Exact match (EM). (b) EM vs. dimension d (n = 20 per point).  
Figure 6: Addition with a supplied recursive rule (100 problems per point). (a) EM vs. operand length, which equals the number of recursive calls. (b) EM vs. verified EM (correct sum and correct state sequence) at five digits.

Appendix H provides results on composition from explicitly stated rules. We distinguish symbolic construction from numerical execution, with performance depending on the model and the supplied decomposition.

Results for applying a rule recursively. Although the rule is fixed, accuracy falls as the number of recursive calls grows (Figure 6a): Qwen3- 32B falls from 90% to 39% between 5 and 35 digits, Llama-3.1-70B-Instruct from 82% to 0% by 35 digits, and the smaller models below 5% by 25 digits. Correct sums do not show that the rule was applied (Figure 6b). At five digits, only 24 of Qwen3-32B’s 90 correct sums come from a correct execution of the recursive rule. In Appendices E and K, we give details and results on evaluating models for recursion on quantification tasks.

Takeaways Our experiments indicate that models have not fully learned to use abstraction or recursion. Their ability to find abstract structure and to apply these structures to new data is considerably below their ability to provide correct answers even on simple problems. All models had trouble with simple recursions. This motivates evaluating abstraction through the preservation of a rule across new instances, rather than treating a correct expression or benchmark answer as sufficient evidence of that capacity.

## 8 Conclusion

Transformers use an elegant implementation of the meaning as use theory. We, like earlier empiricists, have argued that full concept and linguistic mastery requires require at least two principles that complement experience, abstraction and recursion. They are what allows finite experience to support stable rule and concept extension. Thus they are essential for OOD generalization, and thereinlies their pragmatic justification.

We evaluated abstraction capabilities for 10 models on natural language texts describing grade school math problems and on basic geometrical concepts. We also evaluated models for recursion ability on arithmetic and universal quantification. Our results indicate that models have not really mastered these principles, but they provide a way for models to acquire such a mastery.

## 9 Limitations

Our experiments are diagnostic and intentionally narrow. Arithmetic, geometric criteria, and gradeschool word problems are used because the correct extension of each rule is independently specified; we do not claim that they establish the full scope of OOD generalization failure in natural language. Our abstraction paradigm supplies the variables and their bindings, so models identify a function over given quantities rather than decide what to abstract over. The 214 GSM8K-derived pairs are a curated sample whose sources may appear in pretraining data. The arithmetic interventions in Appendix B are not exhaustive; stronger positional biases, process supervision, or tool use may extend extrapolation further, which is consistent with our argument since they add structure beyond finite use. Finally, abstraction and recursion are a proposal about what a purely use-based account lacks, not an implemented solution; operationalizing them remains future work.

## References

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Leandro von Werra, and Thomas Wolf. 2024. Smollm - blazingly fast and remarkably powerful.

Nicholas Asher. 2011. Lexical Meaning in Context: A web ofwords. Cambridge University Press.

Nicholas Asher, Swarnadeep Bhar, Akshay Chaturvedi, Julie Hunter, and Soumya Paul. 2023. Limits for learning with large language models. In 12th Joint Conference on Lexical and Computational Semantics (\*Sem). Association for Computational Linguistics.

Allan Bäck. 2014. Aristotle’s theory of abstraction, volume 73. Springer.

Emily M Bender and Alexander Koller. 2020. Climbing towards nlu: On meaning, form, and understanding in the age of data. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 5185–5198.

Usha Bhalla, Thomas Fel, Can Rager, Sheridan Feucht, Tal Haklay, Daniel Wurgaft, Siddharth Boppana, Matthew Kowal, Vasudev Shyam, Jack Merullo, Atticus Geiger, and Ekdeep Singh Lubana. 2026. Do sparse autoencoders capture concept manifolds? Preprint, arXiv:2604.28119.

Yonatan Bisk, Ari Holtzman, Jesse Thomason, Jacob Andreas, Yoshua Bengio, Joyce Chai, Mirella Lapata, Angeliki Lazaridou, Jonathan May, Aleksandr Nisnevich, Nicolas Pinto, and Joseph P. Turian. 2020. Experience grounds language. CoRR, abs/2004.10151.

Robert Brandom. 1998. Making it explicit: Reasoning, representing, and discursive commitment. Harvard University Press.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, and 12 others. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.

Rudolf Carnap. 1950. Empiricism, semantics and ontology. Revue Internationale de Philosophie, 4.11:20– 40.

Akshay Chaturvedi, Swarnadeep Bhar, Soumadeep Saha, Utpal Garain, and Nicholas Asher. 2024. Analyzing Semantic Faithfulness of Language Models via Input Intervention on Question Answering. Computational Linguistics, pages 1–37.

Hanseul Cho, Jaeyoung Cha, Pranjal Awasthi, Srinadh Bhojanapalli, Anupam Gupta, and Chulhee Yun. 2024. Position coupling: Improving length generalization of arithmetic transformers using task structure. Preprint, arXiv:2405.20671.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

Nicolas Cusanus. 1440. De Docta Ignorantia. Google books reprinted in 2009.

Donald Davidson. 1967. Truth and meaning. Synthese, 17:304–323.

Chunyuan Deng, Zhiqi Li, Roy Xie, Ruidi Chang, and Hanjie Chen. 2026. Language models are symbolic learners in arithmetic. Preprint, arXiv:2410.15580.

Michael Dummett. 1993. The Seas ofLanguage. Oxford University Press.

Thomas Fel, Binxu Wang, Michael A. Lepori, Matthew Kowal, Andrew Lee, Randall Balestriero, Sonia Joseph, Ekdeep S. Lubana, Talia Konkle, Demba Ba, and Martin Wattenberg. 2026. Into the rabbit hull: From task-relevant concepts in dino to minkowski geometry. Preprint, arXiv:2510.08638.

Sheridan Feucht, Tal Haklay, Usha Bhalla, Daniel Wurgaft, Can Rager, Raphaël Sarfati, Jack Merullo, Thomas McGrath, Owen Lewis, Ekdeep Singh Lubana, Thomas Fel, and Atticus Geiger. 2026. Arithmetic in the wild: Llama uses base-10 addition to reason about cyclic concepts. Preprint, arXiv:2605.01148.

Leonardo Fibonacci. 1202. Liber Abaci. republished and translated by Jordi Deulofeu Piquet et Stephen Sanchez (Trad.), Le créateur de la suite mathématique de la beauté : Fibonacci, Barcelone, RBA Coleccionables, 2018.

Siavash Golkar, Mariel Pettee, Michael Eickenberg, Alberto Bietti, Miles Cranmer, Geraud Krawezik, Francois Lanusse, Michael McCabe, Ruben Ohana, Liam Parker, Bruno Régaldo-Saint Blancard, Tiberiu Tesileanu, Kyunghyun Cho, and Shirley Ho. 2024. xval: A continuous numerical tokenization for scientific language models. Preprint, arXiv:2310.02989.

Olivier Gouvert, Julie Hunter, Jérôme Louradour, Christophe Cerisara, Evan Dufraisse, Yaya Sy, Laura Rivière, Jean-Pierre Lorré, and OpenLLM-France community. 2025. The lucie-7b llm and the lucie training dataset: Open resources for multilingual language generation. Preprint, arXiv:2503.12294.

Kaiying Hou, David Brandfonbrener, Sham Kakade, Samy Jelassi, and Eran Malach. 2024. Universal length generalization with turing programs. Preprint, arXiv:2407.03310.

Dieuwke Hupkes, Mario Giulianelli, Verna Dankers, Mikel Artetxe, Yanai Elazar, Tiago Pimentel, Christos Christodoulopoulos, Karim Lasri, Naomi Saphra, Arabella Sinclair, Dennis Ulmer, Florian Schottmann, Khuyagbaatar Batsuren, Kaiser Sun, Koustuv Sinha, Leila Khalatbari, Maria Ryskina, Rita Frieske, Ryan Cotterell, and Zhijing Jin. 2023. A taxonomy and review of generalization research in NLP. Nature Machine Intelligence, 5:1161–1174.

Dhruva Karkada, Daniel J Korchinski, Andres Nava, Matthieu Wyart, and Yasaman Bahri. 2026. Symmetry in language statistics shapes the geometry of model representations. arXiv preprint arXiv:2602.15029.

Daniel Keysers, Nathanael Schärli, Nathan Scales, Hylke Buisman, Daniel Furrer, Sergii Kashubin, Nikola Momchev, Danila Sinopalnikov, Lukasz Stafiniak, Tibor Tihon, Dmitry Tsarkov, Xiao Wang, Marc van Zee, and Olivier Bousquet. 2020. Measuring compositional generalization: A comprehensive method on realistic data. In International Conference on Learning Representations (ICLR).

Najoung Kim and Tal Linzen. 2020. COGS: A compositional generalization challenge based on semantic interpretation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9087–9105.

Saul A Kripke. 1982. Wittgenstein on Rules and Private Language: An Elementary Exposition. Harvard University Press.

Brenden M. Lake and Marco Baroni. 2018. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In Proceedings ofthe 35th International Conference on Machine Learning (ICML).

Andrew K. Lampinen. 2024. Can language models handle recursively nested grammatical structures? a case study on comparing models and humans. Computational Linguistics, 50(4).

Llama Team. 2024. The llama 3 herd of models. CoRR, abs/2407.21783.

Bernard Lonergan. 1958. Insight. Harper & Row Publishers.

R Thomas McCoy, Shunyu Yao, Dan Friedman, Mathew D Hardy, and Thomas L Griffiths. 2024. Embers of autoregression show how large language models are shaped by the problem they are trained to solve. Proceedings of the National Academy of Sciences, 121(41):e2322420121.

Colin McGinn. 1984. Wittgenstein on Meaning. Basil Blackwell Publishers.

Sean McLeish, Arpit Bansal, Alex Stein, Neel Jain, John Kirchenbauer, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Jonas Geiping, Avi Schwarzschild, and Tom Goldstein. 2024. Transformers can do arithmetic with the right embeddings. Preprint, arXiv:2405.17399.

Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2020. A diverse corpus for evaluating and developing English math word problem solvers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 975–984, Online. Association for Computational Linguistics.

Matteo Muffo, Aldo Cocco, and Enrico Bertino. 2022. Evaluating transformer language models on arithmetic operations using number decomposition. In Proceedings ofthe Thirteenth Language Resources and Evaluation Conference, pages 291–297, Marseille, France. European Language Resources Association.

Omar Naim, Swarnadeep Bhar, Jérôme Bolte, and Nicholas Asher. 2026. Ssa: Improving performance with a better scoring function. Preprint, arXiv:2508.14685.

Omar Naim, Jérôme Bolte, and Nicholas Asher. 2025a. Analyzing limits for in-context learning. arXiv preprint arXiv:2502.03503.

Omar Naim, Guilhem Fouilhé, and Nicholas Asher. 2025b. Re-examining learning linear functions in context. In German Conference on Artificial Intelligence (Künstliche Intelligenz), pages 104–117. Springer.

Yaniv Nikankin, Anja Reusch, Aaron Mueller, and Yonatan Belinkov. 2025. Arithmetic without algorithms: Language models solve math with a bag of heuristics. In The Thirteenth International Conference on Learning Representations (ICLR).

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. 2021. Show your work: Scratchpads for intermediate computation with language models. Preprint, arXiv:2112.00114.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, and 1 others. 2022. In-context learning and induction heads. arXiv preprint arXiv:2209.11895.

D. Osherson and S. Weinstein. 1986. Identification in the limit of first order structures. Journal of Philosophical Logic, 15:55–81.

Jorge Pérez, Pablo Barceló, and Javier Marinkovic. 2021. Attention is turing-complete. Journal ofMachine Learning Research, 22(75):1–35.

Steven Pinker and Alan Prince. 1988. On language and connectionism: Analysis of a parallel distributed processing model of language acquisition. Cognition, 28(1-2):73–193.

Alexandre Pouget, Peter Dayan, and Richard Zemel. 2000. Information processing with population codes. Nature Reviews Neuroscience, 1(2):125–132.

Jing Qian, Hong Wang, Zekun Li, Shiyang Li, and Xifeng Yan. 2022. Limitations of language models in arithmetic and symbolic induction. Preprint, arXiv:2208.05051.

Linlu Qiu, Peter Shaw, Panupong Pasupat, Pawel Nowak, Tal Linzen, Fei Sha, and Kristina Toutanova. 2022. Improving compositional generalization with latent structure and data augmentation. In Proceedings of the 2022 conference of the north american chapter ofthe associationfor computational linguistics: Human language technologies, pages 4341– 4362.

Willard Van Orman Quine. 1960. Word and object. Harvard University Press.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, and 1 others. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

D. E. Rumelhart and J. L. McClelland. 1986. On learning the past tenses of english verbs. In Parallel Distributed Processing, Volume 2: Explorations in the Microstructure ofCognition: Psychological and Biological Models. The MIT Press.

J. R. Shoenfield. 1959. On degrees of unsolvability. Annals ofMathematics, 69:644–653.

Arabella Sinclair, Jaap Jumelet, Willem Zuidema, and Raquel Fernández. 2022. Structural persistence in language models: Priming as a window into abstract language representations. Transactions ofthe Associationfor Computational Linguistics, 10:1031–1050.

Yao Tong, Jiayuan Ye, Anastasia Borovykh, and Reza Shokri. 2026. Generalization in llm problem solving: The case of the shortest path. Preprint, arXiv:2604.15306.

Ludwig Wittgenstein. 1953. Philosophical investigations. Blackwells.

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2021. An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080.

Yang Yan, Yu Lu, Renjun Xu, and Zhenzhong Lan. 2025. Do large language models truly grasp addition? a rule-focused diagnostic using two-integer arithmetic. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 13467–13483, Suzhou, China. Association for Computational Linguistics.

Dylan Zhang, Curt Tigges, Zory Zhang, Stella Biderman, Maxim Raginsky, and Talia Ringer. 2024. Transformer-based models are not yet perfect at learning to emulate structural recursion. Preprint, arXiv:2401.12947.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. Opt: Open pre-trained transformer language models. Preprint, arXiv:2205.01068.

Tianyi Zhou, Deqing Fu, Vatsal Sharan, and Robin Jia. 2024. Pre-trained large language models use fourier features to compute addition. Preprint, arXiv:2406.03445.

Tianyi Zhou, Deqing Fu, Mahdi Soltanolkotabi, Robin Jia, and Vatsal Sharan. 2026. Fone: Precise single-token number embeddings via fourier features. Preprint, arXiv:2502.09741.

## A Learning in the limit

Here we investigate learning in the limit or limit computability, which applies to infinitary concepts. Definition 1. (Shoenfield, 1959) A set $S \subseteq \mathbb { N }$ is limit computable iffor $x , s \in \mathbb { N }$ its characteristic function

$$
\phi _ { S } ( x ) = l i m _ { s  \omega } g ( x , s )
$$

for some total computablefunction $g ( x , s )$

Since the Kripkenstein arguments attack the possibility of justifiably inferring a universal rule from finite instances, let us consider universal quantification and learnability in the limit. Universal quantification is a function over pairs of properties or sets into {0, 1} such that

$$
f ( P , Q ) = 1 { \mathrm { ~ i f f ~ } } P \subseteq Q
$$

We consider an arithmetic version of universal quantification in which we set $P = \mathbb { N }$ . We can generalize this argument to arbitrary countable models M, by “linearizing” the domain (assigning each element a natural number) and treat s a sequence of objects as characterizing a finite submodel of M. M is then characterized by an infinite sequence. Quantified statements then correspond to countable unions and intersections of such sequences, creating the Borel Hierarchy.

There is a computable function g(s,x) where x ranges over elements of s and s is a finite initial segment of ω and

$$
g ( x , s ) = 1 { \mathrm { ~ i f f ~ } } \forall x \in s ( x \in P  x \in Q )
$$

Over N, for each $s < \omega , g ( s , x )$ is computable. However, $\{ g ( s , x ) \} _ { s \in \omega }$ doesn’t necessarily converge to 0 or 1 on ω.

As (Osherson and Weinstein, 1986) remark however, a universal statement is computable in the limit with "one hypothesis change" (co-computably enumerable in their terminology). That is, we can learn a universal claim ϕ if we allow one revision to our hypothesis about ϕ: we start out with the assumption that $\phi$ is true, and if we encounter a falsifying instance then we conclude that ϕ is false. Conversely, we can learn in the limit an existential statement ψ over N with one hypothesis change, if we start out with the assumption that ψ is false.

Proposition 1. ϕ is computable in the limit iff it is computably enumerable and co-computably enumerable.

An adaptation of the theorem that a set is computable iff it is recursively enumerable and co recursively enumerable proves this proposition.

Evaluation. Learnability in the limit arguments depend on the assumption that terms have a constant meaning as we evaluate a universal or existentialformula in ever largerfinite models. But this is precisely what the skeptical attack on meanings disputes.

The Kripkenstein argument gives us a constraint on probability distributions. $P$ obeys a Kripkean justification pattern, if $P ( x | \sigma ) \approx P ( y | \sigma )$ for OOD x, y and σ in ID.

Proposition 2. $\Pi _ { 1 } , \Sigma _ { 1 }$ sets are not learnable in the limit with 1 hypothesis change, given objectivefunctions with Kripkean justification patterns. (Asher et al., 2023)

## B Six arithmetical experiments

## B.1 Experimental Setup

We evaluate addition under digit length shifts. For a model with training range $1 , \ldots n ,$ in distribution examples use operands with at most n digits. Near OOD examples extend the operand length slightly beyond the training range and far OOD examples extend it further. We report exact match accuracy on the full generated answer, since digit level metrics can obscure correlated errors across output positions.

Our experiments cover several pretrained causal language models: GPT-2 small (Radford et al., 2019), OPT-125M (Zhang et al., 2022), SmolLM-135M (Allal et al., 2024), Qwen3-0.6B (Qwen Team, 2025), Llama-3.2-1B, Llama-3.1- 70B (Llama Team, 2024), and Lucie-7B (Gouvert et al., 2025). We vary the form of structure added to the model: numerical representation (§B.2), positional scaffolding (§B.3), an explicit rule supplied at inference time (§B.5), and procedural supervision on intermediate states (§B.6).

## B.2 Structured numerical interfaces

Prior work has shown that specialized numerical interfaces, including continuous number tokenization, digit-position embeddings, and Fourier number embeddings, can improve numerical reasoning over ordinary text tokenization (Golkar et al., 2024; McLeish et al., 2024; Zhou et al., 2026). Motivated by this, we compare ordinary text fine-tuning with two structured numerical interfaces.

The first is a column-state decoder.<sup>2</sup> Each number is replaced in the prompt with a special numerical token, while its integer value is passed through a structured numerical encoder, producing one embedding for the whole number. We add a dedicated number decoder for these embeddings.

The second is an auxiliary supervised text method. Here the model keeps the ordinary language-model output head, and answers are still generated as text tokens. Unlike the column-state decoder, the numerical pathway does not replace the text pathway: numerical embeddings are added to ordinary token embeddings at number positions, so the transformer receives both the pretrained text representation and explicit numerical structure.

The distinction is deliberate. For smaller models we replace the weak native numerical pathway, while for stronger pretrained models we use a lighter auxiliary interface motivated by FoNE-style features (Zhou et al., 2026) that preserves the BPE/- text pathway. Table 2 shows the three interfaces on a single problem; implementation details are in Appendix C.

<table><tr><td></td><td>Model input</td><td>Answer produced by</td></tr><tr><td>Plain Col</td><td>372 + 158 =</td><td>LM head, text tokens 530</td></tr><tr><td></td><td>&lt;num&gt; + &lt;num&gt; = values 372, 158 via encoder columns (2, 7, 3), (8, 5, 1)</td><td>number decoder, LSB-first 0, 3, 5 → 530</td></tr><tr><td>Aux</td><td>372 + 158 = et = et + g · num(x)</td><td>LM head, text tokens 530</td></tr></table>

Table 2: The same problem under the three interfaces. Col changes the numeric interface, replacing numbers in the text stream and decoding the answer outside the LM head. Aux leaves that interface intact and changes only what the hidden states are encouraged to represent.

## B.3 Addition under digit-length shift

Table 3 shows a consistent pattern. Plain text models can achieve high in-distribution accuracy, but often fail sharply when answer length moves beyond the training regime. Structured numerical interfaces substantially improve in-distribution and near-OOD accuracy, showing that the models can exploit useful arithmetic structure with a more favorable representation.

The improvement, however, is not equivalent to acquiring the addition rule. For the columnstate and auxiliary interfaces in Table 3, far-OOD addition remains largely unsolved for our tested interfaces. Our point is not that numerical structure is useless, but that better numerical representations and supervised readout, while improving local generalization by themselves, do not specify how the rule should extend.

<table><tr><td>Backbone</td><td>n</td><td>Method</td><td>ID</td><td>Near-OOD</td><td>Far-OOD</td></tr><tr><td>GPT-2 small</td><td>6</td><td>Plain</td><td>30.18%</td><td>0.00%</td><td>0.00%</td></tr><tr><td>GPT-2 small</td><td>6</td><td>Col</td><td>99.50%</td><td>58.34%</td><td>1.36%</td></tr><tr><td>OPT-125M</td><td>6</td><td>Plain</td><td>36.42%</td><td>0.00%</td><td>0.00%</td></tr><tr><td>OPT-125M</td><td>6</td><td>Col</td><td>100.00%</td><td>58.96%</td><td>1.47%</td></tr><tr><td>SmolLM-135M</td><td>6</td><td>Plain</td><td>94.67%</td><td>15.85%</td><td>0.00%</td></tr><tr><td>SmolLM-135M</td><td>6</td><td>Col</td><td>99.30%</td><td>59.27%</td><td>1.47%</td></tr><tr><td>Llama-3.2-1B</td><td>15</td><td>Plain</td><td>94.44%</td><td>0.10%</td><td>0.00%</td></tr><tr><td>Llama-3.2-1B</td><td>15</td><td>Aux</td><td>97.69%</td><td>26.65%</td><td>0.10%</td></tr><tr><td>Qwen3-0.6B</td><td>20</td><td>Plain</td><td>76.12%</td><td>0.00%</td><td>0.00%</td></tr><tr><td>Qwen3-0.6B</td><td>20</td><td>Aux</td><td>98.50%</td><td>73.40%</td><td>10.35%</td></tr><tr><td>Lucie-7B</td><td>30</td><td>Plain</td><td>98.60%</td><td>13.60%</td><td>2.29%</td></tr><tr><td>Lucie-7B</td><td>30</td><td>Aux</td><td>99.11%</td><td>33.55%</td><td>8.18%</td></tr></table>

Table 3: Addition under digit-length shift. n is the maximum operand digit length seen in training; ID uses operands of at most n digits, Near-OOD uses n+1, n+2, and Far-OOD uses n+3, n+4.

The interfaces above change how numbers are represented. A different kind of intervention changes how positions are indexed. We therefore tested an Abacus-style (McLeish et al., 2024) positional scaffold on Qwen3-0.6B, which supplies an explicit coordinate for digit-column alignment.

<table><tr><td>Model</td><td>Method</td><td>n</td><td>ID</td><td>Near-OOD</td><td>Far-OOD</td></tr><tr><td>Qwen3-0.6B</td><td>Plain</td><td>20</td><td>76.12%</td><td>0.00%</td><td>0.00%</td></tr><tr><td>Qwen3-0.6B</td><td>Abacus</td><td>20</td><td>91.80%</td><td>72.85%</td><td>34.20%</td></tr></table>

Table 4: Qwen3-0.6B addition results comparing plain with an abacus-style positional scaffold

We take Table 4 as a positive result. Far-OOD addition is not beyond transformer models, and different forms of structure extend the generalization range to markedly different degrees. This supports rather than weakens the rule-following diagnosis. The Abacus scaffold does not encode the addition algorithm: it supplies no carry computation and no column-update procedure. What it supplies is an inductive bias constraining how the finite training examples are to be extended, and that bias is doing most of the work. At the same time, a substantial gap remains even in the strongest condition. A coordinate system that determines how digits align is not the same as a procedure that executes reliably as the domain grows.

## B.4 Good representation ̸= rule following

Recent mechanistic work suggests that language models often represent numbers through periodic or Fourier-like features, where a number is encoded by its residues modulo periods such as 2, 5, 10, or 100 (Zhou et al., 2024; Feucht et al., 2026). Relatedly, Nikankin et al. (2025) find that language models solve arithmetic through a bag of heuristics implemented by distributed neuron sets rather than a general algorithm, and Deng et al. (2026) report converging evidence on the robustness of model arithmetic. These accounts are consistent with our diagnosis; our question is what follows for how competence is assessed. These periods are natural for base-10 arithmetic: for example, n mod 10 gives the final digit of n, and (a + b) mod 10 gives the final digit of the sum. We used a linear probe to see whether the hidden state at the answer position contains residues of the operands and of the sum. Table 5 reports the modulo-10 probe; Appendix D reports the corresponding probes for other periods.

<table><tr><td>Model</td><td>Method</td><td>a mod 10</td><td>b mod 10</td><td>(a + b) mod 10</td></tr><tr><td>Llama-3.2-1B</td><td>plain</td><td>87.0%</td><td>98.1%</td><td>12.7%</td></tr><tr><td>Llama-3.2-1B</td><td>aux</td><td>99.6%</td><td>99.8%</td><td>84.8%</td></tr></table>

Table 5: Linear-probe accuracy for base-10 residue information at the answer-context hidden state.

In a plain arithmetic-finetuned Llama-3.2-1B model, operand residues such as a mod 10 and b mod 10 are decodable, but the sum residue (a + b) mod 10 is nearly absent. With our numericalauxiliary training, (a + b) mod 10 becomes highly decodable from the answer-context hidden state. However, this representational improvement does not translate into stable length generalization: exact-match accuracy still collapses with addition of numbers with more digits. It is not that the model just lacks numerical information internally. Rather, finite training can induce local arithmetic geometry without yielding a rule-like procedure for extending addition beyond the training range.

We state the conclusion narrowly. Digit-length shift tests two things at once: whether a model can represent numerals of increasing length, and whether it can apply the addition procedure to them. Our probes do not fully separate the two. They show that better internal access to numerical information is not sufficient for stable extension, but not that numeral representation itself has been solved.

## B.5 Giving the rule doesn’t yield rule following

If improved numerical representations do not by themselves yield rule-like extensions, perhaps model failure is due to the relevant rule not being explicitly available to the model.

We tested a prompt-only setting in which the model is given the semi-decision procedure for universal statements from the learning-in-the-limit discussion in Appendix A: assume the statement is true, scan the sequence from left to right, and switch to false once a counterexample is found. We compare a task-only prompt with a rule-guided prompt that states this scanning algorithm and evaluate Llama-3.1-70B-Instruct on sequences of increasing length for the question: Are all numbers in the sequence positive? Details are given in Appendix E.

The rule-guided prompt improves performance on short and moderate sequence lengths but does not eliminate the length cliff: exact-match accuracy still collapses as sequence length increases. This matters for our argument because it separates possession of a rule description from stable rule application. A model can be told the relevant rule and still fail to execute it reliably under domain extension. The issue is therefore not merely whether the rule is expressible in the prompt, but whether the system has a stable procedure for applying it as the domain grows.

## B.6 Providing structure in training

The prompt experiment gives the model the rule at inference time. A stronger intervention is to train the model on the intermediate states of the computation itself. This is close to the Turing–program style scratchpad supervision, where an algorithm is decomposed into local state updates that mimic a Turing–machine computation (Hou et al., 2024). For universal quantification, the intermediate state is the running truth value of the universal claim; the full trace format is given in Appendix F.

We consider two supervision regimes: direct supervision on the input–output mapping, and trace supervision, which trains the model to imitate the intermediate state updates. For each regime, we compare standard softmax scoring with SSA, the alternative scoring function of Naim et al. (2026), that is designed to reduce softmax saturation and preserve more contextual information across tokens.

![](images/c940e97d4ff337d6249a652d75844eb5cf89e13e73670a6bbb62963af614e6ef.jpg)  
Figure 7: Direct supervision trains only on input-output mappings; trace supervision trains on state updates.

Figure 7 shows that trace supervision moves the OOD cliff farther from the training boundary: the model applies the update rule for longer sequences than under direct final-answer supervision. However, the cliff is not eliminated. The gain comes from additional procedural structure, not from ordinary input–output use alone.

Procedural supervision can extend the range over which a learned rule is applied, but it does not remove Kripke’s point that finite traces underdetermine how the procedure should extend beyond the training regime. The experiments in this section share a shape. Each intervention supplies structure that the training distribution does not, and each extends the range over which the rule is applied correctly without fixing how it should extend further.

## C Numerical Encoder and Decoder Implementation Details

This appendix gives implementation details for the two structured numerical interfaces used in Table 3. The main text describes the interfaces at a high level: a column-state decoder, used mainly for smaller models, and an auxiliary supervised text method, used mainly for stronger pretrained models. Here we describe how the numerical encoder, embedding injection, output decoder, auxiliary losses, and evaluation are implemented.

## C.1 Data and evaluation protocol

All arithmetic experiments use synthetically generated addition examples with prompts of the form

$$
a + b =
$$

where a and $b$ are sampled uniformly from specified digit-length ranges. For a training range $1 , \ldots , n ,$ in-distribution examples use operands with at most n digits. Near-OOD examples use operands slightly beyond this range, and far-OOD examples extend the range further. The value of n differs by model family, as reported in Table 3.

We evaluate by exact match on the full generated integer answer. This is an end-to-end metric: a prediction is correct only if the entire generated number matches the true integer answer. We therefore avoid padded digit-slot accuracy, which can be inflated by easy padding positions and does not faithfully measure numeric generation. For textgeneration models, the generated string is parsed as an integer. For column-state models, the digit decoder output is converted directly into an integer. All reported headline results use deterministic decoding.

## C.2 Numerical encoder

Both structured interfaces use a numerical encoder inspired by Fourier-style number representations. Given an operand value x, the encoder combines periodic features, a log-magnitude feature, and a sign feature. Abstractly, the whole-number embedding has the form

$$
\begin{array} { r l } & { \mathrm { { n u m } } ( x ) = W _ { \mathrm { f o u r i e r } } \phi ( x ) } \\ & { \qquad + \mathrm { { M L P } } ( \log _ { 1 0 } ( | x | + 1 ) ) + e _ { \mathrm { s i g n } } ( x ) , } \\ & { \phi ( x ) = \left[ { \cos } \left( \displaystyle \frac { 2 \pi | x | } { T _ { i } } \right) , \right. } \\ & { \qquad \left. { \sin } \left( \displaystyle \frac { 2 \pi | x | } { T _ { i } } \right) \right] _ { i } . } \end{array}
$$

The periods $T _ { i }$ are powers of ten over a wide decimal range, by default 10<sup>−</sup> $^ { - 3 } , 1 0 ^ { - 2 } , \dots , 1 0 ^ { 1 2 }$ . These periodic features make decimal-scale information explicitly available. The log-magnitude term gives a coarse scale signal, and the sign embedding distinguishes positive and negative values. The final vector is projected to the hidden dimension of the backbone language model.

The two interfaces use this numerical embedding differently. The column-state decoder uses it as a replacement embedding for a special numerical token. The auxiliary supervised text method uses it additively, preserving the ordinary text tokens while adding numerical information as a side channel.

## C.3 Column-state decoder

In the column-state interface, each input number is replaced in the prompt by a special numerical token. The integer value of that number is passed separately through the numerical encoder, producing one vector for the whole number. This vector is projected into the backbone embedding space and substituted for the embedding of the special numerical token before the sequence is passed through the transformer.

The output side is also modified. Instead of producing numeric answers through the standard language-model vocabulary, the model uses a dedicated number decoder conditioned on the transformer hidden state at the answer context. The decoder emits the answer least-significant digit first.

For the column-state runs, operands are also represented as a column memory. Each operand is decomposed from right to left into decimal columns, for example

$$
3 7 2 \mapsto ( 2 , 7 , 3 ) , \qquad 1 5 8 \mapsto ( 8 , 5 , 1 ) .
$$

Each column memory entry contains a digit embedding, a column-position embedding, an operand-id embedding, and a sign embedding. This gives the decoder direct access to aligned operand columns, such as the ones digit of a and the ones digit of b. The language-model backbone, numerical encoder, adapter, and number decoder are fine-tuned jointly. Thus, these results should not be assimilated to those of a frozen external calculator attached after the fact.

## C.4 Auxiliary supervised text method

The auxiliary supervised text method keeps the ordinary language-model input and output pathway. Input numbers remain text tokens, and answers are generated with the standard language-model head. Unlike the column-state decoder, this method does not replace the text pathway. Instead, at each token position overlapping a number span with value x, the model adds a gated numerical embedding to the ordinary token embedding:

$$
e _ { t } ^ { \prime } = e _ { t } + g \cdot \mathtt { n u m } ( x ) .
$$

Here $e _ { t }$ is the pretrained token embedding and $g$ is a learned gate. This additive design is motivated by FoNE-style (Zhou et al., 2026) numerical features: it preserves the existing BPE/text pathway while giving the model explicit numerical structure.

Training uses the standard language-model loss together with auxiliary number-oriented losses. These losses encourage hidden states associated with numerical expressions to encode structure such as magnitude, digit position, column information, answer length, and carry state. In the outputreadout implementation used for the strongest auxiliary results, token-level heads predict the numerical chunk represented at a generated numeric token and the least-significant-digit-counted column of that token. Context-level heads predict global numerical information such as the answer length and carry pattern. The auxiliary heads are training-time supervision: the reported auxiliary models still generate answers through the ordinary language-model vocabulary.

This distinction is important. The columnstate decoder changes the numeric output interface, while the auxiliary supervised text method keeps the output interface textual and instead encourages the hidden states to expose numerical structure. The residue probes in Appendix D show that this auxiliary training makes sum-related residue information more linearly decodable, even when far-OOD exact-match generalization remains limited.

## C.5 Training and evaluation

Arithmetic training setup. All arithmetic experiments use synthetic integer expressions and report add-only evaluation. Unless otherwise stated, each run uses 50,000 training examples and 5,000 validation examples, with operands sampled uniformly from the specified digit-length range. Evaluation uses 2,000 examples per split and reports exact match on the full generated answer. Decoding is greedy and deterministic. Plain-text generations are parsed by taking the first whitespace-delimited generated token as the predicted integer; non-integer generations are counted as incorrect. Figure 8 visualizes the same trend across models and interfaces.

All headline arithmetic models are fully finetuned rather than trained with parameter-efficient adapters. We use AdamW with weight decay 0.01, cosine learning-rate decay, gradient clipping at 1.0, and maximum sequence length 128, except for Lucie-7B where we use maximum sequence length 192. Batch size is 32 for GPT-2, OPT-125M, SmolLM-135M, Llama-3.2-1B, and Qwen3- 0.6B, and 4 for Lucie-7B. GPT-2, OPT-125M, and SmolLM-135M are trained for 30 epochs with learning rate $5 \times 1 0 ^ { - 5 }$ for plain fine-tuning; their column-state runs use learning rate $5 \times 1 0 ^ { - 5 }$ for the backbone and $1 0 ^ { - 3 }$ for newly initialized numerical parameters. Llama-3.2-1B and Qwen3-0.6B are trained for 10 epochs; plain runs use learning rate $5 \times 1 0 ^ { - 5 }$ , while auxiliary runs use backbone learning rates $5 \times 1 0 ^ { - 5 }$ , with $1 0 ^ { - 3 }$ for auxiliary parameters. Lucie-7B is trained for 10 epochs in bfloat16 with gradient checkpointing, using batch size 4 and learning rate 10<sup>−5</sup> for plain fine-tuning; its auxiliary run uses backbone learning rate 10<sup>−5</sup> and auxiliary-parameter learning rate $5 \times 1 0 ^ { - 4 }$ . The arithmetic runs use fixed synthetic data seeds, and the reported numbers are averaged over 3 random initializations.

![](images/d382d29efb6eb205b83b908ef739203762d513eb7f43a2cfec9ac0dff61356d1.jpg)

![](images/b8a173b0263e1f965c72937c404fb77076a39d876665f6854dbf99be3ea0828c.jpg)

![](images/e096aa083acf60d5b84f7c5004c1f170afddcf6a1a8203708ad2609b8114ef6b.jpg)

![](images/f3f46f4a85764ef0e91100cec7ecd9da5c60e2345a68d935128ee8d8e7acf51c.jpg)

![](images/d58f55cc6672f49ede39c5e941d3311c6b7ec26afb24c723e27f960d4f391577.jpg)

![](images/c656259cf66d6afd36b329ae3ff107b88c2559ae868cb4a237e04fbae71c03e7.jpg)  
Figure 8: Per-operand-length OOD accuracy trends for addition. The x-axis shows operand length relative to the maximum training operand length: ID, $n + 1 , n + 2 , n + 3$ , and $n + 4 .$ . Accuracy drops sharply as operand length moves farther beyond the training regime.

## D Residue Probes for Arithmetic Geometry

To distinguish representational failures from generation failures, we probe whether arithmetic residue information is linearly decodable from the model hidden state at the answer context. For a period $T ,$ , the residue n mod $T$ captures periodic base-10 structure: for example, n mod 10 is the final digit of n, and $( a + b )$ mod 10 is the final digit of the sum.

We generate 12,000 pairs $a , b \in \{ 0 , 9 9 9 \}$ and split them 75/25 into probe-training and held-out sets. For each pair, we collect hidden states at the last token of $^ { a , }$ the last token of $b ,$ and the answercontext position in prompts of the form $a + b = .$ We then fit ridge probes with regularization $1 0 ^ { - 2 }$ to predict Fourier targets for residues modulo 2, 5, 10, and 100. Held-out residue accuracy is computed by mapping the predicted Fourier vector back to the nearest residue class. Results are reported in Table 6.

<table><tr><td>Model</td><td></td><td>a mod 10 b mod 10</td><td></td><td></td><td></td><td>0 (a + b) mod 2 (a + b) mod 5 (a + b) mod 10 (a + b) mod 100</td></tr><tr><td> $\operatorname { L l a m a } - 3 . 2 – 1 \mathbf { B } + \mathrm { p l a i n } \operatorname { F T }$ </td><td>87.0%</td><td>98.1%</td><td>62.5%</td><td>25.4%</td><td>12.7%</td><td>1.4%</td></tr><tr><td>Llama  $3 . 2 – 1 \mathbf { B } + \mathbf { \bar { a u x } } \mathbf { F T }$ </td><td>99.6%</td><td>99.8%</td><td>100.0%</td><td>95.8%</td><td>84.8%</td><td>9.6%</td></tr></table>

Table 6: Held-out linear-probe accuracy for residue information at the answer-context hidden state. The probe asks whether base-10 arithmetic information is linearly decodable from the model state. Operand residues are already decodable after ordinary arithmetic fine-tuning, but numerical-auxiliary training makes residues of the sum much more decodable. Chance accuracy is $1 / T$ for a modulo-T residue.

## E Algorithms

Quantification: From Section A, we can test quantification in a setting with a linearized domain and a single property. This abstracts a common naturallanguage reasoning pattern: statements such as “all emails from John before July 15 require action” require scanning a finite sequence while preserving the same universal rule as the sequence grows. To test the effectiveness of abstraction and recursion, we then supply the general algorithm used in Section A: assume “every number is positive” is true until you come to a negative number. Then predict false for the rest of the autoregressive predictions. Addition: For two one digit numbers u, v create two registers U, V with $u \in U , v \in V$ . Convert u, v into successor notation; i.e. if u is 3, then set $u = 0 ^ { \prime \prime \prime }$ . Now remove one <sup>′</sup> from $u \in U$ and add it to v in V; i.e. if $v = 0 s ^ { \prime \prime }$ , then after the operation you should have $v = 0 ^ { \prime \prime \prime }$ . Continue this operation until u = 0, and return the result in V.

For two n-digit numbers $x = x _ { 1 } \cdot \cdot \cdot x _ { n }$ and $y =$ $y _ { 1 } \cdots y _ { n }$ , let x<sub>j</sub> , $y _ { j } \in \{ 0 , \ldots , 9 \}$ , where $j = n$ is the units column and $j = 1$ is the most significant column. Create registers $X _ { j } , Y _ { j }$ for each column j, initialized with $X _ { j } = x _ { j }$ and $Y _ { j } = y _ { j }$ . Also create an extra leading register $X _ { 0 } = 0$

Starting from $j = n$ and moving leftward to $j = 1$ , add the contents of $Y _ { j }$ into $X _ { j }$ . If after this operation $X _ { j } \geq 1 0$ , write

$$
X _ { j } : = X _ { j } - 1 0
$$

and increment the next higher column:

$$
X _ { j - 1 } : = X _ { j - 1 } + 1 .
$$

After all columns have been processed, return

$$
X _ { 0 } * X _ { 1 } * \cdots * X _ { n } ,
$$

omitting $X _ { 0 }$ if it is 0, where ∗ denotes digit concatenation.

## F Quantification Trace Diagnostic: Implementation Details

The task is universal quantification over signed integer sequences. Input numbers are represented by tokens of the form P004 for positive integers and N002 for negative integers. The target output is the truth value of the sentence “all numbers so far are positive” at each prefix of the sequence.

Data. Training sequences have lengths sampled uniformly from 1 to 40. We use integer magnitudes from 1 to 99. In 20% of examples all numbers are positive. Otherwise, the first negative position is sampled uniformly from the sequence, all preceding numbers are positive, and later numbers may be positive or negative. Evaluation uses the same number distribution but tests longer sequence lengths: 10, 20, 30, 40, 50, 60, 80, 100.

Training formats. We compare two target formats. In the final-answer format, the model is trained only to produce the prefix truth values:

<bos> <seq> P004 P009 N002 P007   
<ans> T T F F <eos>

In the trace format, the model is trained to produce the intermediate state update before the final answer:

<bos> <seq> P004 P009 N002 P007 <trace>   
S T R P004 O T <sep>   
S T R P009 O T <sep>   
S T R N002 O F <sep>   
S F R P007 O F <sep>   
<ans> T T F F <eos>

Here S marks the previous state, R the read input token, and O the output state after reading that token. The loss is applied only after the target marker, i.e. after <ans> in the final-answer condition and after <trace> in the trace condition.

Model. All four models are trained from scratch with the same decoder-only transformer architecture: 8 layers, hidden size 512, 8 attention heads, MLP size 2048, sinusoidal positional embeddings, maximum sequence length 1024, and dropout 0.0. The vocabulary contains the special tokens, truthstate tokens, and signed integer tokens. We train for 4000 steps with batch size 64, AdamW, learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay 0.01, 200 warmup steps, and bfloat16 precision.

## G GSM8K transfer: dataset variants and complete results

The transfer evaluation uses 214 paired problems derived from a curated subset of GSM8K test questions. Each source problem is paired with a target problem that changes the scenario and bound quantities while preserving the reference expression. Variables denote quantity roles, rather than their order of occurrence. The same source IDs and mathematical relationships are used in all four target conditions described in Table 7.

The scaled-quantity condition does not require every operand or intermediate result to be integral: structural ratios and fractions remain part of the problems. The fractional intervention affects only 66 pairs, so its aggregate score averages changed and unchanged problems. Source questions are identical across conditions. The scaled and perturbed conditions reuse the same generated abstractions; baseline and extended-scale evaluations generate them separately. Variation in abstraction scores therefore reflects separate executions on unchanged source prompts, not abstraction from larger source quantities.

We distinguish direct solving (Eval 1), source abstraction (Eval 2), and transfer (Eval 3). Direct solving receives the target question; abstraction receives the source question and quantity legend; transfer receives the generated expression and supplied target bindings in a second model call. No reference-expression fallback is supplied when abstraction fails. Expressions are scored by exact rational-function equivalence, and numerical answers by exact equality. Figure 9 reports all ten models under these criteria. Llama, OLMo, Qwen2.5, and SmolLM2 labels denote their Instruct variants; OLMo-2-32B abbreviates OLMo-2-0325-32B-Instruct.

The application decline extends beyond the Qwen models discussed in the main text. Llama-3.1-70B-Instruct’s transfer accuracy falls from 59.3% at baseline to 2.3% at extended scale, while its source-abstraction accuracy remains above 85%. OLMo-2-0325-32B-Instruct retains approximately

78% abstraction accuracy but obtains no correct transfer answers at extended scale. These results distinguish generating an appropriate expression from reliably executing it under the evaluated conditions.

Controls and decoding. Transfer receives supplied variable bindings and uses two model calls. At extended scale, Qwen3-32B scores 45.3% on a direct control with the same bindings, compared with 43.5% for transfer and 36.4% for unassisted direct solving. The apparent advantage over unassisted solving therefore does not isolate a benefit from generating the expression. All GSM conditions use greedy decoding, with Qwen thinking disabled, a 256-token abstraction budget, and a 1,024-token answer budget. Exact scoring does not accept rounded approximations to nonterminating fractions.

Scope. The source questions form a curated rather than random sample and may have appeared in pretraining. The authored target scenarios and quantity mappings have not received independent human review. Baseline targets change both context and quantities relative to their sources, and extended-scale quantities need not describe realistic situations. The results characterize exact-answer performance under the stated construction and generation budgets; they do not establish unrestricted generalization or failure under every computational regime.

## H Composition of symbolic functions

We evaluate 50 algebraically distinct rules, ten at each operation count from one to five, using addition, subtraction, multiplication, doubling, and squaring. Each rule has 20 input and story instances, giving 1,000 problems per condition. Numerical inputs are distinct integers from 1 to 9; all intermediate values remain within ±10,000. Symbolic counterparts replace the inputs with a and b. Rules are explicitly supplied, so the experiment tests composition from a stated procedure rather than inference of hidden operations. Operation count refers to the stated procedure, with doubling and squaring each counted as one operation.

We compare direct prompting, which requests only the final answer or expression; provided decomposition, which supplies numbered operations and requests intermediate results; and modelgenerated decomposition, which describes the rule in a paragraph and asks the model to determine its intermediate computations. Direct prompting uses the same numbered procedure as provided decomposition. Model-generated decomposition changes both presentation and response instructions, so this comparison does not isolate a single prompting effect. Numerical answers use exact integer equality and symbolic answers use polynomial equivalence. We report audited scores that normalize unambiguous answer formatting. Each condition uses greedy decoding, a 1,024-token output budget, and disabled Qwen thinking.

<table><tr><td></td><td colspan="4">(a) Direct solving</td><td colspan="3">(b) Abstraction</td><td colspan="4">(c) Transfer</td><td rowspan="2">100</td></tr><tr><td>Qwen3-32B</td><td>91.6</td><td>64.0</td><td>61.7</td><td>36.4</td><td>89.3</td><td>89.3 89.3</td><td>89.3</td><td>86.9</td><td>56.5</td><td>52.8</td><td>43.5</td></tr><tr><td>Qwen3-8B</td><td>90.2</td><td>65.0</td><td>61.7</td><td>37.4</td><td>78.5</td><td>75.7 75.7</td><td>75.2</td><td>72.0</td><td>43.0</td><td>40.2</td><td>30.4</td><td></td></tr><tr><td>Llama-3.1-70B</td><td>71.5</td><td>29.9</td><td>26.6</td><td>4.2</td><td>85.5</td><td>86.0</td><td>86.0 86.9</td><td>59.3</td><td>22.9</td><td>22.4</td><td>2.3</td><td></td></tr><tr><td>Llama-3.1-8B</td><td>40.2</td><td>15.0</td><td>15.0</td><td>2.8</td><td>59.3</td><td>59.8</td><td>59.8 58.9</td><td>22.9</td><td>7.5</td><td>7.5</td><td>1.9</td><td></td></tr><tr><td>OLMo-2-32B</td><td>31.3</td><td>8.4</td><td>7.5</td><td>0.0</td><td>77.6</td><td>78.0</td><td>78.0 78.0</td><td>18.7</td><td>5.1</td><td>6.1</td><td>0.0</td><td></td></tr><tr><td>Qwen3-0.6B</td><td>35.5</td><td>18.2</td><td>17.3</td><td>1.9</td><td>6.1</td><td>6.1</td><td>6.1 6.1</td><td>14.5</td><td>7.0</td><td>7.5</td><td>2.3</td><td>ACcu(%) -50</td></tr><tr><td>Qwen2.5-0.5B</td><td>19.2</td><td>7.0</td><td>6.1</td><td>0.0</td><td>0.9</td><td>0.9</td><td>0.9 0.9</td><td>6.5</td><td>3.3</td><td>2.8</td><td>0.5</td><td></td></tr><tr><td>Llama-3.2-1B</td><td>3.7</td><td>0.5</td><td>0.5</td><td>0.0</td><td>1.9</td><td>1.9</td><td>1.9 1.9</td><td>1.9</td><td>0.0</td><td>0.0</td><td>0.0</td><td></td></tr><tr><td>SmolLM2-360M</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>3.7</td><td>3.7</td><td>3.7 4.2</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td></td></tr><tr><td>SmolLM2-135M</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0 0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td></td></tr><tr><td>Baseline</td><td>Scaled</td><td>Perturbed</td><td></td><td>Extended</td><td>Baseline</td><td>Scaled Perturbed</td><td>Extended</td><td>Baseline</td><td>Scaled</td><td>Perturbed Extended</td><td></td><td>0</td></tr></table>

Figure 9: Complete direct-solving, abstraction, and transfer results on 214 matched pairs per condition. Scaled, perturbed, and extended denote the conditions in Table 7. Source prompts remain unchanged across columns; the scaled and perturbed conditions share generated abstractions. Values are exact numerical or algebraic-equivalence accuracies, as appropriate.
<table><tr><td>Condition</td><td>Largest target input</td><td>Construction</td></tr><tr><td>Baseline</td><td>Unscaled</td><td>Retains the target quantities assigned when the paired problems were con- structed.</td></tr><tr><td>Scaled quantities</td><td> $[ 1 0 ^ { 4 } , 1 0 ^ { 6 } )$ </td><td>Places the largest input in the five- or six-digit range while preserving structural ratios, fractions, and domain constraints.</td></tr><tr><td>Fractional perturbation</td><td> $[ 1 0 ^ { 4 } , 1 0 ^ { 6 } )$ </td><td>Uses the scaled condition, adding 0.01–0.99 to one eligible price or measure- ment in 66 problems. The other 148 problems remain unchanged.</td></tr><tr><td>Extended scale</td><td> $[ 1 0 ^ { 1 2 } , 1 0 ^ { 1 3 } )$ </td><td>Places the largest input in the 13-digit range while preserving the reference expression and domain constraints.</td></tr></table>

Table 7: GSM8K transfer dataset variants, each containing the same 214 pairs. Ranges describe the largest bound input per target problem, not every operand or the answer. Baseline and scaled quantities can overlap; the conditions are not disjoint magnitude bins.

Qwen3-8B obtains 99.8% symbolic accuracy with provided decomposition and 99.3% with model-generated decomposition; its numerical scores are 83.9% and 98.7%. Llama-3.1-8B-Instruct obtains 72.9% and 78.8% symbolically, compared with 97.2% and 84.6% numerically. Under direct prompting, Llama achieves 83.8% symbolically but only 27.5% numerically, despite

99.6% accuracy on the 1,029 unique isolated primitive calculations. Neither representation nor prompting condition is uniformly easier, and difficulty with composition cannot always be reduced to difficulty with the constituent calculations.

The operation-count comparison retains only provided and model-generated decomposition. At five operations, Qwen3-8B obtains 99.0% and 100.0% symbolic accuracy, respectively, compared with 72.5% and 50.0% for Llama-3.1-8B-Instruct. Performance is not uniformly monotonic in operation count. Each count contains different rules, with varying primitive choices, branching structures, and output magnitudes. The curves are therefore descriptive comparisons, not an isolated estimate of the effect of adding one operation.

![](images/1cf766489e28297f7eb4cb7f4c81c58cd4ef68650b6dc1c29ac55ebe0ece099a.jpg)

(a) Symbolic
<table><tr><td rowspan=1 colspan=4>(a) Symbonc</td></tr><tr><td rowspan=1 colspan=1>Qwen3-8B</td><td rowspan=1 colspan=1>88.5</td><td rowspan=1 colspan=1>99.8</td><td rowspan=1 colspan=1>99.3</td></tr><tr><td rowspan=1 colspan=1>Llama-3.1-8B</td><td rowspan=1 colspan=1>83.8</td><td rowspan=1 colspan=1>72.9</td><td rowspan=1 colspan=1>78.8</td></tr><tr><td rowspan=1 colspan=1>Qwen3-0.6B</td><td rowspan=1 colspan=1>21.8</td><td rowspan=1 colspan=1>73.4</td><td rowspan=1 colspan=1>70.8</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-0.5B</td><td rowspan=1 colspan=1>8.1</td><td rowspan=1 colspan=1>23.4</td><td rowspan=1 colspan=1>52.3</td></tr><tr><td rowspan=1 colspan=1>Llama-3.2-1B</td><td rowspan=1 colspan=1>31.8</td><td rowspan=1 colspan=1>35.2</td><td rowspan=1 colspan=1>11.1</td></tr><tr><td rowspan=1 colspan=1>SmolLM2-360M</td><td rowspan=1 colspan=1>7.3</td><td rowspan=1 colspan=1>2.6</td><td rowspan=1 colspan=1>2.0</td></tr><tr><td rowspan=1 colspan=1>SmolLM2-135M</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.2</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>Direct</td><td rowspan=1 colspan=1>Provided</td><td rowspan=1 colspan=1>Model-generated</td></tr><tr><td rowspan=1 colspan=1>decomposition</td><td rowspan=1 colspan=1>decomposition</td></tr></table>

(b) Numerical  
![](images/8c58ebbbb7c8901848bf1ceae5492f0dcfb8540ed7599767b50bb936e28062ab.jpg)  
Figure 10: Symbolic and numerical composition accuracy on 1,000 problems per condition. Direct prompting requests a final output; provided and model-generated decomposition request intermediate results. Scores measure presentation-normalized final-answer correctness, not correctness of every intermediate step.  
Figure 11: Symbolic composition by operation count with provided and model-generated decomposition. Each point averages 200 instances from ten rules. Direct prompting is omitted. These bounded explicit-rule tests do not establish unrestricted compositional generalization.

## I Complete ASDiv addition and subtraction results

Table 8 reports all seven models on the same 190 addition and subtraction problems. The main-text figure focuses on the two 8B models, but their original-to-abstracted gap is not universal: several smaller models match or exceed their numerical scores under five-shot prompting. Numerical answers use numeric equality, whereas abstracted answers use formula exact match rather than algebraic equivalence. The comparison therefore includes differences in both task representation and scoring criterion.

<table><tr><td colspan="3">Zero-shot</td><td colspan="2">Five-shot</td></tr><tr><td>Model</td><td>Orig.</td><td>Abstr.</td><td>Orig.</td><td>Abstr.</td></tr><tr><td>Qwen3-8B</td><td>90.0</td><td>64.7</td><td>91.6</td><td>76.3</td></tr><tr><td>Llama-3.1-8B</td><td>87.9</td><td>56.8</td><td>88.4</td><td>63.2</td></tr><tr><td>Qwen2.5-0.5B</td><td>14.2</td><td>7.9</td><td>52.1</td><td>58.9</td></tr><tr><td>Qwen3-0.6B</td><td>56.3</td><td>24.2</td><td>52.6</td><td>52.6</td></tr><tr><td>SmolLM2-360M</td><td>10.5</td><td>0.5</td><td>20.0</td><td>45.8</td></tr><tr><td>Llama-3.2-1B</td><td>58.4</td><td>7.9</td><td>67.9</td><td>39.5</td></tr><tr><td>SmolLM2-135M</td><td>0.0</td><td>0.0</td><td>1.6</td><td>15.8</td></tr></table>

Table 8: Complete ASDiv addition/subtraction scores (%), with 190 problems per condition. Original scores measure numerical-answer accuracy; abstracted scores measure formula exact match. Llama, Qwen2.5, and SmolLM2 entries use Instruct variants.

Unit sphere (3D) yes Do all points lie on the unit sphere? Test $x ^ { 2 } + y ^ { 2 } + z ^ { 2 } = 1$ for every point.   
$\begin{array} { r l r } {  { \bigl ( - \frac { 7 } { 2 7 } , \frac { \check { \textrm { \scriptsize { 2 } } } ^ { \star } } { 2 7 } , - \frac { 2 6 } { 2 7 } \bigl ) , \bigl ( - \frac { 2 } { 7 } , - \frac { 6 } { 3 5 } , - \frac { 3 3 } { 3 5 } \bigl ) , \bigl ( - \frac { 4 } { 1 3 } , - \frac { 3 } { 1 3 } , - \frac { 1 2 } { 1 3 } \bigl ) } \quad } & { { } } & { } \end{array}$ Convex hull (2D) no Is q in the convex hull of $V _ { 1 } , V _ { 2 } , V _ { 3 } ?$ Solve $q = \textstyle \sum _ { i } \lambda _ { i } V _ { i }$ $\textstyle \sum _ { i } \lambda _ { i } = 1$ ; membership holds iff every $\lambda _ { i } \geq 0 .$   
$\overline { { q } } \overset { \circ } { = } \bigl ( \frac { 4 6 } { 7 } , \frac { 1 9 } { 3 } \bigr ) ; V = ( 3 , - 4 ) , ( 1 3 , - 4 ) , ( 3 , 6 )$   
Collinearity $( d = 1 6 ,$ sparse) yes Are $P _ { 1 } , P _ { 2 } , P _ { 3 }$ collinear? Set $u = P _ { 2 } - P _ { 1 } , v = P _ { 3 } - P _ { 1 } ;$ take the first k with $u _ { k } \neq 0 ,$ set $t = v _ { k } / u _ { k }$ , and test v = t u for every i.   
$P _ { 1 } = \{ 4 \colon - 3 , 5 \colon 3 , 6 \colon 3 , 1 2 \colon 7 \} , P _ { 2 } = \{ 4 \colon - 7 , 5 \colon 9 , 6 \colon -$ $2 , 1 2 { : } 4 \} , P _ { 3 } = \{ 4 { : } - 1 5 , 5 { : } 2 1 , 6 { : } - 1 \hat { 2 } , 1 2 { : } - 2 \}$

Table 10: Example problems. Each prompt states the rule and asks for the substitution and exact calculation before a final yes/no. Sparse vectors list nonzero coordinates only.

## J Geometry rule-application experiment

<table><tr><td>Model</td><td>EM</td><td>Verified EM</td><td>Unverified</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>80.0</td><td>69.0</td><td>11.0</td></tr><tr><td>Qwen3-8B</td><td>66.2</td><td>64.8</td><td>1.4</td></tr><tr><td>Qwen3-32B</td><td>62.0</td><td>61.6</td><td>0.4</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>58.2</td><td>37.2</td><td>21.0</td></tr><tr><td>OLMo-2-0325-32B-Instruct</td><td>57.4</td><td>17.6</td><td>39.8</td></tr><tr><td>Qwen3-0.6B</td><td>51.2</td><td>8.6</td><td>42.6</td></tr><tr><td>Qwen2.5-0.5B-Instruct</td><td>22.2</td><td>3.4</td><td>18.8</td></tr><tr><td>SmolLM2-360M-Instruct</td><td>21.2</td><td>0.0</td><td>21.2</td></tr><tr><td>Llama-3.2-1B-Instruct</td><td>12.8</td><td>0.2</td><td>12.6</td></tr><tr><td>SmolLM2-135M-Instruct</td><td>0.6</td><td>0.0</td><td>0.6</td></tr></table>

Table 9: Geometry results (% of 500). EM is finalanswer exact match. Verified EM and Unverified partition EM: unverified responses have a correct answer but a derivation with at least one error or no adequate derivation.

Data. Each of the ten families has 25 matched yes/no pairs; negatives perturb one coordinate or scale one point. All labels are computed with exact rational arithmetic (Table 10).

Verification. Every response with a correct final answer was audited. Each displayed claim was checked against the input with exact rational arithmetic, and the derivation was labeled correct, incorrect (at least one uncorrected error, including errors that do not change the answer), or insufficient (no error, but no adequate derivation, e.g., a bare “Yes”). Only correct counts toward verified EM.

ADD(A, B, C, O):   
record the entry state (A, B, C, O)   
if A == 0 and B == 0:   
if C == 1: O = "1" + O   
return O with leading zeros removed, or "0"   
T = (A % 10) + (B % 10) + C   
D = T % 10   
return ADD(A // 10, B // 10, T // 10, str(D) + O)   
Start with ADD(A, B, 0, "").   
95 + 17: (95,17,0,"") → (9,1,1,"2") → (0,0,1,"12") → 112

Table 11: The recursive rule supplied in the prompt (abridged from the prompt’s register definitions), with an example execution. A, B are integer input registers, C a carry bit, and O a string that keeps its leading zeros until the base case.

## K Recursive addition: setup and complete results

Data. We sample 700 addition problems, 100 at each operand length $n \in \{ 5 , 1 0 , \ldots , 3 5 \}$ . Both operands have exactly n digits and are drawn uniformly; duplicate pairs, in either order, are rejected. Carry patterns are neither balanced nor selected.

Conditions. Every problem appears in three conditions. Direct asks for the sum only. CoT asks for a brief plan and a step-by-step calculation, followed by Answer: <sum>. Recursive rule adds the procedure in Table 11 and asks the model to output every entry state, including the initial state and the base case, as one JSON object per line with keys step, A, B, C, O, followed by Answer: <sum>. An n-digit problem requires n recursive calls and n + 1 recorded states. Each call uses only singlecolumn addition, integer division, remainder and string concatenation.

Decoding. All models run in vLLM with greedy decoding, an 8,192-token context, and the same 7,168-token output budget in every condition; Qwen3 thinking is disabled.

Scoring. EM reads the final answer line, tolerating presentation variants such as markdown or a restated “the sum is”, and counts unparseable or truncated responses as incorrect. Verified EM additionally requires the recorded states to equal the reference sequence. We extract every JSON state, tolerating code fences, prose, extra keys, numeric or string registers, and step labels. The output register O must keep its leading zeros, as the prompt requires. Comparing O numerically and ignoring states after the base case still yields no correct execution. We also check each call locally, from the model’s own previous state: whether the last digits of A and B are removed, whether the new carry equals $\lfloor { T } / 1 0 \rfloor$ , and whether O gains the digit T mod 10.

(b) CoT  
![](images/545b53da145083bacbb29a4d77d0fca17f44a98f14bbe85b898dd553521cc539.jpg)

![](images/7ece315a40af07d1c60aeb118c22851af636aeb38fa11b557b8b0465438a57f2.jpg)

(c) Recursive rule  
![](images/9b145bd17efec57fa48d217684dcccf9599f3aae69f4cb4f96a12efb3b135854.jpg)  
Figure 12: Answer EM (%) for direct prompting, CoT and the supplied recursive rule on the same 100 problems per length.

Results. Figure 12 compares the three conditions at 5–35 digits, where all fifteen model–condition runs are complete. Supplying the rule does not remove the decline with length. For Qwen3-32B, the recursive prompt gives higher EM than both baselines from 20 digits on (39% vs. 26% direct and 14% CoT at 35 digits; 28% vs. 13% direct at 50). For the other models it stays below the better baseline at every length except one (Qwen3- 8B at 10 digits). At 50 digits our Llama-3.1-70B-Instruct run covers only 28 of the 100 problems (none correct), so Figure 6 stops at 45 digits.

Qwen3-32B removes digits correctly in more than 99.9% of calls. Its new carry, however, is correct in only 49.4% of calls, about what a constant guess achieves (carries are 0 in 49% of reference steps), and its output digit in only 14.2%. In 941 of its 1,000 responses it also continues past the base case. Its answers are largely independent of its trace: only 28 of its 594 correct sums equal the value its final recorded state returns. Table 12 shows a typical case.

<table><tr><td>Step</td><td>A</td><td>B</td><td>Ref. (C, O) Qwen3-32B</td></tr><tr><td>0</td><td>55900</td><td>40560 0, &quot;&quot;</td><td>0, &quot;&quot;</td></tr><tr><td>1</td><td>5590</td><td>4056 0, &quot;0&quot;</td><td>0, &quot;0&quot;</td></tr><tr><td>2</td><td>559</td><td>405 0, &quot;60&quot;</td><td>0, &quot;00&quot;</td></tr><tr><td>3</td><td>55</td><td>40 0 1, &quot;460&quot;</td><td>0, &quot;000&quot;</td></tr><tr><td>4</td><td>5</td><td>4 0,  $" 6 4 6 0 "$ </td><td>0, &quot;0000&quot;</td></tr><tr><td>5</td><td>0</td><td>0  $0 , \ ^ { \prime \prime } 9 6 4 6 0 ^ { \prime \prime }$ </td><td>0, &quot;00000&quot;</td></tr><tr><td>6</td><td>0</td><td>0 一</td><td>1, &quot;00000&quot;</td></tr><tr><td>Answer line</td><td></td><td>96460</td><td>96460</td></tr><tr><td>Returned by final state</td><td></td><td>96460</td><td>100000</td></tr></table>

Table 12: Qwen3-32B on 55900 + 40560. It updates A and B correctly but prepends 0 to O at every step, adds a state after the base case, and still gives the correct sum.