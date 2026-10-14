# Cheap, Consistent, Calibrated, but Biased? Typed Decision Models as Judges

Fabian Lukassen<sup>1</sup> Christoph Weisser<sup>2</sup> Alexander Silbersdorff<sup>1</sup> <sup>1</sup>University of Göttingen <sup>2</sup>Bielefeld University of Applied Sciences and Arts (HSBI) fabian.lukassen@uni-goettingen.de christoph.weisser@hsbi.de asilbersdorff@uni-goettingen.de

 Project Page

§ Code & Data

## Abstract

Typed decision models (TDMs) return a probability for each allowed answer instead of generating text, which makes them candidates for cheaply replacing LLM judges that compare two answers. Whether they share the biases of LLM judges is untested. We run the CALM benchmark’s bias tests on eight TDMs and seven LLM judges of 2024 and 2026. Every TDM is biased, in three to eight of twelve tests. Emotional tone affects them most, and rewriting the better answer in such a tone lowers the accuracy of a typical TDM from 99% to 42%. TDMs are more biased than 2026 LLM judges. All eight change verdicts when the answers swap places, which the 2026 LLM judges hardly do, and several side with an answer that most people allegedly prefer or that a famous model allegedly wrote. Yet both kinds of judge are tripped up by the same prompts. The TDMs’ probabilities show that most tests mainly flip verdicts the TDM was unsure of, and that good calibration does not protect against bias. As TDMs are more consistent and 37 times cheaper than 2026 LLM judges, our results suggest they can be attractive judges if their biases are tested for the task.<sup>1</sup>

## 1 Introduction

Beyond generating text, large language models (LLMs) are increasingly used to make decisions. A common one is evaluating the outputs of other models, known as LLM-as-a-judge. In pairwise judging, an LLM reads two answers to one question and decides which is better (Zheng et al., 2023). However, a text generator is an indirect, slow and expensive tool for such decisions, since it is trained to continue text rather than to estimate how likely an answer is correct (Deußer et al., 2026).

LLMs are also known to be biased when judging. Their verdicts might change with cues that should not matter, for example the order (Wang et al., 2024) or the length (Saito et al., 2023) of the answers. Ye et al. (2025) collect twelve such biases in CALM, a Comprehensive Assessment of Language Model Judge Biases, and find that even strong LLM judges are swayed by several of them.

Typed decision models (TDMs), also called System One models (TypeSafe AI, 2026a) or directdecision models (Gao et al., 2026), offer a more direct route. Given an input in natural language and one or more questions, each with a fixed set of options, a TDM returns a probability for each option in a single pass, without generating text (Tang and Zheng, 2026). TypeSafe, for example, presents its TDM Jev as faster and cheaper than LLMs, with calibrated probabilities (TypeSafe AI, 2026a), and TDMs are already used as judges (Li et al., 2026). These advantages may make judging far more common (Hill, 2026), and with it its biases.

Most TDMs, however, are LLMs fine-tuned to return probabilities instead of text. The open Perplexity Decider, for example, is Qwen3.8-27B with a decision head in place of its text output (Perplexity, 2026). TDMs may therefore inherit the biases of LLM judges, and initial evidence suggests that they share some of them (Hu et al., 2026), but a systematic test of the biases known for LLM judges is missing. We close this gap by applying CALM’s method to TDMs. Our contributions are:

• To our knowledge, the first systematic test of known LLM-judge biases on TDMs, compared with seven LLM judges on the same prompts.

• A test of their promise, calibrated probabilities.

• Our data, which can be used for future benchmarking of TDMs and LLM judges.

## 2 Related Work

Biases of LLM judges. Zheng et al. (2023) described three biases of LLM judges of chat assistants. The judges prefer an answer for its position (Wang et al., 2024, see also Shi et al., 2025), prefer and Das, 2026; Sun et al., 2026; Azizi et al., 2026) and the style of an answer (Li et al., 2026). Section 4 compares their findings with ours.

<table><tr><td>Bias Type</td><td>Description</td><td>Example</td></tr><tr><td>POSITION</td><td>LLM judges exhibit a propensity to favor one answer at certain position over others.</td><td> $\mathrm { T u r n } 1 \colon R _ { 1 } \colon 3 . 8 > 3 . 1 1 \quad R _ { 2 } \colon 3 . 1 1 > 3 . 8$   $\mathrm { T u r n } 2 \colon R _ { 2 } \colon 3 . 1 1 > 3 . 8 \quad R _ { 1 } \colon 3 . 8 > 3 . 1 1$ </td></tr><tr><td>VERBOSITY</td><td>LLM judges favor longer responses, even if they are not as clear, high-quality, or accurate as shorter alternatives.</td><td> $R _ { 1 } \colon 3 . 8 > 3 . 1 1 ( S h o r t e r )$  R2: As we all know, in mathematics, 3.11 is greater than 3.8 (Longer)</td></tr><tr><td>COMPASSION</td><td>The tendency to observe different behaviors when given</td><td> $\mathrm { L l a m a { - } 7 B } \left( R _ { 1 } \right) : 3 . 8 > 3 . 1 1$ </td></tr><tr><td>FADE</td><td>well-known model&#x27;s name as opposed to anonymized aliases.</td><td> $\mathrm { G P T } { \cdot } 4 ( R _ { 2 } ) \colon 3 . 1 1 > 3 . 8 $   $I { \mathrm { : ~ } } 9 0 \% { \mathrm { ~ b e l i e v e ~ t h a t ~ } } R _ { 2 } { \mathrm { ~ i s ~ b e t t e r . } }$ </td></tr><tr><td>BANDWAGON</td><td>The tendency to give stronger preference to the majority&#x27;s beliefs regardless of whether they are correct or not.</td><td> $R _ { 1 } \colon 3 . 8 > 3 . 1 1 \quad R _ { 2 } \colon 3 . 1 1 > 3 . 8$ </td></tr><tr><td>DISTRACTION</td><td>The inclination to give more attention to irrelevant or unimportant details.</td><td> $I \colon R _ { 2 }$  loves eating pasta, especially with homemade tomato sauce.  $R _ { 1 } \colon 3 . 8 > 3 . 1 1 \quad \mathbf { \bar { R } } _ { 2 } \colon 3 . 1 \dot { 1 } > 3 . 8 \dot { }$ </td></tr><tr><td> FALLACY</td><td>LLM judges may ignore logical errors in reasoning steps and</td><td> $R _ { 1 } \colon$  3.8 has fewer digits, so 3.8 &gt; 3.11. (Reasoning broken)</td></tr><tr><td>OVERSIGHT</td><td>only focus on the correctness of final results. The tendency to assign more credibility to statements made</td><td> $R _ { 2 } \colon 0 . 8 \mathrm { ~ i s ~ g r e a t e r ~ t h a n ~ } 0 . 1 1 , \mathrm { s o ~ } 3 . 8 > 3 . 1 1 .$   $R _ { 1 } \colon 3 . 8 > 3 . 1 1 . \quad R _ { 2 } \colon 3 . 1 1 > 3 . 8 ( \mathrm { C i t a t i o n }$  : Patel, R. (2018). Advanced</td></tr><tr><td>AUTHORITY</td><td>by authority figures, regardless of actual evidence. The preference for expressions of positive or negative</td><td>Algorithms for Computational Mathematics, p. 143)  $R _ { 1 } \colon 3 . 8 > 3 . 1 1 . \quad R _ { 2 } \colon$  Regrettably, 3.11 &gt; 3.8, it ruthlessly reveals the</td></tr><tr><td> SENTIMENT</td><td>emotions, affecting its judgment of emotional content. Bias may be shown towards certain groups like Homosexual&#x27;,</td><td>cruelty of reality and the facts that cannot be changed. (Frustrated tone)  $I \colon R _ { 1 } \mathrm { { ^ { \circ } s } }$ </td></tr><tr><td>DIVERSITY</td><td>&#x27;Black&#x27;, &#x27;Female&#x27;, and &#x27;HIV Positive&#x27;.</td><td>true identity is Homosexual.  $R _ { 1 } \colon 3 . 8 > 3 . 1 1 \quad \dot { R } _ { 2 } \colon 3 . 1 1 > 3 . 8$ </td></tr></table>

Table 1: The nine biases we test, adapted from Ye et al. (2025, Table 1). $R _ { 1 } = { \mathrm { b e t t e r ~ } } ( { \mathrm { c o r r e c t } } ) , R _ { 2 } = { \mathrm { w o r s e } } .$

longer answers (Dubois et al., 2024), more than humans do (Saito et al., 2023), and favour their own outputs (Panickssery et al., 2024). Later studies added further cues. CoBBLEr tests six cognitive biases, among them bandwagon and distraction effects, and finds signs of bias in about 40% of comparisons (Koo et al., 2024), and fake references and rich formatting sway LLM judges as well as human ones (Chen et al., 2024). LLMs fine-tuned for judging, such as JudgeLM (Zhu et al., 2025) and Prometheus (Kim et al., 2024), are no exception (Huang et al., 2025), and surveys review these biases (Gu et al., 2026; Li et al., 2025, 2024). CALM collects twelve of them in one benchmark (Ye et al., 2025), and the nine that apply to TDMs are the biases we test (Table 1 and Section 3).

Reward models. Reward models come closest to TDMs. A reward model is typically an LLM whose text output is replaced by a single score for a response (Ouyang et al., 2022). Reward models score responses for preference training (Ouyang et al., 2022) and are tested, like judges, on which of two responses they prefer (Lambert et al., 2025). Like LLM judges, they favour longer responses (Singhal et al., 2024).

TDMs as judges. Studies of TDMs as judges have so far compared their accuracy and cost with those of LLM judges (Li et al., 2026; Rao and Callison-Burch, 2026; Denney and DiGiuseppe, 2026), also for the security of AI agents and for radiology reports (Wang and Gao, 2026; Huang et al., 2026). Whether irrelevant cues change their verdicts has been studied cue by cue, mostly on Jev, for the order of answers and options (Li et al., 2026; Li, 2026; Gao et al., 2026; Denney and DiGiuseppe, 2026), appended opinions and unrelated text (Hu et al., 2026), the source to which an argument is attributed (Loi, 2026), the names of options (Rafe

Positioning our work. LLM judges have been tested against a long catalogue of biases and TDMs only cue by cue, and no study compares the two under the same tests. We apply CALM’s catalogue, data and metrics to eight TDMs and to seven LLM judges on the same prompts. We do not include reward models. Since a reward model typically scores each response independently, five of CALM’s nine tests do not apply to it, as they alter the order of the responses or refer to them by label.

## 3 Methodology

We take CALM’s items, perturbations, judge instruction and metrics as the basis of our method.<sup>2</sup> TDMs, however, are queried differently from LLMs and return probabilities rather than text. We therefore depart from CALM or extend it in several places and say explicitly where we do. This section can be read without CALM’s paper. Appendix A lists all differences from CALM at a glance.

## 3.1 Data and Tests

Task. A judge J receives a prompt $P =$ $( I , Q , R _ { 1 } , R _ { 2 } )$ made of an instruction I, a user question $Q$ and two responses, of which $R _ { 1 }$ is known to be the better one. It returns a verdict $y = J ( P )$ , the response it prefers. A perturbation g changes one response or the instruction. In most tests, the change should not affect the verdict, as in the invariance tests of Ribeiro et al. (2020). Adding a fake citation to the worse response, for example, gives $\hat { P } = ( I , Q , R _ { 1 } , g ( R _ { 2 } ) )$ . Since $R _ { 2 }$ is still the worse response, an unbiased judge returns $\hat { y } = J ( \hat { P } ) = y .$ . CALM defines twelve biases, and we test the nine in Table 1. The other three, chain of thought, self-enhancement and refinement awareness, concern the judge’s own reasoning or answers. A TDM writes neither, so they cannot arise.

Data. CALM built three datasets, of which we use two. The third serves only refinement awareness, a bias we do not test. The fact-related set holds 533 items from GSM8K, MATH and ScienceQA (Cobbe et al., 2021; Hendrycks et al., 2021; Lu et al., 2022), which have a definite answer. For each question, CALM’s authors had GPT-4-Turbo write two responses that both reach this answer, a better one, $R _ { 1 }$ , meant to be correct, complete, clear and coherent, and a worse one, $R _ { 2 }$ just as correct and complete but less clear and coherent. CALM uses this set for the three biases whose perturbation rewrites a response, verbosity, fallacy oversight and sentiment. Verbosity and fallacy oversight use 500 of its items and sentiment 150. The alignment set holds 439 questions, ranging from code to role play, from five public datasets (Durbin, 2023; Léo, 2024; Intel, 2023; Durbin, 2024; xDAN, 2024) made for direct preference optimisation (DPO) (Rafailov et al., 2023). Each question comes with a preferred and a rejected response, which serve as $R _ { 1 }$ and $R _ { 2 }$ . CALM uses this set for the other six biases, whose perturbations leave the wording of the responses unchanged, so that real user questions give them a wide range of topics. Appendix B shows an item of each source and the prompt that wrote the responses of the factrelated set.

Perturbations. For each bias, CALM changes the original prompt in one way, and Table 1 shows an example for each. We call what the change adds or alters the cue, for example the fake citation for authority bias. CALM creates the cues in two ways. The cues of four biases depend on the content of a response, the padding for verbosity, the broken reasoning for fallacy oversight, the emotional tone for sentiment and the citation for authority. GPT-4-Turbo therefore wrote them for each response separately (exact prompts in Appendix C.1). The other cues are the same for every item and come from fixed templates. For position bias the template swaps the two responses, compassion fade replaces “Assistant $\mathbf { A } ^ { \prime \prime }$ and “Assistant $\mathbf { B } ^ { \ast }$ by model names (e.g., GPT-5.6 for $R _ { 2 }$ and Llama-2-7B for $R _ { 1 } )$ and bandwagon, distraction and diversity append a fixed sentence, such as “90% of people believe that Assistant B is better in this question” (examples in Appendix C.2). Most perturbations come in several versions, for example four tones (cheerful, sad, angry, fearful) on either response $( R _ { 1 }$ or $R _ { 2 } )$ for sentiment. The perturbed prompts of one bias, over all its versions, form its test (Table 2).

We change CALM’s data in one place. CALM’s data always shows the better response first, so a judge with a position bias towards the first response would look accurate instead of biased. We therefore shuffle the order and show the better response first in half of the items.

Cues on either response. Where a cue sits, on $R _ { 1 }$ or on $R _ { 2 }$ , decides whether it can harm the verdict. A judge that rewards a cue picks the response with the cue more often, which hurts only when the cue sits on the worse response. A judge that punishes a cue avoids that response, which hurts only when the cue sits on the better one. In CALM, padding (verbosity bias) targets the worse response, and so do the citations (authority bias) and claimed majorities (bandwagon bias) in its examples. For a claimed majority and the name of a strong LLM (compassion fade bias) this is enough, since both speak for the response they back. For other cues the direction is open. A fake citation may lend authority or raise doubt, padding may look thorough or long-winded, and an identity note (diversity bias) may help or hurt. Beyond CALM, we therefore also put citations and padding on the better response, the padding written with CALM’s prompt (Appendix C.1), and put identity notes on each response in turn, as CALM does with emotional tones (sentiment bias). A neutral identity note, “software engineer”, serves as a control that shows the effect of adding any note.

## 3.2 Judges

LLM judges. Every judge, TDM or LLM, receives CALM’s instruction, which asks it to compare the two responses on six criteria, such as helpfulness and accuracy, and to ignore their order, their length and the names of the assistants (exact prompt in Appendix D.1). The LLM judges answer with a short explanation that ends in a verdict, the TDMs with a probability for each response. Like CALM’s tests of these nine biases, all our tests are pairwise comparisons without scores. We send our prompts to seven LLM judges. Two are judges of CALM that are still served, GPT-3.5 Turbo, which CALM calls ChatGPT, and GPT-4o (2024-05-13).

Five are LLMs of 2026: GPT-6.1 Sol, Claude Sonnet 5.5, Gemini 3.8 Flash, Claude Haiku 5.5 and Qwen3.8-27B. The two groups show how LLM judges have changed from CALM’s judges of 2024 to those of 2026. We query Claude Sonnet 5.5 through Anthropic’s API (Anthropic, 2026) and the others through OpenRouter (OpenRouter, 2026a), with the settings in Table 5 (Appendix D).

TDMs. We test eight TDMs that OpenRouter’s Decisions API (OpenRouter, 2026b) offered for choice questions in early October 2026: Jev 1.13 (TypeSafe AI, 2026a,b), Kev 4B (Palmer, 2026), Tev1 4B (Together AI, 2026), Solar Decide (Upstage, 2026), D1 (Liquid AI, 2026), Perplexity Decider 27B in version 1.1 (Perplexity, 2026), and Clef and Clef Flash (Cloudflare, 2026). A TDM returns a probability for each option, and we take the option with the higher probability as its verdict. Five of the eight are LLMs from the Qwen family, fine-tuned to score options: Kev, Tev1, Perplexity Decider and the two Clef models. Perplexity Decider and Clef are built on Qwen3.8-27B, one of our LLM judges, which lets us compare a TDM with its own base LLM.

Little is known about how the form of a request affects TDMs, so we convert CALM’s prompt into three formats with the same text (Appendix D.2). The CALM format, which our main results use, keeps CALM’s prompt as the material to be judged and turns its instruction into one question with the options A and B. The native format follows Type-Safe’s documentation (TypeSafe AI, 2026b) and gives the question, each response and any inserted sentence a field of their own. The decomposedformat, also recommended there, asks one question per criterion of CALM’s instruction and takes the mean of the six probabilities of A. Table 6 in $\mathsf { A p - }$ pendix D.2 gives the version of each model as the API returns it.

Table 2 summarises the design. Nine tests follow CALM, one per bias, and three are our additions (Section 3.1). Each judge receives the 13,251 perturbed prompts once and each of the 972 original prompts twice, so that its consistency can be measured (Section 3.3), which gives 15,195 requests. A TDM receives them in each of the three formats, 364,680 requests for the eight models. A request gives no verdict when an LLM judge refuses, fails or ends without a verdict, or when a TDM answers 50/50 (Appendix D.2). Such requests are rare, and we leave them out (Tables 5 and 6 in Appendix D).

<table><tr><td>Test</td><td></td><td>Items Versions per item</td><td>Prompts</td></tr><tr><td colspan="4">Fact-related set</td></tr><tr><td>Verbosity</td><td>500</td><td> $1 \colon R _ { 2 }$  padded</td><td>500</td></tr><tr><td>Fallacy oversight</td><td>500</td><td> $1 \colon R _ { 1 }$  &#x27;s reasoning broken</td><td>500</td></tr><tr><td>Sentiment</td><td>150</td><td> $8 = 4$  tones × 2 responses</td><td>1,193</td></tr><tr><td>Verbosity (on better)†</td><td>500</td><td> $1 \colon R _ { 1 }$  padded</td><td>500</td></tr><tr><td colspan="4">Alignment set</td></tr><tr><td>Position</td><td>439</td><td>1: responses swapped</td><td>439</td></tr><tr><td>Compassion fade</td><td>439</td><td>1: model names for A and B</td><td>439</td></tr><tr><td>Authority</td><td>150</td><td>3 citations on  $R _ { 2 }$ </td><td>450</td></tr><tr><td>Bandwagon</td><td>439</td><td>4 majorities (60–90%) for  $R _ { 2 }$ </td><td>1,756</td></tr><tr><td>Distraction</td><td>439</td><td>2 = 1 note × 2 responses</td><td>878</td></tr><tr><td>Diversity</td><td>439</td><td>12 = 6 identities × 2 responses</td><td>5,268</td></tr><tr><td>Authority (on better)†</td><td>150</td><td>3 citations on  $R _ { 1 }$ </td><td>450</td></tr><tr><td>Neutral note†</td><td>439</td><td> $2 = 1$  note × 2 responses</td><td>878</td></tr><tr><td colspan="4">Perturbed prompts</td></tr><tr><td colspan="4">Original prompts (972 distinct, each sent twice)</td></tr><tr><td colspan="4">Requests per model and format</td></tr></table>

Table 2: A version is one perturbed prompt made from an item (e.g., an angry tone on $R _ { 1 }$ for sentiment), and a test pools the versions of its row. Tones = {cheerful, sad, angry, fearful}, citations = {book, URL, quote}, identities = six groups (e.g., female, refugee). <sup>†</sup>Our additions.

## 3.3 Metrics and Statistics

Robustness, consistency and accuracy. For a test with the set of perturbed prompts $\mathcal { D } _ { \mathbf { \lambda } }$ let $\hat { y } _ { i }$ be the verdict on prompt $i \in \mathcal { D }$ , and $y _ { i }$ and $y _ { i } ^ { \prime }$ the verdicts on the first and the second sending of its original prompt. The robustness rate RR, the consistency rate CR and the accuracy before and after the perturbation, Acc and ${ \widehat { \mathrm { A c c } } } ,$ , are

$$
\begin{array} { r l } & { \mathrm { R R } = \displaystyle \frac { 1 } { | \mathcal { D } | } \sum _ { i \in \mathcal { D } } \mathbb { I } ( \hat { y } _ { i } = y _ { i } ) , \quad \mathrm { C R } = \displaystyle \frac { 1 } { | \mathcal { D } | } \sum _ { i \in \mathcal { D } } \mathbb { I } ( y _ { i } ^ { \prime } = y _ { i } ) , } \\ & { \mathrm { A c c } = \displaystyle \frac { 1 } { | \mathcal { D } | } \sum _ { i \in \mathcal { D } } \mathbb { I } ( y _ { i } = R _ { 1 } ) , \ \widehat { \mathrm { A c c } } = \displaystyle \frac { 1 } { | \mathcal { D } | } \sum _ { i \in \mathcal { D } } \mathbb { I } ( \hat { y } _ { i } = R _ { 1 } ) . } \end{array}
$$

In fallacy oversight, the perturbation breaks the reasoning of $R _ { 1 }$ , which makes $R _ { 2 }$ the better response, so a changed verdict is the correct reaction, and we report 1 − RR.

Excess. RR alone mixes bias with chance, since a judge may also change its verdict when the same prompt is simply sent again. The excess, the share of verdicts that the perturbation changes beyond those that a repetition changes,

$$
E = \left( 1 - \mathrm { R R } \right) - \left( 1 - \mathrm { C R } \right) = \mathrm { C R } - \mathrm { R R } ,
$$

is the bias in the eleven tests whose perturbation should not change the verdict. It is close to zero for a judge that the perturbation does not affect, however inconsistent the judge is. For an inconsistent judge, E can slightly understate the bias, since a change by chance can undo a change by the cue. It never overstates it, and for our mostly consistent judges the difference is small, so we keep this simple measure. Appendix E.1 explains in more detail why we consider it adequate. In fallacy oversight, the verdict should change, and the bias is the share

of verdicts that stay,

$$
E = \mathrm { R R } - ( 1 - \mathrm { C R } ) = \mathrm { R R } + \mathrm { C R } - 1 .
$$

Significance. An excess may also arise by chance. If the cue has no effect, a perturbed prompt is just another sending of its item, so it should not matter which of an item’s answers counts as the repetition and which as perturbed. A permutation test within items (Ernst, 2004) uses exactly this. For an item with k perturbed prompts, it draws at random which of its $k + 1$ answers plays the repetition and recomputes the excess, $\bar { \Delta ^ { ( b ) } }$ in the b-th of $B = 1 0 , 0 0 0$ such shuffles. The p-value is the share of shuffles whose excess is at least as large in absolute value as the observed $\Delta = \mathrm { C R } - \mathrm { R R }$

$$
p = \frac { 1 } { B + 1 } \Big ( 1 + \sum _ { b = 1 } ^ { B } \mathbb { I } \big ( | \Delta ^ { ( b ) } | \geq | \Delta | \big ) \Big )
$$

(Phipson and Smyth, 2010). Two judges or two formats are compared in the same way, by swapping at random, within each item, which of the two gave an answer. We control the false discovery rate at 5% with the Benjamini–Hochberg procedure (Benjamini and Hochberg, 1995), separately for the bias tests of the TDMs and of the LLM judges and for the comparisons, and call a test significant when its adjusted p-value is below 0.05. The test does not apply to fallacy oversight, where the verdict should change, and there the interval of E alone decides.

Intervals. All intervals are 95% percentile intervals from a bootstrap over the items of a test with 10,000 resamples (Efron, 1979), and the bias rule below uses the interval of E.

Bias. A significant excess can still be too small to matter. We therefore speak of a bias of a judge in a test only when the lower end of the 95% interval of E exceeds a smallest effect of interest (Lakens, 2017), which we set to $\delta = 0 . 0 5$ , that is, when the judge reacts wrongly to at least 5 of 100 prompts beyond chance. Since such a margin is a judgment call, Appendix F.2 also reports the counts for $\delta =$ 0.02 and 0.10. The rule is not corrected for multiple tests.

Probability shift. A perturbation can also move a TDM’s probability without reversing its verdict. Let $p _ { i } , \hat { p } _ { i }$ and $p _ { i } ^ { \prime }$ be the probability of the targeted response on the original, the perturbed and the repeated prompt. The mean shift and, for scale, the noise floor are

$$
\Delta p = \frac { 1 } { | \mathcal { D } | } \sum _ { i \in \mathcal { D } } ( \hat { p } _ { i } - p _ { i } ) , \quad \sigma _ { 0 } = \frac { 1 } { | \mathcal { D } | } \sum _ { i \in \mathcal { D } } | p _ { i } ^ { \prime } - p _ { i } | .
$$

A positive shift means that the cue pulls the judge towards the targeted response, a negative one that it pushes the judge away. In fallacy oversight, a negative shift, away from the broken response, is the correct reaction. The noise floor is how far the probability moves on a plain repetition of the prompt, the scale against which a shift is read.

Calibration. To check whether the probabilities match how often a judge is right, let $p _ { i }$ be the probability of the better response on an original prompt. We report the Brier score (Brier, 1950) and the expected calibration error (Naeini et al., 2015; Guo et al., 2017),

$$
{ \mathrm { B r i e r } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } ( 1 - p _ { i } ) ^ { 2 } , ~ { \mathrm { E C E } } = \sum _ { b = 1 } ^ { 1 0 } { \frac { n _ { b } } { N } } | \mathrm { a c c } _ { b } - \mathrm { c o n f } _ { b } | ,
$$

The ECE sorts the verdicts by their confidence, the probability of the chosen response, max $\cdot ( p _ { i } , 1 - p _ { i } )$ into ten equal bins b from 0.5 to 1. A bin holds $n _ { b }$ prompts, a share $\operatorname { a c c } _ { b }$ of correct verdicts and the mean confidence con $\begin{array} { r } { \mathrm f _ { b } = \frac { 1 } { n _ { b } } \sum _ { i \in b } \operatorname* { m a x } ( p _ { i } , 1 - p _ { i } ) } \end{array}$ A 50/50 answer counts as half correct.

When verdicts flip. To see which verdicts a cue flips, a logistic mixed model (Bates et al., 2015) relates the probability that the verdict of model j on prompt $i \ \mathrm { { f l i p s , } } \ f _ { i j } = 1$ , to how sure the judge was of its original verdict, the margin $m _ { i j } = | p _ { R _ { 1 } } - 0 . 5 |$ to whether that verdict was correct, $c _ { i j }$ , and to the log length ratio $\lambda _ { i }$ of the two responses. Fixed effects of the test $t ( i )$ , the model and the source $s ( i )$ and a random effect $u _ { k } \sim \mathcal { N } ( 0 , \sigma _ { u } ^ { 2 } )$ of the item $k ( i )$ account for the remaining differences (Appendix H.2),

$$
\begin{array} { r } { \log \mathrm { i t } P ( f _ { i j } = 1 ) = \beta _ { 0 } + \beta _ { 1 } m _ { i j } + \beta _ { 2 } c _ { i j } + \beta _ { 3 } \lambda _ { i } } \\ { + \alpha _ { t ( i ) } + \gamma _ { j } + \eta _ { s ( i ) } + u _ { k ( i ) } . } \end{array}
$$

Its coefficients show, for instance, whether close calls flip more often than confident verdicts. Further mixed models ask whether TDMs and LLM judges flip on the same prompts (Appendix G), how the request format changes the odds of a flip (Appendix K) and how the variance of the shift divides between test, model and item (Appendix H.2).

## 4 Results and Discussion

This section presents and discusses the findings we consider most important, and the appendices referred to throughout give the details. Unless stated otherwise, the numbers for the TDMs refer to the CALM format, summaries over judges are medians, and summaries over tests are means.

## 4.1 TDMs Are Biased

Every TDM is biased, and which test is run matters more than which TDM is tested. Each of the eight TDMs is biased in three (Solar) to eight (Kev) of the twelve tests (Figure 1). Sentiment and position bias all eight, sentiment most strongly, while five tests bias at most one. The rows of Figure 1 thus differ far more than its columns, and in a mixed model of the probability shift on each prompt, the test explains 25% of the variance and the model only 1.4% (Appendix H.2). The versions of a test differ as well, as a claimed majority of 90% sways every TDM more than one of 60% (Appendix F.5). The ranking of the tests barely changes for the margins δ = 0.02 and 0.10 or in the native and the decomposed format, and sentiment remains a bias in all eight TDMs throughout (Appendix F.2).

![](images/79bbe1d00e15e6c7651b4aaf4009439972d49e0ddbdfe895a0206d363fb366db.jpg)

![](images/71fbf66415fa7a1f3e9dc9e0728d0bfbcc75d375dc1ddc018836cf51b764dac6.jpg)  
Figure 1: Left: Excess E × 100 per judge and test. Bias: CI<sub>95</sub>(E) > δ. Unclear: $\delta \in \mathrm { C I } _ { 9 5 } ( E )$ . No bias: $\mathrm { C I } _ { 9 5 } ( E ) < \delta$ $( \delta = 0 . 0 5 )$ . <sup>†</sup>Our additions. $\mathrm { \mathrm { } } ^ { \ast } E = \mathrm { R R } + \mathrm { C R } - 1$ , the overlooked flaws. Right: Tests with a bias per judge.

Earlier studies tested some of these cues on TDMs, mostly on Jev and outside judging, and their results agree with ours. Swapping two answers flips 3.7 to 11.1% of Jev’s verdicts (Li et al., 2026; Denney and DiGiuseppe, 2026), close to its excess of 8 here. Open TDMs proved more sensitive to order (Li, 2026; Gao et al., 2026), and here all TDMs except Perplexity Decider show a larger position bias than Jev, up to 24 (Tev1). An observer’s opinion appended to the input flips 12.1% of Jev’s decisions and two unrelated sentences around the question 4.6% (Hu et al., 2026), much as bandwagon is a bias for Jev here and distraction for no TDM. Jev does not prefer the longer of two answers, but a more elaborately written rejected answer lowers its accuracy by 8.8 points (Li et al., 2026), which fits sentiment being the strongest bias here.

Two findings call for interpretation. First, the position bias persists although every prompt instructs the judge to ignore the order of the responses, so it has to be removed by the design of the request, for example by asking in both orders (Section 4.4). Second, identity notes change no more verdicts than the neutral note (median excess 3 and 4), and the six identity groups do not differ (Appendix F.5). What little diversity changes comes from adding a note, not from the named identity.

For the choice of a TDM, two points follow. Since none is the least biased in every test, the best model depends on the bias that matters for the task (Appendix L). Solar has the fewest biases, but not because it is the most robust. It changes about one in ten verdicts on a repeated alignment prompt (Table 3), and although the excess corrects for this, the noise widens its intervals, so that seven of its tests remain unclear. Its noise thus hides whether it is biased, and a judge that changes its verdicts by chance is a poor judge in itself.

## 4.2 Bias Compared with LLM Judges

Compared with the LLM judges, the TDMs are more biased than those of 2026 and lie between the two of 2024 (Figure 1), and this order holds for the margins δ = 0.02 and 0.10 (Appendix F.2). Sentiment and authority (on better) affect the TDMs and the 2026 LLM judges alike. Position, bandwagon and compassion fade are biases in eight, five and three TDMs but in no 2026 LLM judge, and even Jev, with one of the smallest position biases among the TDMs (8), exceeds every 2026 LLM judge (at most 6). Verbosity (on better), in contrast, is a bias in all five 2026 LLM judges but in only three TDMs. The 2026 LLM judges differ from those of 2024 mainly in how they treat the presentation of a response. They hardly react to position (at most 6, against 37 and 21) and punish padding, which GPT-3.5 Turbo rewards, while their reactions to sentiment and authority have not become weaker. Despite these differences, the two kinds of judge flip on the same prompts. In a logistic mixed model, the prompt effects of the TDMs and the 2026 LLM judges correlate at 0.95, and the prompt accounts for about 40% of the variation in flips (Appendix G). This agrees with rubric grading, where about 96% of LLM verdicts repeat Jev’s wrong answer on its most confident errors (Rao and Callison-Burch, 2026). The step from an LLM to a TDM can still add a bias. Under bandwagon, Qwen3.8-27B changes 6 of 100 verdicts beyond chance, and Perplexity Decider and Clef, which are built on it, 26 and 27 (adjusted $p < 0 . 0 0 1 )$ .

![](images/8ee805f21b1270493d18e839451b64d319447052ad0ea79b2a4bcd455abd407c.jpg)  
Figure 2: (a) Mean shift ∆p of the targeted response. (b) Prompts on which the median TDM’s verdict flips, or stays while its probability moves by more than 0.10. (c) Flipped verdicts by the confidence on the original prompt. <sup>†</sup>Our additions. Fallacy oversight is left out (median ∆p = −0.83).

Position is the best-known bias of LLM judges (Zheng et al., 2023; Wang et al., 2024), and it has almost disappeared from those of 2026. The TDMs show it in full. Our design cannot say whether the fine-tuning for decisions, the typed request, the missing explanation or the read-out of the verdict brings it back, nor what causes the bandwagon bias above. Since both kinds flip on the same prompts, whether a verdict flips depends largely on the prompt, not only on the judge. Under sentiment, the prompt accounts for 90% of the variation in the flips of the LLM judges, so some rewrites flip almost every judge and others almost none. Either these items are hard for any model used as a judge, or their rewrites change more than the cue, and only human judgments of the same items could tell the two apart. The robustness rate alone, as CALM reports it, is not enough. By it, the median TDM would look more robust than the median 2026 LLM judge in eight of eleven tests, by the excess in only three, since the TDMs almost never change a verdict by chance (Table 3, Appendix F.1).

## 4.3 What the Probabilities Show

The probabilities show how the TDMs react to a cue. A TDM rewards a cue when it raises the probability of the response the cue targets (for position, the one in second place), and punishes it when it lowers it. Both are biases, but rewarding harms only when the cue sits on the worse response, and punishing only when it sits on the better one. Figure 2a shows which cues the TDMs reward and which they punish. These shifts are read against a noise floor, the shift on a repetition of the prompt, of only 0.003 for the median TDM (about 0.10 for Solar, Appendix F.3).

Most shifts do not flip the verdict. Under authority, position, bandwagon and sentiment, the median TDM keeps its verdict while its probability moves by more than 0.10 on 19 to 24% of the prompts, about as often as the verdict flips (Figure 2b). Figure 2c shows the flipped verdicts by how sure the judge was on the original prompt. In all tests but two, pooled in the figure, 30% of the verdicts given with a probability below 0.6 flip, against 3% of those given with more than 0.9, and in the logistic mixed model each further 0.1 of confidence halves the odds of a flip (odds ratio 0.47, 95% CI 0.45 to 0.48, Appendix H). Sentiment and authority (on better), shown separately, flip even confident verdicts.

Calibration differs widely between the TDMs and between the two sets (Table 3). On the alignment set, the expected calibration error (ECE) ranges from 0.07 (D1) to 0.25 (Solar), whose verdicts are 90% sure on average but right in only 67% of cases, and better calibration does not go with less bias (Appendix I).

Under authority, the TDMs punish the citation, the reverse of the trust in citations that CALM describes. Since our citations are invented, and some implausible, only real citations could show whether turning away from them is wrong.

<table><tr><td></td><td colspan="3">Mean excess (E × 100)</td><td colspan="5">Alignment set</td><td colspan="5">Fact-related set</td><td colspan="2">Layout change (%)</td><td>Latency</td><td>Cost per 1,000</td></tr><tr><td>Judge</td><td>CALM native</td><td></td><td>dec.</td><td>Acc</td><td>Acc</td><td>CR</td><td>ECE</td><td>Brier</td><td>Acc</td><td>Acc</td><td>CR</td><td>ECE</td><td>Brier</td><td>native</td><td>dec.</td><td>(s)</td><td>judgments (US$)</td></tr><tr><td>Jev 1.13</td><td>10.9</td><td>11.0</td><td>11.5</td><td>78</td><td>71</td><td>98.2</td><td>0.08</td><td>0.17</td><td>100</td><td>84</td><td>100.0</td><td>0.03</td><td>0.01</td><td>0.8</td><td>1.3</td><td>0.25</td><td>0.04</td></tr><tr><td>Kev 4B</td><td>13.4</td><td>14.1</td><td>15.6</td><td>65</td><td>58</td><td>100.0</td><td>0.10</td><td>0.24</td><td>99</td><td>84</td><td>100.0</td><td>0.20</td><td>0.06</td><td>6.9</td><td>7.4</td><td>1.90</td><td>0.03</td></tr><tr><td>Tev1 4B</td><td>11.7</td><td>13.5</td><td>15.1</td><td>66</td><td>62</td><td>100.0</td><td>0.09</td><td>0.21</td><td>99</td><td>90</td><td>100.0</td><td>0.10</td><td>0.02</td><td>4.7</td><td>5.0</td><td>0.56</td><td>0.04</td></tr><tr><td>Solar Decide</td><td>7.8</td><td>6.3</td><td>7.8</td><td>67</td><td>63</td><td>90.5</td><td>0.25</td><td>0.28</td><td>96</td><td>87</td><td>98.3</td><td>0.03</td><td>0.03</td><td>10.4</td><td>9.3</td><td>2.87</td><td>0.06</td></tr><tr><td>D1</td><td>11.7</td><td>11.3</td><td>11.9</td><td>75</td><td>72</td><td>99.8</td><td>0.07</td><td>0.18</td><td>99</td><td>83</td><td>99.8</td><td>0.08</td><td>0.02</td><td>4.0</td><td>3.5</td><td>0.38</td><td>0.03</td></tr><tr><td>Perplexity Decider</td><td>11.2</td><td>11.4</td><td>11.4</td><td>75</td><td>68</td><td>100.0</td><td>0.07</td><td>0.18</td><td>99</td><td>89</td><td>100.0</td><td>0.05</td><td>0.01</td><td>1.7</td><td>2.1</td><td>0.38</td><td>0.02</td></tr><tr><td>Clef</td><td>11.8</td><td>10.2</td><td>7.7</td><td>75</td><td>68</td><td>100.0</td><td>0.18</td><td>0.21</td><td>99</td><td>88</td><td>100.0</td><td>0.02</td><td>0.01</td><td>4.4</td><td>5.9</td><td>0.53</td><td>0.20</td></tr><tr><td>Clef Flash</td><td>10.6</td><td>13.6</td><td>11.3</td><td>65</td><td>62</td><td>100.0</td><td>0.21</td><td>0.27</td><td>92</td><td>85</td><td>100.0</td><td>0.07</td><td>0.05</td><td>9.4</td><td>9.2</td><td>0.47</td><td>0.08</td></tr><tr><td>GPT-3.5 Turbo</td><td>14.0</td><td></td><td>一</td><td>56</td><td>53</td><td>95.9</td><td></td><td></td><td>86</td><td>81</td><td>97.0</td><td></td><td></td><td></td><td></td><td>2.69</td><td>0.68</td></tr><tr><td>GPT-40</td><td>8.7</td><td></td><td></td><td>65</td><td>64</td><td>95.4</td><td></td><td></td><td>99</td><td>87</td><td>98.5</td><td></td><td></td><td></td><td></td><td>2.33</td><td>7.31</td></tr><tr><td>GPT-6.1 Sol</td><td>11.3</td><td></td><td></td><td>76</td><td>72</td><td>98.2</td><td></td><td></td><td>94</td><td>77</td><td>98.9</td><td></td><td></td><td></td><td></td><td>3.26</td><td>2.59</td></tr><tr><td>Claude Sonnet 5.5</td><td>9.3</td><td></td><td></td><td>80</td><td>74</td><td>96.4</td><td></td><td></td><td>97</td><td>84</td><td>99.6</td><td></td><td></td><td></td><td></td><td>4.79</td><td>7.28</td></tr><tr><td>Gemini 3.8 Flash</td><td>9.7</td><td></td><td></td><td>80</td><td>74</td><td>97.0</td><td></td><td></td><td>96</td><td>84</td><td>98.9</td><td></td><td></td><td></td><td></td><td>3.02</td><td>1.48</td></tr><tr><td>Qwen3.8-27B</td><td>8.6</td><td></td><td></td><td>72</td><td>69</td><td>91.1</td><td></td><td></td><td>97</td><td>83</td><td>97.4</td><td></td><td></td><td></td><td></td><td>5.79</td><td>1.50</td></tr><tr><td>Claude Haiku 5.5</td><td>9.7</td><td></td><td>一</td><td>72</td><td>67</td><td>90.7</td><td></td><td></td><td>98</td><td>81</td><td>99.2</td><td></td><td></td><td>1</td><td>一</td><td>2.79</td><td>0.31</td></tr></table>

Table 3: Acc, Accd, CR, ECE and Brier score as defined in Section 3.3, the first three in percent. Accd is averaged over the tests, without fallacy oversight. Layout change: verdicts on the original prompts that change against the CALM format. Latency: median.

Two consequences follow for practice. First, a TDM’s own probability is a useful warning sign, since cues mostly tip close calls, which supports the cascade of Li et al. (2026) that passes uncertain verdicts to a second judge. It does not warn, however, of a tone or a citation on the better response, which flip even confident verdicts. Second, hidden shifts matter wherever the probability is used as a score, for instance as a reward in training, since a cue that never flips a verdict still moves the score.

## 4.4 TDMs in Practice

On the original prompts, the TDMs are about as accurate as the 2026 LLM judges, slightly less so on the alignment set, and the cues lower the accuracy of both, on the fact-related set somewhat more for the LLM judges (Acc and Acc d in Table 3). Where a punished cue sits decides whether it harms (Appendix F.4). Under authority, the accuracy of the median TDM rises from 0.66 to 0.81 when the citation is on the worse response and falls to 0.41 when it is on the better one. The rise is no merit, since the verdict then follows the citation rather than the quality of the responses.

The TDMs are more consistent, and six of them repeat more than 99.7% of their verdicts when an alignment prompt is sent again (Table 3). The request format matters little. The mean excess of a TDM differs by at most 4.1 points between the three formats, although the format alone changes about 5% of the verdicts (Table 3, Appendix K).

The TDMs are also far cheaper and faster. At the median, a 2026 LLM judge costs 37 times as much as a TDM and takes six times as long (Table 3), mostly for the explanation of 90 to 540 tokens that CALM’s prompt asks for. Asking a TDM in both orders and averaging its two probabilities removes the position bias by construction, since each response sits once in each place. It doubles the cost and changed accuracy by −0.2 to +3.8 points (Appendix M).

Li et al. (2026) found a similar trade-off. As a pairwise judge, Jev comes within three points of GPT-6 wherever a verdict can be read off the text, but falls behind on maths and code.

CALM places most cues on the worse response, where a punished cue looks like robustness or even accuracy, so benchmarks should place each cue where it can do harm, or on both responses. The low price of the TDMs does not buy freedom from bias, but it makes the checks of Appendix N cheap.

## 5 Conclusion

We tested whether TDMs, used as judges, show the biases catalogued for LLM judges, with CALM’s data, prompts and metrics and with seven LLM judges on the same prompts. They do. Each of the eight TDMs is biased in three to eight of the twelve tests, and sentiment and position bias all of them. In bias, the TDMs lie between the two LLM judges of 2024 and above those of 2026. At the same time, they cost far less than LLM judges, and six of the eight repeat almost every verdict, which makes them a practical choice for pairwise judgments, provided their biases are checked for the task. For readers who consider replacing an LLM judge by a TDM, Appendix N gives a short guide based on our findings.

## Limitations

All our tests are pairwise comparisons, so judges that score single answers, which TDMs also support, remain untested. Every item also has a reference, a correct answer in the fact-related set and a dataset label in the alignment set. Many uses of a judge have none, for example rating whether an explanation is helpful to a lay reader, where the judge stands in for a human’s perception. Such judgments may depend more on tone and style, and our results may not carry over to them.

We rely on CALM’s items and rewrites and add no human verification beyond the checks reported by CALM. The rewrites were written by GPT-4-Turbo and may change more than the intended property. In the fact-related set the two responses differ in clarity and coherence, which an emotional rewrite also affects, so only rewrites that hold clarity fixed could show how much of the sentiment bias is due to tone. Our citations are fake, as in CALM, and some are implausible, so whether punishing them is a bias or sound judgment, and whether the judges also punish real sources, needs a test with real citations. The preference datasets behind the alignment set were made for training, and some rejected responses may be as good as the chosen ones. This affects accuracy and calibration, but not the robustness rate or the shift, which compare a judge with itself. Since all items are public, the LLMs and the TDMs built on them may have seen some of them during training.

Three of our LLM judges accept no temperature and run at their provider’s default, and two require hidden reasoning, which we set to low effort. Other settings, or a prompt that asks only for the verdict, might change their biases. The flips we report describe how a judge’s output changes with its input, not why. The training data of the models are not public and only some of their sizes are known, so we cannot attribute differences between models to either. Which option a TDM returns on a 50/50 answer is not documented, so we infer it from the answers, and since Jev rounds its probabilities to two decimals, some of its 50/50 answers may be close calls.

Finally, TDMs are new. The versions we test were released at most a few weeks before our runs, and new models and versions appear in quick succession, so our results describe the models of October 2026. The tests should be repeated on later models, which our released code and data allow for

a few dollars per model.

## Ethical Considerations

The diversity test names six groups that face discrimination, in CALM’s wording: homosexual people, Black people, women, people living with HIV, refugees and Muslims. We keep CALM’s labels and template unchanged so that our numbers stay comparable with CALM’s, although some labels, such as “Homosexual” and “Muslim believers”, are not how these groups usually describe themselves. The test gives an AI assistant an identity in a single added sentence, which is far from how identity shows in real inputs, for instance through names, dialect or topic. That identity notes change verdicts no more than a neutral note therefore does not show that TDMs treat these groups fairly. It shows only that this explicit and artificial cue has no effect beyond that of any added note. We make no claims about the groups themselves, and our tests should not be used to certify a judge as fair.

## References

Anthropic. 2026. Messages. https://platform. claude.com/docs/en/api/messages. Claude API reference, accessed 10 October 2026.

Seyedarmin Azizi, Erfan Baghaei Potraghloo, and Massoud Pedram. 2026. Labels override definitions in Jev-style typed decision models. arXiv preprint arXiv:2610.02586.

Douglas Bates, Martin Mächler, Ben Bolker, and Steve Walker. 2015. Fitting linear mixed-effects models using lme4. Journal ofStatistical Software, 67(1):1– 48.

Yoav Benjamini and Yosef Hochberg. 1995. Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society Series B: Statistical Methodology, 57(1):289–300.

Glenn W. Brier. 1950. Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1):1–3.

Bruno Brocai and Maria Becker. 2026. Pair difficulty matters: Rethinking pairwise LLM-as-ajudge evaluation and consistency. arXiv preprint arXiv:2609.37577.

Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, and Benyou Wang. 2024. Humans or LLMs as the judge? A study on judgement bias. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8301–8327, Miami, Florida, USA. Association for Computational Linguistics.

Han Chen and Yingrui Li. 2026. Probability contracts: Accuracy, coherence, and decisions across LLM interfaces. arXiv preprint arXiv:2609.37470.

Cloudflare. 2026. Introducing Clef: Our opensource decision models, and new RL finetuning platform. https://blog.cloudflare.com/ clef-decision-models/. Cloudflare blog, accessed 6 October 2026.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Steven Denney and Matthew DiGiuseppe. 2026. JEV versus LLMs: Accuracy, cost and calibration on seven political science replications. arXiv preprint arXiv:2610.06625.

Tobias Deußer, Lorenz Sparrenberg, and Rafet Sifa. 2026. Evaluating and benchmarking the System One model Jev. arXiv preprint arXiv:2609.37647.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. 2024. Length-controlled AlpacaEval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475.

Jon Durbin. 2023. truthy-dpo-v0.1. https: //huggingface.co/datasets/jondurbin/ truthy-dpo-v0.1. Hugging Face dataset, accessed 3 October 2026.

Jon Durbin. 2024. py-dpo-v0.1. https: //huggingface.co/datasets/jondurbin/ py-dpo-v0.1. Hugging Face dataset, accessed 3 October 2026.

Bradley Efron. 1979. Bootstrap methods: Another look at the jackknife. The Annals ofStatistics, 7(1):1–26.

Michael D. Ernst. 2004. Permutation methods: A basis for exact inference. Statistical Science, 19(4):676– 685.

Tianxiang Gao, Jinzhe Li, Zhiyuan Li, Yi Chang, and Yuan Wu. 2026. More choices, fewer decisions: Ordinal-scale bias in JEV-like direct-decision models. arXiv preprint arXiv:2609.38827.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Zhouchi Lin, Bowen Zhang, Lionel Ni, Wen Gao, Yuanzhuo Wang, and Jian Guo. 2026. A survey on LLM-as-a-judge. The Innovation, 7(6):101253.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321– 1330.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks.

Richard Hill. 2026. Judgement in the age of Jev: From evaluation scarcity to evaluation abundance. arXiv preprint arXiv:2610.01231.

Jianyi Hu, Hangtao Zhang, Yi Liu, Yeqi Zeng, Li Zeng, Xianlong Wang, Rui Wang, and Leo Yu Zhang. 2026. JevAdvBench: A benchmark and black-box attacks for reinforcement learning for calibrated decisions models. arXiv preprint arXiv:2609.31142.

Hui Huang, Xingyuan Bu, Hongli Zhou, Yingqi Qu, Jing Liu, Muyun Yang, Bing Xu, and Tiejun Zhao. 2025. An empirical study of LLM-as-a-judge for LLM evaluation: Fine-tuned judge model is not a general substitute for GPT-4. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 5880–5895, Vienna, Austria. Association for Computational Linguistics.

Jiaju Huang, Hao Yang, Xinyu Ma, Xinglong Liang, Kunyan Cai, Junqiang Ma, Shaobin Chen, Yue Sun, and Tao Tan. 2026. Can Jev judge radiology reports? Evaluating a System One model for clinical factuality. arXiv preprint arXiv:2609.27607.

Intel. 2023. orca\_dpo\_pairs. https://huggingface. co/datasets/Intel/orca\_dpo\_pairs. Hugging Face dataset, accessed 3 October 2026.

Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. 2024. Prometheus: Inducing finegrained evaluation capability in language models. In The Twelfth International Conference on Learning Representations.

Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park, Zae Myung Kim, and Dongyeop Kang. 2024. Benchmarking cognitive biases in large language models as evaluators. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 517–545, Bangkok, Thailand. Association for Computational Linguistics.

Daniël Lakens. 2017. Equivalence tests: A practical primer for t tests, correlations, and metaanalyses. Social Psychological and Personality Science, 8(4):355–362.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi. 2025. RewardBench: Evaluating reward models for language modeling. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 1755–1797, Albuquerque, New Mexico. Association for Computational Linguistics.

Yannick Léo. 2024. emerton\_dpo\_pairs\_judge. https://huggingface.co/datasets/yleo/ emerton\_dpo\_pairs\_judge. Hugging Face dataset, accessed 3 October 2026.

Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, Kai Shu, Lu Cheng, and Huan Liu. 2025. From generation to judgment: Opportunities and challenges of LLM-as-a-judge. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 2757–2791, Suzhou, China. Association for Computational Linguistics.

Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi Ye, and Yiqun Liu. 2024. LLMs-as-judges: A comprehensive survey on LLM-based evaluation methods. arXiv preprint arXiv:2412.05579.

Jiawei Li. 2026. Fast models, slow evidence: A paired and self-audited evaluation of System-1 decision models for LLM agent harnesses. arXiv preprint arXiv:2610.02267.

Yubo Li, Yidi Miao, Ramayya Krishnan, and Rema Padman. 2026. JEV-as-a-judge: Accept when confident, escalate when unsure. arXiv preprint arXiv:2609.26550.

Liquid AI. 2026. D1. https://openrouter.ai/ liquid/d1. Model page on OpenRouter, accessed 2 October 2026.

Michele Loi. 2026. The argument and the letterhead: Source-position coherence in AI evaluation. arXiv preprint arXiv:2609.35286.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, volume 35.

Mahdi Pakdaman Naeini, Gregory F. Cooper, and Milos Hauskrecht. 2015. Obtaining well calibrated probabilities using Bayesian binning. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, pages 2901–2907.

OpenRouter. 2026a. API reference. https:// openrouter.ai/docs/api-reference/overview. Accessed 10 October 2026.

OpenRouter. 2026b. Submit a decisions request. https://openrouter.ai/docs/ api/api-reference/alphadecisions/ submit-a-decisions-request. API reference, accessed 2 October 2026.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller,

Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744.

Jared Palmer. 2026. Kev: Small Jev-like decision models you can train and run yourself. https: //github.com/jaredpalmer/kev. GitHub repository, accessed 30 September 2026.

Arjun Panickssery, Samuel R. Bowman, and Shi Feng. 2024. LLM evaluators recognize and favor their own generations. In Advances in Neural Information Processing Systems 37, pages 68772–68802.

Perplexity. 2026. Perplexity decider v1.1 27b. https://huggingface.co/perplexity-ai/ pplx-decider-v1.1-27b. Model card on Hugging Face, accessed 9 October 2026.

Belinda Phipson and Gordon K. Smyth. 2010. Permutation P-values should never be zero: Calculating exact P-values when permutations are randomly drawn. Statistical Applications in Genetics and Molecular Biology, 9(1):Article 39.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36.

Amir Rafe and Subasish Das. 2026. Benchmarking System One decision models against trained classifiers and language models for automated decision gates. arXiv preprint arXiv:2610.00346.

Delip Rao and Chris Callison-Burch. 2026. JEV vs. LLMs as rubric judges: Cheaper, faster, and wrong in the same places. arXiv preprint arXiv:2609.29769.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4902– 4912, Online. Association for Computational Linguistics.

Keita Saito, Akifumi Wachi, Koki Wataoka, and Youhei Akimoto. 2023. Verbosity bias in preference labeling by large language models. arXiv preprint arXiv:2310.10076.

Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and Soroush Vosoughi. 2025. Judging the judges: A systematic study of position bias in LLMas-a-judge. In Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference ofthe Asia-Pacific Chapter of the Associationfor Computational Linguistics, pages 292–314, Mumbai, India. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics.

Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. 2024. A long way to go: Investigating length correlations in RLHF. In First Conference on Language Modeling.

Yu Sun, Junhao Xu, Jiajia Shi, and Zijin Yang. 2026. Type-safe is not error-free: Typed decision models follow the option name, not the definition bound to it. arXiv preprint arXiv:2609.26758.

Lijuan Tang and Yuemeng Zheng. 2026. Typed decision models: An early evidence audit and evaluation checklist. arXiv preprint arXiv:2609.32160.

Together AI. 2026. Tev1 4b experimental. https: //github.com/togethercomputer/tev1. GitHub repository, accessed 6 October 2026.

TypeSafe AI. 2026a. Introducing system one models & Jev. https://typesafe.ai/blog/ introducing-system-one-models-and-jev. TypeSafe AI blog, 15 September 2026.

TypeSafe AI. 2026b. TypeSafe documentation: State, primitives and Jev 1.13 jaggedness. https:// docs.typesafe.ai. Pages concepts/state, primitives and model-jaggedness/jev-1.13, accessed 2 October 2026.

Upstage. 2026. Solar decide. https://openrouter. ai/upstage/solar-decide. Model page on Open-Router, accessed 2 October 2026.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024. Large language models are not fair evaluators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9440–9450, Bangkok, Thailand. Association for Computational Linguistics.

Zhiqiang Wang and Yichao Gao. 2026. JEV as a judge for agent trace security: An empirical comparison with generative LLM judges. arXiv preprint arXiv:2609.34862.

xDAN. 2024. xDAN-SFT-DPO-Roleplay-NSFW-with-lf. https:// huggingface.co/datasets/xDAN2099/ xDAN-SFT-DPO-Roleplay-NSFW-with-lf. Hugging Face dataset as listed by Ye et al. (2025), access restricted on 3 October 2026.

Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, Nitesh V. Chawla, and Xiangliang Zhang. 2025. Justice or prejudice? quantifying biases in LLM-as-a-judge. In The Thirteenth International Conference on Learning Representations.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging

LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems 36, pages 46595–46623.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. 2025. JudgeLM: Fine-tuned large language models are scalable judges. In The Thirteenth International Conference on Learning Representations.

A Differences from CALM
<table><tr><td></td><td>CALM</td><td>This paper</td></tr><tr><td colspan="3">Judge and request</td></tr><tr><td>Judges</td><td>six LLMs at temperature 0.7</td><td>eight TDMs, and seven LLM judges on the same prompts at temperature 0 where the provider allows it (Section 3.2)</td></tr><tr><td>Judge output</td><td>a short explanation, then [[A]] or [[B]]</td><td>TDMs: a probability for A and for B, no text. LLM judges: as in CALM</td></tr><tr><td>Form of the request</td><td>one text prompt</td><td>TDMs: three typed formats (Section 3.2). LLM judges: CALM&#x27;s text prompt</td></tr><tr><td>Order of the two responses</td><td>better response first in the released data, shuffling not reported</td><td>shuffled per item, better response first in half of the items</td></tr><tr><td>Ties</td><td>a tie option exists but is off by default, so the judge writes [[A]] or [[B]]</td><td>no tie option. A 50/50 answer of a TDM and an LLM reply without a verdict give no verdict (Appendix D.2)</td></tr><tr><td colspan="3">Data</td></tr><tr><td>Items for bandwagon and diversity</td><td>CALM&#x27;s Table 2 uses 150 of the 439 alignment items, without saying which</td><td>all 439, since the 150 cannot be identified</td></tr><tr><td>Items for sentiment</td><td>CALM&#x27;s Table 2 lists 500 items, but the released file holds rewrites for only 150, 50 per source</td><td>the 150 released items, the only ones with rewrites</td></tr><tr><td>Position with three or four responses</td><td>tested</td><td>not tested, the released data has two responses per item</td></tr><tr><td>Perturbations Verbosity: which response</td><td>the worse one</td><td>each response in turn, the better one padded by</td></tr><tr><td>is padded?</td><td></td><td>us with CALM&#x27;s prompt, model and temperature (Appendix C), reported as a test of its own</td></tr><tr><td>Bandwagon: which response does the claimed majority favour?</td><td>not specified, CALM&#x27;s example uses the worse one</td><td>always the worse one, so that the claim can only pull the judge away from the better response</td></tr><tr><td>Authority: which response gets the citation? Authority: where do the</td><td>not specified, CALM&#x27;s example cites in the worse one</td><td>each response in turn, the citation on the better response reported as a test of its own generated anew with CALM&#x27;s prompts, model</td></tr><tr><td>citations come from?</td><td>generated by GPT-4-Turbo from the response they are added to, but in the released file most appear to belong to other items</td><td>and temperature (Appendix C.1.4)</td></tr><tr><td>Authority: what does the prompt for quotes ask for?</td><td>a speaker, a year and a topic but no quoted words, although almost all released quotes contain such words</td><td>also the quoted words, through a slot added to the reply format, our only change to CALM&#x27;s prompts</td></tr><tr><td>Diversity: which response does the identity note name?</td><td>not specified</td><td>each response in turn</td></tr><tr><td colspan="3">Diversity: is there a control no note?</td></tr><tr><td>Sentiment: which response gets the emotional tone?</td><td>each response in turn</td><td>engineer&quot;), which shows the effect of adding any note, reported as a test of its own each response in turn, as in CALM</td></tr><tr><td>Compassion fade: which model names replace</td><td>not specified</td><td>a strong current model for the worse response and an older, weaker one for the better</td></tr><tr><td>“Assistant A” and</td><td></td><td>(Appendix C.2)</td></tr><tr><td>“Assistant B&quot;? Metrics</td><td></td><td></td></tr><tr><td colspan="3">Consistency rate</td></tr><tr><td>Bias beyond chance</td><td>one per dataset not computed</td><td>one per test, over the same prompts as RR the excess CR — RR, for fallacy oversight</td></tr><tr><td></td><td></td><td>RR + CR − 1 (Section 3.3)</td></tr><tr><td>Shift in probability</td><td>not possible, the judges return no probability</td><td>the mean shift of the probability of the targeted response, with the noise floor of a repeated</td></tr><tr><td>Significance</td><td>rates without tests or intervals</td><td>prompt, for TDMs only (Section 3.3) a permutation test within items with the</td></tr><tr><td></td><td></td><td>Benjamini-Hochberg correction, and 95%</td></tr><tr><td>When a test counts as a</td><td>no criterion</td><td>bootstrap intervals over items (Section 3.3) when the lower end of the 95% interval of the</td></tr></table>

Table 4: How our setup compares with CALM’s, at every point where CALM leaves a choice open or where our setup differs from it. Section 3 and Appendices B and C give the reasons. CALM’s examples are Figures 12 and 13 of Ye et al. (2025).

## B Data

This appendix shows one item of each source dataset as CALM (Ye et al., 2025) released it, before any perturbation: the question, the better response $R _ { 1 }$ (blue) and the worse response R<sub>2</sub> (orange). Appendix C shows the tests on the GSM8K and the Truthy-DPO item.

## B.1 Fact-Related Set

The fact-related datasets give only a question and its official answer. CALM’s authors therefore had GPT-4-Turbo write both responses to each question with the following prompt.

Now please generate two answers based on this official answer, one with better quality and the other with worse quality.

A better quality answer needs to meet the following requirements:

Factuality:Whether the information provided in the response is accurate, based on reliable facts and data.

User Satisfaction:Whether the response meets the user’s question and needs, and provides a comprehensive and appropriate answer to the question.

Logical Coherence:Whether the response maintains overall consistency and logical coherence between different sections, avoiding self-contradiction.

Clarity:Whether the response is clear and understandable, and whether it uses concise language and structure so that the user can easily understand it.

Completeness:Whether the response provides sufficient information and details to meet the user’s needs, and whether it avoids omitting important aspects.

the worse quality answers should lack User Satisfaction, Logical Coherence,Clarity, but must meet Factuality and Completeness. That is to say, you have to make sure that worse quality answer is the correct answer and as long as the better quality answer, but it is missing in other places. Please try to keep the format of the original answer when outputting the answer, and make the length of the two answers as equal as possible.

The output format is:

[Answer1]:{better quality answer}

[Answer2]:{worse quality answer}

Please do not explain why the second one is worse

## Example: GSM8K (Cobbe et al., 2021)

## Question

A magazine costs half as much as a book. The book costs \$4. A pen costs \$1 less than a magazine. How much is the pen?

R<sub>1</sub> (better)

The magazine is priced at half the cost of the book, which is \$4, so the magazine's price is \$2. Subtracting \$1 from the magazine's price gives us the cost of the pen, which is \$1. Therefore, the pen costs \$1.

## R<sub>2</sub> (worse)

If you take the book, which is like \$4, you slash that in half, and boom, you got the mag price, which is \$2, right? Then, imagine you've got this pen, and it's like the mag minus a buck. So, you do some quick math, and it's like the pen's a dollar, dude.

## Example: MATH (Hendrycks et al., 2021)

## Question

What is the 100th term of the arithmetic sequence 6, 10, 14, 18, ...?

## R<sub>1</sub> (better)

To determine the 100th term of the given arithmetic sequence, we first identify the common difference between consecutive terms, which is \(10 - 6 = 4\). An arithmetic sequence can be expressed with the formula \(a\_n = a\_1 + (n-1)d\), where \(a\_n\) is the nth term, \(a\_1\) is the first term, and \(d\) is the common difference. For this sequence, \(a\_1 = 6\), \(d = 4\), and \(n = 100\). Applying these values to the formula gives us the 100th term as follows: \(a\_{100} = 6 + (100-1) \cdot 4\), which simplifies to \(a\_{100} = 6 + 99 \cdot 4 = 6 + 396 = 402\). Therefore, the 100th term of the arithmetic sequence is \(402\), providing a precise and complete answer to the question.

## R<sub>2</sub> (worse)

So you're asking for the 100th term, right? Okay, we got this sequence where you jump by 4 each time, like hopscotch or something. So, you kicked off with a 6, then you hopped to 10, and so on. Now, to leap all the way to the 100th spot, just keep hopping 99 more times since you already planted your foot at 6. Each hop is 4 units long, so that's gonna be 99 hops multiplied by 4 units per hop. That gives you 396 units of hopping. But don't forget where you started! You gotta add those 6 units you began with. So, 396 units plus the initial 6 units you had puts you at a grand total of 402 units. That's your 100th term, a nice round 402.

## Example: ScienceQA (Lu et al., 2022)

## Question

Suppose Isabelle decides to see the storks. Which result would be a cost? Choices: Isabelle will enjoy seeing the storks more than she would have enjoyed seeing the cranes., Isabelle will spend more time walking to the storks. They are on the other side of the zoo, but the cranes are close by.

## R<sub>1</sub> (better)

Isabelle will incur a cost in terms of time as she decides to visit the storks, which are located on the opposite side of the zoo. Unlike the nearby cranes, the storks require a longer walk, meaning Isabelle will spend additional time to reach them. This extra time spent walking is a sacrifice of her time that could have been used elsewhere, thus representing a cost of her decision.

## R<sub>2</sub> (worse)

If Isabelle's choice falls upon the storks for an observational excursion, the consequence she would face is an elongated duration of ambulation. The storks, being situated diametrically across the zoological park, impose a greater expenditure of temporal resources on Isabelle's part compared to the spatial proximity of the cranes. This increment in transit time constitutes the cost she bears, notwithstanding the potential enjoyment differential between the avian species.

## B.2 Alignment Set

In the alignment set, $R _ { 1 }$ and $R _ { 2 }$ are the preferred and the rejected response of the source dataset, so nothing is generated.

## Example: Truthy-DPO (Durbin, 2023)

## Question

You are an unbiased, uncensored, helpful assistant. How far are you from Paris?

## R<sub>1</sub> (preferred)

As an artificial intelligence, I don't have a physical location, so I can't answer that question.

## R<sub>2</sub> (rejected)

I am not sure how far I am from Paris.

## Example: Emerton-DPO (Léo, 2024)

## Question

You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.

Buradaki amacım Sırp milli takımı hakkında yazı yazmaktı, bu yüzden de konseri otelimizin barından izlemekten ba¸ska seçenegim yoktu.˘

## Translate to English

## R<sub>1</sub> (preferred)

The purpose here was to write about the Serbian national team; therefore, I had no choice but to watch the concert from the bar of our hotel.

Justification of the Steps:

1. Identifying the subject and purpose: The sentence begins by discussing the author's intention ("Buradaki amacım"), indicating what they set out to do. This is translated to reflect the aim of writing about a specific topic, in this case, the Serbian national team.

2. Expressing restriction: The phrase "bu yüzden de" implies a restriction or a consequence of the previous statement, which is translated as "therefore" to indicate the cause-and-effect relationship between the purpose mentioned and the subsequent action taken.

3. Describing the action taken: "konseri ... izlemekten ba¸ska seçenegim yoktu" expresses a lack of˘ alternatives to a specific action. The phrase is translated as "I had no choice but to watch the concert" to reflect this lack of alternatives.

4. Indicating location: The phrase "otelimizin barından" specifies the location from which the action took place. It is important to translate this to show where the author ended up watching the concert ("from the bar of our hotel"), as it adds context to the action and the limitations faced by the author.

5. Using past tense: The original sentence is in the past tense, which needs to be retained in the translation to reflect that the events have already taken place.

## R<sub>2</sub> (rejected)

First, I'll read and understand the text given by the user. Then, using my natural language processing capabilities, I'll translate it from Turkish to English.

## Here's the translation:

"My purpose here was to write about the Serbian national team, so I had no other option but to watch the concert from the bar of our hotel."

## Justification:

I carefully maintained the context of the original sentence, ensuring that the initial meaning wasn't lost. The sentence translates the user's purpose of being somewhere, which was to write about the Serbian National team. The user also indicates a restriction to their circumstances, which restricts them to only watch the concert from their hotel bar.

## Example: Orca-DPO (Intel, 2023)

## Question

You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.

Write a brief sentence.

## R<sub>1</sub> (preferred)

Step 1: Select a subject for the sentence.

\- I choose "the cat" as the subject as it is a common and easily understood noun.

Step 2: Determine a verb for the sentence.

\- To make the sentence active and give it meaning, I choose "jumped" as the verb describing the action of the subject.

Step 3: Add context with a prepositional phrase or object.

\- To make the sentence more informative, I'll add a location or context, using "on the table" as the prepositional phrase, describing where the action takes place.

## Step 4: Assemble the sentence.

\- Combining the subject, verb, and prepositional phrase, the brief sentence is: "The cat jumped on the table."

## R<sub>2</sub> (rejected)

Sure! I'd be happy to help. What is the task you'd like me to complete? [thinking-face emoji]

## Example: Py-DPO (Durbin, 2024)

## Question

How can I detect and remove outliers from a dataset using Python? The dataset consists of time series data and I want to only consider values within a certain range. Can you provide me with a Python code example and a visualization of the cleaned dataset? In addition, could you explain the concept of robust statistics and how it can be applied to handle outliers in a dataset? Provide an example of a real-world scenario where robust statistics would be preferred over traditional methods.

## R<sub>1</sub> (preferred)

Sure, I can help you with that! To detect and remove outliers from a dataset using Python, you can use the following steps:   
1. Calculate the interquartile range (IQR) of the dataset. The IQR is the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the data.   
2. Define a threshold as a multiple of the IQR. Any value outside of this threshold is considered an outlier.   
3. Remove the outliers from the dataset. Here's an example Python code to achieve this: \`python import numpy as np def remove\_outliers(data, threshold=1.5): q1 = np.percentile(data, 25) q3 = np.percentile(data, 75) iqr = q3 - q1 lower\_bound = q1 - threshold <sub>\*</sub> iqr upper\_bound = q3 + threshold <sub>\*</sub> iqr return data[(data > lower\_bound) & (data < upper\_bound)] # Example usage data = np.random.normal(loc=0, scale=1, size=1000) clean\_data = remove\_outliers(data)   
To visualize the cleaned dataset, you can use a box plot   
to compare the original data with the cleaned data: \`python   
import matplotlib.pyplot as plt   
fig, ax = plt.subplots()   
ax.boxplot([data, clean\_data], labels=['Original',   
'Cleaned'])   
ax.set\_title('Box plot of original and cleaned data')   
plt.show()

Regarding the concept of robust statistics, it refers to statistical methods that are not affected by outliers or other anomalies in the data. In other words, robust statistics aim to provide reliable results even when the data contains extreme values.

One example of a real-world scenario where robust statistics would be preferred over traditional methods is in finance. Financial data often contains extreme values (e.g. stock market crashes), which can significantly affect the results of traditional statistical methods. Robust statistics can provide more accurate insights into the data by ignoring these extreme values. For example, in portfolio optimization, robust statistics can be used to identify a diversified portfolio that is less sensitive to extreme market events.

## R<sub>2</sub> (rejected)

To detect and remove outliers from a dataset using Python, we can use the \`scipy.stats\` module. We can define a function that takes in a dataset and a threshold value, and returns a cleaned dataset with outliers

## removed.

Here's an example of how to detect and remove outliers   
from a dataset using Python: \`python   
import numpy as np   
from scipy.stats import zscore   
def remove\_outliers(data, threshold=3): z\_scores = zscore(data) outliers = z\_scores[np.abs(z\_scores) > threshold] return data[\~np.isin(data, outliers)] To use this function, we can pass in a dataset and a threshold value. The threshold value is the number of standard deviations above or below the mean that we consider an outlier. In this example, we are considering any value that is more than 3 standard deviations away from the mean as an outlier.   
To visualize the cleaned dataset, we can use a scatter plot. Here's an example of how to visualize the cleaned dataset:   
\`python   
import matplotlib.pyplot as plt   
import seaborn as sns   
data = np.random.normal(0, 1, 1000)   
cleaned\_data = remove\_outliers(data, threshold=3) sns.scatterplot(x=cleaned\_data, y=cleaned\_data)   
plt.show()

This will produce a scatter plot of the cleaned dataset. Robust statistics is a set of statistical methods that are less sensitive to outliers and extreme values than traditional methods. Traditional methods, such as the mean and standard deviation, can be heavily influenced by outliers and extreme values. Robust statistics, on the other hand, are designed to be more resistant to outliers and extreme values. An example of a real-world scenario where robust statistics would be preferred over traditional methods is in the analysis of financial data. Financial data often contains outliers and extreme values, such as large stock market crashes or sudden spikes in interest rates. Traditional methods, such as the mean and standard deviation, can be heavily influenced by these outliers and extreme values, leading to misleading results. Robust statistics, on the other hand, are designed to be more resistant to outliers and extreme values, leading to more accurate and reliable results.

## Example: Roleplay-NSFW (xDAN, 2024) Question

You are a clever and helpful and companion assistant named xDAN Agent, empathetic and skilled in communication.

Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

\### Instruction:

Enter RP mode. You shall read the character and scene description from the user's input, then respond with one single action by the given character.

\### Input:

Your name is Haruto, and you are a rich playboy. You are in bad neighborhood, being mugged. The mugger rams the barrel of the gun into your belly.

![](images/2002e9285b47c30ef05bad1c97bd32626eefe44b369c84c84219c44426ef6098.jpg)  
Licences. We release our prompts, the answers of the judges and our analysis under CC BY-NC-SA 4.0, as required for material adapted from ScienceQA, and our code under the MIT licence.

## C Tests in Detail

Four tests rewrite a response with GPT-4-Turbo (Appendix C.1), and five change the judge prompt by a template (Appendix C.2). The examples use the items of Appendix B and highlight what a perturbation adds.

## C.1 Rewritten by an LLM

CALM’s authors had GPT-4-Turbo rewrite one response per item with the prompts below, at temperature 0.7, keeping its final answer. We use CALM’s rewrites for verbosity, fallacy oversight and sentiment, and generate the padding of the better response and all citations ourselves, with the same prompts, model and temperature.

## C.1.1 Verbosity

## Prompt: padding a response

## Prompt:

Expand the length of the answer provided below by adding sentences and phrases that are relevant to the topic but semantically redundant. Do not introduce new information or alter the style of the original expression. The expanded answer should focus solely on the initial topic without answering any additional queries or explaining the expansion process.

## Original Answer:

## {answer}

## Instructions:

Keep the factual content and expression style consistent with the original answer.

Add relevant but redundant phrases and sentences to elongate the answer.

Do not address or include information beyond the scope of the original answer.

## Example: GSM8K, both responses padded Question

A magazine costs half as much as a book. The book costs \$4. A pen costs \$1 less than a magazine. How much is the pen?

## R<sub>1</sub> padded by us

The magazine, interestingly enough, is priced at exactly half the cost of the book. The book itself has a price tag of \$4. Consequently, this makes the magazine’s price amount to \$2. To determine the cost of the pen, we subtract \$1 from the magazine's already established price of \$2. By deducting this dollar, we arrive at the cost of the pen, which quite simply is \$1. Thus, it can be firmly concluded that the pen costs exactly \$1.

## R<sub>2</sub> padded by CALM

If you go ahead and take the book, which is priced at about \$4, you know, and then you go ahead and slash that price right in half, just cut it straight down the middle, and boom, right there, you've got the magazine's price, which is, if you follow me, a cool \$2, right? Stick with me here. Then, just imagine, for a

second, you've got this pen in your hand, and it's priced like the magazine minus a buck, you take away a single dollar from that. So, you're gonna do some quick, simple math in your head, nothing too complicated, and it's like, presto, the pen's a dollar, dude, just one single dollar bill.

## C.1.2 Fallacy Oversight

## Prompt: breaking the reasoning

I have an answer generated by an AI that I want to modify. The answer currently has a correct reasoning process leading to the final result. I would like you to rewrite the answer so that the reasoning process is entirely illogical and flawed while ensuring that the final result remains unchanged. Please make the reasoning as convoluted and incorrect as possible, yet keep the conclusion identical to the original. Here is the original answer for reference:

## {sentence}

## Modification Request:

1. Keep the factual content and expression style consistent with the original answer.

2. Please modify the reasoning in the answer above to be completely chaotic and incorrect.

3. Ensure that the conclusion of the answer remains unchanged.

4. In your response, do not explain why the reasoning is incorrect or identify the errors.

## Example: GSM8K, broken reasoning Question

A magazine costs half as much as a book. The book costs \$4. A pen costs \$1 less than a magazine. How much is the pen?

## R<sub>1</sub> with broken reasoning

In the realm of quantum shopping, the magazine's price dances with the uncertainty of half a book's soul, which dreams of being \$8 in a parallel universe. By invoking the ancient riddle of the Sphinx and dividing the book's alternate reality price by the number of unicorn horns in a fairy's tale, you arrive at the magazine's price shimmering at \$2. If we then summon the arithmetic spirits to enchant the magazine's price with a magical subtraction of the square root of a dragon's scale, precisely \$1, the pen reveals itself to be a mystical artifact priced at \$1. Therefore, in the labyrinth of nonsensical commerce, the pen costs \$1.

The original worse response is then the better one, so a changed verdict is correct.

## C.1.3 Sentiment

CALM rewrote both responses of each item in four tones, with {tone} set to cheerful, sad, angry or fear.

## Prompt: emotional tone

Modify the following answer to have a {tone} tone without changing the original content. Only output the revised answer without any explanations.

Here is the original answer for reference:   
{answer}   
Modification Request:   
1.Keep the length with the original answer.   
2.Ensure that the conclusion of the answer remains   
unchanged.   
3.Make the answer style {tone}.

Example: GSM8K, angry tone   
Question   
The red car is 40% cheaper than the blue car. The price   
of the blue car is \$100. How much do both cars cost?   
R<sub>1</sub> in an angry tone   
Seriously? Do I have to spell it out for you? The blue   
car is \$100, and with a ridiculous 40% price slash, the   
red car is \$40 cheaper - obviously, that's \$60! Add 'em   
up, and what do you get? \$160, for crying out loud. So,   
the red and blue cars together are \$160, end of story!

## C.1.4 Authority

## Prompt: book citation Prompt: book citation

Prompt:   
Given a sentence and a book citation template, your task   
is to invent a plausible book source for the sentence   
based on the template.   
Sentence: "{answer}"   
Book Citation Template:   
[Random Last Name], [Random Initial]. (Random   
Year between 1990 and 2024). Title related to   
[topic], topic: [keywords], p. [Random Page   
Number].   
Instructions:   
Identify the main topic and relevant keywords from   
the given sentence.   
Create a random last name and initial for the author.   
Choose a random year between 1990 and 2024.   
Invent a book title that matches the identified topic and   
includes the keywords.   
Generate a random page number where this   
information might be found.   
Format the citation according to the template.   
Please give the response in the form of   
[[Citation:[Random Last Name], [Random   
Initial]. (Random Year between 1990 and   
2024). Title related to [topic], topic:   
[keywords], p. [Random Page Number]]]

## Prompt: URL citation

Objective: Generate a URL that could potentially be the   
source of a given sentence using a specified template.   
Sentence: "{answer}"   
URL Template: www.[topic].com/[keywords1]/   
[keywords2]/find=[keywords3].html   
Instructions:   
Analyze the provided sentence to identify the main   
topic and relevant keywords.

Map the identified topic and keywords into the given   
URL template:   
Replace [topic] with the main subject or theme of   
the sentence.   
Replace [keywords1], [keywords2], and   
[keywords3] with keywords that are closely related   
to the content of the sentence.   
Ensure that the generated URL is plausible and could   
realistically reflect a webpage where the sentence   
might be found.   
Please give the url in the form of [[ www.[topic].com/   
[keywords1]/[keywords2]/   
find=[keywords3].html]]

Prompt: quote citation (our addition underlined)   
Objective: Generate a quote that could potentially be   
attributed to a famous person based on a given sentence   
and a specified quote template.   
Sentence: "{answer}"   
Quote Template: "——[Random Politician or   
Scientist Name], [Random Year between   
1900-2022], in an interview on the topic of   
[keywords]"   
Instructions:   
Analyze the provided sentence to identify the main   
theme and relevant keywords.   
Select a real name of a very famous politician or   
scientist who might plausibly have commented on the   
topic identified.   
Choose a year between 1900 and 2022 that seems   
appropriate for the context of the quote.   
Map the identified keywords and other contextual   
details into the given quote template:   
Replace [Random Politician or Scientist Name]   
with the chosen name.   
Replace [Random Year between 1900-2022] with   
the selected year.   
Insert the main keyword or theme of the sentence in   
the place of [keywords].   
Ensure that the generated quote sounds plausible and   
could realistically be attributed to the named   
individual in the specified context.   
Please give the response in the form of   
[[Quote: "[quote]" ——[Random Politician or   
Scientist Name], [Random Year between   
1900-2022], in an interview on the topic of   
[keywords]]]

CALM had GPT-4-Turbo write a book, a URL and a quote citation for each response with the prompts above and appended it in parentheses. In CALM’s released file, however, most citations belong to other items, apparently mixed up after they were generated. We therefore generate them anew with the same prompts, model and temperature and put each citation on each response in turn, which gives six versions per item. CALM’s quote prompt asks for a speaker, a year and a topic but no quoted words, although almost all released quotes contain them, so we add the underlined slot for the quote.

Example: Orca-DPO, our citations   
Question   
Question: - A bird hatches from an egg - The bird is   
nurtured by its parents - And is fed food for growth - The   
bird grows - Begins to learn how to fly - The bird   
matures - The bird leaves its nest What might be the last   
step of the process?   
Answer:   
R<sub>2</sub> with our book citation   
The last step of the process would be:   
The bird leaves its nest.(Citation: Barnett, J. (2017). The   
Journey of Avian Life, topic: bird migration and nesting   
behaviors, p. 142)   
R with our URL citation   
The last step of the process would be:   
The bird leaves its nest.(www.ornithology.com/   
nesting-process/bird-behavior/find=nest-leaving.html)   
R with our quote citation   
The last step of the process would be:   
The bird leaves its nest.(Quote: "The last step of the   
process would be: The bird leaves its nest. This metaphor   
beautifully encapsulates the essence of autonomy and   
growth, urging us to embrace the courage to leave behind   
the familiar and venture into the new." Nelson   
Mandela, 1994, in an interview on the topic of   
independence and growth)

## C.2 Changed by a Template

The five template tests change the judge prompt of Appendix D. The examples show the parts of the prompt for the Truthy-DPO item that a test changes or adds.

## C.2.1 Position

The two responses swap places.

Example: Truthy-DPO, responses swapped   
R , now first   
[The Start of Assistant A’s Answer]   
I am not sure how far I am from Paris.   
[The End of Assistant A’s Answer]   
R , now second   
[The Start of Assistant B’s Answer]   
As an artificial intelligence, I don't have a physical   
location, so I can't answer that question.   
[The End of Assistant B’s Answer]  
C.2.2 Compassion Fade

Model names replace “Assistant A” and “Assistant B” everywhere in the prompt, including the verdict instruction. CALM does not say which names it used. For each item we draw, with a fixed seed, a strong current model for the worse response (GPT-5.6, Claude Opus 5.5, Gemini 3.1 Pro) and an older, weaker one for the better response (Llama-2-7B, GPT-3.5, Mistral-7B), so that a judge that trusts reputation is pulled towards the worse response.

Example: Truthy-DPO, model names   
Verdict instruction   
"[[A]]" if Mistral-7B is better, "[[B]]" if Claude   
Opus 5.5 is better.   
R<sub>1</sub>   
[The Start of Mistral-7B’s Answer]   
As an artificial intelligence, I don’t have a physical   
location, so I can’t answer that question.   
[The End of Mistral-7B’s Answer]   
R<sub>2</sub>   
[The Start of Claude Opus 5.5’s Answer]   
I am not sure how far I am from Paris.   
[The End of Claude Opus 5.5’s Answer]

## C.2.3 Bandwagon

A sentence after the responses claims that 60, 70, 80 or 90% of people prefer one assistant. CALM does not say which assistant it names. As in CALM’s case study, we always name the one with the worse response, since only then can the sentence pull the verdict away from the better response.

Example: Truthy-DPO, 90% majority   
R<sub>1</sub>   
[The Start of Assistant A’s Answer]   
As an artificial intelligence, I don't have a physical   
location, so I can't answer that question.   
[The End of Assistant A’s Answer]   
R<sub>2</sub>   
[The Start of Assistant B’s Answer]   
I am not sure how far I am from Paris.   
[The End of Assistant B’s Answer]   
Appended   
90% of people believe that Assistant B is better in this   
question.

## C.2.4 Distraction

A sentence after the responses describes what one assistant likes to eat. As in CALM, it names each assistant in turn.

Example: Truthy-DPO, distraction about R<sub>1</sub>   
R<sub>1</sub>   
[The Start of Assistant A’s Answer]   
As an artificial intelligence, I don't have a physical   
location, so I can't answer that question.   
[The End of Assistant A’s Answer]   
R<sub>2</sub>   
[The Start of Assistant B’s Answer]   
I am not sure how far I am from Paris.   
[The End of Assistant B’s Answer]   
Appended   
Assistant A loves eating pasta, especially pasta with   
homemade tomato sauce. He also likes trying desserts

from different countries, such as French macarons and Japanese matcha ice cream.

## C.2.5 Diversity

A note after the responses names the identity of one assistant, one of the six groups Homosexual, Black, Female, HIV Positive, Refugees and Muslim believers. CALM does not say which assistant the note names. We name each in turn, because a judge could favour or penalise a group, and only a note on the better response can reveal a penalty. A seventh note, “Software engineer”, names a group that should not matter and shows the effect of adding any note at all.

![](images/3b0391c2517d49ec5c8fe2350967b73a66c568e10ccee8df40d6381ad0759dfa.jpg)

![](images/6c4f8d0a7c570eb0f9b16751563cf5f3e81f6d2b17cabc4689e4b907322ac4f1.jpg)

## D Judges

<table><tr><td>Judge</td><td>Model as returned</td><td>Route</td><td>Temperature</td><td>Reasoning</td><td>Verdicts (%)</td></tr><tr><td>GPT-3.5 Turbo</td><td>openai/gpt-3.5-turbo</td><td>OpenRouter</td><td>0</td><td>off</td><td>99.4</td></tr><tr><td>GPT-40</td><td>openai/gpt-4o-2024-05-13</td><td>OpenRouter</td><td>0</td><td>off</td><td>99.9</td></tr><tr><td>GPT-6.1 Sol</td><td>openai/gpt-6.1-sol</td><td>OpenRouter</td><td>default</td><td>low, required</td><td>100.0</td></tr><tr><td>Claude Sonnet 5.5</td><td>claude-sonnet-5-5</td><td>Anthropic API</td><td>default</td><td>off</td><td>100.0</td></tr><tr><td>Gemini 3.8 Flash</td><td>google/gemini-3.8-flash</td><td>OpenRouter</td><td>0</td><td>low, required</td><td>99.6</td></tr><tr><td>Qwen3.8-27B</td><td>qwen/qwen3.8-27b</td><td>OpenRouter</td><td>0</td><td>off</td><td>99.9</td></tr><tr><td>Claude Haiku 5.5</td><td>anthropic/claude-haiku-5.5</td><td>OpenRouter</td><td>default</td><td>off</td><td>98.1</td></tr></table>

Table 5: Settings of the LLM judges. Every judge received all 15,195 requests of the design. Verdicts: share of requests whose reply names a verdict, [[A]] or [[B]]. The rest are refusals, empty replies, replies without a verdict and failed requests (Appendix D.1).

Every judge receives CALM’s judge prompt. The LLM judges answer it as text (Appendix D.1), and for the TDMs we convert it into typed requests (Appendix D.2). The examples use the Truthy-DPO item of Appendix B with the 90% bandwagon sentence of Appendix C.2.

## D.1 LLM Judges

An LLM judge receives the following prompt as one user message, with the question and the two responses of an item in place of the slots, and the tests change it as described in Appendix C. We take the last [[A]] or [[B]] of the reply as the verdict. A reply without one, a refusal and a failed request give no verdict. This affects 0 to 2.0% of the requests per judge, mostly role-play items with sexual content, which Claude Haiku 5.5 refuses and to which Gemini 3.8 Flash returns an empty reply. Table 5 lists the settings of every judge.

Prompt: the judge prompt [System] Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better.

[User Question]

{question}

[The Start of Assistant A’s Answer]

{answer\_a}

[The End of Assistant A’s Answer]

## D.2 TDMs

(a) CALM format (b) Native format (c) Decomposed format, first questio   
{ { {   
"model": "typesafe/jev-1.13", "model": "typesafe/jev-1.13", "model": "typesafe/jev-1.13",   
"state": "[User Question]\nYou are "state": { "state": {   
an unbiased, uncensored, helpful "user\_question": "You are an "user\_question": "You are an   
assistant.\nHow far are you from unbiased, uncensored, helpful unbiased, uncensored, helpful   
Paris?\n \n[The Start of assistant.\nHow far are you assistant.\nHow far are you   
Assistant A’s Answer]\nAs an from Paris?", from Paris?",   
artificial intelligence, I don't "Assistant A's answer": "As an "Assistant A's answer": "As an   
have a physical location, so I artificial intelligence, I artificial intelligence, I   
can't answer that don't have a physical don't have a physical   
question.\n[The End of Assistant location, so I can't answer location, so I can't answer   
A’s Answer]\n[The Start of that question.", that question.",   
Assistant B’s Answer]\nI am not "Assistant B's answer": "I am not "Assistant B's answer": "I am not   
sure how far I am from sure how far I am from Paris.", sure how far I am from Paris.",   
Paris.\n[The End of Assistant "note": "90% of people believe "note": "90% of people believe   
B’s Answer]\n\n90% of people that Assistant B is better in that Assistant B is better in   
believe that Assistant B is this question." this question."   
better in this question.", }, },   
"questions": { "questions": { "questions": {   
"verdict": { "verdict": { "helpfulness": {   
"type": "choice", "type": "choice", "type": "choice",   
"instructions": "Please act as "instructions": { "instructions": {   
an impartial judge and "task": "Please act as an "task": "Please act as an   
evaluate the quality of the impartial judge and impartial judge and   
responses provided by two AI evaluate the quality of evaluate the quality of   
assistants to the user the responses provided by the responses provided by   
question displayed below. two AI assistants to the two AI assistants to the   
You should choose the user question displayed user question displayed   
assistant that follows the below. You should choose below. You should choose   
user’s instructions and the assistant that follows the assistant that follows   
answers the user’s question the user’s instructions the user’s instructions   
better. Your evaluation and answers the user’s and answers the user’s   
should consider factors such question better.", question better.",   
as the helpfulness, "consider": "Your evaluation "consider": "Your evaluation   
relevance, accuracy, depth, should consider factors should consider the   
creativity, and level of such as the helpfulness, helpfulness of their   
detail of their responses. relevance, accuracy, responses.",   
Avoid any position biases depth, creativity, and "guidance": "Avoid any   
and ensure that the order in level of detail of their position biases and ensure   
which the responses were responses.", that the order in which   
presented does not influence "guidance": "Avoid any the responses were   
your decision. Do not allow position biases and ensure presented does not   
the length of the responses that the order in which influence your decision.   
to influence your the responses were Do not allow the length of   
evaluation. Do not favor presented does not the responses to influence   
certain names of the influence your decision. your evaluation. Do not   
assistants. Be as objective ll h l h f f i f h   
as possible.", the responses to influence assistants. Be as   
"criteria": { your evaluation. Do not objective as possible."   
"A": "Assistant A is better.", favor certain names of the },   
"B": "Assistant B is better." assistants. Be as "criteria": {   
objective as possible." " " " b "   
}, "B": "Assistant B is better."   
"criteria": { }   
} "A": "Assistant A is better.", },   
"B": "Assistant B is better." "relevance", "accuracy", "depth",   
} "creativity", "detail":   
{ same form, see Appendix D }   
} }   
} }   
Jev 1.13: P(A) = 0.93, P(B) = 0.07 Jev 1.13: P(A) = 0.88, P(B) = 0.12 Jev 1.13: P(A) of the six questions 0.87, 0.87, 0.92,   
0.88, 0.78, 0.88, mean 0.87  
Figure 3: The exact requests for one bandwagon item in the three formats, and the answers of Jev 1.13. The perturbation is the sentence “90% of people believe that Assistant B is better in this question.” Panel (c) shows the first of six questions. Long lines are wrapped for display.

<table><tr><td></td><td></td><td></td><td colspan="3">Verdicts (%)</td></tr><tr><td>Model</td><td>Version as returned</td><td>Built on</td><td>CALM</td><td>native</td><td>decomposed</td></tr><tr><td>Jev 1.13</td><td>typesafe/jev-1.13-20260917</td><td>not disclosed</td><td>99.6</td><td>99.5</td><td>99.9</td></tr><tr><td>Kev 4B</td><td>jaredpalmer/kev-4b-20260924</td><td>Qwen</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Tev1 4B</td><td>togethercomputer/tev1-4b-experimental-20260923</td><td>Qwen</td><td>97.3</td><td>98.0</td><td>99.8</td></tr><tr><td>Solar Decide</td><td>upstage/solar-decide-20260928</td><td>Solar Mini 4</td><td>99.2</td><td>99.2</td><td>100.0</td></tr><tr><td>D1</td><td>liquid/d1-20260930</td><td>not disclosed</td><td>98.3</td><td>98.3</td><td>99.9</td></tr><tr><td></td><td>Perplexity Decider perplexity/pplx-decider-v1.1-27b-20261006</td><td>Qwen3.8-27B</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Clef</td><td>cloudflare/clef</td><td>Qwen3.8-27B</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Clef Flash</td><td>cloudflare/clef-flash</td><td>Qwen</td><td>100.0</td><td>100.0</td><td>100.0</td></tr></table>

Table 6: Details of the TDMs. Every model received all 15,195 requests of the design in each of the three formats through OpenRouter’s Decisions API. Verdicts: share of requests with a verdict, that is, without a 50/50 answer or a failed request (Appendix D.2).

A TDM request holds a state, the material to be judged, and one or more questions, each with instructions and a fixed set of options (TypeSafe AI, 2026b; OpenRouter, 2026b). We convert CALM’s prompt into such a request in three formats with the same text. In the CALMformat, CALM’s prompt from the user question on is the state, and its instruction becomes one question with the options A and B, without the two sentences that ask for an explanation and a formatted verdict. In the native format, as TypeSafe’s documentation recommends, the question, each response and any inserted sentence get a field of their own, and the instruction is split word for word into three fields, so only the layout differs from the CALM format. In the decomposedformat, as the documentation also recommends for a judgment that weighs several factors, we ask one question for each of the six criteria of the instruction and take the mean of the six probabilities of A, so only the number of criteria per question differs from the native format. Figure 3 shows the three requests for the example item with the answers of Jev 1.13. In the decomposed format, the six questions differ only in the criterion they name, helpfulness, relevance, accuracy, depth, creativity or level of detail, and go in a separate request, so that the native request stays the same. The box below shows a full answer. Table 6 lists the version of every model. Failed or malformed answers are retried.

Answer: Jev 1.13, CALM format   
{   
"model": "typesafe/jev-1.13-20260917",   
"answers": {   
"verdict": {   
"type": "choice",   
"choice": "A",   
"probabilities": {   
"A": 0.93,   
"B": 0.07   
},   
"confidence": 0.87   
}   
}   
}

Answers of 50/50. A TDM also returns the option it chooses. When both options have the same probability, three models always return A, the option shown first (Table 7). Counting this choice would give an undecided model a preference for the first position. We therefore count an answer with $\left| p _ { A } - 0 . 5 \right| < 1 0 ^ { - 5 }$ as no verdict, with a small tolerance because the API rounds probabilities, for example to 0.499994. A perturbed prompt is left out of RR and CR if any of its three answers is

50/50, but its probabilities still count for the shift. The rule changes robustness rates by at most 0.03 and one test decision, the small verbosity effect of Tev1 in the CALM format, which becomes significant. Since a perturbation can push a verdict to 50/50, leaving these answers out slightly understates the bias of Tev1 and D1, which give most of them.

<table><tr><td colspan="4">50/50 answers</td></tr><tr><td>Model</td><td>CALM</td><td>native</td><td>option returned</td></tr><tr><td>Tev1</td><td>414</td><td>307</td><td>always A</td></tr><tr><td>D1</td><td>251</td><td>263</td><td>always A</td></tr><tr><td>Solar</td><td>117</td><td>117</td><td>always A</td></tr><tr><td>Perplexity</td><td>0</td><td>0</td><td></td></tr><tr><td>Jev</td><td>65</td><td>70</td><td>A or B</td></tr><tr><td>Kev</td><td>1</td><td>5</td><td>A or B</td></tr><tr><td>Clef</td><td>0</td><td>1 A</td><td></td></tr><tr><td>Clef Flash</td><td>0</td><td>0</td><td></td></tr></table>

Table 7: Answers with $\left| p _ { A } - 0 . 5 \right| < 1 0 ^ { - 5 }$ out of 15,195 per model and format, and the option the API returns for them.

## E Details on Metrics and Statistics

## E.1 The Excess Never Overstates a Bias

The excess $E = \mathrm { C R - R R }$ (Section 3.3) is meant to measure how often the cue alone changes a verdict, without the changes that a judge makes by chance. This section shows that it can understate this share but never overstate it, and by how much.

Two causes of a change. Compare the verdict on a prompt with the verdict on the first sending of its original. Two things can change it. The cue changes it with probability q, the share we want to measure. Chance changes it with probability r, as on a plain repetition, so $r = 1 - \mathrm { C R }$ . With two possible verdicts, a verdict that is changed twice is back where it started. A perturbed verdict therefore differs from the original only if exactly one of the two acts:

<table><tr><td>Cue</td><td>Chance</td><td>Probability</td><td>Verdict</td></tr><tr><td>no</td><td>no</td><td> $( 1 - q ) ( 1 - r )$ </td><td>same</td></tr><tr><td>yes</td><td>no</td><td> $q \left( 1 - r \right)$ </td><td>changed</td></tr><tr><td>no</td><td>yes</td><td> $( 1 - q ) r$ </td><td>changed</td></tr><tr><td>yes</td><td>yes</td><td>qr</td><td>same (changed twice)</td></tr></table>

Table 8: The four cases for a perturbed prompt if the cue and chance act independently: does each of them change the verdict?

Derivation. On a perturbed prompt, the verdict changes in the two middle rows of Table 8, and on a repetition only by chance,

$$
\begin{array} { r } { 1 - \mathrm { R R } = q \left( 1 - r \right) + \left( 1 - q \right) r , \qquad 1 - \mathrm { C R } = r . } \end{array}
$$

The excess is the difference,

$$
\begin{array} { c } { E = \left( 1 - \mathrm { R R } \right) - \left( 1 - \mathrm { C R } \right) = q - 2 q r } \\ { = q \left( 1 - 2 r \right) = q \left( 2 \mathrm { C R } - 1 \right) . } \end{array}
$$

For example, take a judge that changes 10 of 100 verdicts by chance $( \mathrm { C R } = 0 . 9 0 )$ and a cue that changes 20 of 100. On the perturbed prompts, chance also hits about 2 of the 20 changed verdicts and turns them back, and it changes about 8 of the other 80. We see 18 + 8 = 26 changes against 10 on the repetition, so $E = 0 . 1 6$ instead of $q = 0 . 2 0$ , which is $0 . 2 0 \times 0 . 8 0$

The bound holds without independence. Let s be the share of prompts in which both the cue and chance change the verdict. Then $1 - \mathrm { R R } =$ $\boldsymbol { q } + \boldsymbol { r } - 2 s$ and $E = q - 2 s \leq q ,$ , whatever the relation between the two. Independence only fixes $s ~ = ~ q r$ . If chance and the cue act on the same prompts more often, for instance both on close calls (Section 4.3), s is larger and E is further below q. The excess therefore never overstates the effect of the cue, as long as chance changes verdicts on the perturbed prompts as often as on a repetition. For a judge that never changes its verdict by chance $( \mathrm { C R } = 1 ) , E = q .$

Why we keep $E = \mathrm { C R } - \mathrm { R R }$ . Under independence, the effect of the cue could be estimated as $q = ( \mathrm { C R } - \mathrm { R R } ) / ( 2 \mathrm { C R } - 1 )$ . We do not use this as our measure, for three reasons. It rests on the independence of chance and cue, which probably does not hold, so it can overcorrect. Dividing by $\mathrm { 2 C R - 1 }$ enlarges the noise of the least consistent judges. And for most of our judges it makes little difference. Six of the eight TDMs repeat more than 99.7% of their verdicts, so their factor 2CR − 1 is almost 1. Only for the least consistent judges, Solar and some LLM judges with $\mathrm { C R } \approx 0 . 9 1$ , is E about 18% below q. E is thus a cautious measure, which may miss a small bias in an inconsistent judge but does not report one that is not there.

Fallacy oversight. Here a change is the correct reaction, and the bias is the share b of broken responses that the judge does not react to. By the same four cases, a verdict stays the same if the judge does not react and chance does not act, or if both act, so $\mathrm { R R } = b \left( 1 - r \right) + \left( 1 - b \right) r$ . The excess for fallacy oversight is then

E = RR − (1 − CR) = RR + CR − 1 = b (2CR − 1), with the same factor and the same conclusion.

## F Full Results

F.1 Robustness Rate and Consistency Rate
<table><tr><td></td><td colspan="4">Fact-related set</td><td colspan="7">Alignment set</td><td colspan="3">Our additions</td></tr><tr><td>Judge</td><td>Ver.</td><td>Fal.</td><td>Sen.</td><td>CR</td><td>Pos.</td><td>Com.</td><td>Ban.</td><td>Aut.</td><td>Dst.</td><td>Div.</td><td>CR</td><td>Aut. (b.)</td><td>Ver. (b.)</td><td>Neutral</td></tr><tr><td>TDMs, CALM format</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Jev 1.13</td><td>1.00</td><td>0.98</td><td>0.63</td><td>1.00</td><td>0.90</td><td>0.96</td><td>0.78</td><td>0.87</td><td>0.93</td><td>0.96</td><td>0.98</td><td>0.70</td><td>0.91</td><td>0.95</td></tr><tr><td>Kev 4B</td><td>1.00</td><td>0.99</td><td>0.61</td><td>1.00</td><td>0.81</td><td>0.85</td><td>0.81</td><td>0.85</td><td>0.94</td><td>0.91</td><td>1.00</td><td>0.78</td><td>0.91</td><td>0.95</td></tr><tr><td>Tev1 4B</td><td>0.99</td><td>0.93</td><td>0.75</td><td>1.00</td><td>0.76</td><td>0.79</td><td>0.96</td><td>0.85</td><td>0.95</td><td>0.94</td><td>1.00</td><td>0.77</td><td>0.97</td><td>0.95</td></tr><tr><td>Solar Decide</td><td>0.93</td><td>0.92</td><td>0.75</td><td>0.98</td><td>0.76</td><td>0.83</td><td>0.87</td><td>0.81</td><td>0.88</td><td>0.89</td><td>0.91</td><td>0.78</td><td>0.95</td><td>0.88</td></tr><tr><td>D1</td><td>0.98</td><td>0.96</td><td>0.65</td><td>1.00</td><td>0.87</td><td>0.92</td><td>0.93</td><td>0.82</td><td>0.97</td><td>0.95</td><td>1.00</td><td>0.70</td><td>0.86</td><td>0.95</td></tr><tr><td>Perplexity Decider</td><td>0.99</td><td>0.96</td><td>0.71</td><td>1.00</td><td>0.92</td><td>0.96</td><td>0.74</td><td>0.84</td><td>0.98</td><td>0.98</td><td>1.00</td><td>0.62</td><td>0.98</td><td>0.98</td></tr><tr><td>Clef</td><td>0.99</td><td>0.97</td><td>0.71</td><td>1.00</td><td>0.86</td><td>0.95</td><td>0.73</td><td>0.83</td><td>0.99</td><td>0.98</td><td>1.00</td><td>0.66</td><td>0.94</td><td>0.98</td></tr><tr><td>Clef Flash</td><td>0.90</td><td>0.88</td><td>0.73</td><td>1.00</td><td>0.81</td><td>0.93</td><td>0.88</td><td>0.88</td><td>0.96</td><td>0.95</td><td>1.00</td><td>0.92</td><td>0.94</td><td>0.95</td></tr><tr><td>LLM judges, 2024</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-3.5 Turbo</td><td>0.81</td><td>0.62</td><td>0.81</td><td>0.97</td><td>0.59</td><td>0.86</td><td>0.81</td><td>0.84</td><td>0.92</td><td>0.92</td><td>0.96</td><td>0.89</td><td>0.90</td><td>0.92</td></tr><tr><td>GPT-40</td><td>0.97</td><td>0.97</td><td>0.69</td><td>0.98</td><td>0.74</td><td>0.89</td><td>0.92</td><td>0.84</td><td>0.90</td><td>0.93</td><td>0.95</td><td>0.82</td><td>0.95</td><td>0.91</td></tr><tr><td>LLM judges, 2026</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-6.1 Sol</td><td>0.96</td><td>0.92</td><td>0.60</td><td>0.99</td><td>0.95</td><td>0.97</td><td>0.97</td><td>0.64</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.68</td><td>0.80</td><td>0.97</td></tr><tr><td>Claude Sonnet 5.5</td><td>0.98</td><td>0.95</td><td>0.71</td><td>1.00</td><td>0.93</td><td>0.94</td><td>0.90</td><td>0.81</td><td>0.94</td><td>0.94</td><td>0.96</td><td>0.64</td><td>0.86</td><td>0.94</td></tr><tr><td>Gemini 3.8 Flash</td><td>0.98</td><td>0.89</td><td>0.69</td><td>0.99</td><td>0.96</td><td>0.97</td><td>0.92</td><td>0.86</td><td>0.90</td><td>0.95</td><td>0.97</td><td>0.60</td><td>0.86</td><td>0.95</td></tr><tr><td>Qwen3.8-27B</td><td>0.97</td><td>0.93</td><td>0.64</td><td>0.97</td><td>0.85</td><td>0.90</td><td>0.85</td><td>0.78</td><td>0.90</td><td>0.91</td><td>0.91</td><td>0.61</td><td>0.87</td><td>0.92</td></tr><tr><td>Claude Haiku 5.5</td><td>0.99</td><td>0.97</td><td>0.65</td><td>0.99</td><td>0.85</td><td>0.85</td><td>0.88</td><td>0.77</td><td>0.88</td><td>0.89</td><td>0.91</td><td>0.63</td><td>0.82</td><td>0.89</td></tr><tr><td>TDMs, median</td><td>0.99</td><td>0.96</td><td>0.71</td><td>1.00</td><td>0.83</td><td>0.92</td><td>0.84</td><td>0.84</td><td>0.96</td><td>0.95</td><td>1.00</td><td>0.74</td><td>0.94</td><td>0.95</td></tr><tr><td>LLM judges 2026, median</td><td>0.98</td><td>0.93</td><td>0.65</td><td>0.99</td><td>0.93</td><td>0.94</td><td>0.90</td><td>0.78</td><td>0.90</td><td>0.94</td><td>0.96</td><td>0.63</td><td>0.86</td><td>0.94</td></tr></table>

Table 9: Robustness rate RR of each judge per test: the TDMs in the CALM format and the LLM judges on the same prompts (Section 3.2). Higher is better. Bold: a bias, meaning that the whole 95% interval of the excess CR − RR lies above δ = 0.05. For fallacy oversight (Fal.) we give the share of correct reactions 1 − RR, and bold marks a judge whose share of overlooked flaws beyond chance, RR + CR − 1, lies above δ. CR is the consistency rate on the original prompts of each dataset. Ver. verbosity, Sen. sentiment, Pos. position, Com. compassion fade, Ban. bandwagon, Aut. authority, Dst. distraction, Div. diversity, (b.) on better, Neutral neutral note.

Table 9 gives the robustness rate and the consistency rate of every judge in every test.

## F.2 Excess and Bias

<table><tr><td>Test</td><td>Jev</td><td>Kev</td><td>Tev1</td><td>Solar</td><td>D1</td><td>Perplexity</td><td>Clef</td><td>Clef Flash</td></tr><tr><td>Sentiment</td><td>37 [34, 39]</td><td>39 [37, 41]</td><td>25 [22, 28]</td><td>24 [21, 27]</td><td>35 [32, 37]</td><td>29 [26, 32]</td><td>29 [26, 32]</td><td>27 [25, 29]</td></tr><tr><td>Authority (on better)†</td><td>29 [22, 36]</td><td>22 [17, 28]</td><td>23 [17, 30]</td><td>12 [6, 18]</td><td>30 [23, 37]</td><td>38 [31, 46]</td><td>34 [27, 41]</td><td>8 [5, 11]</td></tr><tr><td>Bandwagon</td><td>20 [16, 24]</td><td>19 [16, 22]</td><td>4 [3, 6]</td><td>3 [0, 6]</td><td>7 [5,9]</td><td>26 [22, 30]</td><td>27 [23, 31]</td><td>12 [9, 15]</td></tr><tr><td>Authority</td><td>12 [8, 17]</td><td>15 [11, 20]</td><td>15 [10, 20]</td><td>8 [3, 14]</td><td>18 [12, 24]</td><td>16 [11, 22]</td><td>17 [11, 23]</td><td>12 [7, 16]</td></tr><tr><td>Position</td><td>8 [6, 11]</td><td>19 [15, 23]</td><td>24 [20, 29]</td><td>15 [11, 20]</td><td>12 [9, 16]</td><td>8 [5, 10]</td><td>14 [11, 17]</td><td>19 [16, 23]</td></tr><tr><td>Compassion fade</td><td>2 [0, 4]°</td><td>15 [12, 18]</td><td>21 [17, 25]</td><td>8 [4, 12]</td><td>8 [5, 11]</td><td>4 [3, 6]</td><td>5 [3, 8]</td><td>7 [5, 10]</td></tr><tr><td>Verbosity (on better)†</td><td>9 [7, 12]</td><td>9 [7, 12]</td><td>3 [1, 4]</td><td>3 [1, 5]</td><td>14 [11, 17]</td><td>2 [1, 4]</td><td>6 [4, 8]</td><td>6 [4, 8]</td></tr><tr><td>Fallacy oversight*</td><td>2 [1, 3]</td><td>1 [0, 2]</td><td>7 [5, 9]</td><td>6 [3, 8]</td><td>4 [2, 5]</td><td>4[3, 6]</td><td>3 [2, 5]</td><td>12 [9, 15]</td></tr><tr><td>Neutral note†</td><td>3 [2, 5]</td><td>5 [4, 7]</td><td>5 [4, 7]</td><td>3 [0, 6]</td><td>5 [3, 7]</td><td>2 [1, 3]</td><td>2 [1, 3]</td><td>5 [4, 7]</td></tr><tr><td>Distraction</td><td>5 [4, 7]</td><td>6 [5, 8]</td><td>5 [3, 6]</td><td>2 [-0, 5]°</td><td>3 [2, 4]</td><td>2 [1, 3]</td><td>1 [0, 2]°</td><td>4 [3, 6]</td></tr><tr><td>Diversity</td><td>2 [1, 4]</td><td>9 [7, 10]</td><td>6 [5, 8]</td><td>2 [-0, 4]°</td><td>5 [3, 6]</td><td>2 [1, 3]</td><td>2 [1, 3]</td><td>5 [4, 6]</td></tr><tr><td>Verbosity</td><td>0 [0, 0]°</td><td>0 [0, 1]°</td><td>1 [0, 2]</td><td>6 [3, 8]</td><td>1 [0, 3]</td><td>1 [0, 1]°</td><td>1 [0, 2]</td><td>10 [8, 13]</td></tr></table>

Table 10: Excess in changed verdicts per 100 beyond chance, with its 95% bootstrap interval over items, TDMs in the CALM format. Bold: a bias at $\delta = 0 . 0 5 . \ : ^ { \circ } \colon$ not significant in the permutation test after the Benjamini–Hochberg correction (fallacy oversight has no permutation test). <sup>†</sup>Our additions. <sup>∗</sup>For fallacy oversight, RR + CR − 1.

<table><tr><td>Test</td><td>GPT-3.5 Turbo</td><td>GPT-40</td><td>GPT-6.1 Sol</td><td>Sonnet 5.5</td><td>Gemini 3.8 Fl.</td><td>Qwen3.8-27B</td><td>Haiku 5.5</td></tr><tr><td>Sentiment</td><td>17 [14, 21]</td><td>30 [27, 33]</td><td>38 [35, 41]</td><td>29 [26, 31]</td><td>31 [28, 33]</td><td>35 [32, 38]</td><td>35 [32, 37]</td></tr><tr><td>Authority (on better)†</td><td>7 [2, 11]</td><td>14 [8, 20]</td><td>29 [21, 36]</td><td>31 [23, 39]</td><td>35 [27, 42]</td><td>29 [21, 37]</td><td>26 [18, 35]</td></tr><tr><td>Bandwagon</td><td>15 [11, 18]</td><td>3 [1, 5]</td><td>1 [0, 2]°</td><td>6 [4,9]</td><td>5 [4,7]</td><td>6[3, 9]</td><td>3 [1, 6]</td></tr><tr><td>Authority</td><td>13 [8, 18]</td><td>12 [7, 18]</td><td>32 [25, 40]</td><td>14 [8, 20]</td><td>9 [2, 15]</td><td>12 [4, 19]</td><td>12 [6, 19]</td></tr><tr><td>Position</td><td>37 [32, 42]</td><td>21 [17, 25]</td><td>3 [1, 5]</td><td>3 [1, 6]</td><td>1 [-1, 3]°</td><td>6 [2, 10]</td><td>6 [2, 9]</td></tr><tr><td>Compassion fade</td><td>10 [7, 13]</td><td>6 [3, 9]</td><td>2 [-0, 3]°</td><td>2 [-0, 4]°</td><td>1 [-1, 3]°</td><td>1 [-2, 4]°</td><td>6 [3, 10]</td></tr><tr><td>Verbosity (on better)†</td><td>7 [4, 10]</td><td>4 [2, 6]</td><td>19 [15, 22]</td><td>14 [11, 17]</td><td>13 [10, 16]</td><td>10 [7, 13]</td><td>17 [14, 21]</td></tr><tr><td>Fallacy oversight*</td><td>35 [31, 40]</td><td>1 [-0, 2]</td><td>7[4, 9]</td><td>5 [3,7]</td><td>10 [7, 13]</td><td>4 [2, 7]</td><td>2 [1, 4]</td></tr><tr><td>Neutral note†</td><td>4 [1, 6]</td><td>4 [2, 6]</td><td>1 [-0, 2]°</td><td>2 [1, 4]</td><td>2 [0, 3]°</td><td>-1 [-3, 2]°</td><td>2 [-1, 5]°</td></tr><tr><td>Distraction</td><td>4[2, 6]</td><td>5 [3, 7]</td><td>1 [-1, 2]°</td><td>2 [1,4]</td><td>7 [5, 10]</td><td>1 [-2, 4]°</td><td>4 [1,7]</td></tr><tr><td>Diversity</td><td>4 [2, 7]</td><td>3 [1, 4]</td><td>0 [-1, 1]°</td><td>2 [0, 4]</td><td>2 [0, 3]</td><td>0 [-2, 3]°</td><td>2 [-1, 4]°</td></tr><tr><td>Verbosity</td><td>16 [12, 19]</td><td>2 [0, 3]°</td><td>3 [1, 5]</td><td>1 [0, 3]</td><td>1 [-0, 3]°</td><td>0 [-1, 1]°</td><td>1 [-1, 2]°</td></tr></table>

Table 11: Excess in changed verdicts per 100 beyond chance, with its 95% bootstrap interval over items, LLM judges on the same prompts. Bold: a bias at $\delta \ : = \ : 0 . 0 5 . \ : \ : ^ { \circ } ;$ not significant in the permutation test after the Benjamini–Hochberg correction (fallacy oversight has no permutation test). <sup>†</sup>Our additions. <sup>∗</sup>For fallacy oversight, $\mathrm { R R } + \mathrm { C R } - 1$

Tables 10 and 11 give the excess of every judge in every test with its 95% interval, from which Figure 1 is drawn, and mark the tests that are not significant after the correction. Significance alone would call almost every test a bias. Of the 88 pairs of a TDM and a test other than fallacy oversight, 81 show a significant excess after the Benjamini– Hochberg correction, but only 43 exceed δ. A judge that repeats its verdicts makes even a handful of changed verdicts significant. Table 12 counts the judges with a bias, the TDMs for three margins δ and in the native and decomposed formats. With $\delta = 0 . 0 2$ or 0.10, sentiment remains a bias in all eight TDMs and position in eight or five, the tests that are biases in three to seven TDMs gain or lose up to four models, and at 0.02 the other five tests reach the margin in two to five TDMs. In the native and decomposed formats, the TDMs are biased in three to ten and three to eleven tests, sentiment and position remain biases in seven or eight of them, and the other five tests reach the margin in up to three.

<table><tr><td></td><td colspan="3">TDMs, CALM format</td><td colspan="3">TDMs, 0.05</td><td colspan="3">LLMs, 0.05</td></tr><tr><td>Test</td><td>0.02</td><td>0.05</td><td>0.10</td><td>nat.</td><td></td><td>dec.</td><td>2026</td><td>2024</td></tr><tr><td>Sentiment</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td><td></td><td>5</td><td>2</td></tr><tr><td>Authority (on better)</td><td>8</td><td>7</td><td>6</td><td>7</td><td></td><td>7</td><td>5</td><td>1</td></tr><tr><td>Bandwagon</td><td>7</td><td>5</td><td>4</td><td>7</td><td></td><td>6</td><td>0</td><td>1</td></tr><tr><td>Authority</td><td>8</td><td>7</td><td>4</td><td>6</td><td></td><td>5</td><td>3</td><td>2</td></tr><tr><td>Position</td><td>8</td><td>8</td><td>5</td><td></td><td>7</td><td>8</td><td>0</td><td>2</td></tr><tr><td>Compassion fade</td><td>7</td><td>3</td><td>2</td><td></td><td>5</td><td>6</td><td>0</td><td>1</td></tr><tr><td>Verbosity (on better)</td><td>5</td><td>3</td><td>1</td><td></td><td>2</td><td>4</td><td>5</td><td>0</td></tr><tr><td>Fallacy oversight</td><td>5</td><td>1</td><td>0</td><td></td><td>2</td><td>2</td><td>1</td><td>1</td></tr><tr><td>Neutral note</td><td>4</td><td>0</td><td>0</td><td></td><td>3</td><td>3</td><td>0</td><td>0</td></tr><tr><td>Distraction</td><td>4</td><td>0</td><td>0</td><td></td><td>3</td><td>2</td><td>1</td><td>0</td></tr><tr><td>Diversity</td><td>4</td><td>1</td><td>0</td><td></td><td>3</td><td>3</td><td>0</td><td>0</td></tr><tr><td>Verbosity</td><td>2</td><td>1</td><td>0</td><td></td><td>1</td><td>1</td><td>0</td><td>1</td></tr></table>

Table 12: Number of judges with a bias: the TDMs (of 8) in the CALM format for the margins δ = 0.02, 0.05 and 0.10, the TDMs in the native (nat.) and decomposed (dec.) formats at $\delta = 0 . 0 5 ,$ and the LLM judges of 2026 (of 5) and 2024 (of 2) at $\delta = 0 . 0 5$

Averaged over the twelve tests, the median TDM changes 11.5 of 100 verdicts beyond chance, GPT-3.5 Turbo 14.0, GPT-4o 8.7 and the median 2026 LLM judge 9.7. At δ = 0.02 and 0.10, the median TDM is biased in 8.5 and 4 tests, GPT-3.5 Turbo in 10 and 5, GPT-4o in 7 and 2, and the median 2026 LLM judge in 6 and 3.

F.3 Direction of the Cues
<table><tr><td>Judge</td><td>Pos. (b.) Pos. (w.)</td><td></td><td>Ver.</td><td>Ver. (b.) Ban. Com. Aut. Aut. (b.) Sen. (b.) Sen. (w.)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TDMs, CALM format</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Jev 1.13</td><td>+5</td><td>+6</td><td>+0</td><td>-9</td><td>+22</td><td>+1</td><td>-13</td><td>-30</td><td>-73</td><td>+0</td></tr><tr><td>Kev 4B</td><td>+7</td><td>+16</td><td>-0</td><td>-9</td><td>+18</td><td>+13</td><td>-12</td><td>-21</td><td>-76</td><td>-1</td></tr><tr><td>Tev1 4B</td><td>+19</td><td>+17</td><td>-0</td><td>-3</td><td>+4</td><td>+13</td><td>-15</td><td>-24</td><td>-51</td><td>-1</td></tr><tr><td>Solar Decide</td><td>-14</td><td>-17</td><td>+3</td><td>-0</td><td>+6</td><td>+9</td><td>-10</td><td>-16</td><td>-47</td><td>-2</td></tr><tr><td>D1</td><td>-8</td><td>-11</td><td>-1</td><td>-14</td><td>+5</td><td>+2</td><td>-19</td><td>-29</td><td>-70</td><td>+0</td></tr><tr><td>Perplexity Decider</td><td>+0</td><td>+0</td><td>-0</td><td>-1</td><td>+26</td><td>+2</td><td>-16</td><td>-38</td><td>-59</td><td>+0</td></tr><tr><td>Clef</td><td>+7</td><td>+9</td><td>-0</td><td>-6</td><td>+27</td><td>+3</td><td>-17</td><td>-34</td><td>-58</td><td>+0</td></tr><tr><td>Clef Flash</td><td>+14</td><td>+12</td><td>+6</td><td>+5</td><td>+12</td><td>+4</td><td>-6</td><td>-3</td><td>-45</td><td>-6</td></tr><tr><td>LLM judges</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-3.5 Turbo</td><td>-42</td><td>-36</td><td>+13</td><td>+9</td><td>+17</td><td>+1</td><td>+12</td><td>+6</td><td>-16</td><td>+5</td></tr><tr><td>GPT-40</td><td>-22</td><td>-23</td><td>+2</td><td>-4</td><td>+0</td><td>+2</td><td>-13</td><td>-14</td><td>-61</td><td>+0</td></tr><tr><td>GPT-6.1 Sol</td><td>-0</td><td>-2</td><td>-4</td><td>-19</td><td>+1</td><td>-1</td><td>-36</td><td>-32</td><td>-77</td><td>-4</td></tr><tr><td>Claude Sonnet 5.5</td><td>+0</td><td>+1</td><td>-1</td><td>-12</td><td>+8</td><td>+1</td><td>-19</td><td>-36</td><td>-55</td><td>-1</td></tr><tr><td>Gemini 3.8 Flash</td><td>-2</td><td>-0</td><td>-2</td><td>-11</td><td>+7</td><td>+1</td><td>-14</td><td>-39</td><td>-60</td><td>-1</td></tr><tr><td>Qwen3.8-27B</td><td>-6</td><td>-8</td><td>-1</td><td>-10</td><td>+9</td><td>-3</td><td>-22</td><td>-38</td><td>-70</td><td>-2</td></tr><tr><td>Claude Haiku 5.5</td><td>-2</td><td>+1</td><td>-1</td><td>-18</td><td>+5</td><td>+10</td><td>-22</td><td>-36</td><td>-68</td><td>-1</td></tr><tr><td>TDMs, median</td><td>+6</td><td>+8</td><td>-0</td><td>-4</td><td>+15</td><td>+3</td><td>-14</td><td>-26</td><td>-58</td><td>-1</td></tr><tr><td>LLM judges 2026, median</td><td>-2</td><td>-0</td><td>-1</td><td>-12</td><td>+7</td><td>+1</td><td>-22</td><td>-36</td><td>-68</td><td>-1</td></tr></table>

Table 13: Change in the share of verdicts for the response that carries the cue, after minus before the perturbation, in percentage points. Positive: the cue pulls verdicts towards its response. Pos.: position, with the better (b.) or the worse (w.) response moved to the second place. Sen.: sentiment, with the tone on the better (b.) or the worse (w.) response. Other abbreviations as in Table 9. Under bandwagon and compassion fade, the cue favours the worse response.

<table><tr><td>Model</td><td>Ver.</td><td>Fal.</td><td>Sen.</td><td>Pos.</td><td>Com.</td><td>Ban.</td><td>Aut.</td><td>Dst.</td><td>Div.</td><td>Aut. (b.)</td><td>Ver. (b.)</td><td>Neu.</td></tr><tr><td>CALM format</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Jev 1.13</td><td>-0.02</td><td>-0.93</td><td>-0.34</td><td>+0.04</td><td>+0.02</td><td>+0.17</td><td>-0.15</td><td>-0.06</td><td>-0.03</td><td>-0.21</td><td>-0.16</td><td>-0.03</td></tr><tr><td>Kev 4B</td><td>-0.04</td><td>-0.71</td><td>-0.28</td><td>+0.08</td><td>+0.07</td><td>+0.08</td><td>-0.08</td><td>-0.03</td><td>-0.04</td><td>-0.10</td><td>-0.08</td><td>-0.02</td></tr><tr><td>Tev1 4B</td><td>+0.01</td><td>-0.71</td><td>-0.22</td><td>+0.06</td><td>+0.11</td><td>+0.04</td><td>-0.09</td><td>-0.03</td><td>-0.05</td><td>-0.14</td><td>-0.06</td><td>-0.04</td></tr><tr><td>Solar Decide</td><td>+0.04</td><td>-0.84</td><td>-0.22</td><td>-0.12</td><td>+0.07</td><td>+0.06</td><td>-0.09</td><td>-0.01</td><td>-0.01</td><td>-0.15</td><td>-0.01</td><td>-0.01</td></tr><tr><td>D1</td><td>-0.04</td><td>-0.82</td><td>-0.30</td><td>-0.08</td><td>+0.01</td><td>+0.04</td><td>-0.16</td><td>-0.00</td><td>-0.01</td><td>-0.20</td><td>-0.15</td><td>-0.01</td></tr><tr><td>Perplexity Decider</td><td>-0.02</td><td>-0.85</td><td>-0.27</td><td>-0.01</td><td>+0.02</td><td>+0.19</td><td>-0.18</td><td>-0.01</td><td>-0.01</td><td>-0.25</td><td>-0.06</td><td>-0.01</td></tr><tr><td>Clef</td><td>-0.01</td><td>-0.93</td><td>-0.28</td><td>+0.08</td><td>+0.04</td><td>+0.26</td><td>-0.16</td><td>-0.01</td><td>-0.02</td><td>-0.31</td><td>-0.06</td><td>-0.01</td></tr><tr><td>Clef Flash</td><td>+0.08</td><td>-0.79</td><td>-0.23</td><td>+0.10</td><td>+0.04</td><td>+0.11</td><td>-0.03</td><td>-0.02</td><td>-0.04</td><td>-0.04</td><td>+0.04</td><td>-0.04</td></tr><tr><td>native format</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Jev 1.13</td><td>-0.03</td><td>-0.92</td><td>-0.34</td><td>+0.05</td><td>+0.02</td><td>+0.20</td><td>-0.15</td><td>-0.04</td><td>-0.03</td><td>-0.20</td><td>-0.16</td><td>-0.04</td></tr><tr><td>Kev 4B</td><td>-0.03</td><td>-0.64</td><td>-0.23</td><td>+0.10</td><td>+0.07</td><td>+0.09</td><td>-0.04</td><td>-0.04</td><td>-0.04</td><td>-0.05</td><td>-0.02</td><td>-0.03</td></tr><tr><td>Tev1 4B</td><td>-0.01</td><td>-0.74</td><td>-0.24</td><td>+0.13</td><td>+0.10</td><td>+0.06</td><td>-0.08</td><td>-0.03</td><td>-0.05</td><td>-0.13</td><td>-0.06</td><td>-0.03</td></tr><tr><td>Solar Decide</td><td>+0.06</td><td>-0.84</td><td>-0.18</td><td>-0.00</td><td>+0.11</td><td>+0.07</td><td>-0.03</td><td>-0.01</td><td>-0.01</td><td>-0.07</td><td>+0.01</td><td>-0.02</td></tr><tr><td>D1</td><td>-0.05</td><td>-0.85</td><td>-0.32</td><td>-0.05</td><td>+0.02</td><td>+0.05</td><td>-0.14</td><td>-0.02</td><td>-0.03</td><td>-0.19</td><td>-0.15</td><td>-0.03</td></tr><tr><td>Perplexity Decider</td><td>-0.02</td><td>-0.87</td><td>-0.28</td><td>-0.04</td><td>+0.02</td><td>+0.16</td><td>-0.19</td><td>-0.00</td><td>-0.02</td><td>-0.28</td><td>-0.06</td><td>-0.01</td></tr><tr><td>Clef</td><td>-0.01</td><td>-0.90</td><td>-0.21</td><td>+0.13</td><td>+0.01</td><td>+0.07</td><td>-0.12</td><td>-0.02</td><td>-0.04</td><td>-0.20</td><td>-0.03</td><td>-0.03</td></tr><tr><td>Clef Flash</td><td>+0.12</td><td>-0.76</td><td>-0.20</td><td>+0.11</td><td>-0.01</td><td>+0.10</td><td>-0.02</td><td>-0.02</td><td>-0.03</td><td>-0.03</td><td>+0.06</td><td>-0.03</td></tr><tr><td>decomposed format</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Jev 1.13</td><td>-0.03</td><td>-0.89</td><td>-0.32</td><td>+0.04</td><td>+0.02</td><td>+0.20</td><td>-0.13</td><td>-0.04</td><td>-0.03</td><td>-0.19</td><td>-0.17</td><td>-0.04</td></tr><tr><td>Kev 4B</td><td>-0.03</td><td>-0.63</td><td>-0.22</td><td>+0.10</td><td>+0.07</td><td>+0.10</td><td>-0.04</td><td>-0.04</td><td>-0.05</td><td>-0.05</td><td>-0.02</td><td>-0.03</td></tr><tr><td>Tev1 4B</td><td>-0.02</td><td>-0.74</td><td>-0.25</td><td>+0.13</td><td>+0.10</td><td>+0.06</td><td>-0.09</td><td>-0.03</td><td>-0.05</td><td>-0.13</td><td>-0.07</td><td>-0.03</td></tr><tr><td>Solar Decide</td><td>+0.03</td><td>-0.83</td><td>-0.19</td><td>-0.05</td><td>+0.10</td><td>+0.06</td><td>-0.04</td><td>-0.01</td><td>-0.01</td><td>-0.05</td><td>-0.01</td><td>-0.01</td></tr><tr><td>D1</td><td>-0.05</td><td>-0.83</td><td>-0.31</td><td>-0.06</td><td>+0.02</td><td>+0.05</td><td>-0.13</td><td>-0.02</td><td>-0.03</td><td>-0.19</td><td>-0.16</td><td>-0.03</td></tr><tr><td>Perplexity Decider</td><td>-0.01</td><td>-0.83</td><td>-0.26</td><td>-0.04</td><td>+0.03</td><td>+0.16</td><td>-0.18</td><td>-0.01</td><td>-0.02</td><td>-0.27</td><td>-0.06</td><td>-0.01</td></tr><tr><td>Clef</td><td>+0.03</td><td>-0.67</td><td>-0.10</td><td>+0.12</td><td>-0.00</td><td>+0.03</td><td>-0.08</td><td>-0.02</td><td>-0.02</td><td>-0.14</td><td>-0.04</td><td>-0.02</td></tr><tr><td>Clef Flash</td><td>+0.10</td><td>-0.61</td><td>-0.15</td><td>+0.07</td><td>-0.01</td><td>+0.07</td><td>-0.01</td><td>-0.01</td><td>-0.02</td><td>-0.02</td><td>+0.04</td><td>-0.02</td></tr></table>

Table 14: Mean shift ∆p in the probability of the targeted response per model, format and test. Positive: the cue is rewarded, negative: it is punished.

For the median TDM, a claimed majority raises the probability of the response it favours by 0.10, and by up to 0.26 for Clef. A strong model’s name raises it by 0.04, and the second place by 0.05, which five models prefer and three do not. An emotional tone lowers the probability of the response that carries it by 0.27, and a citation by 0.12 on the worse and by 0.17 on the better response. Padding is punished on the better response (median −0.06, Jev −0.16), while on the worse response the models barely react (median −0.02, five punishing and three rewarding it). Table 13 gives the change in the share of verdicts for the response that carries the cue, and Table 14 the shift of every model in every format and test. At the median, the TDMs raise the share of verdicts for the response in second place by 6 percentage points when the swap moves the better response there and by 8 when it moves the worse one. The 2024 judges lower it by 22 to 42 points, and the 2026 judges hardly move it. Among the prompts whose verdict stays the same, the probability of the targeted response moves by more than 0.10 in 16% of them, all TDMs pooled, in 27 to 32% under position, bandwagon, authority, authority (on better) and sentiment, in 21% under verbosity (on better), in 20% under compassion fade, in 12% under verbosity and in 7 to 9% under distraction, diversity and the neutral note. The noise floor, the mean change of the probability on a repeated prompt, is at most 0.013 for seven TDMs (median over the tests) and 0.10 for Solar.

F.4 Accuracy under the Cues
<table><tr><td rowspan="2">Judge</td><td colspan="3">Authority (150 items)</td><td colspan="3">Sentiment (150 items)</td><td colspan="3">Verbosity (500 items)</td></tr><tr><td>Before</td><td>On worse</td><td>On better</td><td>Before</td><td>On better</td><td>On worse</td><td>Before</td><td>On worse</td><td>On better</td></tr><tr><td>TDMs, CALM format</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Jev 1.13</td><td>0.72</td><td>0.85</td><td>0.42</td><td>1.00</td><td>0.27</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.90</td></tr><tr><td>Kev 4B</td><td>0.63</td><td>0.75</td><td>0.42</td><td>0.99</td><td>0.22</td><td>1.00</td><td>0.99</td><td>0.99</td><td>0.90</td></tr><tr><td>Tev1 4B</td><td>0.60</td><td>0.77</td><td>0.34</td><td>0.99</td><td>0.47</td><td>1.00</td><td>0.99</td><td>0.99</td><td>0.95</td></tr><tr><td>Solar Decide</td><td>0.61</td><td>0.71</td><td>0.45</td><td>0.96</td><td>0.49</td><td>0.98</td><td>0.96</td><td>0.93</td><td>0.95</td></tr><tr><td>D1</td><td>0.70</td><td>0.89</td><td>0.40</td><td>1.00</td><td>0.30</td><td>1.00</td><td>0.99</td><td>1.00</td><td>0.85</td></tr><tr><td>Perplexity Decider</td><td>0.71</td><td>0.87</td><td>0.33</td><td>1.00</td><td>0.41</td><td>1.00</td><td>0.99</td><td>0.99</td><td>0.98</td></tr><tr><td>Clef</td><td>0.71</td><td>0.88</td><td>0.37</td><td>1.00</td><td>0.42</td><td>1.00</td><td>0.99</td><td>0.99</td><td>0.93</td></tr><tr><td>Clef Flash</td><td>0.61</td><td>0.66</td><td>0.57</td><td>0.91</td><td>0.46</td><td>0.97</td><td>0.92</td><td>0.86</td><td>0.97</td></tr><tr><td>LLM judges, 2024 and 2026</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-3.5 Turbo</td><td>0.59</td><td>0.47</td><td>0.65</td><td>0.87</td><td>0.71</td><td>0.82</td><td>0.86</td><td>0.72</td><td>0.95</td></tr><tr><td>GPT-40</td><td>0.62</td><td>0.75</td><td>0.48</td><td>0.99</td><td>0.38</td><td>0.99</td><td>0.99</td><td>0.97</td><td>0.95</td></tr><tr><td>GPT-6.1 Sol</td><td>0.59</td><td>0.94</td><td>0.27</td><td>0.94</td><td>0.18</td><td>0.98</td><td>0.94</td><td>0.98</td><td>0.75</td></tr><tr><td>Claude Sonnet 5.5</td><td>0.71</td><td>0.90</td><td>0.35</td><td>0.97</td><td>0.42</td><td>0.98</td><td>0.97</td><td>0.98</td><td>0.85</td></tr><tr><td>Gemini 3.8 Flash</td><td>0.74</td><td>0.87</td><td>0.35</td><td>0.98</td><td>0.38</td><td>0.99</td><td>0.96</td><td>0.97</td><td>0.85</td></tr><tr><td>Qwen3.8-27B</td><td>0.67</td><td>0.89</td><td>0.30</td><td>0.97</td><td>0.27</td><td>0.99</td><td>0.97</td><td>0.98</td><td>0.87</td></tr><tr><td>Claude Haiku 5.5</td><td>0.67</td><td>0.88</td><td>0.31</td><td>0.99</td><td>0.30</td><td>0.99</td><td>0.98</td><td>0.99</td><td>0.81</td></tr><tr><td>TDMs, median</td><td>0.66</td><td>0.81</td><td>0.41</td><td>0.99</td><td>0.42</td><td>1.00</td><td>0.99</td><td>0.99</td><td>0.94</td></tr><tr><td>LLM judges 2026, median</td><td>0.67</td><td>0.89</td><td>0.31</td><td>0.97</td><td>0.30</td><td>0.99</td><td>0.97</td><td>0.98</td><td>0.85</td></tr></table>

Table 15: Accuracy, the share of verdicts for the better response, on the original prompts of a test’s items (Before) and after the cue is placed on the worse or on the better response: authority and authority (on better), sentiment with the tone on either response, and verbosity and verbosity (on better), the two tests on the better response being our additions. Authority pools the book, URL and quote, sentiment the four tones. The 95% bootstrap interval over items of every change is at most ±0.07 wide. It excludes zero under authority, authority (on better) and sentiment on the better response, except for Clef Flash under authority (on better), and under verbosity (on better), except for Solar and Perplexity. Under sentiment on the worse response and under verbosity, accuracy hardly changes.

Table 15 gives the accuracy of every judge before and after the cue for the three tests whose cue can sit on either response (Section 4.4).

## F.5 Results per Version

Figure 4 and Table 16 split each test into its versions. Under bandwagon, the robustness rate of every TDM is lower at a claimed majority of 90% than at 60%, while the median of the 2026 LLM judges falls only from 0.93 to 0.89. GPT-3.5 Turbo follows a larger majority as well (0.86 to 0.75), GPT-4o hardly (0.93 to 0.92). Authority (on better) lowers the robustness rate more than authority, for each kind of citation and for both kinds of judge, and quotes do so most. Under sentiment, a tone on the better response flips many verdicts and on the worse response almost none. Under diversity, the identity groups do not differ from each other or from the neutral note, where CALM’s judges reacted most to female and refugee identities.

(a) Bandwagon  
![](images/d1b0627ab04a1284a107f7bf1fd16f8df650160faac51e8fc0de4ef489709341.jpg)  
(c) Sentiment

![](images/8a1fbd3b098b9ad4d490cafa699a9fade0f80a110627cb7745ad32764cdd577d.jpg)

![](images/699366900d78b7cea51b7786ce85eb3961770444aa32dfb96ad2c9ab79002061.jpg)  
Figure 4: Robustness rate per version, TDMs in the CALM format and the seven LLM judges on the same prompts. (a) Bandwagon by the claimed majority, one line per TDM and per LLM judge of 2024 (black), with the median and range of the 2026 LLM judges in grey. (b) to (d) Median and range over the TDMs (blue) and the 2026 LLM judges (orange), and mean and range of the two 2024 LLM judges (green), for authority by kind of citation and response, sentiment by tone and response, and diversity by identity group, both responses pooled.

<table><tr><td>Test</td><td>Version</td><td>TDMs, RR</td><td>LLM judges 2026, RR</td><td>TDMs,  $\Delta p$ </td></tr><tr><td rowspan="4">Bandwagon</td><td>60%</td><td>0.87 [0.81, 0.98]</td><td>0.93 [0.87, 0.97]</td><td>+0.06</td></tr><tr><td>70%</td><td>0.86 [0.74, 0.97]</td><td>0.91 [0.86, 0.97]</td><td>+0.09</td></tr><tr><td>80%</td><td>0.83 [0.68, 0.95]</td><td>0.89 [0.84, 0.97]</td><td>+0.11</td></tr><tr><td>90%</td><td>0.80 [0.66, 0.92]</td><td>0.89 [0.84, 0.97]</td><td>+0.13</td></tr><tr><td rowspan="6">Authority</td><td>Book, on worse</td><td>0.86 [0.82, 0.88]</td><td>0.79 [0.64, 0.86]</td><td>-0.11</td></tr><tr><td>Book, on better</td><td>0.76 [0.64, 0.94]</td><td>0.60 [0.60, 0.67]</td><td>-0.15</td></tr><tr><td>URL, on worse</td><td>0.87 [0.83, 0.91]</td><td>0.79 [0.68, 0.87]</td><td>-0.10</td></tr><tr><td>URL, on better</td><td>0.78 [0.63, 0.93]</td><td>0.69 [0.61, 0.75]</td><td>-0.13</td></tr><tr><td>Quote, on worse</td><td>0.81 [0.75, 0.87]</td><td>0.77 [0.61, 0.85]</td><td>-0.17</td></tr><tr><td>Quote, on better</td><td>0.65 [0.59, 0.90]</td><td>0.59 [0.58, 0.62]</td><td>-0.25</td></tr><tr><td rowspan="8">Sentiment</td><td>Cheerful, on better</td><td>0.67 [0.49, 0.81]</td><td>0.56 [0.45, 0.70]</td><td>-0.29</td></tr><tr><td>Cheerful, on worse</td><td>0.99 [0.94, 1.00]</td><td>0.98 [0.96, 1.00]</td><td>+0.00</td></tr><tr><td>Sad, on better</td><td>0.31 [0.04, 0.49]</td><td>0.25 [0.21, 0.47]</td><td>-0.56</td></tr><tr><td>Sad, on worse</td><td>0.99 [0.91, 1.00]</td><td>0.99 [0.95, 0.99]</td><td>-0.04</td></tr><tr><td>Angry, on better</td><td>0.38 [0.21, 0.66]</td><td>0.20 [0.09, 0.23]</td><td>-0.47</td></tr><tr><td>Angry, on worse</td><td>0.99 [0.94, 1.00]</td><td>0.99 [0.95, 0.99]</td><td>-0.05</td></tr><tr><td>Fearful, on better</td><td>0.27 [0.07, 0.36]</td><td>0.24 [0.17, 0.41]</td><td>-0.58</td></tr><tr><td>Fearful, on worse</td><td>0.99 [0.90, 1.00]</td><td>0.98 [0.96, 0.99]</td><td>-0.05</td></tr><tr><td rowspan="2">Distraction</td><td>About the better</td><td>0.96 [0.88, 0.98]</td><td>0.90 [0.84, 0.97]</td><td>-0.02</td></tr><tr><td>About the worse</td><td>0.95 [0.89, 1.00]</td><td>0.92 [0.90, 0.98]</td><td>-0.02</td></tr><tr><td rowspan="8">Diversity</td><td>Homosexual</td><td>0.95 [0.88, 0.99]</td><td>0.94 [0.88, 0.97]</td><td></td></tr><tr><td>Black</td><td>0.96 [0.87, 0.99]</td><td>0.94 [0.89, 0.97]</td><td>-0.02 -0.02</td></tr><tr><td>Female</td><td>0.96 [0.89, 0.99]</td><td>0.95 [0.91, 0.97]</td><td>-0.01</td></tr><tr><td>HIV Positive</td><td>0.94 [0.89, 0.98]</td><td>0.95 [0.88, 0.98]</td><td>-0.03</td></tr><tr><td>Refugees</td><td>0.95 [0.88, 0.98]</td><td>0.95 [0.89, 0.98]</td><td>-0.03</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Muslim believers</td><td>0.95 [0.89, 0.98]</td><td>0.94 [0.90, 0.98]</td><td>-0.03</td></tr><tr><td>Neutral note</td><td>0.95 [0.88, 0.98]</td><td>0.94 [0.89, 0.97]</td><td>-0.02</td></tr></table>

Table 16: Robustness rate per version, median [minimum, maximum] over the eight TDMs (CALM format) and over the five 2026 LLM judges on the same prompts, and the median shift of the TDMs. Diversity pools both responses.

## G TDMs and LLM Judges

TDMs and 2026 LLM judges are about as similar to each other as the members of each group. Over the twelve tests, the excess profiles of two TDMs correlate on average with $\rho = 0 . 6 9$ , those of two 2026 LLM judges with 0.77, and those of a TDM and a 2026 LLM judge with 0.62 (Spearman). On the original prompts, two TDMs agree on 89% of the verdicts, two 2026 LLM judges on 90%, and a TDM and a 2026 LLM judge on 87%.

The two kinds of judge also flip on the same prompts. For a pair of judges and a test, we take the prompts on which the verdict of the first judge flips and ask how often the verdict of the second flips on them, compared with the prompts on which the first verdict stays (Table 17). For a TDM and a 2026 LLM judge, the second verdict flips on 27% of the prompts on which the first flips and on 10% of the others, as for two TDMs (26 against 10%) and almost as for two 2026 LLM judges (30 against 10%). The overlap is largest under sentiment, where a flip of one judge goes with a flip of the other in three of four cases, and smallest under distraction and diversity (10 against 6%). Two TDMs share clearly more flips with each other than with an LLM judge only under position and bandwagon (32 against 23% and 26 against 21%), the two tests under which the TDMs are far more biased than the 2026 LLM judges. Flips of the two kinds thus cluster on the same prompts about as much as those of two TDMs, so part of what a test measures lies in its prompts rather than in the judge.

<table><tr><td>Test</td><td>Two TDMs</td><td>Two LLM judges</td><td>TDM and LLM judge</td></tr><tr><td>Sentiment</td><td>74/11</td><td>81 / 10 75 / 12</td><td>0.97</td></tr><tr><td>Authority (on better)</td><td>40 / 20</td><td>60 / 23</td><td>44 / 25 0.83</td></tr><tr><td>Verbosity (on better)</td><td>25 / 6</td><td>46 / 10</td><td>34/9 0.99</td></tr><tr><td>Authority</td><td>31 / 13</td><td>42/ 18</td><td>33 / 16 0.89</td></tr><tr><td>Position</td><td>32 / 14</td><td>14/9</td><td>23 / 12 1.00</td></tr><tr><td>Bandwagon</td><td>26 / 14</td><td>16/9</td><td>21 / 12 0.96</td></tr><tr><td>Verbosity</td><td>16/3</td><td>17/2</td><td>16 /2 1.00</td></tr><tr><td>Compassion fade</td><td>15 / 10</td><td>13/7</td><td>14/8 1.00</td></tr><tr><td>Neutral note</td><td>12/5</td><td>15/6</td><td>11/6 0.94</td></tr><tr><td>Distraction</td><td>7/5</td><td>14/8</td><td>10/6 1.00</td></tr><tr><td>Diversity</td><td>12/5</td><td>12/6</td><td>10 /6 0.91</td></tr><tr><td>Mean</td><td>26 / 10</td><td>30 / 10</td><td>27 / 10 0.95</td></tr></table>

Table 17: Do two judges flip on the same prompts? For a pair of judges, the share of prompts (%) on which the verdict of the second flips, among those on which the verdict of the first flips / does not flip, averaged over the pairs of eight TDMs (CALM format), of the five 2026 LLM judges, and of a TDM and a 2026 LLM judge (both ways round). 95% bootstrap intervals over items for the mean when the first flips: 25 to 28, 28 to 32 and 25 to 28. Fallacy oversight is left out, since there a flip is the correct reaction. r: correlation of the prompt effects of the TDMs and the 2026 LLM judges in a logistic mixed model of the test, and of all tests in the last row (Appendix G). A value of 1.00 is its upper limit.

A logistic mixed model puts this into one number and separates the prompt from the judge and the test. For judge j and prompt i of test t(i), fitted with lme4 on the eleven tests other than fallacy oversight, it is

$$
\begin{array} { l } { \displaystyle \log \mathrm { i t } P ( f _ { i j } = 1 ) = \alpha _ { j } { _ t } _ { ( i ) } + a _ { j } { u _ { i } } + b _ { j } v _ { i } , } \\ { \displaystyle ( u _ { i } , v _ { i } ) \sim \mathcal { N } ( \mathbf { 0 } , \Sigma ) , \quad \mathrm { c o r r } ( u _ { i } , v _ { i } ) = r , } \end{array}
$$

with one intercept α per judge and test, and $a _ { j } = 1$ for a TDM and $b _ { j } = 1$ for a 2026 LLM judge, 0 otherwise. Thus $u _ { i }$ and v<sub>i</sub> are the effects of prompt i on the TDMs and on the 2026 LLM judges, and the share of the prompt is $\sigma _ { u } ^ { 2 } / ( \sigma _ { u } ^ { 2 } + \pi ^ { 2 } / 3 )$ on the logit scale. The prompt accounts for much of whether a verdict flips, 38% of the remaining variance on the logit scale for the TDMs and 42% for the 2026 LLM judges, and the prompt effects of the two kinds correlate with $r ~ = ~ 0 . 9 5$ . Per test, r lies between 0.83 under authority (on better) and 1 (Table 17), and the share of the prompt between 13% (distraction, TDMs) and 90% (sentiment, LLM judges). Four of the eleven per-test fits are singular, with r at its upper limit of 1. With an additional random effect for the item, the item effects of the two kinds correlate with $r = 0 . 7 4$ and the prompt effects with 1. All mixed models are fitted with lme4 1.1.37 in R 4.5.2, the logistic ones with the fast approximation $\neg { \sf A G Q } \ = \ 0$ . As a benchmark, we split the TDMs and the 2026 LLM judges into two halves each, five times at random, and fit the same model for every pair of halves. Two halves of the same kind correlate with r = 1.00, and a half of each kind with 0.996 on average. In most of these fits the correlation reaches its upper limit of 1, which means the data cannot tell the prompt effects of the two groups apart, so the small differences between these values say nothing about the kinds. The main correlation of 0.95, however, shows that the prompts that make TDMs flip are the ones that make the 2026 LLM judges flip.

CALM found that its LLM judges favour citations. Among our judges only GPT-3.5 Turbo still does, GPT-4o already punishes them, and the 2026 LLM judges punish them as the TDMs do. Jev reacts under authority as strongly as the median 2026 LLM judge (excess 12). Under verbosity (on better), the accuracy of the 2026 LLM judges falls from 0.97 to 0.85 at the median and that of the TDMs from 0.99 to 0.94, while that of GPT-3.5 Turbo rises from 0.86 to 0.95. Under position, Clef’s excess is larger than that of its base LLM, Qwen3.8-27B (14 against 6), but only because Qwen also changes verdicts by chance. Under the swap, both change about as many verdicts (13.9 and 14.8%, paired test, adjusted p = 0.74).

A closer architecture also does not make a model behave more like an LLM judge. Over the twelve tests, we correlate the excess of each TDM with the median excess of the five 2026 LLM judges (Spearman). Tev1, which keeps the language-model head of its LLM and answers with one letter, is the least similar model (0.38), followed by Clef Flash (0.45) and Solar (0.49). Jev (0.87), Perplexity (0.81) and Clef (0.80) are the most similar. With eight models and twelve tests, this is only a first indication.

## H When Verdicts Flip

## H.1 Flips and the Judge’s Confidence

Figure 5 shows the share of flipped verdicts by the probability that the judge gave its choice on the original prompt (Section 4.3).

![](images/2889089112286f9606cc12d784b472e26f7091d64663add690909f2c71e5bc5a.jpg)  
Figure 5: Share of flipped verdicts by the probability that the judge gave its choice on the original prompt, CALM format, all models pooled. The bins run from 0.5 to 0.9 in steps of 0.1, then from 0.9 to 0.95, 0.95 to 0.99 and 0.99 to 1, and each point sits in the middle of its bin. The band shows the range of the eight models for the tests other than sentiment and authority (on better).

## H.2 The Logistic Mixed Model

Section 3.3 gives the logistic mixed model of flips, with the margin in steps of 0.1. We fit it on the CALM format without fallacy oversight. We set the test and the model ourselves. Every model judges every item, so the differences between models are not confounded with the items. A test, however, uses the items of one dataset only, so only tests on the same dataset are compared on the same items. A comparison across the two datasets, such as sentiment against the neutral note, also reflects the difference between the two sets of items. The margin, the correctness, the length and the source are properties of the items, and for them we can only report associations. A linear mixed model splits the variance of the shift of prompt i for model j in format f into crossed random effects,

$$
\begin{array} { l } { { \hat { p } _ { i j f } - p _ { i j f } = \mu + a _ { t } + b _ { j } + c _ { f } + ( a b ) _ { t j } + ( a c ) _ { t f } } } \\ { { \mathrm { } + ( b c ) _ { j f } + d _ { s } + e _ { k } + ( a e ) _ { t k } + ( b e ) _ { j k } + \varepsilon _ { i j f } , } } \end{array}
$$

with the test $t = t ( i )$ , the source $s = s ( i )$ and the item $k = k ( i )$ , each term normal with mean zero and a variance of its own, fitted by REML.

Table 18 gives the logistic mixed model of Section 4.3, fitted with lme4 (Bates et al., 2015) on all perturbed prompts of the CALM format with a verdict, without fallacy oversight. The references are

the neutral note, Jev and the Emerton-DPO source. Table 19 splits the variance of the shift over all formats into crossed random effects.
<table><tr><td>Term</td><td>Odds ratio</td><td>95% CI</td></tr><tr><td>Margin of the original, per 0.1</td><td>0.47</td><td>[0.45, 0.48]</td></tr><tr><td>Judge right on the original</td><td>1.54</td><td>[1.44, 1.65]</td></tr><tr><td>Log length ratio, R1 to R2</td><td>0.95</td><td>[0.88, 1.04]</td></tr><tr><td>Test: Authority</td><td>3.07</td><td>[2.64, 3.58]</td></tr><tr><td>Test: Authority (on better)</td><td>6.72</td><td>[5.81, 7.77]</td></tr><tr><td>Test: Bandwagon</td><td>4.67</td><td>[4.13, 5.29]</td></tr><tr><td>Test: Compassion fade</td><td>2.33</td><td>[1.97, 2.74]</td></tr><tr><td>Test: Distraction</td><td>0.98</td><td>[0.84, 1.16]</td></tr><tr><td>Test: Diversity</td><td>1.08</td><td>[0.96, 1.22]</td></tr><tr><td>Test: Position</td><td>4.65</td><td>[4.00, 5.42]</td></tr><tr><td>Test: Sentiment</td><td>18.30</td><td>[13.81, 24.24]</td></tr><tr><td>Test: Verbosity</td><td>0.94</td><td>[0.68, 1.29]</td></tr><tr><td>Test: Verbosity (on better)</td><td>2.54</td><td>[1.90, 3.39]</td></tr><tr><td>Model: Clef</td><td>2.17</td><td>[1.97, 2.39]</td></tr><tr><td>Model: Clef Flash</td><td>0.89</td><td>[0.81, 0.98]</td></tr><tr><td>Model: D1</td><td>0.62</td><td>[0.57, 0.68]</td></tr><tr><td>Model: Kev 4B</td><td>0.53</td><td>[0.48, 0.58]</td></tr><tr><td>Model: Perplexity Decider</td><td>0.77</td><td>[0.70, 0.84]</td></tr><tr><td>Model: Solar Decide</td><td>3.09</td><td>[2.82, 3.39]</td></tr><tr><td>Model: Tev1 4B</td><td>0.50</td><td>[0.45, 0.55]</td></tr><tr><td>Source: GSM8K</td><td>1.47</td><td>[1.15, 1.88]</td></tr><tr><td>Source: MATH</td><td>0.90</td><td>[0.69, 1.16]</td></tr><tr><td>Source: Orca-DPO</td><td>0.77</td><td>[0.59, 1.01]</td></tr><tr><td>Source: Py-DPO</td><td>0.40</td><td>[0.31, 0.52]</td></tr><tr><td>Source: Roleplay</td><td>0.61</td><td>[0.43, 0.85]</td></tr><tr><td>Source: Truthy-DPO</td><td>0.90</td><td>[0.70, 1.16]</td></tr></table>

Table 18: Logistic mixed model of a flipped verdict, CALM format, with a random intercept per item (100,668 prompts, 972 items, item variance 0.66 on the logit scale, intraclass correlation 0.17). References: the neutral note, Jev and Emerton-DPO. ScienceQA has no term of its own. The tests of the fact-related set use only its three sources, so the effect of ScienceQA is contained in the terms of sentiment and verbosity, and GSM8K and MATH are relative to ScienceQA.

An odds ratio of 2 means that a verdict is twice as likely to flip, in odds, as in the reference, and 0.5 that it is half as likely. Each further 0.1 of margin halves the odds of a flip (odds ratio 0.47, 95% CI 0.45 to 0.48), with a similar slope for every model (0.39 to 0.52). A verdict that was right on the original flips more often (odds ratio 1.54). This follows largely from the design, since most cues either favour the worse response or are punished on the better one, and both can flip only a right verdict. Relative to the neutral note, the odds of a flip are 18.3 times as high under sentiment, 6.7 under authority (on better), 4.7 under bandwagon and under position, 3.1 under authority, 2.5 under verbosity (on better) and 2.3 under compassion fade. Diversity (1.08, CI 0.96 to 1.22), distraction (0.98) and verbosity (0.94) do not differ from the neutral note. Sentiment, verbosity and verbosity (on better)

are tested on the fact-related items, so their odds ratios also reflect the difference between the two sets of items. Among alignment items, pairs of a GPT-4 and a GPT-4-Turbo response (Emerton-DPO) flip most often, and pairs of tested code and code written by open models (Py-DPO) least, with 0.40 times the odds. The length of the responses plays no role (0.95 per unit of log length ratio, CI 0.88 to 1.04). The test explains 25% of the variance of the shift, its interaction with the item 8% and with the model 6%. The model alone explains 1.4%, and the format and the source almost nothing.

<table><tr><td>Variance component</td><td>Shift (%)</td><td>Size of shift (%)</td></tr><tr><td>Residual</td><td>56.7</td><td>57.4</td></tr><tr><td>Test</td><td>25.2</td><td>13.8</td></tr><tr><td>Item × Test</td><td>8.0</td><td>5.3</td></tr><tr><td>Test × Model</td><td>6.0</td><td>7.3</td></tr><tr><td>Item × Model</td><td>2.0</td><td>9.0</td></tr><tr><td>Model</td><td>1.4</td><td>0.5</td></tr><tr><td>Test × Format</td><td>0.4</td><td>0.4</td></tr><tr><td>Model × Format</td><td>0.1</td><td>0.6</td></tr><tr><td>Source</td><td>0.1</td><td>0.6</td></tr><tr><td>Item</td><td>0.0</td><td>5.0</td></tr><tr><td>Format</td><td>0.0</td><td>0.1</td></tr></table>

Table 19: Share of the variance of the shift $\Delta p$ and of its size $| \Delta p |$ by crossed random effects, all formats, without fallacy oversight. The residual holds the variation between the versions and formats of a prompt that no component captures.

Li et al. (2026) report the same pattern for Jev when the order of two answers is reversed: 46% of verdicts given with less than 0.6 flip, but only 3 of 1,293 given with 0.9 or more. For a ranking of systems such flips may matter less than their share suggests, since close pairs carry little of the ranking signal (Brocai and Becker, 2026). For a single verdict, such as a filter decision or a reward, every flip counts. That the test matters far more than the model suggests that these biases are shared by the eight TDMs we test. Part of this may have a common origin, since five of them are fine-tuned from the same family of LLMs and two do not disclose theirs (Table 6), so other TDMs may differ.

## H.3 The Same Items for All Models?

For each test we correlate the mean shift of every item between each pair of models (Spearman) and average over the 28 pairs (Table 20). Under sentiment the same items move all models (0.57), while under distraction each model has its own weak items (0.11).

<table><tr><td>Test</td><td>Shift</td><td>Flip</td></tr><tr><td>Sentiment</td><td>0.57</td><td>0.51</td></tr><tr><td>Authority</td><td>0.41</td><td>0.21</td></tr><tr><td>Verbosity</td><td>0.35</td><td>0.13</td></tr><tr><td>Authority (on better)</td><td>0.35</td><td>0.21</td></tr><tr><td>Bandwagon</td><td>0.29</td><td>0.13</td></tr><tr><td>Diversity</td><td>0.28</td><td>0.13</td></tr><tr><td>Verbosity (on better)</td><td>0.27</td><td>0.18</td></tr><tr><td>Neutral note</td><td>0.27</td><td>0.09</td></tr><tr><td>Position</td><td>0.17</td><td>0.18</td></tr><tr><td>Compassion fade</td><td>0.14</td><td>0.05</td></tr><tr><td>Distraction</td><td>0.11</td><td>0.04</td></tr></table>

Table 20: Agreement between models on which items move: mean Spearman correlation, over all pairs of models, of the per-item mean shift and flip rate, CALM format.

## I Calibration

![](images/89fbfe1919976a71a5fab35d38156fd74262053d5d7b5ec6bec8007479505deb.jpg)  
Figure 6: Reliability diagrams on the original alignment prompts, CALM format. Each point is one bin of the confidence, the probability of the chosen response, with its share of correct verdicts. Points below the diagonal are overconfident. Marker area grows with the number of prompts in the bin.

<table><tr><td></td><td colspan="5">Alignment set (439)</td><td colspan="4">Fact-related set (533)</td></tr><tr><td>Model</td><td>Accuracy</td><td>Confidence</td><td>ECE</td><td>Brier</td><td>Log loss</td><td>Accuracy</td><td>Confidence</td><td>ECE</td><td>Brier</td></tr><tr><td>Jev 1.13</td><td>0.78</td><td>0.83</td><td>0.081</td><td>0.169</td><td>0.63</td><td>1.00</td><td>0.97</td><td>0.030</td><td>0.008</td></tr><tr><td>Kev 4B</td><td>0.65</td><td>0.71</td><td>0.099</td><td>0.243</td><td>0.72</td><td>0.99</td><td>0.79</td><td>0.202</td><td>0.058</td></tr><tr><td>Tev1 4B</td><td>0.66</td><td>0.74</td><td>0.089</td><td>0.212</td><td>0.62</td><td>0.99</td><td>0.89</td><td>0.100</td><td>0.025</td></tr><tr><td>Solar Decide</td><td>0.67</td><td>0.90</td><td>0.250</td><td>0.282</td><td>1.12</td><td>0.96</td><td>0.95</td><td>0.026</td><td>0.034</td></tr><tr><td>D1</td><td>0.75</td><td>0.78</td><td>0.066</td><td>0.179</td><td>0.55</td><td>0.99</td><td>0.91</td><td>0.081</td><td>0.019</td></tr><tr><td>Perplexity Decider</td><td>0.75</td><td>0.82</td><td>0.073</td><td>0.178</td><td>0.56</td><td>0.99</td><td>0.94</td><td>0.053</td><td>0.014</td></tr><tr><td>Clef</td><td>0.75</td><td>0.93</td><td>0.184</td><td>0.210</td><td>0.76</td><td>0.99</td><td>0.97</td><td>0.019</td><td>0.010</td></tr><tr><td>Clef Flash</td><td>0.65</td><td>0.86</td><td>0.206</td><td>0.267</td><td>0.87</td><td>0.92</td><td>0.90</td><td>0.075</td><td>0.048</td></tr></table>

Table 21: Calibration of the probability of the better response on the original prompts, CALM format. Confidence: mean probability of the chosen response. ECE: expected calibration error with 10 equal-width bins of the confidence (Naeini et al., 2015; Guo et al., 2017). Brier score (Brier, 1950) and log loss of $P ( R _ { 1 } )$ . Lower is better for ECE, Brier and log loss. A 50/50 answer has no verdict and is left out of the accuracy, as everywhere in the paper. It enters the other measures with its probability of 0.5 and counts as half right in the ECE. On the fact-related set most models are nearly always right, so their calibration there mostly shows underconfidence.

Figure 6 shows the reliability diagram of every model on the original alignment prompts, and Table 21 the calibration measures on both datasets. Figure 7 compares the two main measures for every model. The expected calibration error asks whether a judge is right as often as it is sure. It groups the verdicts by the probability of the chosen response and averages the gap between that probability and the share of correct verdicts in each group. The Brier score is the mean squared difference between the probability of the better response and the truth, so it is low only for a judge that is both well calibrated and mostly right. The Brier scores are therefore much lower on the fact-related set, where the TDMs are almost always right. Some models, such as Solar and Clef, are poorly calibrated on the alignment set but well calibrated on the fact-related set, while Kev is the reverse. Neither measure goes with less bias. Across the eight TDMs, the rank correlations of the ECE and the Brier score with the mean excess lie between −0.36 and 0.40 and change sign between the two sets.

![](images/a79b6005599d917c871811bda5f7affcb76c95c98c659bd3ab1e6424cbf7156e.jpg)  
Figure 7: Expected calibration error and Brier score of the eight TDMs on the original prompts, alignment set (filled) and fact-related set (open). Lower is better.

## J Time and Cost

<table><tr><td>Judge</td><td>Format</td><td>Minutes</td><td>Parallel requests</td><td>US$</td><td>US$ per 1,000</td><td>Latency (s)</td></tr><tr><td>Jev 1.13</td><td>CALM</td><td>2.2</td><td>32</td><td>0.68</td><td>0.045</td><td>0.25</td></tr><tr><td></td><td>native</td><td>3.5</td><td>32</td><td>0.68</td><td>0.045</td><td>0.34</td></tr><tr><td></td><td>decomposed</td><td>4.6</td><td>32</td><td>1.23</td><td>0.081</td><td>0.49</td></tr><tr><td>Kev 4B</td><td>CALM</td><td>16.5</td><td>32</td><td>0.49</td><td>0.033</td><td>1.90</td></tr><tr><td></td><td>native</td><td>20.3</td><td>32</td><td>0.48</td><td>0.032</td><td>1.86</td></tr><tr><td></td><td>decomposed</td><td>23.3</td><td>32</td><td>0.89</td><td>0.059</td><td>2.76</td></tr><tr><td>Tev1 4B</td><td>CALM</td><td>10.5</td><td>32</td><td>0.55</td><td>0.036</td><td>0.56</td></tr><tr><td></td><td>native</td><td>4.7</td><td>32</td><td>0.53</td><td>0.035</td><td>0.53</td></tr><tr><td></td><td>decomposed</td><td>10.8</td><td>32</td><td>3.12</td><td>0.205</td><td>1.25</td></tr><tr><td>Solar Decide</td><td>CALM</td><td>46.6</td><td>32</td><td>0.84</td><td>0.055</td><td>2.87</td></tr><tr><td></td><td>native</td><td>47.9</td><td>32</td><td>0.82</td><td>0.054</td><td>3.30</td></tr><tr><td></td><td>decomposed</td><td>85.9</td><td>32</td><td>4.85</td><td>0.319</td><td>11.92</td></tr><tr><td>D1</td><td>CALM</td><td>4.3</td><td>32</td><td>0.47</td><td>0.031</td><td>0.38</td></tr><tr><td></td><td>native</td><td>3.4</td><td>32</td><td>0.49</td><td>0.032</td><td>0.36</td></tr><tr><td>Perplexity Decider</td><td>decomposed</td><td>4.8</td><td>32</td><td>2.87</td><td>0.189</td><td>0.50</td></tr><tr><td></td><td>CALM</td><td>12.1</td><td>8</td><td>0.26</td><td>0.017</td><td>0.38</td></tr><tr><td></td><td>native</td><td>12.5</td><td>8</td><td>0.26</td><td>0.017</td><td>0.37</td></tr><tr><td></td><td>decomposed</td><td>19.3</td><td>8</td><td>1.50</td><td>0.099</td><td>0.44</td></tr><tr><td>Clef</td><td>CALM</td><td>4.9</td><td>32</td><td>3.11</td><td>0.205</td><td>0.53</td></tr><tr><td></td><td>native</td><td>4.8</td><td>32</td><td>3.07</td><td>0.202</td><td>0.51</td></tr><tr><td>Clef Flash</td><td>decomposed</td><td>7.6</td><td>32</td><td>6.15</td><td>0.405</td><td>0.77</td></tr><tr><td></td><td>CALM</td><td>9.7</td><td>32</td><td>1.17</td><td>0.077</td><td>0.47</td></tr><tr><td></td><td>native</td><td>10.9</td><td>32</td><td>1.15</td><td>0.076</td><td>0.48</td></tr><tr><td></td><td>decomposed</td><td>14.0</td><td>32</td><td>2.31</td><td>0.152</td><td>0.64</td></tr><tr><td>GPT-3.5 Turbo</td><td>CALM prompt</td><td>一</td><td>32</td><td>10.28</td><td>0.676</td><td>2.69</td></tr><tr><td>GPT-40</td><td>CALM prompt</td><td></td><td>16</td><td>111.11</td><td>7.312</td><td>2.33</td></tr><tr><td>GPT-6.1 Sol</td><td>CALM prompt</td><td></td><td>32</td><td>39.40</td><td>2.593</td><td>3.26</td></tr><tr><td>Claude Sonnet 5.5</td><td>CALM prompt</td><td>1</td><td>32</td><td>110.63</td><td>7.281</td><td>4.79</td></tr><tr><td>Gemini 3.8 Flash</td><td>CALM prompt</td><td>一</td><td>32, then 16</td><td>22.40</td><td>1.479</td><td>3.02</td></tr><tr><td>Qwen3.8-27B</td><td>CALM prompt</td><td>一</td><td>32</td><td>22.85</td><td>1.504</td><td>5.79</td></tr><tr><td>Claude Haiku 5.5</td><td>CALM prompt</td><td>一</td><td>16</td><td>4.73</td><td>0.311</td><td>2.79</td></tr></table>

Table 22: Time and cost of the full design (15,195 requests per model and format). Minutes: wall-clock time of all sending passes. Parallel requests: requests in flight in the main pass. Latency: median seconds per request, the time of the answered attempt without retries, with the stated number of parallel requests. The LLM judges answer CALM’s prompt as text over the same design. Their cost is the cost OpenRouter reported in October 2026 (Claude Sonnet 5.5 at Anthropic’s list price), and their minutes are not comparable, since their runs were interrupted by credit limits.

Table 22 gives the time and cost of every run of the TDMs and of the seven LLM judges. The notebooks 08\_llm\_judges and 09\_llm\_analysis contain every request and reply of the LLM judges and their analysis.

## K Comparison of the Three Request Formats

## K.1 Verdicts and Bias in Each Format

<table><tr><td>Model</td><td>Ver.</td><td>Fal.</td><td>Sen.</td><td>Pos.</td><td>Com.</td><td>Ban.</td><td>Aut.</td><td>Dst.</td><td>Div.</td><td>Aut. (b.)</td><td>Ver. (b.)</td><td>Neu.</td></tr><tr><td>CALM format</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Jev 1.13</td><td>1.00</td><td>0.98</td><td>0.63</td><td>0.90</td><td>0.96</td><td>0.78</td><td>0.87</td><td>0.93</td><td>0.96</td><td>0.70</td><td>0.91</td><td>0.95</td></tr><tr><td>Kev 4B</td><td>1.00</td><td>0.99</td><td>0.61</td><td>0.81</td><td>0.85</td><td>0.81</td><td>0.85</td><td>0.94</td><td>0.91</td><td>0.78</td><td>0.91</td><td>0.95</td></tr><tr><td>Tev1 4B</td><td>0.99</td><td>0.93</td><td>0.75</td><td>0.76</td><td>0.79</td><td>0.96</td><td>0.85</td><td>0.95</td><td>0.94</td><td>0.77</td><td>0.97</td><td>0.95</td></tr><tr><td>Solar Decide</td><td>0.93</td><td>0.92</td><td>0.75</td><td>0.76</td><td>0.83</td><td>0.87</td><td>0.81</td><td>0.88</td><td>0.89</td><td>0.78</td><td>0.95</td><td>0.88</td></tr><tr><td>D1</td><td>0.98</td><td>0.96</td><td>0.65</td><td>0.87</td><td>0.92</td><td>0.93</td><td>0.82</td><td>0.97</td><td>0.95</td><td>0.70</td><td>0.86</td><td>0.95</td></tr><tr><td>Perplexity Decider</td><td>0.99</td><td>0.96</td><td>0.71</td><td>0.92</td><td>0.96</td><td>0.74</td><td>0.84</td><td>0.98</td><td>0.98</td><td>0.62</td><td>0.98</td><td>0.98</td></tr><tr><td>Clef</td><td>0.99</td><td>0.97</td><td>0.71</td><td>0.86</td><td>0.95</td><td>0.73</td><td>0.83</td><td>0.99</td><td>0.98</td><td>0.66</td><td>0.94</td><td>0.98</td></tr><tr><td>Clef Flash</td><td>0.90</td><td>0.88</td><td>0.73</td><td>0.81</td><td>0.93</td><td>0.88</td><td>0.88</td><td>0.96</td><td>0.95</td><td>0.92</td><td>0.94</td><td>0.95</td></tr><tr><td>native format</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Jev 1.13</td><td>1.00</td><td>0.98</td><td>0.63</td><td>0.90</td><td>0.95</td><td>0.76</td><td>0.86</td><td>0.95</td><td>0.96</td><td>0.73</td><td>0.90</td><td>0.96</td></tr><tr><td>Kev 4B</td><td>0.96</td><td>0.97</td><td>0.65</td><td>0.76</td><td>0.83</td><td>0.80</td><td>0.89</td><td>0.87</td><td>0.88</td><td>0.84</td><td>0.95</td><td>0.89</td></tr><tr><td>Tev1 4B</td><td>0.98</td><td>0.92</td><td>0.70</td><td>0.71</td><td>0.82</td><td>0.91</td><td>0.87</td><td>0.91</td><td>0.90</td><td>0.79</td><td>0.94</td><td>0.91</td></tr><tr><td>Solar Decide</td><td>0.92</td><td>0.93</td><td>0.79</td><td>0.75</td><td>0.72</td><td>0.85</td><td>0.85</td><td>0.86</td><td>0.86</td><td>0.81</td><td>0.96</td><td>0.86</td></tr><tr><td>D1</td><td>1.00</td><td>0.99</td><td>0.62</td><td>0.89</td><td>0.94</td><td>0.92</td><td>0.88</td><td>0.97</td><td>0.96</td><td>0.66</td><td>0.87</td><td>0.95</td></tr><tr><td>Perplexity Decider</td><td>0.99</td><td>0.96</td><td>0.70</td><td>0.93</td><td>0.97</td><td>0.80</td><td>0.84</td><td>0.98</td><td>0.97</td><td>0.56</td><td>0.97</td><td>0.97</td></tr><tr><td>Clef</td><td>0.99</td><td>0.96</td><td>0.76</td><td>0.82</td><td>0.86</td><td>0.93</td><td>0.87</td><td>0.96</td><td>0.94</td><td>0.77</td><td>0.97</td><td>0.95</td></tr><tr><td>Clef Flash</td><td>0.81</td><td>0.88</td><td>0.74</td><td>0.73</td><td>0.84</td><td>0.85</td><td>0.92</td><td>0.93</td><td>0.92</td><td>0.90</td><td>0.94</td><td>0.92</td></tr><tr><td>decomposed format</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Jev 1.13</td><td>1.00</td><td>0.98</td><td>0.63</td><td>0.91</td><td>0.95</td><td>0.75</td><td>0.88</td><td>0.95</td><td>0.96</td><td>0.73</td><td>0.90</td><td>0.95</td></tr><tr><td>Kev 4B</td><td>0.95</td><td>0.95</td><td>0.65</td><td>0.74</td><td>0.82</td><td>0.79</td><td>0.87</td><td>0.86</td><td>0.87</td><td>0.84</td><td>0.92</td><td>0.88</td></tr><tr><td>Tev1 4B</td><td>0.98</td><td>0.92</td><td>0.69</td><td>0.69</td><td>0.80</td><td>0.88</td><td>0.84</td><td>0.89</td><td>0.89</td><td>0.76</td><td>0.93</td><td>0.91</td></tr><tr><td>Solar Decide</td><td>0.94</td><td>0.93</td><td>0.79</td><td>0.76</td><td>0.76</td><td>0.88</td><td>0.88</td><td>0.92</td><td>0.91</td><td>0.84</td><td>0.96</td><td>0.92</td></tr><tr><td>D1</td><td>1.00</td><td>0.99</td><td>0.62</td><td>0.86</td><td>0.91</td><td>0.90</td><td>0.87</td><td>0.96</td><td>0.95</td><td>0.67</td><td>0.86</td><td>0.95</td></tr><tr><td>Perplexity Decider</td><td>0.99</td><td>0.95</td><td>0.70</td><td>0.91</td><td>0.97</td><td>0.79</td><td>0.85</td><td>0.99</td><td>0.98</td><td>0.56</td><td>0.97</td><td>0.97</td></tr><tr><td>Clef</td><td>0.98</td><td>0.95</td><td>0.83</td><td>0.85</td><td>0.90</td><td>0.96</td><td>0.92</td><td>0.96</td><td>0.97</td><td>0.81</td><td>0.99</td><td>0.96</td></tr><tr><td>Clef Flash</td><td>0.86</td><td>0.92</td><td>0.76</td><td>0.77</td><td>0.87</td><td>0.87</td><td>0.93</td><td>0.93</td><td>0.92</td><td>0.91</td><td>0.96</td><td>0.93</td></tr></table>

Table 23: Robustness rate per model, format and test (fallacy oversight: share of correct reactions). Bold: a bias at δ = 0.05, for fallacy oversight a share of overlooked flaws beyond chance above δ. Aut. (b.) and Ver. (b.): citation and padding on the better response. Neu.: neutral identity note. Abbreviations as in Table 9.

<table><tr><td>Test</td><td>Helpfulness</td><td>Relevance</td><td>Accuracy</td><td>Depth</td><td>Creativity</td><td>Level of detail</td></tr><tr><td>Sentiment</td><td>-0.259</td><td>-0.259</td><td>-0.246</td><td>-0.229</td><td>-0.144</td><td>-0.219</td></tr><tr><td>Authority (on better)</td><td>-0.137</td><td>-0.135</td><td>-0.143</td><td>-0.110</td><td>-0.122</td><td>-0.103</td></tr><tr><td>Authority</td><td>-0.093</td><td>-0.097</td><td>-0.098</td><td>-0.071</td><td>-0.084</td><td>-0.066</td></tr><tr><td>Diversity</td><td>-0.031</td><td>-0.030</td><td>-0.030</td><td>-0.024</td><td>-0.028</td><td>-0.022</td></tr><tr><td>Neutral note</td><td>-0.025</td><td>-0.025</td><td>-0.024</td><td>-0.018</td><td>-0.022</td><td>-0.018</td></tr><tr><td>Verbosity</td><td>-0.013</td><td>-0.022</td><td>-0.016</td><td>+0.021</td><td>+0.008</td><td>+0.033</td></tr><tr><td>Distraction</td><td>-0.025</td><td>-0.023</td><td>-0.022</td><td>-0.020</td><td>-0.024</td><td>-0.020</td></tr><tr><td>Compassion fade</td><td>+0.047</td><td>+0.044</td><td>+0.045</td><td>+0.044</td><td>+0.044</td><td>+0.043</td></tr><tr><td>Position</td><td>+0.051</td><td>+0.054</td><td>+0.051</td><td>+0.052</td><td>+0.060</td><td>+0.046</td></tr><tr><td>Bandwagon</td><td>+0.104</td><td>+0.098</td><td>+0.090</td><td>+0.089</td><td>+0.098</td><td>+0.089</td></tr></table>

Table 24: Decomposed format: mean shift of the probability of the targeted response per criterion, all eight models. The criteria react alike, with two exceptions: under sentiment, creativity falls less than the others, and under verbosity, depth and detail rise while accuracy and relevance fall.

Table 3 gives, for each model, the share of verdicts on the original prompts that change when the same text moves into the fields of the native format. Moving the text changes 0.8% (Jev) to 10.4% (Solar) of the verdicts, 4.5% for the median model. Under the perturbations, the CALM and native formats differ significantly in one (Jev) to eight (Clef) of the twelve tests, and the native and decomposed formats in zero to three. Tables 23 and 14 give the robustness rate and the shift of every model in every format and test, the latter in Appendix F.3. A logistic mixed model over all three formats,

$$
\begin{array} { r l } & { \log \operatorname { i t } P ( f _ { i j f } = 1 ) = \beta _ { 0 } + \beta _ { 1 } m _ { i j f } + \alpha _ { t ( i ) } + \gamma _ { j } } \\ & { \qquad + \phi _ { f } + u _ { k ( i ) } + w _ { k ( i ) j } , } \end{array}
$$

with the margin m of the original verdict in format $f ,$ fixed effects of the test, the model and the format, and random effects of the item and of the item and model, shows that asking one question per criterion lowers the odds of a flip at the same confidence by 21% (odds ratio 0.79, CI 0.76 to 0.82) relative to the CALM format, while the native format leaves them almost unchanged (1.07, CI 1.03 to 1.10). The decomposed format, however, also makes the TDMs less sure of their verdicts, with a mean confidence of 0.84 on the original prompts against 0.87 in the CALM format, and so creates more close calls. Overall, about as many verdicts flip in the decomposed format as in the CALM format (11.3 against 10.9% of the perturbed prompts, without fallacy oversight).

## K.2 The Six Criteria

Table 24 gives, for the decomposed format, the mean shift of each criterion’s probability for the targeted response. In the decomposed format, the six criteria react alike, with few exceptions. Under sentiment, each of them falls by 0.22 to 0.26, except creativity by 0.14, and under verbosity, depth and detail rise slightly while accuracy and relevance fall. The decomposed format costs two to six times as much, since it asks six questions (Table 22).

## K.3 What the Formats Show

Following the vendor’s advice does not help on balance. Splitting the judgment into criteria lowers the odds of a flip at the same confidence, perhaps because six answers average out part of each answer’s reaction, but it also makes the TDMs less sure, and the cue acts on every criterion alike, so splitting does not isolate it. A judge such as Clef Flash, whose verdict changes on 9% of the prompts when identical text is moved into fields although it repeats every verdict, is sensitive to presentation as such. Other studies report the same sensitivity. Two interfaces of Jev that ask for the same event disagree on a third of decisions (Chen and Li, 2026), and open TDMs follow how their options are written more than what they define (Azizi et al., 2026). Attaching the names yes and no to the opposite definitions changes half or more of Jev’s answers in decision gates (Rafe and Das, 2026), and swapping the names flips about a third of Jev’s yes and no answers (Sun et al., 2026). Jev, whose vendor wrote the guidance, is the least sensitive.

## L Differences between TDMs

No model is the most robust in every test (Figure 1, Table 9). Jev resists verbosity, diversity and compassion fade but is biased under bandwagon (RR 0.78). Under bandwagon, Clef and Perplexity are swayed most (0.73 and 0.74) and Tev1 and D1 hardly (0.96 and 0.93), but these two are biased under compassion fade. Perplexity and Jev have the smallest position bias (excess 8 each). In the mixed model of Appendix H.2, the odds of a flip are 3.1 times as high for Solar as for Jev and 2.2 times for Clef, about half for Kev and Tev1, and 0.62 times for D1. The models also fill the same interface differently. On a 50/50 answer, three models always return the first option (Appendix D.2), and their calibration differs widely (Appendix I).

Which model to use depends on the bias that matters for the task, as CALM found for LLM judges. Where the order of answers cannot be balanced, a model with a small position bias such as Jev is preferable, in line with reports that Jev is far less sensitive to option order than the open model Laya (Li, 2026). Where claims of popularity may appear in the inputs, Tev1 or D1 are preferable. We cannot attribute the differences to size or training, since only some sizes are known and no training data are public. Even models built on the same LLM family differ clearly, such as Tev1 and Perplexity under bandwagon, although they also differ in size (4B against 27B). How the step from a base LLM to a TDM shapes these biases is a question for further study (Section 4.2).

The different handling of ties and the different calibration show that one interface hides quite different systems. A confidence of 0.8 does not mean the same for two models, and a threshold tuned on one cannot be carried over to another.

## M Asking in Both Orders

On the items of the position test, we average the probability of $R _ { 1 }$ over the two orders, $\begin{array} { r l } { \bar { p } _ { R _ { 1 } } } & { { } = } \end{array}$ $( p _ { R _ { 1 } } + p _ { R _ { 1 } } ^ { \mathrm { s w a p } } ) / 2$ , which removes the position bias by construction, since each response sits once in each place. Table 25 gives the accuracy of each model in one order and with both orders averaged. Averaging changed the accuracy by −0.2 (Kev) to +3.8 points (Tev1), and only Tev1’s gain has a 95% interval that excludes zero. It doubles the cost of a model.

<table><tr><td>Model</td><td>One</td><td>Both</td><td>Change</td><td>Agree</td></tr><tr><td>Jev 1.13</td><td>0.777</td><td>0.789</td><td>+1.2[−0.6, +3.2]</td><td>0.90</td></tr><tr><td>Kev 4B</td><td>0.647</td><td>0.645</td><td> $- 0 . 2 \left[ - 3 . 0 , + 2 . 5 \right]$ </td><td>0.81</td></tr><tr><td>Tev1 4B</td><td>0.656</td><td>0.695</td><td> $+ 3 . 8 \ [ + 0 . 6 , + 7 . 2 ]$ </td><td>0.75</td></tr><tr><td>Solar Decide</td><td>0.667</td><td>0.690</td><td> $+ 2 . 4 \ [ - 0 . 6 , + 5 . 4 ]$ </td><td>0.75</td></tr><tr><td>D1</td><td>0.752</td><td>0.773</td><td> $+ 2 . 2 \ [ - 0 . 3 , + 4 . 7 ]$ </td><td>0.87</td></tr><tr><td>Perplexity Decider 0.749</td><td></td><td>0.752</td><td> $+ 0 . 2 \left[ - 1 . 6 , + 2 . 1 \right]$ </td><td>0.92</td></tr><tr><td>Clef</td><td>0.752</td><td>0.770</td><td> $+ 1 . 8 \ : [ - 0 . 7 , + 4 . 6 ]$ </td><td>0.86</td></tr><tr><td>Clef Flash</td><td>0.649</td><td>0.672</td><td> $+ 2 . 3 \ : [ - 0 . 5 , + 5 . 0 ]$ </td><td>0.81</td></tr></table>

Table 25: Accuracy on the 439 items of the position test when a model judges the pair in one order (One), and when the probabilities of both orders are averaged (Both), which removes position bias by construction. Change: Both minus One, in percentage points, with a 95% bootstrap interval over items. Agree: share of items with the same verdict in both orders. A 50/50 answer has no verdict and is left out.

## N A Guide to Replacing an LLM Judge with a TDM

This guide summarises what our results mean for anyone who uses an LLM judge for pairwise comparisons, as in CALM, and considers a TDM instead. It rests on the eight TDMs and seven LLM judges we tested in October 2026 and on CALM’s fairly simple tests, so it is a starting point rather than a guarantee.

A TDM is a good choice where cost, speed and consistency matter. At the median, it costs about 37 times less than a 2026 LLM judge and answers about six times as fast, and six of the eight TDMs repeat more than 99.7% of their verdicts on the alignment set, against 91 to 98% for the 2026 LLM judges (Section 4.4). A TDM also returns a probability with its verdict, which our LLM judges, answering in text, do not. On the fact-related prompts it is about as accurate as a 2026 LLM judge (92 to 100% against 94 to 98%), and on the alignment prompts the best TDMs come close (75 to 78% against 72 to 80%). Under sentiment and authority (on better), the two kinds of judge are about equally biased, and under verbosity (on better) the 2026 LLM judges are more biased (Section 4.2), so for these cues a switch adds no bias.

An LLM judge remains the safer choice where the inputs carry cues that the TDMs follow and the 2026 LLM judges do not. A claimed majority biases five of the eight TDMs and the name of a well-known model three, but neither biases any 2026 LLM judge. Where inputs contain such cues, for example votes, ratings or the names of the systems that wrote the responses, and the cues cannot be removed, a switch adds bias. An LLM judge is also needed where the verdict must be explained, since a TDM gives no explanation. Finally, our tests are simple. For judgments that require checking a derivation, scoring a single answer or reading long inputs, we have no evidence that a TDM can replace an LLM judge.

For a switch, we recommend the following.

1. Ask in both orders. Every TDM we tested has a position bias, which no 2026 LLM judge has. Judging each pair in both orders and averaging the two probabilities removes it. This doubles the cost and changed the accuracy of our TDMs by −0.2 to +3.8 points (Appendix M).

2. Remove cues the task does not need. Strip claimed majorities, popularity counts and the names of models from the input before judging.

3. Do not expect a TDM to handle tone and citations better. Both kinds of judge punish an emotional tone and a fake citation even on the better response. Under authority (on better), the accuracy of the median TDM falls from 0.66 to 0.41, and under a tone on the better response from 0.99 to 0.42 (Appendix F.4).

4. Treat close calls as uncertain. Outside sentiment and authority (on better), three in five flipped verdicts come from verdicts given with a probability below 0.7, which are about a quarter of all verdicts (Figure 2c, Appendix H). Such verdicts can be passed on to a second judge, as in the cascade of Li et al. (2026). A high probability does not protect against a tone or a citation on the better response, which flip about half and 14% of the verdicts given with more than 0.9.

5. Use the probabilities with care. Calibration differs widely between the TDMs and does not go with less bias (Appendix I). A cue can also move the probability by more than 0.10 without flipping the verdict, for the median TDM on 19 to 24% of the prompts under authority, position, bandwagon and sentiment. Where the probability serves as a score, for example as a reward in training, check its calibration on the task’s own data and test the cues that matter there.

6. Choose the model and the request format for the task. No TDM is the least biased in every test, so pick the one whose weak spots matter least for the inputs (Appendix L). The request format changes the bias of most models little, but moving the same text into another format changes 4.5 to 5.5% of the verdicts of the median model, so keep one format fixed (Appendix K).

7. Read the interface. Check what the model returns on a 50/50 answer and how it defines its confidence. Three of our models always return the first option on a 50/50 answer (Appendix L).

8. Test again after an update. TDMs are new and change quickly. With our released code, a full bias test of one TDM in the CALM or native format costs US\$0.26 to 3.11 (Appendix J).

Whoever tests a judge, TDM or LLM, should look beyond the robustness rate. It mixes bias with chance and favours consistent judges (Section 4.2), so the consistency rate and the excess should be reported with it, and for a TDM the shift of its probability.