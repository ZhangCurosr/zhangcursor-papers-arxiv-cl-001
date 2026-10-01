# Cheap to Draw, Expensive to Trust: Certifying Test-Time Scaling Curves

Sohail (Neel) Sarkar

Shakuntala Baichoo

PMCC AI Lab, Peter Munk Cardiac Centre

University Health Network, Toronto, Ontario, Canada

sohail.sarkar@uhn.ca shakuntala.baichoo@uhn.ca

## Abstract

Sampling several answers and keeping the one a verifier scores highest is one of the simplest ways to buy accuracy at test time. Its efect is reported as a scaling curve: accuracy against the number k of sampled answers. The curve is cheap to draw and expensive to trust. A budget read of it is chosen after looking at every point, so only a band that covers all budgets at once protects the choice, and on a 100-question benchmark a fixed exact-binomial design needs 192,000 generated answers to certify 64 budgets to within ±1/32 at 95%. Most of that cost pays for the wrong uncertainty. A benchmark is a fixed list of questions; at budget 64, about three quarters of the variance of a selected answer’s correctness lies between questions, and an audit that revisits every question need not pay for it. We derive the minimax cost of certifying the whole curve, up to logarithmic factors. It has three parts: calibrating the tail of the score distribution, telling the questions apart, and within-question noise summed along the curve. At a single benchmark the last part sharpens to the variance of one answer’s influence under the best allocation of answers to questions, which every valid audit pays and an audit that learns the allocation attains, up to a logarithm, as the precision grows. A paired audit built on an exponential inequality for two independent draws at the same question needs no pilot. On 185 held-out score pools it uses 0.74 times the answers of the cheapest competing certified audit at 64 budgets and 0.53 times at 1,024, and on a newly generated MMLU-Pro study it certified the curve with 79,133 answers, within 0.6% of what a cost law fitted beforehand predicted from the study’s within-question variance. The same paths certify pass@k and majority voting, and the bands extend to populations of questions and to answers that depend on earlier ones.

Keywords: test-time scaling; best-of-k selection; simultaneous confidence bands; confidence sequences;   
minimax lower bounds; stratified sampling; multilevel Monte Carlo; language model evaluation.

## 1 Introduction

Sampling several answers from a language model and returning the one a verifier scores highest is among the simplest ways to buy accuracy with computation at test time (Chen et al., 2021; Nakano et al., 2021; Brown et al., 2024; Saad-Falcon et al., 2025). Its efect is reported as a scaling curve: the accuracy of the selected answer at each budget k = 1, . . . , K on a fixed benchmark (Figure 1a). Readers use the curve to decide how many samples are worth paying for, and they usually decide on a plateau. On the 185 held-out score pools studied below, the median curve comes within 1/32 of its best value by k = 10. The diferences that matter are a few points.

A budget read of a curve is chosen after looking at every point. Only a band that holds at all budgets at once protects that choice. Pointwise intervals do not, and neither does the bootstrap: on the same pools, a withinquestion bootstrap band at nominal level 95% missed the exact curve in 58 of 925 runs. Valid bands exist, and they are expensive. For 100 MMLU-Pro questions, one path of 64 answers per question, 6,400 answers in all, already gives an unbiased estimate of every point. Certifying the curve to within ±1/32 with a fixed exact-binomial design takes 192,000 (Figure 1c).

Most of that cost pays for the wrong uncertainty. The benchmark is a fixed list, and the accuracy it reports is an average over that list. An audit that samples questions at random pays for which questions it happened to draw. At budget 64, about three quarters of the variance of the selected answer’s correctness is of that kind (Figure 1b). Visiting every question in every round removes it, as stratified sampling removes between-stratum variance (Cochran, 1977). Balance by itself buys nothing, though. An exact-binomial audit costs the same on balanced rounds as on random ones (Section 8.3), because its interval cannot see a smaller variance. The saving appears only when the confidence sequence can see it, and its size depends on how the remaining variance accumulates along a whole curve. Working this out is the subject of the paper.

What a certified curve costs. We model an audit that may choose questions, grow paths of answers, query correctness and stop, all adaptively, and must return a band of half-width ε that covers every budget with probability $1 - \delta ,$ whatever the answers look like. Three resources are counted: generated answers, correctness queries and question visits. The results are these.

• The price on a fixed list (Theorem 4). On a benchmark of M questions, certifying all budgets up to K takes $K / \varepsilon + \operatorname* { m i n } ( M , \varepsilon ^ { - 2 } ) + s / \varepsilon ^ { 2 }$ generated answers up to logarithmic factors, where s bounds the within-question variance summed along the curve. The three terms are three obstructions: a rare high-scoring answer that decides the largest budgets, the identity of the questions, and within-question noise. Each has a matching lower bound, and the first is paid at every law. Correctness queries obey a law of the same shape, and labels for the whole curve cost what labels for its first point cost, up to logarithms. A coupling of nested winners on one path (Lemma 3) is what lets one path serve every budget. For fresh questions from a population the price is $K / \varepsilon ^ { 2 }$ instead, while simultaneity over budgets adds only a log log K term to the number of questions (Proposition 2).

• The price at one benchmark (Theorem 6). At a fixed benchmark every valid audit pays $K / \varepsilon + \Gamma / \varepsilon ^ { 2 }$ , where Γ is the variance of one answer’s influence under the best allocation of answers to questions, and an audit that learns that allocation attains $\Gamma / \varepsilon ^ { 2 }$ up to a logarithmic factor as $\varepsilon \to 0$ . At a fixed precision no audit matches every benchmark’s own optimum (Proposition 5).

• A paired audit (Theorem 9). Two fresh paths per question and an exponential inequality for pairs of Bernoulli draws (Lemma 7) give anytime-valid simul taneous bands whose penalty measures the withinquestion variance without a pilot. An exact-binomial final look with a corrected level (Lemma 8) ends the audit at a fixed round. Run beside the multilevel audit behind Theorem 4, it gives one audit with the minimax rate at no more than 32/31 of its own cost, plus K answers (Corollary 11).

• Other curves and settings (Section 7). The same machinery certifies pass@k and majority voting, populations of questions, and trajectories whose answers depend on earlier ones. When score percentiles are unknown, the minimax mean squared error of a single question’s curve is asymptotically $4 / e$ times its value with known percentiles whenever generated answers, not labels, are the bottleneck (Theorem 13).

• Evidence (Section 8). On 185 held-out pools the paired audit uses 0.74, 0.66 and 0.53 times the generated answers of the cheapest competing certified audit at $K = 6 4$ , 256 and 1024, and it is cheaper on all eight generator–benchmark answer sets. It missed none of its 3,255 runs against the exact curves. Its cost follows the three terms of the theory $( R ^ { 2 } = 0 . 9 9 5 )$ , and a cost law fitted before a new study predicted that study’s bill within 0.6%.

Outline. Section 2 sets up the problem. Sections 3– 5 derive the price of a certified curve, first for fresh questions, then on a fixed list, then at a single benchmark. Section 6 gives the paired audit and Section 7 extends it. Section 8 tests it on stored score pools and on a newly generated study. Long proofs are in Appendices A–F and experimental details in Appendix G.

## 2 The problem

Questions, answers and scores. A benchmark is a known list $\mathcal { X } = \{ 1 , \ldots , M \}$ of questions with equal weights. At question x a generated answer carries a numerical verifier score S and a correctness bit Y. The pairs (S, Y ) of diferent answers are independent and identically distributed given x, with an unknown law $P _ { x }$ The score can be a reward model, a unit-test count or the model’s own mean token log-probability; nothing below depends on where it comes from. A path is a sequence of answers at one question, and $W _ { k }$ is the correctness of the first answer attaining the highest score among its first k. The target is the best-of-k curve

$$
\begin{array} { l } { \theta _ { k } = \displaystyle \frac { 1 } { M } \sum _ { x = 1 } ^ { M } p _ { k } ( x ) , \qquad p _ { k } ( x ) = \mathbb { E } ( W _ { k } \mid x ) , } \\ { 1 \le k \le K . } \end{array}\tag{1}
$$

Two other curves live on the same list. Pass@k counts a question as solved when some answer among the first k is correct (Chen et al., 2021). Majority voting counts it as solved when the most frequent valid answer among the first k is correct, with plurality ties broken uniformly at random and no valid answer counted as incorrect (Wang et al., 2023).

Audits and what they pay. An audit may choose questions, stop paths, query correctness and stop, all adaptively. It must return intervals $I _ { k }$ of width at most 2ε with

$$
\mathbb { P } \{ \theta _ { k } \in I _ { k } \mathrm { ~ f o r ~ a l l ~ } k \le K \} \ge 1 - \delta
$$

for every answer law. It pays in three currencies: generated answers N, correctness queries m (one per queried

![](images/2c898cd5e41e3cc1349c617d3849bb440ac6a2f8ae8e4a57f911a4bda09b5e4d.jpg)

![](images/c89fe2cfb34968ab1fea79195440c227a5f21faeeb674964ce2356ed4ae23670.jpg)

$$
K = 6 4
$$

![](images/0f57df843849a2348285b61a41b434768c80476c4b872098894b7660d16cc553.jpg)  
Figure 1. Why a certified curve can be cheap. (a) Accuracy curves $p _ { k } ( x )$ of single questions spread from 0 to 1, while the benchmark average rises smoothly (MATH500, answers from Llama-3.1-8B-Instruct, one reward model; 60 of the 250 questions are drawn). (b) Median share of the variance of the selected answer’s correctness that lies between questions, with interquartile band, over the 185 held-out pools. (c) Generated answers needed for 100 MMLU-Pro questions and 64 budgets: an unbiased estimate of every point, a band from a fixed exact-binomial design, and a band from the paired audit of Section 6, both at half-width $1 / 3 2$ with 95% simultaneous coverage.

answer occurrence), and question visits n (one per path started). Answers usually dominate the bill, since each needs a forward pass of the model, while a query may be cheap when the grader is a string comparison. We report all three. Unless stated otherwise $\delta = 0 . 0 5$ and $\varepsilon = 1 / 3 2$

What a band buys. Simultaneity is what makes a budget choice safe. The following facts use nothing but coverage, so they hold whatever rule picked the budget after seeing the band.

Proposition 1 (Decisions from a band). Suppose $[ l _ { k } , u _ { k } ]$ covers $\theta _ { k }$ for every $k \leq K$ , and $u _ { k } - l _ { k } \leq 2 \varepsilon$

(i) The set $\{ k : u _ { k } \geq \operatorname* { m a x } _ { j } l _ { j } \}$ contains every budget that maximizes $\theta _ { k }$

(ii) Every k with $l _ { k } \ge \operatorname* { m a x } _ { j } u _ { j } - \tau$ has $\theta _ { k } \geq \operatorname* { m a x } _ { j } \theta _ { j } - \tau ;$ the budget that maximizes $l _ { k }$ qualifies with $\tau = 2 \varepsilon$

(iii) For known costs $c ( k )$ and a price $\lambda \geq 0$ , a maximizer $\widehat { k } \ o f \ ( l _ { k } + u _ { k } ) / 2 - \lambda c ( k )$ has $\operatorname* { m a x } _ { k } \{ \theta _ { k } -$ $\lambda c ( k ) \} - \{ \theta _ { \widehat { k } } - \lambda c ( \widehat { k } ) \} \leq 2 \varepsilon$

(iv) If a second system has a band $[ l _ { k } ^ { \prime } , u _ { k } ^ { \prime } ]$ covering its curve $\theta _ { k } ^ { \prime }$ , then $l _ { k } > u _ { k } ^ { \prime }$ implies $\theta _ { k } > \theta _ { k } ^ { \prime } .$ , and both bands cover at once with probability at least $1 - \delta - \delta ^ { \prime }$

Proof. If $k ^ { * }$ maximizes $\theta ,$ then $u _ { k ^ { * } } \geq \theta _ { k ^ { * } } \geq \theta _ { j } \geq l _ { j }$ for every $j ,$ , which is (i). For (ii), $\theta _ { k } \ge l _ { k } \ge$ max<sub>j</sub> $u _ { j } - \tau \geq$ max<sub>j</sub> $\theta _ { j } - \tau ;$ and max<sub>k</sub> $l _ { k } \ge \operatorname* { m a x } _ { j } ( u _ { j } - 2 \varepsilon )$ . For (iii), write $c _ { k } = ( l _ { k } + u _ { k } ) / 2 .$ , so $| c _ { k } - \theta _ { k } | \leq \varepsilon ;$ for any $\mathfrak { a } , \theta _ { k } - \lambda c ( k ) \le$ $c _ { k } + \varepsilon - \lambda c ( k ) \leq c _ { \widehat { k } } + \varepsilon - \lambda c ( \widehat { k } ) \leq \theta _ { \widehat { k } } + 2 \varepsilon - \lambda c ( \widehat { k } )$ . Part (iv) is a union bound. □

Pointwise intervals give none of this: a budget that looks best among K pointwise intervals is best by selec tion as much as by accuracy.

Two kinds of variance. Write

$$
\begin{array} { l } { \displaystyle \bar { v } _ { k } = \frac { 1 } { M } \sum _ { x } p _ { k } ( x ) \{ 1 - p _ { k } ( x ) \} , } \\ { \displaystyle \Sigma _ { \mathrm { w } } = \sum _ { j = 1 } ^ { K } \underset { j \le k \le K } { \operatorname* { m a x } } \bar { v } _ { k } . } \end{array}\tag{2}
$$

For a question drawn uniformly from the list, the law of total variance splits the variance of $W _ { k }$ as

$$
\theta _ { k } ( 1 - \theta _ { k } ) = \bar { v } _ { k } + \mathrm { V a r } _ { x } p _ { k } ( x ) ,\tag{3}
$$

a within-question part and a between-question part. The second part is what random questions cost and balanced visits avoid. The tail sum $\Sigma _ { \mathrm { w } }$ charges the answer at position $j$ of a path for the noisiest budget from $j$ on that still needs it. If every budget had within-question variance v, then $\Sigma _ { \mathrm { w } } = K v$ . When the selected answer at a question becomes reliably right or reliably wrong as k grows, $\Sigma _ { \mathrm { w } }$ is much smaller: on the MMLU-Pro study of Section 8.6, $\Sigma _ { \mathrm { w } } = 2 . 5 7$ against a worst case of $K / 4 = 1 6$ at $K = 6 4$

## 3 Fresh questions: the population price

Start with the setting that most evaluation work assumes. Each path begins at a fresh question drawn from a population, and the target is the population average $\theta _ { k } = \mathbb { E } _ { X } p _ { k } ( X )$ . Every draw then carries the betweenquestion variance, and the price is the following (proof in Appendix B).

Proposition 2 (Fresh questions). Let $0 < \varepsilon \le 1 / 3 2$ $0 < \delta \leq 1 / 1 6$ and $J = 1 { + } \lfloor \log _ { 1 6 } K \rfloor$ . Under deterministic resource caps, with an abort counted as a failure, every valid audit needs

$$
\begin{array} { r l } & { n \gtrsim \frac { \log ( J / \delta ) } { \varepsilon ^ { 2 } } , \qquad m \gtrsim \frac { J \log ( J / \delta ) } { \varepsilon ^ { 2 } } , } \\ & { N \gtrsim \frac { K \log ( 1 / \delta ) } { \varepsilon ^ { 2 } } , } \end{array}
$$

and one predetermined record audit attains all three orders together.

The three currencies behave very diferently. Generation is where the curve is expensive: every question costs about K answers, because the largest budget needs a full path. Labels are cheap. A path needs correctness only at its score records, the answers that beat every earlier score, and when scores do not tie, the kth answer is a record with probability $1 / k$ whatever the question $\mathrm { ( R e n y i }$ , 1962); so a path of length K needs about $\textstyle H _ { K } = \sum _ { j \leq K } 1 / j$ ≈ log K labels, and the lower bound shows that a factor of that size is unavoidable. Questions are nearly free: simultaneity over all K budgets adds only log J ≈ log log K to the log(1/δ) that a single budget pays.

The reason is a geometric fact about winners. The winners at budgets $k \leq \ell$ coincide unless the best of the first ℓ answers comes after position k, which happens with probability at most $1 - k / \ell$ . So

$$
{ \mathbb E } \{ ( W _ { k } - W _ { \ell } ) ^ { 2 } \mid x \} \leq 1 - \frac { k } { \ell } \leq \log \frac { \ell } { k } ,
$$

and budgets within a factor $e ^ { r ^ { 2 } }$ of each other move together to within r in root mean square. At resolution r the curve has only about log $K / r ^ { 2 }$ distinct budgets, and a chaining bound turns that count into a log log K price for the supremum over all of them. A fixed list changes everything else.

## 4 A fixed list of questions

Estimating every budget separately would cost K times one budget. It does not, because two winners on the same path rarely disagree.

Lemma 3 (Nested winners). For $a \leq k \leq 2 a$ , under first-maximum selection and with ties allowed,

$$
{ \mathbb E } \{ ( W _ { k } - W _ { a } ) ^ { 2 } \mid x \} \leq p _ { a } ( x ) \{ 1 - p _ { a } ( x ) \} .
$$

Proof. Fix x and suppose first that the highest score on any path is attained once, almost surely. Take a path of length k with $a \leq k \leq 2 a$ . Call its first a positions block A and its last a positions block B; they share the $2 a - k$ positions in the middle (Figure 2). Let U, V and W be the correctness of the winners of A, of B and of the whole path. The winner of the whole path is the winner of A or the winner of B, so $W \in \{ U , V \}$ . If $U = V$ , both agree with $W \colon \operatorname { i f } U \neq V$ , exactly one of them does. Hence $\mathbb { P } ( W \neq U ) + \mathbb { P } ( W \neq V ) = \mathbb { P } ( U \neq V )$ . Exchanging the unshared parts of A and B does not change the law of the path, leaves the winner of the whole path in place, and swaps U with V. The two probabilities on the left are therefore equal, and $\mathbb { P } ( W \neq U ) = { \frac { 1 } { 2 } } \mathbb { P } ( U \neq V )$

Given the shared positions, U and V are independent with a common conditional mean g. Hence $\mathbb { E } ( U V ) = \mathbb { E } g ^ { 2 }$ $\mathbb { E } U = \mathbb { E } V = \mathbb { E } g = p _ { a } ( x )$ , and

$$
\begin{array} { r l } & { \mathbb { P } ( W \neq U ) = \frac { 1 } { 2 } \big ( \mathbb { E } U + \mathbb { E } V - 2 \mathbb { E } U V \big ) } \\ & { \quad \quad = p _ { a } ( x ) \{ 1 - p _ { a } ( x ) \} - \mathrm { V a r } ( g ) } \\ & { \quad \quad \le p _ { a } ( x ) \{ 1 - p _ { a } ( x ) \} . } \end{array}
$$

Since $W = W _ { k }$ and $U = W _ { a }$ , this is the claim. When $k = 2 a$ the blocks are disjoint, g is constant and equality holds.

For ties, attach to every answer an independent uniform key and rank by score, then by key. Given the scores, the correctness bits of diferent answers are independent, and answers with equal scores share one conditional correctness law, so the selected correctness has the same law under this rule as under first-maximum selection; in particular $p _ { a } ( x )$ is unchanged. If the path of length k reaches a score strictly above the maximum of its first a positions, the two winners carry diferent scores under either rule, and the disagreement event has the same probability under both. If the two maxima are equal, first-maximum selection keeps the same answer, so $W _ { k } = W _ { a }$ , while the randomized rule may move to another tied answer. The disagreement probability under first-maximum selection is therefore at most the one under the randomized rule, which the argument above bounds. □

Budgets a factor two apart thus difer by at most the within-question noise of the smaller one, and none of the between-question variance enters. The rest of the theory is about how this noise adds up along a curve. Let $\mathcal { C } ( M , K , v , s )$ be the laws with continuous conditional score distributions, max<sub>k</sub> $\bar { v } _ { k } \ \leq \ v$ and $\Sigma _ { \mathrm { w } } \ \leq \ s .$ , where $0 \leq v \leq 1 / 4$ and $0 \le s \le K v$ . Put $M _ { \varepsilon } = \operatorname* { m i n } ( M , \varepsilon ^ { - 2 } )$ 2 $a _ { * } = \operatorname* { m i n } ( v , s )$ with $a _ { * } \{ 1 + \log ( s / a _ { * } ) \} = 0$ when $a _ { * } = 0$ $\textstyle H _ { K } = \sum _ { j < K } 1 / j$ and $\kappa _ { \delta } = \mathrm { k l } ( 1 - \delta , \delta )$ , the Kullback– Leibler divergence between Bernoulli laws with means $1 - \delta$ and δ. Audits must stay valid for all laws, inside the class and outside it.

Theorem 4 (Price of a certified curve). Let $0 < \delta \le 1 / 4$ and $0 < \varepsilon \le 1 / 3 2$

(i) Over $\mathcal { C } ( M , K , v , s )$ the minimax expected number of generated answers is

$$
\widetilde \Theta \Bigl ( \frac { K } { \varepsilon } + M _ { \varepsilon } + \frac { s } { \varepsilon ^ { 2 } } \Bigr ) .\tag{4}
$$

(ii) For every valid audit there is a law in $\mathcal { C } ( M , K , v , s )$ at which its expected number of correctness queries is at least a constant, depending only on $\delta ,$ , times

$$
\frac { H _ { K } } { \varepsilon } + M _ { \varepsilon } + \frac { a _ { * } \{ 1 + \log ( s / a _ { * } ) \} } { \varepsilon ^ { 2 } } .\tag{5}
$$

block A: the first a answers, winner $U = W _ { a }$

$$
a \leq k \leq 2 a
$$

$$
W _ { k } = V
$$

Figure 2. The coupling behind Lemma 3. The highest score of the whole path lies in block A or in block $B ,$ so the winner at budget k is the winner of one of the two blocks. Exchanging the unshared parts of the blocks swaps their winners and leaves the law of the path unchanged.

(iii) One audit, told neither v nor s, attains both (4) and (5), up to factors that are polynomial in log $\{ K / ( \varepsilon \delta ) \}$

Three obstructions. Each term of (4) is forced by a diferent kind of law (Appendix C).

• Calibration. A valid audit must allow for a rare answer that scores above everything seen so far and carries the minority label. If such an answer turns up with probability $3 \varepsilon / K$ per draw, $\theta _ { K }$ moves by more than $2 \varepsilon .$ so the audit has to generate about $K / \varepsilon$ answers before it can rule the answer out. This term is present even when correctness is deterministic, and it is paid at every law: every valid audit generates at least $7 \kappa _ { \delta } K / ( 3 2 \varepsilon )$ answers in expectation, whatever the law.

• Telling questions apart. Give each question a hidden correctness bit that scores do not reveal. An interval of width 2ε for the average needs most of the bits once $M \leq \varepsilon ^ { - 2 }$ , and about $\varepsilon ^ { - 2 }$ of them otherwise.

• Within-question noise. A law whose winners change label within a question forces $s / \varepsilon ^ { 2 }$ answers, and the same law forces the label term $\dot { a } _ { * } \{ 1 + \log ( s / a _ { * } ) \} / \varepsilon ^ { 2 } $ one law makes both resources expensive at once.

Labels follow the same pattern with 1/ε in place of $K / \varepsilon \colon$ in $\widetilde { \Theta }$ form they cost $1 / \varepsilon + M _ { \varepsilon } + { a _ { * } } / { \varepsilon ^ { 2 } }$ , and (ii) shows that the factors $H _ { K }$ and $1 + \log ( s / a _ { * } )$ cannot be removed. The first of them is paid at every law as well. For every continuous-score law, every valid audit uses at least (1 + $\lfloor \log _ { 1 6 } K \rfloor ) 3 \kappa _ { \delta } / ( 3 2 \varepsilon )$ correctness queries in expectation, one $3 \kappa _ { \delta } / ( 3 2 \varepsilon )$ for each of the disjoint score scales on which the winners of budgets 1, 16, 256, . . . live (Fitas, 2026). Labels for the whole curve cost what labels for $\theta _ { 1 }$ alone cost, up to these logarithms: a law whose withinquestion variance halves with each budget already forces $\kappa _ { \delta } a _ { * } / ( 2 6 \varepsilon ^ { 2 } )$ queries for $\theta _ { 1 }$ . The larger budgets are paid for in generated answers, not in grading. If correctness stays noisy after the score is seen, with conditional success probability between η and 1 − η, the price of labels rises to $( 1 + \lfloor \log _ { 1 6 } K \rfloor ) \kappa _ { \delta } \eta c _ { 0 } ^ { 2 } / ( 3 6 \varepsilon ^ { 2 } )$ at every such law once $\varepsilon \le \eta c _ { 0 } / 6$ , with $c _ { 0 } = e ^ { - 1 / 4 } - e ^ { - 4 }$

An example. On our MMLU-Pro study $( M = 1 0 0 ,$ $K = 6 4 , \varepsilon = 1 / 3 2 )$ the three terms are 2,048, 100 and 2,628. With the unrestricted envelope $s = K / 4$ the last term would be 16,384: the measured within-question variance is six times smaller than the worst case, and the last term decides the bill. These are rate terms, not a predicted bill; logarithmic factors and the start-up of Section 6 multiply them. Over the 185 held-out pools, $\Sigma _ { \mathrm { w } }$ is a quarter of its worst case at $K = 6 4$ and 6% of it at $K = 1 0 2 4$

The multilevel audit. The upper bound in (iii) is a multilevel Monte Carlo audit (Giles, 2015; Rhee and Glynn, 2015). It estimates $\theta _ { 1 }$ and adds dyadic corrections: at level ℓ a path of length $b _ { \ell } = \operatorname* { m i n } ( 2 ^ { \ell } , K )$ contributes $W _ { \mathrm { m i n } ( k , b _ { \ell } ) } - W _ { 2 ^ { \ell - 1 } }$ to every budget $k > 2 ^ { \ell - 1 }$ and nothing to smaller budgets. The corrections telescope to $\theta _ { k } - \theta _ { 1 }$ Lemma 3 bounds the second moment of each correction by $\bar { v } _ { 2 ^ { \ell - 1 } }$ , and stopping each level on its own makes its sample size follow its variance. Summing over levels turns these variances into $\Sigma _ { \mathrm { w } }$ . The audit identifies the rate; the audit we run in practice is the one in Section 6.

## 5 One benchmark at a time

Theorem 4 is a statement about classes of laws. At a fixed precision no audit can be optimal law by law, and the reason is a guess-and-verify argument (Narayanan et al., 2024, Claim 1.8): an audit that guesses a benchmark’s correctness table can check its guess with about $\log ( 2 / \delta ) / \varepsilon$ labels, and no single audit is that cheap at every table.

Proposition 5 (No audit is optimal at every law). Let $\delta \leq 1 / 4 , \varepsilon \leq 1 / 1 2 8 , K = 1 , M = \lfloor \pi / ( 1 0 2 4 \varepsilon ^ { 2 } ) \rfloor$ and $q = \lceil \log ( 2 / \delta ) / \{ - \log ( 1 - \varepsilon ) \} \rceil$ . For every valid audit there is a law, with continuous scores and deterministic correctness, at which its expected numbers of generated answers, correctness queries and question visits are each at least $M / 4$ , while another valid audit uses exactly q of each. The ratio $M / ( 4 q )$ is of order $1 / \{ \varepsilon \log ( 2 / \delta ) \}$ }.

Proof. A cheap audit for one table. Fix $f : \{ 1 , \dots , M \} $ {0, 1} with average ${ \check { f } } .$ The audit $A _ { f }$ draws q questions independently and uniformly from the list, generates one answer at each and queries its correctness. If every value agrees with $f ,$ it returns $[ \bar { f } - \varepsilon , \bar { f } + \varepsilon ] ;$ ; otherwise it runs a fixed valid audit at level $\delta / 2$ , such as a fixed Hoefding design, and returns its interval. Let $d = M ^ { - 1 } \sum _ { x } \mathbb { P } ( Y \neq$ $f ( x ) \mid x )$ . If $d \leq \varepsilon .$ , then $\begin{array} { r } { | \theta _ { 1 } - \bar { f } | \le M ^ { - 1 } \sum _ { x } \bar { | p _ { 1 } ( x ) - } } \end{array}$ $f ( x ) | = d \leq \varepsilon$ and the first interval covers. If $l > \varepsilon .$ the first interval is returned with probability $( 1 - d ) ^ { q } \leq$ $( 1 - \varepsilon ) ^ { q } \leq \delta / 2$ . So $A _ { f }$ is valid at every law, and at a law with $Y = f ( x )$ it uses exactly q answers, q queries and q visits.

Every audit is expensive at some table. Let scores be independent and uniform on [0, 1], let $Y = f ( x )$ , and draw the M bits $f ( x )$ independently and uniformly. Scores, unqueried answers and the audit’s own randomness carry no information about the bits, so after any transcript the bits of questions without a queried answer are independent and uniform given the transcript. Let u be their number. Given the revealed bits, $\theta _ { 1 }$ is a known shift of $\mathrm { B i n } ( u , 1 / 2 ) / M$ , and an interval of width 2ε contains at most $2 \varepsilon M + 1$ of its possible values. If $u \ge M / 2$ , the interval therefore contains $\theta _ { 1 }$ with conditional probability at most

$$
\begin{array} { r l } & { ( 2 \varepsilon M + 1 ) \underset { j } { \mathrm { m a x } } \mathbb { P } \{ \mathrm { B i n } ( u , 1 / 2 ) = j \} } \\ & { \le ( 2 \varepsilon M + 1 ) \sqrt { \frac { 2 } { \pi u } } } \\ & { \le 4 \varepsilon \sqrt { \frac { M } { \pi } } + \frac { 2 } { \sqrt { \pi M } } } \\ & { \le \frac 1 8 + \frac { 2 } { \sqrt { 5 0 \pi } } < \frac { 1 } { 2 } , } \end{array}
$$

because $M \leq \pi / ( 1 0 2 4 \varepsilon ^ { 2 } )$ and $M \geq \lfloor \pi \cdot 1 2 8 ^ { 2 } / 1 0 2 4 \rfloor = 5 0$ Averaging the coverage requirement over the prior gives $\begin{array} { r } { 1 - \delta \le \mathbb { P } ( u < M / 2 ) + \frac { 1 } { 2 } \mathbb { P } ( u \ge M / 2 ) } \end{array}$ , so more than $M / 2$ bits are revealed with probability at least $1 - 2 \delta \ge 1 / 2$ and the expected number D of revealed bits is at least $M / 4$ under the prior. Some table $f$ attains $\mathbb { E } _ { f } D \ \geq$ $M / 4$ . Each revealed bit needs an answer generated at its question, a correctness query and a visit, so all three expected counts are at least $M / 4$ at that law, where $A _ { f }$ uses q of each. Finally, $q \leq 1 + \log ( 2 / \delta ) / \varepsilon$ and $M \overset { \cdot } { \geq }$ $\pi / ( 1 0 2 4 \varepsilon ^ { 2 } ) - 1$ make $M / ( 4 q )$ of order $1 / \{ \varepsilon \log ( 2 / \delta ) \}$ . □

What survives is the leading term as $\varepsilon \to 0$ at a fixed benchmark. For budget k let $\widetilde { W } _ { k }$ be the correctness of the highest-scoring answer among k, averaged over tied maxima, so that $\mathbb { E } ( \widetilde { W } _ { k } \mid x ) = p _ { k } ( x )$ . Let

$$
v _ { k } ( x ) = k ^ { 2 } \operatorname { V a r } \{ \mathbb { E } ( \widetilde { W } _ { k } \mid Z _ { 1 } , x ) \mid x \} ,
$$

where $Z _ { 1 }$ is one of the k answers. This is the variance per answer of the all-subsets average of $\widetilde { W } _ { k }$ over many answers at x, the first term of its Hoefding decomposition (Hoefding, 1948). If a fraction $a _ { x }$ of $N$ answers goes to question x, the benchmark estimate at budget k has variance close to $\textstyle \sum _ { x } v _ { k } ( x ) / ( M ^ { 2 } a _ { x } N )$ , and

$$
\Gamma = \operatorname* { m i n } _ { a } \operatorname* { m a x } _ { k \leq K } \frac { 1 } { M ^ { 2 } } \sum _ { x = 1 } ^ { M } \frac { v _ { k } ( x ) } { a _ { x } } ,\tag{6}
$$

with a ranging over the simplex, is the smallest worstbudget variance per answer.

Theorem 6 (Price at one benchmark). Let $0 < \delta < 1 / 2$ and fix a law on the list.

(i) For $0 < \varepsilon \le 1 / 3 2$ every valid audit has $\mathbb { E N } \geq$ $( \kappa _ { \delta } / 1 2 8 ) ( K / \varepsilon + \Gamma / \varepsilon ^ { 2 } )$ , and every family of valid audits has lim in $\mathrm { f } _ { \varepsilon  0 } \varepsilon ^ { 2 } \mathbb { E } N \geq \kappa _ { \delta } \Gamma / 2$

(ii) One valid audit, whose schedule does not depend on the law, has

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } \upsilon \varepsilon ^ { 2 } \mathbb { E } N \leq 2 \Gamma \log ( 2 K / \delta ) .
$$

$$
( i i i ) \ \Gamma \leq \mathrm { m a x } _ { k } \ : k { \bar { v } } _ { k } \leq \Sigma _ { \mathrm { w } } .
$$

Part (i) is a bound at every law rather than at a worst case, and (ii) matches its leading term up to the factor $4 \log ( 2 K / \delta ) / \kappa _ { \delta }$ . The audit behind (ii) spends a pilot to bound each $v _ { k } ( x )$ , gives question x answers in proportion to $\{ \Sigma _ { k } \lambda _ { k } v _ { k } ( x ) \} ^ { 1 / 2 }$ for a least favorable mixture λ of budgets, as in Neyman allocation (Cochran, 1977; Carpentier et al., 2015), and averages each question’s answers over all k-subsets. It labels every answer (Appendix D). By (iii), $\Sigma _ { \mathrm { w } } / \Gamma$ is what an audit gives up by reading one winner per path and visiting every question equally.

The gap is large on measured benchmarks (Figure 3). On the MMLU-Pro study $\Gamma = 0 . 2 8$ against $\Sigma _ { \mathrm { w } } = 2 . 5 7$ and over the held-out pools the median of $\Sigma _ { \mathrm { w } } / \Gamma$ is 12, with interquartile range 8.4 to 21.9. Most of it is allocation. The model is right, or wrong, at least 98% of the time on 58 of the 100 MMLU-Pro questions, and the optimal allocation puts 90% of its answers on 23 questions. Finding those questions costs answers, though. At $\varepsilon = 1 / 3 2$ the pilot of the audit behind (ii) alone would use 179,200 answers, more than twice the whole bill of the paired audit below, and Proposition 5 says that no audit finds them for free at a fixed precision.

## 6 The paired audit

Stratified sampling removes the between-question variance only if the confidence sequence can measure the variance that is left. A plug-in estimate of each question’s mean does this slowly. Early on, the error of the plug-in center is itself a between-question deviation, so the penalty charges exactly the variance that balance was supposed to remove. Two independent draws at the same question avoid the problem: their squared diference has mean exactly twice the within-question variance, whatever the question’s mean.

(a) within-question price, K = 64  
![](images/c533cf3be4d5f078ea6cae75a25d7bb35a38d3465812dd066805333de76a9d85.jpg)

(b) the allocation behind Γ, MMLU-Pro  
![](images/398f61bfb4731daaace3108086fa331a96d56eed6317486354cf7343df79bc42.jpg)  
Figure 3. The price at one benchmark. (a) Γ of Theorem 6 against $\Sigma _ { \mathrm { w } }$ at $K = 6 4$ for the 185 held-out pools and the MMLU-Pro study (law of the first 4,000 reference answers per question). The median of $\Sigma _ { \mathrm { w } } / \Gamma$ is 12. (b) The allocation of answers to questions that attains Γ on the MMLU-Pro study, sorted by share.

Lemma 7 (Paired exponential inequality). Let A, B be independent Bernoulli(p) variables, $0 \leq \lambda < 1$ and $\psi ( \lambda ) = - \log ( 1 - \lambda ) - \lambda$ . Then

$$
{ \mathbb E } \exp \{ \pm \lambda ( A + B - 2 p ) - \psi ( \lambda ) ( A - B ) ^ { 2 } \} \le 1 .\tag{7}
$$

Proof. Let $q = 1 - p$ and $u = 2 \lambda$ . The pair $( A , B )$ takes the values $( 0 , 0 ) , ( 1 , 1 )$ and the two mixed values with probabilities $q ^ { 2 } , p ^ { 2 }$ and 2pq, and $e ^ { \lambda - \psi ( \lambda ) } = e ^ { 2 \lambda } ( 1 - \lambda )$ Enumerating,

$$
\begin{array} { r l } & { \mathbb { E } e ^ { \lambda ( A + B - 2 p ) - \psi ( \lambda ) ( A - B ) ^ { 2 } } } \\ & { = e ^ { - u p } \big [ q ^ { 2 } + p ^ { 2 } e ^ { u } + 2 p q e ^ { u } ( 1 - u / 2 ) \big ] } \\ & { = e ^ { - u p } \big [ q ^ { 2 } + e ^ { u } \{ p ( 1 + q ) - u p q \} \big ] . } \end{array}
$$

It remains to show $e ^ { u p } - q ^ { 2 } - e ^ { u } \{ p ( 1 + q ) - u p q \} \geq 0$ for $u \geq 0$ . Expand in powers of u. The coeficient of $u ^ { m } / m !$ is $1 - q ^ { 2 } - p ( 1 + q ) = 0$ for $m = 0 ,$ , and $p ^ { m } - p ( 1 + q ) + m p q$ q for $m \geq 1$ , which vanishes for $m = 1$ and $m = 2$ . For $m \geq 3 ~ \mathrm { i t }$ t equals

$$
\begin{array} { c } { p ^ { m } + ( m - 2 ) p - ( m - 1 ) p ^ { 2 } } \\ { = p \big \{ p ^ { m - 1 } + ( m - 2 ) \begin{array} { r l } \end{array} } & { } \\ { - ( m - 1 ) p \big \} \geq 0 , } \end{array}
$$

because $p ^ { m - 1 } = ( 1 - q ) ^ { m - 1 } \geq 1 - ( m - 1 ) q$ by Bernoulli’s inequality. This proves the inequality with the sign +. Applying it to 1−A and 1−B, which are Bernoulli(q) and have the same squared diference, gives the sign −.

Penalized exponential inequalities of this kind go back to Fan et al. (2015) and are the basis of betting confidence sequences (Howard et al., 2021; Waudby-Smith and Ramdas, 2024). Estimating a variance from squared diferences is classical as well (Wang and Ramdas, 2025). What the pair buys here is exactness: for two Bernoulli draws the penalty needs no estimated center and no correction term, so the within-question variance enters the band from the first pair on.

The audit. The audit runs in rounds, and each round visits every question once with a fresh path. Two rounds make a pair.

1. Grow only what is needed. In pair $j ,$ every path is grown to the largest budget whose interval is still wider than 2ε, and correctness is queried only at score records.

2. Update the band. For each unresolved budget k, let $T _ { j }$ be the sum of the 2M selected-correctness values of the pair and $D _ { j }$ the number of questions whose two values disagree. Multiplying (7) over questions and pairs gives two nonnegative supermartingales per budget, and Ville’s inequality (Ville, 1939) with a union over the 2K sides gives, at every pair,

$$
\begin{array} { r } { \theta _ { k } \in \frac { \sum _ { j } \lambda _ { j } T _ { j } } { 2 M \sum _ { j } \lambda _ { j } } \ \pm \ \frac { \sum _ { j } \psi ( \lambda _ { j } ) D _ { j } + \log ( 2 K / \alpha _ { S } ) } { 2 M \sum _ { j } \lambda _ { j } } , } \\ { \alpha _ { S } = 0 . 9 5 \delta . \qquad } \end{array}\tag{8}
$$

The audit keeps the running intersection of these intervals.

3. Bet on the variance seen so $f a r .$ . The bet $\lambda _ { j } ~ =$ min $\{ \lambda _ { \mathrm { m a x } } , \varepsilon / ( \varepsilon + \hat { v } _ { j } ) \}$ maximizes $\lambda \varepsilon - \psi ( \lambda ) \hat { v } _ { j }$ , where $\hat { v } _ { j }$ is half the disagreement rate of earlier pairs, started with one pseudo-pair of variance $1 / 4$ and floored at min $( 1 0 ^ { - 4 } , \varepsilon / 1 0 0 )$

4. Retire and stop. A budget retires when its interval has width at most 2ε. The remaining 0.05 δ goes to an exact-binomial (Clopper and Pearson, 1934) interval at a fixed final round, the smallest even number of rounds at which every possible count gives an interval of width at most 2ε. That round ends the audit.

Figure 4 follows one audit on one stored pool. Budgets resolve at diferent times, and paths shrink as they do: after four pairs only the smallest budgets are still open, and the last pairs generate a few answers per question.

A final look with unequal means. The pooled count at the final round is a sum of independent Bernoulli variables whose average mean is $\theta _ { k }$ but whose individual means difer, one per question. Hoefding (1956, Theorem 4) bounds $\mathbb { P } ( S \le b )$ by the binomial tail when $b \leq n \theta _ { k } - 1$ . The counts just below the mean are not covered, and there the comparison can fail. With $n = 2$ and means $( 1 , 2 \theta - 1 )$ for some $\theta > 1 / 2$ , the count is at most 1 with probability $2 ( 1 - \theta )$ , more than the binomial value $1 - \theta ^ { 2 } . \mathrm { A t } \theta = ( 1 - a ) ^ { 1 / 2 }$ an exact-binomial test at level a rejects whenever the count is at most 1, which happens with probability $2 \{ 1 - ( 1 - a ) ^ { 1 / 2 } \} > a$ . A small correction of the level repairs this.

Lemma 8 (Exact-binomial tails for unequal Bernoulli sums). Let $B \sim \mathrm { B i n } ( n , \theta )$ and let S be a sum of n independent Bernoulli variables whose means average to θ. $I f 0 < a < 1 / 4$ and $a ^ { \prime } = a / ( 1 + a )$ , a lower-tail binomial test at level a<sup>′</sup> rejects with probability at most a under S. The same holds for the upper tail.

Proof. Let b be the largest count the test rejects, so $\mathbb { P } ( B \le b ) \le a ^ { \prime } < 1 / 4$ . Suppose first $b \leq n - 2$ , and suppose $b > n \theta - 1$ . With $h = b + 1 \leq n - 1$ we have $\theta <$ $h / n$ , and since binomial lower tails decrease in the success probability, $\mathbb { P } ( B \le b ) \ge \mathbb { P } \{ \mathrm { B i n } ( n , h / n ) \le h - 1 \}$ . The sum of $h + 1$ independent Bernoulli $\{ h / ( h + 1 ) \}$ variables, padded with $n - h - 1$ zeros, has mean $h ,$ so Hoefding’s comparison at $h - 1$ , one below the mean, gives

$$
\begin{array} { l l r } {  { \mathbb { P } \{ \mathrm { B i n } ( n , h / n ) \le h - 1 \} } } \\ & { \ge \mathbb { P } \{ \mathrm { B i n } ( h + 1 , h / ( h + 1 ) ) \le h - 1 \} } \\ & { = 1 - \Big ( \displaystyle \frac { h } { h + 1 } \Big ) ^ { h } \displaystyle \frac { 2 h + 1 } { h + 1 } \ge \displaystyle \frac { 1 } { 4 } , } \end{array}
$$

because the subtracted term decreases in h from $3 / 4$ at $h = 1$ . This contradicts $\mathbb { P } ( B \le b ) < 1 / 4$ , so $b \leq n \theta - 1$ ， and Hoefding’s comparison gives $\mathbb { P } ( S \le b ) \le \mathbb { P } ( B \le b ) \le$ $a ^ { \prime } \leq a . { \mathrm { ~ I f ~ } } b = n - 1$ , rejection means $1 - \theta ^ { n } \leq a ^ { \prime } .$ , and a union bound over the n variables gives $\mathbb { P } ( S \le n - 1 ) \le$ $n ( 1 - \theta ) \leq - n \log \theta \leq - \log ( 1 - a ^ { \prime } ) = \log ( 1 + a ) \leq a$ Applying the argument to the complements proves the upper tail. □

The final look inverts each of the 2K tails at $a / ( 1 + a )$ with $a = 0 . 0 5 \delta / ( 2 K )$ . Since $a / ( 1 + a )$ difers from a by a factor $1 + a \le 1 + 1 0 ^ { - 3 }$ , the final round is the same as with the uncorrected level in every configuration we ran.

Theorem 9 (Paired audit). For every fixed $\lambda _ { \operatorname* { m a x } } \in ( 0 , 1 )$ , the paired audit has simultaneous coverage at least $1 - \delta$ for every answer law and stops by a deterministic round cap. With L logarithmic in $K , 1 / \varepsilon$ and $1 / \delta _ { ; }$ , and constants depending on $\lambda _ { \mathrm { m a x } }$ 2

$$
\begin{array} { l } { \displaystyle \mathbb { E } N = O \bigg ( K M + L \Big [ \frac { K } { \varepsilon } + \frac { \Sigma _ { \mathrm { w } } } { \varepsilon ^ { 2 } } \Big ] \bigg ) , } \\ { \displaystyle \mathbb { E } m \leq H _ { K } \mathbb { E } n . } \end{array}\tag{9}
$$

Proof of coverage. Fix a budget k and write $A _ { x j } , \ B _ { x j }$ for the selected-correctness values of the two paths at question x in pair $j .$ The bet $\lambda _ { j }$ is fixed before either path of the pair is generated, and the paths of diferent questions are independent. Multiplying (7) over questions and then over pairs shows that

$$
\exp \Biggl \{ \begin{array} { l } { { \displaystyle \sum _ { j \le J } \lambda _ { j } \biggl ( \sum _ { x } ( A _ { x j } + B _ { x j } ) - 2 M \theta _ { k } \biggr ) } } \\ { { \displaystyle - \sum _ { j \le J } \psi ( \lambda _ { j } ) \sum _ { x } ( A _ { x j } - B _ { x j } ) ^ { 2 } } } \end{array} \Biggr \}
$$

is a nonnegative supermartingale in $^ { J , }$ and so is its mirror image. Ville’s inequality at level $\alpha _ { S } / ( 2 K )$ for each of the 2K processes, and a union bound, give (8) at all pairs at once with probability at least $1 - \alpha _ { S }$ , and running intersections keep this event. A budget that has retired needs no further values: attach a latent full path to every scheduled visit, and note that the audit reads the winner at budget k only while k is active, so no unobserved value enters the sequence. At the final round, Lemma 8 bounds each of the 2K tails by $a = \alpha _ { C } / ( 2 K )$ with $\alpha _ { C } = 0 . 0 5 \delta$ The final interval is intersected with the betting interval; if the intersection is empty, which happens only outside the coverage event, a point is returned. Coverage holds with probability at least $1 - \alpha _ { S } - \alpha _ { C } = 1 - \delta$ □

The cost bound is proved in Appendix E. The calculation behind it is short. Replace the penalty in (8) by its mean and suppose the bet uses the true $\bar { v } _ { k }$ With $\lambda = \varepsilon / ( \varepsilon + \bar { v } _ { k } )$ , the penalty part of the radius is $\psi ( \lambda ) \bar { v } _ { k } / \lambda \leq \varepsilon / 2$ , and the other part, $L / ( n \lambda )$ after n question visits, falls below $\varepsilon / 2$ once

$$
n _ { k } = 2 L \left( \bar { v } _ { k } + \varepsilon \right) / \varepsilon ^ { 2 } , \qquad L = \log ( 2 K / \alpha _ { S } ) .\tag{10}
$$

An answer at position j is generated only while some budget $k \geq j$ is unresolved. After a start-up of one complete pair of rounds, 2KM answers, the audit therefore generates about $\begin{array} { r } { \sum _ { j } \operatorname* { m a x } _ { k \geq j } n _ { k } = 2 L ( K / \varepsilon + \Sigma _ { \mathrm { w } } / \varepsilon ^ { 2 } ) } \end{array}$ answers. Section 8.5 fits this shape to measured costs and finds coeficients 2.47, 1.38 and 1.75 where the calculation gives 2, 2 and 2. The start-up is small when $M \varepsilon ^ { 2 }$ is small; Section 8.4 shows where it is not.

![](images/436bf1c46525025fd394b9cb022ab1d80d94c0672909a580c3ed91a390ab5538.jpg)

![](images/8273e84d7279bf849cf6551bcc99b43f3f05a3587d16e1b7cd24bc6cd9ecd49a.jpg)

![](images/0e837cf18f5fc6c757114acda5b553dbf640adc30d77e8d47e535545e2111d58.jpg)  
Figure 4. One paired audit, traced round by round (the stored run on the MATH500 pool of Figure 1a, 250 questions, $K = 6 4 )$ . (a) Answers generated per question visit in each pair of rounds; paths stop at the largest unresolved budget. (b) Half-width of the band at three budgets; a budget retires when it reaches $1 / 3 2 ,$ , and the final look is at round 18. (c) Answers used so far, against the fixed exact-binomial design and the cheapest of the competing certified audits on the same pool. The audit stops after 12 rounds with 127,500 answers, and its band covers the exact curve.

Explicit constants. The constants hidden in (9) come from the adaptive bet. A variant that also runs fixed dyadic bets turns the calculation into a proof with small constants.

Proposition 10 (Paired audit with a grid of bets). Suppose the audit also runs, for every budget, the bets $\lambda _ { h } ~ = ~ 2 ^ { - h - 2 }$ for $h \ = \ 0 , \ldots , H \ - - \ 1$ with $H \_ =$ $1 + \lceil \log _ { 2 } \{ ( 1 / 4 + \varepsilon ) / \varepsilon \} \rceil$ , each giving the interval $T / n \pm$ $\{ \psi ( \lambda _ { h } ) D + L ^ { \prime } \} / ( \lambda _ { h } n )$ after n rows of complete pairs, where T and D are pooled success and disagreement counts and ${ \cal L } ^ { \prime } = \log \{ 2 K ( H + 1 ) / \alpha _ { S } \}$ , and intersects all its intervals. Let n be the number of rows at the final round. Then for every $\eta \in ( 0 , 1 )$ 2

$$
\begin{array} { c } { { \mathbb { E } N \leq \operatorname* { m i n } \Bigl \{ K n _ { F } , 2 K M + \bigl \{ 1 6 L ^ { \prime } + \log ( K / \eta ) \bigr \} } } \\ { { \Bigl ( \displaystyle \frac { K } { \varepsilon } + \frac { \Sigma _ { \mathrm { w } } } { \varepsilon ^ { 2 } } \Bigr ) + \eta K n _ { F } \Bigr \} . } } \end{array}
$$

Proof. Fix k and write $v = \bar { v } _ { k }$ . The grid contains a bet with $\varepsilon / \{ 8 ( v + \varepsilon ) \} < \lambda \le \varepsilon / \{ 4 ( v + \varepsilon ) \} \le 1 / 4$ , and $\psi ( \lambda ) \leq$ $\lambda ^ { 2 }$ there. The count D is a sum of independent Bernoulli variables with total mean nv, so $\bar { \mathbb { E } } e ^ { \hat { D ^ { \prime } } / 2 } \leq \exp \lbrace ( e ^ { 1 / 2 } -$ 1)nv} and $\mathbb { P } ( D > 2 n v + 2 u ) \le e ^ { - u }$ . Outside that event the radius of this bet is at most

$$
2 \lambda v + \frac { 2 \lambda u } { n } + \frac { L ^ { \prime } } { \lambda n } \leq \frac { \varepsilon } { 2 } + \frac { 1 } { n } \Big \{ \frac { \varepsilon u } { 2 ( v + \varepsilon ) } + \frac { 8 ( v + \varepsilon ) L ^ { \prime } } { \varepsilon } \Big \} ,
$$

which is at most ε once $n \geq ( v + \varepsilon ) ( 1 6 L ^ { \prime } + u ) / \varepsilon ^ { 2 }$ . Apply this at the K deterministic row counts $n _ { k }$ given by that bound, rounded up to a multiple of 2M, with $u = \log ( K / \eta )$ . Outside an event of probability $\eta ,$ every budget k is resolved within $n _ { k }$ rows or at the final round.

An answer at position j is generated only while some budget $k \geq j$ is unresolved, so $\begin{array} { r } { N \leq \sum _ { j } \operatorname* { m a x } _ { k \geq j } n _ { k } \leq } \end{array}$ $2 K M + \left\{ 1 6 L ^ { \prime } + \log ( K / \eta ) \right\} \sum _ { i }$ ma $\bar { \cdot } k \geq j  \big ( \bar { v } _ { k } + \varepsilon \big ) / \varepsilon ^ { 2 }$ , and $\textstyle \sum _ { j }$ ma ${ \mathrm { x } } _ { k \geq j } ( { \bar { v } } _ { k } + \varepsilon ) = { \mathrm { ~ } } { \mathrm { \Sigma } } _ { \mathrm { w } } + { \bar { K } } \varepsilon$ . Outside that event, $N \leq K n _ { F }$ □

Taking $\eta = \varepsilon$ gives the order of (9) with the constant 16 in place of the calculation’s 2.

One audit for the rate and for practice. The startup KM is absent from Theorem 4, so the paired audit attains the rate (4) only when $M \lesssim 1 / \varepsilon + s / ( K \varepsilon ^ { 2 } )$ . The multilevel audit attains the rate everywhere, but with poor constants. Running the two side by side keeps the best of both.

Corollary 11 (Portfolio). Fix $\rho \in ( 0 , 1 )$ . Run the paired audit at level $( 1 - \rho ) \delta$ and the multilevel audit of Theorem $4 ( i i i )$ at level ρδ on separate fresh answers, one path at a time, always advancing the audit whose answer count divided by its weight, 1 − ρ or ρ, is smaller, and return the band of the first to finish. The result is a valid audit, and on every run

$$
N \leq \operatorname* { m i n } \Bigl \{ \frac { N _ { \mathrm { p a i r e d } } } { 1 - \rho } , \frac { N _ { \mathrm { m u l t i l e v e l } } } { \rho } \Bigr \} + K ,
$$

where $N _ { \mathrm { p a i r e d } }$ and $N _ { \mathrm { m u l t i l e v e l } }$ are the answers each audit would use alone on the same answers.

Proof. Write $w _ { 1 } = 1 - \rho$ and $w _ { 2 } = \rho$ , and let $N _ { i }$ be the answers audit i has used so far. Each step generates one path, of at most K answers, for the audit with the smaller ratio $N _ { i } / w _ { i }$ . After every step, $N _ { j } \leq w _ { j } N _ { i } / w _ { i } + K$ for $i \neq$ j: at the last step at which audit j advanced, $N _ { j } / w _ { j } \le$

$N _ { i } / w _ { i }$ ; since then $N _ { i }$ has not decreased, and that step added at most $K$ answers to $N _ { j }$ . Hence $N = N _ { 1 } + N _ { 2 } \leq$ $N _ { i } / w _ { i } + K$ for both i. Each audit’s state depends only on its own answers, so until the first completion $N _ { i }$ is at most the count the audit would use alone, and the bound follows. Each audit covers every $\theta _ { k }$ with probability at least $1 - w _ { i } \delta$ whatever the other does, so both cover with probability at least $1 - \delta$ and the returned band covers. Both stop by deterministic caps, so the portfolio does too. □

With $\rho = 1 / 3 2$ the portfolio attains (4) up to a factor 32 and uses at most 32/31 of the answers of the paired audit at level $3 1 \delta / 3 2 .$ , plus K. The bound concerns generated answers only.

## 7 Other curves, other settings

## 7.1 Pass@k and majority voting

Coverage of the paired audit uses only that each path yields a correct-or-incorrect outcome at every budget, computed from the first k answers, and that paths are independent given the question. Pass@k and majority voting have that form, so the paired audit certifies them unchanged; only the label count difers, since pass@k needs every answer before the first correct one checked and majority voting needs a grade for each distinct valid answer.

A path can also serve every budget at once through its subsets. A path of length $b \geq k$ contains $\binom { b } { k }$ subsets of size k, and each is distributed as k fresh answers. Averaging a curve’s outcome over all of them gives an unbiased estimate $U _ { x , k }$ of $p _ { k } ( x )$ , a U-statistic (Hoefding, 1948). For best-of-k, sorting the path by score gives the average in closed form: a group of d tied answers with c answers strictly below it contributes its average correctness times $\{ \binom { c + \bar { d } } { k } - \binom { c } { k } \} / \binom { b } { k }$ , which matches firstmaximum selection in expectation (Nakano et al., 2021; Gao et al., 2023). For pass@k with c correct answers on the path, $\begin{array} { r } { U _ { x , k } = 1 - \binom { b - c } { k } / \binom { b } { k } } \end{array}$ (Chen et al., 2021). For majority voting we average over random subsets, an incomplete U-statistic that is still unbiased.

These averages are bounded but not binary, so the paired inequality does not apply. A bounded-mean inequality does: for $Z ~ \in ~ [ - 1 , 1 ]$ and $0 ~ \leq ~ \lambda ~ < ~ 1$ $e ^ { \lambda z - \psi ( \lambda ) z ^ { 2 } } \leq 1 + \lambda z \mathrm { o n } [ - 1 , 1 ]$ , hence $\mathbb { E } e ^ { \lambda ( Z - \mathbb { E } Z ) - \psi ( \lambda ) Z ^ { 2 } } \preceq$ 1. With $Z = U _ { x , k } - c _ { x , k }$ for a prediction $c _ { x , k } \in [ 0 , 1 ]$ built from earlier rounds, multiplying over the questions of a round and over rounds gives two nonnegative supermartingales per budget, and Ville’s inequality gives simultaneous bands. We call this the all-subsets audit. It needs the correctness of every answer on the path, records or not, so it uses more labels than the paired audit unless one grade can serve every repeat of an answer (Section 8.6).

First-success stopping for pass@k. For pass@k alone a path can stop at its first correct answer. With $H = \operatorname* { m i n } \{ j : Y _ { j } = 1 \}$ , or $K + 1$ if none of the first K answers is correct, the indicator $\mathbf { 1 } \{ H \leq k \}$ is the pass@k outcome of the path at every $k \leq K$ . On n paths at questions drawn independently and uniformly from the list, the stopping positions are independent and identically distributed, and the Dvoretzky–Kiefer–Wolfowitz inequality with the constant of Massart (1990) gives one band of radius $\{ \log ( 2 / \delta ) / ( 2 n ) \} ^ { 1 / 2 }$ for every k at once (Dvoretzky et al., 1956). A path at question x uses $\textstyle \sum _ { j < K } \{ 1 - p _ { 1 } ( x ) \} ^ { j }$ answers in expectation and needs the correctness of each of them.

## 7.2 One grade per distinct answer

On multiple-choice or short-answer benchmarks the grader is a deterministic function of the question and the extracted answer. Reusing one grade for every repeat of a question–answer pair changes no observation of any audit: bands, coverage and answer counts are identical on every run, and only the number of distinct grader calls falls. The same holds for any curve computed from the extracted answers. In the MMLU-Pro study below, fewer than 400 distinct grader calls certify each curve.

## 7.3 A population of questions

The target so far is the listed questions. For a population of unseen questions the between-question variance returns, and Proposition 2 gives its price. A fixed list can still serve a population. Draw a panel of m questions independently from the population, audit the panel as a fixed list at level $\delta ,$ and add

$$
r _ { m } = \left\{ \frac { \log ( 2 K / \alpha ) } { 2 m } \right\} ^ { 1 / 2 }
$$

to every radius. By Hoefding’s inequality (Hoefding, 1963) and a union over budgets and sides, the panel average of $p _ { k }$ is within $r _ { m }$ of the population value at every k with probability at least $1 - \alpha .$ , so the widened band covers the population curve with probability at least $1 - \delta - \alpha$ . The audit then pays the within-question price for the panel and the population pays once, through m.

## 7.4 Answers that depend on earlier answers

The theory above assumes that answers on a path are independent given the question. Agents that revise, search or reuse context break this. Coverage of a curve does not need independence within a path. Let $Z _ { 1 } , \ldots , Z _ { n }$ be independent complete trajectories from one fixed generation policy whose budgets are prefixes of each other, so that later answers may depend on earlier ones. Let $W _ { k } ( Z )$ be the correctness of the first highest-scoring answer among

the first k, and

$$
\begin{array} { r l } & { V = \displaystyle \sum _ { k = 2 } ^ { K } \mathbb { P } \{ W _ { k } \neq W _ { k - 1 } \} , } \\ & { R = \mathbb { E } \{ \mathrm { n u m b e r ~ o f ~ s t r i c t ~ s c o r e ~ r e c o r d s ~ i n ~ } Z \} . } \end{array}
$$

Since the winner changes only at a strict record, $0 \leq V \leq$ $R - 1 \leq K - 1$

Proposition 12 (Bands for dependent trajectories). $F o r$ a universal constant $C _ { \delta }$

$$
\mathbb { E } \operatorname* { m a x } _ { k \leq K } \Big | \frac { 1 } { n } \sum _ { i = 1 } ^ { n } W _ { k } ( Z _ { i } ) - \mathbb { E } W _ { k } \Big | \leq C \sqrt { \frac { \log ( 2 + V ) } { n } } ,
$$

and with probability at least $1 - \delta$ the maximum exceeds this bound by at most $\sqrt { \log ( 1 / \delta ) / ( 2 n ) }$ . With B independent vectors $\sigma ^ { ( b ) } o f$ Rademacher signs and $A _ { B } \ =$ $\begin{array} { r } { B ^ { - 1 } \sum _ { b < B } \operatorname* { m a x } _ { k \leq K } \left| n ^ { - 1 } \sum _ { i } \sigma _ { i } ^ { ( b ) } W _ { k } ( Z _ { i } ) \right| } \end{array}$ , the radius

$$
2 A _ { B } + 2 { \sqrt { \frac { \log ( 3 / \delta ) } { 2 B } } } + 3 { \sqrt { \frac { \log ( 3 / \delta ) } { 2 n } } }
$$

covers every $\mathbb { E } W _ { k }$ simultaneously with probability at least $1 - \delta$ , without knowledge of V .

Proof. Let $\tau _ { 1 } = 0$ and $\textstyle \tau _ { k } = \sum _ { i = 2 } ^ { k } \mathbb { P } ( W _ { j } \neq W _ { j - 1 } )$ . For $0 < r \leq 1$ , group consecutive budgets by $\lfloor \tau _ { k } / r ^ { 2 } \rfloor$ ; there are at most $1 + \lfloor V / r ^ { 2 } \rfloor$ groups. Within a group $[ a , b ]$ the pointwise minimum and maximum of the $W _ { k }$ form a bracket with $\begin{array} { r } { \mathbb { E } \{ \operatorname* { m a x } _ { a \leq k \leq b } W _ { k } - \operatorname* { m i n } _ { a \leq k \leq b } W _ { k } \} ^ { 2 } \leq } \end{array}$ $\textstyle \sum _ { j = a + 1 } ^ { b } \mathbb { P } ( W _ { j } \neq W _ { j - 1 } ) < r ^ { 2 }$ . The bracketing entropy integral is at most $\begin{array} { r } { \int _ { 0 } ^ { 1 } \sqrt { \log ( 2 + V / r ^ { 2 } ) } d r \leq \sqrt { \log ( 2 + V ) } + } \end{array}$ $\textstyle \int _ { 0 } ^ { 1 } { \sqrt { 2 \log ( 1 / r ) } } d r$ , and the bracketing maximal inequality for bounded classes (van der Vaart and Wellner, 1996, Theorem 2.14.2) gives the bound in expectation. Changing one trajectory moves the maximum by at most $1 / n$ so the bounded-diferences inequality (McDiarmid, 1989) gives the high-probability statement. For the computable radius, symmetrization bounds the expected maximum error by twice the expected Rademacher supremum. Both the maximum error and the conditional expected Rademacher supremum change by at most $1 / n$ when one trajectory changes, and two bounded-diference events account for $3 \sqrt { \log ( 3 / \delta ) / ( 2 n ) }$ . Given the trajectories, each simulated supremum lies in [0, 1], and Hoefding’s inequality adds $2 \sqrt { \log ( 3 / \delta ) / ( 2 B ) }$ . A union bound over the three events finishes the proof. □

A complete trajectory costs K answers and, when generation needs no labels, at most its number of records in queries. The proposition gives coverage and a sample size from any upper bound on V. It does not optimize acquisition, and it covers one fixed policy.

## 7.5 The price of unknown score percentiles

Every audit above treats the verifier score as an ordinal quantity: only the order of the scores on a path matters. If the population distribution of scores at a question were known, each answer’s percentile would be known too, and labels could be spent exactly where the winners of each budget land (Fitas, 2026). How much is that knowledge worth? We answer for one question and a mean-squared-error target, where the answer is exact.

All answers are independent and identically distributed from one score–correctness law. In the oracle experiment the auditor is given the score distribution, so it observes each answer’s percentile; in the unknown experiment it observes scores only. In both it may pool answers across budgets and acquire up to $C$ answers and T correctness labels adaptively, each answer labeled at most once. Let

$$
R = \operatorname* { i n f } _ { \mathcal { A } } \operatorname* { s u p } _ { P } \operatorname* { m a x } _ { k \leq K } \mathbb { E } _ { P } ( \widehat { \theta } _ { k } - \theta _ { k } ) ^ { 2 }
$$

be the minimax worst-budget mean squared error, with $R _ { \mathrm { o r a c l e } }$ and $R _ { \mathrm { u n k n o w n } }$ its values in the two experiments. We allow caps that hold on every run and caps that hold in expectation uniformly over laws.

Theorem 13 (Unknown score percentiles). $I f K \to \infty ,$ T / log $K \to \infty , C / K \to \infty$ and $C \geq T$ , then under either kind of cap

$$
\begin{array} { r } { R _ { \mathrm { o r a c l e } } = ( 1 + o ( 1 ) ) \operatorname* { m a x } \Bigl \{ \cfrac { \log K } { 1 6 T } , \frac { K } { 8 C } \Bigr \} , } \\ { R _ { \mathrm { u n k n o w n } } = ( 1 + o ( 1 ) ) \operatorname* { m a x } \Bigl \{ \cfrac { \log K } { 1 6 T } , \frac { K } { 2 e C } \Bigr \} . } \end{array}
$$

Hence $R _ { \mathrm { u n k n o w n } } / R _ { \mathrm { o r a c l e } }  4 / e$ when

$$
\operatorname* { l i m } \operatorname* { s u p } C \log K / ( K T ) \leq 2 ,
$$

and $R _ { \mathrm { u n k n o w n } } / R _ { \mathrm { o r a c l e } }  1$ when

$$
\operatorname* { l i m } \operatorname* { i n f } C \log K / ( K T ) \geq 8 / e .
$$

With $x = C \log { K } / ( K T )$ , the label term dominates the oracle rate when $x \ge 2$ and the unknown rate when $x \ge 8 / e$ which gives the two limits and the curve of Figure 5. An estimator that uses only the observed order of the scores attains the unknown rate. The lower bounds allow any adaptive acquisition and delayed labeling. So $4 / e \approx 1 . 4 7$ is the exact asymptotic price of not knowing the percentiles when answers are the bottleneck. When labels are, the knowledge is worth nothing to first order. The proof, in Appendix F, rests on a variance bound for the complete-pool U-statistic that keeps every order of overlap between subsets and holds uniformly over score– correctness laws,

$$
\operatorname { V a r } ( U _ { n , k } ) \leq { \frac { k } { 2 e n } } + \left( { \frac { k } { n } } \right) ^ { 2 } + { \frac { 1 } { n } } ,\tag{11}
$$

![](images/9aee644cf539fc993b96d3bb337e00ec5eaa114ff5ddc3991780cd791417d576.jpg)  
Figure 5. Theorem 13: the leading ratio of the minimax errors without and with known score percentiles, as a function of the balance C log $K / ( K T )$ between generated answers C and labels T.

where $U _ { n , k }$ averages the selected label over all subsets of size k of n answers. The same constant $4 / e$ appears in the work of Fitas (2026) with a diferent meaning: there it compares an envelope design with the variance-optimal design, both with known percentiles.

## 7.6 Spending labels when percentiles are known

When the percentiles are known, a band of fixed width calls for a diferent allocation of labels than a small mean squared error does. Let P be a known $K \times d$ matrix whose row k gives the probability that the budget-k winner falls in each of d score cells. For a proposal q on the cells, draw T independent cells, query their labels, and estimate

$$
\widehat { \theta } _ { k } = \textstyle { \frac { 1 } { 2 } } + \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \frac { P _ { k , I _ { i } } } { q _ { I _ { i } } } \Big ( Y _ { i } - \textstyle { \frac { 1 } { 2 } } \Big ) .
$$

With $\begin{array} { r } { V _ { k } ( q ) = \sum _ { i } P _ { k i } ^ { 2 } / q _ { i } , W _ { k } ( q ) = \operatorname* { m a x } _ { i } P _ { k i } / q _ { i } } \end{array}$ and $x =$ $\log ( 2 K / \delta )$ , Bernstein’s inequality gives each budget the radius

$$
r _ { k } ( T , q ) = \frac { ( W _ { k } + 1 ) x } { 6 T } + \sqrt { \frac { V _ { k } x } { 2 T } + \Big ( \frac { ( W _ { k } + 1 ) x } { 6 T } \Big ) ^ { 2 } } ,
$$

and $r _ { k } \ \leq \ \varepsilon$ holds exactly when $T \geq x \{ V _ { k } ( q ) / ( 2 \varepsilon ^ { 2 } ) +$ $( W _ { k } ( q ) + 1 ) / ( 3 \varepsilon ) \}$ . The smallest suficient number of labels is therefore governed by the convex objective

$$
C _ { \rho } ( q ) = \operatorname* { m a x } _ { k } \{ V _ { k } ( q ) + \rho W _ { k } ( q ) \} , \qquad \rho = 2 \varepsilon / 3 ,
$$

not by $V _ { k }$ or $W _ { k }$ alone. Writing $B _ { ( k , i ) , j } ~ = ~ P _ { k j } ^ { 2 } ~ +$ $\rho P _ { k i } \mathbf { 1 } \{ j = i \}$ gives $\begin{array} { r } { C _ { \rho } ( q ) = \operatorname* { m a x } _ { k , i } \sum _ { j } B _ { ( k , i ) , j } / q _ { j } } \end{array}$ , and convex duality gives the identity

$$
\operatorname* { m i n } _ { \boldsymbol { q } \in \Delta _ { d } } C _ { \boldsymbol { \rho } } ( \boldsymbol { q } ) = \operatorname* { m a x } _ { \lambda \in \Delta _ { K d } } \Bigl [ \sum _ { j } \Bigl ( \sum _ { \boldsymbol { k } , i } \lambda _ { \boldsymbol { k } i } B _ { ( \boldsymbol { k } , i ) , j } \Bigr ) ^ { 1 / 2 } \Bigr ] ^ { 2 } ,
$$

with the inner minimum at $q _ { j }$ proportional to the square root for fixed λ. Strict feasibility and divergence at any zero coordinate give strong duality, so any feasible q and any λ bracket the optimum. Table 1 compares three proposals on two complete pools. The fixed-width proposal saves 8.50% and 4.74% of the labels of the envelope proposal $q \propto \operatorname* { m a x } _ { k } P _ { k } .$ , and 3.26% and 11.68% of those of the variance-optimal proposal. The variance objective ma $\mathrm { x } _ { k } V _ { k }$ can have several near-minimizers with diferent label counts, and that column reports the one our solver returns.

## 8 Experiments

## 8.1 Stored score pools

The primary evidence is 185 score pools from Saad-Falcon et al. (2025). Eight answer sets (GPQA, MATH500, MMLU and MMLU-Pro questions, each answered by Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct, with 100 answers per question) are each scored by 19 to 27 reward models and verifiers; one answer set and one scorer make one pool. An audit draws answers uniformly with replacement from a question’s stored answers, so every pool defines an exact answer law whose curve we compute in closed form, ties included, and every miss is visible. The questions of each benchmark were split in half once, before any tuning. Twenty-six settings of stratified betting audits, three paired and twenty-three per-row, were compared on the first halves and the best was frozen. The held-out pools use the other halves, 250 to 360 questions each. Thirty-two further pools (Weaver’s combined score, CodeRM unit tests, CodeContests) come from sources seen during development and are reported separately. We run five audits per pool at $K \in \{ 6 4 , 2 5 6 , 1 0 2 4 \}$ with $\varepsilon = 1 / 3 2$ and $\delta = 0 . 0 5$ . Appendix G gives licenses, the protocol and every competing design.

The curves themselves are flat where it matters (Figure 6): the median held-out curve comes within $1 / 3 2$ of its best value by budget 10, 126 of the 185 by budget 16 and 174 by budget 64. At $K = 6 4$ , no curve among all 217 pools sits more than 1/32 below its best value over budgets up to 64.

The competition. The comparator on each pool is the cheapest of three certified audits, chosen with hindsight on that pool: an exact-binomial audit that stops revealing paths once every budget is settled, nested exactbinomial looks with budget retirement, and a rank-based stopping audit (Appendix G.3). All three draw questions at random. At $K = 6 4$ and 256 we also ran a fixed Hoefding design, a fixed exact-binomial design and the record design of Fitas (2026); none of them is cheaper than the best of the three on any of the 370 held-out cases.

Table 1. Suficient label draws for a band of half-width 1/32 at level 95% over $K = 1 0 0$ budgets with known percentiles, on two complete HumanEval+ pools scored by Llama-3.1-70B unit tests (Ma et al., 2025; Liu et al., 2023). The designs receive the same pool; answer generation is not charged. Last column: relative gap between the feasible fixed-width proposal and its dual lower bound.
<table><tr><td>solutions from</td><td>envelope</td><td>variance-optimal</td><td>fixed-width</td><td>primal-dual gap</td></tr><tr><td>Llama-3-8B</td><td>6,681</td><td>6,319</td><td>6,113</td><td>0.215%</td></tr><tr><td>Llama-3-70B</td><td>5,398</td><td>5,822</td><td>5,142</td><td>0.233%</td></tr></table>

![](images/e37b2e596515b564ebdc46ed7128de1a69b21cba996826131496585d6f8360d5.jpg)  
Figure 6. First budget at which the exact curve of a held-out pool comes within 1/32 of its best value over budgets up to 1024.

## 8.2 The held-out comparison

Table 2 and Figure 7a give the result. The paired audit is cheaper on at least 173 of 185 pools at every horizon and on all eight answer sets, and the saving grows with K. It also needs fewer question visits (median ratios 0.84, 0.72 and 0.67) and fewer labels. Across all 217 stored pools, including the 32 seen during development, it missed none of its 3,255 runs against the exact curves. The competitors missed 17 times in their 13,020 runs, all within their δ. On the 32 development pools the median ratios are 0.52, 0.43 and 0.36.

## 8.3 Where the saving comes from

Figure 7b takes the audit apart. Balance alone buys nothing: nested exact-binomial looks cost the same on balanced rounds as on random questions (ratio of the two 1.01, 1.00 and 1.03), because an exact-binomial interval cannot see a smaller variance. A stratified betting sequence with per-question plug-in centering (Waudby-Smith and Ramdas, 2024) captures much of the saving, and the paired inequality takes the rest. The paired audit beats that sequence, and a version of it with the paired audit’s bet and cap, on every one of the 185 pools at every horizon; against the better of the two its median saving is 9%, 11% and 17%.

## 8.4 When the audit loses, and why the saving grows

The ten held-out pools where the paired audit loses at K = 64 are those with little variance to remove. Their median within-question share of variance at budget 64 is 0.45, against 0.24 for the rest, and across all 185 pools the Spearman correlation between that share and the cost ratio is 0.82.

Precision matters in the way the cost law says (Figure 8a). We reran the frozen audit at $K = 6 4$ with $\varepsilon = 1 / 1 6$ and $\varepsilon = 1 / 6 4$ , after recording the prediction that the start-up of two complete rounds would dominate at coarse precision and fade at fine precision. At $\varepsilon = 1 / 1 6$ two complete rounds already exceed what a fixed design needs, and the paired audit loses on every held-out pool (median ratio 1.40). $\mathrm { A t } ~ \varepsilon = 1 / 6 4$ it wins on 179 of 185 (median 0.51).

Why does the saving grow with K? Part of the answer is the pools. A stored question has only 100 answers, so at budget k its top-scored answer appears with probability $1 - 0 . 9 9 ^ { k }$ : 0.47 at $k = 6 4$ and 0.92 at 256. Large budgets then redraw a small set, and $\Sigma _ { \mathrm { w } }$ falls from 24% of its worst case at $K = 6 4$ to 6% at $K = 1 0 2 4$ (Figure 8c). To separate this efect from the audit, we built a pool from the first 4,000 reference answers per question of the MMLU-Pro study below, where the top answer appears with probability 0.016, 0.062 and 0.226 at the three horizons and $\Sigma _ { \mathrm { w } }$ stays between 12% and 16% of its worst case. The paired audit still wins by more at larger K. Its ratio to the cheapest competitor is 0.46, 0.38 and 0.33 (Figure 8b), with no miss in 75 runs.

Table 2. Paired audit on the 185 held-out pools, relative to the cheapest competing certified audit on each pool (median ratio of five-run means). Brackets: 95% bootstrap interval over the eight answer sets. Labels are compared with the cheapest competitor in labels.
<table><tr><td> $K$ </td><td>answers</td><td>interval</td><td>pools cheaper</td><td>labels</td></tr><tr><td>64</td><td>0.74</td><td>[0.67, 0.88]</td><td> $1 7 5 / 1 8 5$ </td><td>0.73</td></tr><tr><td>256</td><td>0.66</td><td>[0.59, 0.73]</td><td>175/185</td><td>0.68</td></tr><tr><td>1024</td><td>0.53</td><td>[0.47, 0.58]</td><td>173/185</td><td>0.61</td></tr></table>

![](images/2d769b1c5fdf64c4fa2c91f6116396e384219b9e07d58d1cd23791550031b5ed.jpg)  
answers, paired audit / cheapest competitor

(b) what each ingredient buys  
![](images/6af7446d299d41ac003dd228d4898fca4ab3f5fce0089afe6d9f5677bb61e29f.jpg)  
Figure 7. (a) Distribution of the ratio of generated answers, paired audit to the cheapest competing certified audit, over the 185 held-out pools. Left of the dashed line the paired audit is cheaper than every competitor on that pool. (b) Median ratio for the designs of the ablation in Section 8.3; every design uses the same retirement of resolved budgets.

## 8.5 The cost law

The audit’s cost follows the three terms of Theorem 9. Across 1,085 pool, horizon and precision settings, a nonnegative least-squares fit of $a K M + L ( b K / \varepsilon + c \Sigma _ { \mathrm { w } } / \varepsilon ^ { 2 } )$ with $L = \log ( 2 K / \alpha _ { S } )$ has $R ^ { 2 } = 0 . 9 9 5$ and median relative error 4.7% (Figure 9b). Replacing $\Sigma _ { \mathrm { w } }$ by K max<sub>k</sub> $\bar { v } _ { k }$ the worst-case envelope of the same variance, lowers $R ^ { 2 }$ to 0.906 and raises the median error to 18.1%, so the tail sum is the right quantity. Fitted at K = 64 alone, the law predicts the settings with K = 256, K = 1024 and $\varepsilon = 1 / 6 4$ with median errors 2.3%, 5.2% and 7.0%, and those with $\varepsilon = 1 / 1 6 .$ where the start-up dominates, with 11.3%.

The step that matters most is prospective (Figure 9a). An earlier fit, on two audits per pool at $K = 6 4$ , gave $N \approx 2 . 1 6 K M + 1 5 . 1 3 K / \varepsilon + 1 2 . 8 8 \Sigma _ { \mathrm { w } } / \varepsilon ^ { 2 } \ : ( R ^ { 2 } = 0 . 9 5 8$ median error 2.7%), and these coeficients went into the analysis of the MMLU-Pro study before any of its answers was generated. Given the study’s $\Sigma _ { \mathrm { w } } = 2 . 5 7$ , computed from its separate reference stream, the law predicted 78,656 answers; the audit used 79,133. Resampling the 100 question-level reference curves 2,000 times puts the prediction’s 5–95% range at 69,096 to 89,152 answers.

## 8.6 A newly generated study

We fixed the protocol first: 100 MMLU-Pro questions (Wang et al., 2024) in mathematics, physics, chemistry and engineering; the gpt-6-luna model at temperature 1 with eight answers per request; the mean token logprobability of an answer as its verifier score; three independent audit streams and a separate reference stream. The model produced 716,384 answers for \$17.24, 409,448 of them for the reference. Every audit reads its stream in generation order, so a run costs exactly what it would have cost with answers generated on demand. Table 3 gives the cost of each certified curve and Figure 10 the bands of the first stream.

The best-of-k curve costs 79,133 answers, 41% of the fixed exact-binomial design, and \$1.91 against \$4.61. Pass@k and majority voting are cheaper still with the all-subsets audit, and because the grader is deterministic, fewer than 400 distinct grader calls certify each curve. All 18 bands contain the reference curves at every budget. The closest band edge is a majority-voting band at budget 55, $5 . 3 \times 1 0 ^ { - 4 }$ from the reference value, below the reference’s own bootstrap standard error of $7 . 9 \times 1 0 ^ { - 4 }$ there.

(a) precision, all 217 pools  
![](images/75b0319294fa2b6f4ade9be47399485eb866d9b0bdc07d89152fc73edb3956d7.jpg)

(b) the saving grows with K  
![](images/24611f76489b126282e6c4a139f1740afc2e3a8255e28b10a399d065c941729b.jpg)

![](images/fe05d4e0813adb305b04bfbc1044fe4e0fae9b40037f5ff45e7ff57a5ad5ead9.jpg)  
Figure 8. (a) Cost ratio at K = 64 for every stored pool and three precisions, against $M \varepsilon ^ { 2 } ;$ the start-up of two complete rounds decides the ratio at the right. (b) Median ratio on the held-out pools, whose questions have 100 stored answers, and on a pool with $4 { , } 0 0 0$ answers per question built from the MMLU-Pro reference stream. (c) $\Sigma _ { \mathrm { w } }$ as a fraction of its worst case $K / 4$ in the same two settings (held-out: median).

Table 3. MMLU-Pro study, $M = 1 0 0 , K = 6 4 , \varepsilon = 1 / 3 2 , \delta = 0 . 0 5 \colon$ means over three audit streams. Grader calls reuse one grade per distinct question–answer pair; dollars are the generation cost at the price we paid per answer. The designs below the rule certify best-of-k with fixed sample sizes.
<table><tr><td>curve</td><td>audit</td><td>answers</td><td>labels</td><td>grader calls</td><td>cost ($)</td></tr><tr><td>best-of-k</td><td>paired</td><td>79,133</td><td>6,160</td><td>256</td><td>1.91</td></tr><tr><td>best-of-k</td><td>all-subsets</td><td>78,933</td><td>55,590</td><td>385</td><td>1.91</td></tr><tr><td>pass@k</td><td>paired</td><td>60,467</td><td>5,943</td><td>285</td><td>1.46</td></tr><tr><td>pass@k</td><td>all-subsets</td><td>49,067</td><td>49,067</td><td>385</td><td>1.18</td></tr><tr><td>majority voting</td><td>paired</td><td>62,267</td><td>58,999</td><td>356</td><td>1.50</td></tr><tr><td>majority voting</td><td>all-subsets</td><td>49,067</td><td>49,067</td><td>385</td><td>1.18</td></tr><tr><td>best-of-k</td><td>fixed exact binomial</td><td>192,000</td><td></td><td></td><td>4.61</td></tr><tr><td>best-of-k</td><td>record design (Fitas, 2026)</td><td>257,216</td><td></td><td></td><td>6.18</td></tr><tr><td>best-of-k</td><td>fixed Hoeffding</td><td>262,400</td><td></td><td></td><td>6.31</td></tr></table>

What the bands say. The three curves tell diferent stories. Pass@k climbs from 0.736 to 0.960, and in every stream its bands rule out every budget below 12 as a best budget (Proposition 1(i)). Best-of-k, with the model’s own log-probability as the verifier, gains only three points, from 0.736 to 0.767 on the reference curve, and majority voting reaches 0.787. Most of what sampling could buy on these questions is out of the verifier’s reach, as Stroebl et al. (2026) found for imperfect verifiers, and the bands certify it. In every stream, whichever audit certified each curve, the pass@k band at budget 64 lies at least 0.125 above the best-of-k band, so by Proposition 1(iv) pass@64 exceeds best-of-64 accuracy by more than 12 points with probability at least 90%.

(a) fixed before the study, K = 64  
![](images/74f3c322bcd1befa34163f46ce14ffe93a8401dc1c97600724b74390f370ef17.jpg)

(b) one law, 1,085 settings  
![](images/c847bf832a69aecb87b093bc9af7f764291f27753186b9e60c059c883d09b7d4.jpg)  
Figure 9. The cost law. (a) $N \approx 2 . 1 6 K M + 1 5 . 1 3 K / \varepsilon + 1 2 . 8 8 \Sigma _ { \mathrm { w } } / \varepsilon ^ { 2 }$ , fitted to two paired audits per stored pool at $K = 6 4$ and $\varepsilon = 1 / 3 2 \ ( R ^ { 2 } = 0 . 9 5 8 )$ , against measured cost. The star is the new MMLU-Pro study: the coeficients were fixed before any of its answers existed, and its $\Sigma _ { \mathrm { w } }$ comes from the separate reference stream. (b) One law $a K M + L ( b K / \varepsilon + c \Sigma _ { \mathrm { w } } / \varepsilon ^ { 2 } )$ for all horizons and precisions $( R ^ { 2 } = 0 . 9 9 5 )$

![](images/7f8b5aa69859b86f8083af68f4e1eb300c347ae461c897a1e97ddbf7bb63b345.jpg)

![](images/10499e7df7c36295d0a80c9030c6f6b99cb7ccaa8ac7cfa54f40a79ef295244d.jpg)

(c) majority voting  
![](images/39585810184b5e36311abea408327a068d51e895279c4ec0899785ec1e7a0013.jpg)  
Figure 10. Certified curves for 100 MMLU-Pro questions (first audit stream): simultaneous bands of half-width at most 1/32 at level 95% from the paired audit and the all-subsets audit, with reference curves estimated from 409,448 separately generated answers. Shaded in (b): budgets that the pass@k bands rule out as a best budget.

## 9 Related work

Test-time scaling. Repeated sampling with a selection rule is a standard way to trade computation for accuracy. Pass@k was introduced for code generation with an unbiased estimator (Chen et al., 2021), large-scale sampling and filtering produced competition-level programs (Li et al., 2022), best-of-n selection with a reward model has an unbiased estimator of its own (Nakano et al., 2021; Gao et al., 2023), and majority voting over sampled reasoning paths is self-consistency (Wang et al., 2023). Brown et al. (2024) report how coverage grows with the number of samples, and Saad-Falcon et al. (2025) combine weak verifiers to close part of the gap between generation and verification. Stroebl et al. (2026) show that verifier errors bound what resampling can gain, which is why the budget choice matters. Kazdan et al. (2025) predict pass@k at large k from a beta-binomial model of per-question success rates and give harder questions more samples, and Chen et al. (2026) stop parallel reasoning traces once the majority vote is unlikely to change. These works report or predict curves; they do not certify them.

Statistics of model evaluation. Miller (2024) splits the variance of benchmark accuracy by the law of total variance and recommends several answers per question. Its target is a population of unseen questions, which keeps the between-question term that a fixed list removes. Prediction-powered inference (Angelopoulos et al., 2023), active testing (Kossen et al., 2021) and anytime-valid evaluation with e-processes (Zhou et al., 2026) save labels when estimating a single score; bands for tuning curves (Lourie et al., 2024) cover the best score among trials, not the correctness of a verifier’s choice. Here the object is a whole curve produced by paths of generated answers, and generation, not labeling, is the main cost.

Sequential inference and sampling design. Our bands are confidence sequences (Howard et al., 2021; Waudby-Smith and Ramdas, 2024) built from the penalized exponential inequality of Fan et al. (2015) and Ville’s inequality (Ville, 1939). Estimating a variance from squared diferences of pairs is classical (Wang and Ramdas, 2025); the paired inequality is its exact form for Bernoulli draws. Balanced rounds are stratified sampling with questions as strata, and the allocation behind Γ is Neyman allocation for a least favorable mixture of budgets (Cochran, 1977), learned from a pilot as in adaptive stratified sampling (Carpentier et al., 2015). The upper bound of Theorem 4 is multilevel Monte Carlo (Giles, 2015; Rhee and Glynn, 2015) with a coupling of nested winners. The all-subsets estimates are U-statistics (Hoefding, 1948), and the label counts rest on the theory of records (Rényi, 1962). The lower bounds use the change-of-measure argument for adaptive experiments (Kaufmann et al., 2016) and the guess-and-verify argument against instance optimality of Narayanan et al.

(2024).

The closest work. Fitas (2026) studies the same curve for a population of questions. That paper proves that auditing all widths up to N needs labels in proportion to 1 + log N and generated answers in proportion to N, gives a known-percentile design and a record design with simultaneous bands, and asks whether simultaneous confidence needs an extra logarithm. Our label lower bounds reuse its disjoint score scales. For fresh questions and fixed-width bands under deterministic caps, Proposition 2 shows that simultaneity costs generated answers no extra logarithm, and for one question and mean squared error, Theorem 13 gives the leading constants and the exact price of unknown percentiles. The fixed-list results of Sections 4–6 have no counterpart there.

## 10 Discussion

A certified scaling curve on a fixed benchmark costs three things: calibration of the score tail, the identity of the questions, and within-question noise summed along the curve. The last term is where a random audit overpays, because most of the variance it pays for lies between questions, and on measured benchmarks the within-question sum is a fraction of its worst case. Revisiting questions in balanced pairs and retiring budgets as they resolve turns that into a practical audit, and a cost law with three terms tells in advance what it will spend.

Theorem 6 marks where the next saving lies. At a single benchmark the within-question term shrinks to Γ, about a tenth of $\Sigma _ { \mathrm { w } }$ on our data, and most of that saving comes from not spending answers on questions the model almost always gets right or almost always gets wrong. An audit that learns this allocation reaches Γ as the precision grows. At the precisions of Section 8 its pilot costs more than the paired audit’s whole bill, and Proposition 5 rules out a free version, so the open problem is an audit that learns the allocation as it goes, without a separate pilot, and pays for learning only what it uses.

The results have a definite scope. The fixed-list target is the listed questions; for a population the betweenquestion variance returns, and a random panel (Section 7.3) is the way back. The paired audit pays a start-up of two complete rounds, which is why it loses at coarse precision on a large benchmark, and the cost law says in advance when that happens. The price of unknown percentiles is a mean-squared-error statement for one question. Dependent trajectories are covered for one fixed generation policy.

In practice the recommendation is short. Report a simultaneous band, not a curve; choose budgets from the band with Proposition 1; revisit every question in pairs of rounds; and use the cost law with a rough value of $\Sigma _ { \mathrm { w } }$ to budget the audit before generating anything. On our 100-question MMLU-Pro study with 64 budgets, the best-of-k band cost \$1.91 of generation.

Data. The stored pools come from public releases: the Weaver collection on Hugging Face (Saad-Falcon et al., 2025), released under the MIT license according to its dataset cards; the execution results released with the CodeRM repository (Ma et al., 2025); and the CodeContests test split (Li et al., 2022), whose DeepMind-provided non-code materials are CC BY 4.0 (third-party materials may have separate terms). MMLU-Pro (Wang et al., 2024) is released under the MIT license. The MMLU-Pro answers were generated for this study with the protocol of Appendix G.7.

Use of AI tools. Generative AI tools assisted with code implementation, experimental design and analysis, literature review, figure preparation, and drafting and editing this manuscript. The authors are responsible for the claims, proofs, results, references, and final text.

## References

Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., and Zrnic, T. (2023). Prediction-powered inference. Science, 382(6671):669–674.

Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., Ré, C., and Mirhoseini, A. (2024). Large language monkeys: Scaling inference compute with repeated sampling. arXiv:2407.21787.

Carpentier, A., Munos, R., and Antos, A. (2015). Adaptive strategy for stratified Monte Carlo sampling. Journal of Machine Learning Research, 16:2231–2271.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. (2021). Evaluating large language models trained on code. arXiv:2107.03374.

Chen, W., Li, P., Liu, M., Su, W., and Xie, T. (2026). MARS: Margin-adversarial risk-controlled stopping for parallel LLM test-time scaling. arXiv:2606.12935.

Clopper, C. J. and Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika, 26(4):404–413.

Cochran, W. G. (1977). Sampling Techniques. Wiley, New York, third edition.

Dvoretzky, A., Kiefer, J., and Wolfowitz, J. (1956). Asymptotic minimax character of the sample distribution function and of the classical multinomial estimator. The Annals of Mathematical Statistics, 27(3):642–669.

Fan, X., Grama, I., and Liu, Q. (2015). Exponential inequalities for martingales with applications. Electronic Journal of Probability, 20, no. 1, 1–22.

Fitas, R. (2026). The geometry of AI validation: From structural blindness to reusable audits. arXiv:2608.21496v2. Version 2, 11 September 2026.

Gao, L., Schulman, J., and Hilton, J. (2023). Scaling laws for reward model overoptimization. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 10835–10866.

Giles, M. B. (2015). Multilevel Monte Carlo methods. Acta Numerica, 24:259–328.

Gill, R. D. and Levit, B. Y. (1995). Applications of the van Trees inequality: A Bayesian Cramér–Rao bound. Bernoulli, 1(1–2):59–79.

Hoefding, W. (1948). A class of statistics with asymptotically normal distribution. The Annals of Mathematical Statistics, 19(3):293–325.

Hoefding, W. (1956). On the distribution of the number of successes in independent trials. The Annals of Mathematical Statistics, 27(3):713–721.

Hoefding, W. (1963). Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30.

Howard, S. R., Ramdas, A., McAulife, J., and Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. The Annals of Statistics, 49(2):1055–1080.

Kaufmann, E., Cappé, O., and Garivier, A. (2016). On the complexity of best-arm identification in multiarmed bandit models. Journal of Machine Learning Research, 17(1):1–42.

Kazdan, J., Schaefer, R., Allouah, Y., Sullivan, C., Yu, K., Levi, N., and Koyejo, S. (2025). Eficient prediction of pass@k scaling in large language models. arXiv:2510.05197.

Kossen, J., Farquhar, S., Gal, Y., and Rainforth, T. (2021). Active testing: Sample-eficient model evaluation. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 5753–5763.

Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago, A., et al. (2022). Competition-level code generation with AlphaCode. arXiv:2203.07814.

Liu, J., Xia, C. S., Wang, Y., and Zhang, L. (2023). Is your code generated by ChatGPT really correct? rigorous evaluation of large language models for code generation. In Advances in Neural Information Processing Systems.

Lourie, N., Cho, K., and He, H. (2024). Show your work with confidence: Confidence bands for tuning curves. In Proceedings of the 2024 Conference of the North American Chapter ofthe Association for Computational Linguistics, pages 3455–3472.

Ma, Z., Zhang, X., Zhang, J., Yu, J., Luo, S., and Tang, J. (2025). Dynamic scaling of unit tests for code reward modeling. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6917–6935.

Massart, P. (1990). The tight constant in the Dvoretzky– Kiefer–Wolfowitz inequality. The Annals of Probability, 18(3):1269–1283.

McDiarmid, C. (1989). On the method of bounded differences. In Surveys in Combinatorics, volume 141 of London Mathematical Society Lecture Note Series, pages 148–188. Cambridge University Press.

Miller, E. (2024). Adding error bars to evals: A statistical approach to language model evaluations. arXiv:2411.00640.

Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain, S., Kosaraju, V., Saunders, W., et al. (2021). WebGPT: Browser-assisted questionanswering with human feedback. arXiv:2112.09332.

Narayanan, S., Rozhoň, V., Tětek, J., and Thorup, M. (2024). Instance-optimality in I/O-eficient sampling and sequential estimation. In 2024 IEEE 65th Annual Symposium on Foundations of Computer Science (FOCS), pages 658–688.

Rényi, A. (1962). Egy megfigyeléssorozat kiemelkedő elemeiről. Magyar Tudományos Akadémia Matematikai és Fizikai Osztályának Közleményei, 12:105–121. English title: On the extreme elements of a series of observations.

Rhee, C.-H. and Glynn, P. W. (2015). Unbiased estimation with square root convergence for SDE models. Operations Research, 63(5):1026–1043.

Saad-Falcon, J., Buchanan, E. K., Chen, M., Huang, T.-H., McLaughlin, B., Bhathal, T., Zhu, S., Athiwaratkun, B., Sala, F., Linderman, S., Mirhoseini, A., and Ré, C. (2025). Weaver: Shrinking the generationverification gap by scaling compute for verification. In Advances in Neural Information Processing Systems.

Stepanov, A. (2021). On the mathematical theory of records. Communications in Mathematics, 29:151–162.

Stroebl, B., Kapoor, S., and Narayanan, A. (2026). The limits of inference scaling through resampling. In International Conference on Learning Representations.

Tsybakov, A. B. (2009). Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer.

van der Vaart, A. W. and Wellner, J. A. (1996). Weak Convergence and Empirical Processes: With Applications to Statistics. Springer, New York.

Varah, J. M. (1975). A lower bound for the smallest singular value of a matrix. Linear Algebra and its Applications, 11(1):3–5.

Ville, J. (1939). Étude critique de la notion de collectif. Gauthier-Villars, Paris.

Wang, H. and Ramdas, A. (2025). Sharp matrix empirical Bernstein inequalities. In Advances in Neural Information Processing Systems.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. (2023). Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., Li, T., Ku, M., Wang, K., Zhuang, A., Fan, R., Yue, X., and Chen, W. (2024). MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track.

Waudby-Smith, I. and Ramdas, A. (2024). Estimating means of bounded random variables by betting. Journal of the Royal Statistical Society Series B, 86(1):1–27.

Zhou, Z., Ye, Z., Chen, Z., Li, B., and Liu, F. (2026). CELEUS: Certifiable and eficient LLM evaluation via e-processes. arXiv:2606.20820.

## Appendix

The appendix gives the proofs that do not appear in the main text (Appendices A–F) and the details of the experiments (Appendix G).

## A Proofs: conventions

Throughout, a valid audit returns intervals of width at most 2ε that cover every $\theta _ { k }$ at once with probability at least $1 - \delta$ at every law. Two laws $P , Q$ with $| \theta _ { k } ( P ) - \theta _ { k } ( Q ) | > 2 \varepsilon$ for some k are separated by every valid audit: the event that its interval at budget k contains $\theta _ { k } ( P )$ has probability at least $1 - \delta$ under $P$ and at most $\delta$ under $Q ,$ so this binary test forces the Kullback–Leibler divergence between the two laws of the whole transcript to be at least $\kappa _ { \delta } = \mathrm { k l } ( 1 - \delta , \delta )$ (Kaufmann et al., 2016). The audit’s own choices carry no information about the law, so the chain rule charges each observation its conditional divergence. When Q changes only the answer laws $P _ { x }$ , the transcript divergence is therefore at most $\begin{array} { r l } { \sum _ { x } C ( x ) \operatorname { K L } ( P _ { x } \| Q _ { x } ) } \end{array}$ , where $C ( x )$ is the expected number of answers generated at x under $P ;$ revealing every correctness value for free can only increase it. We refer to this as testing.

## B Proof of Proposition 2

In this appendix every path starts at a fresh question X drawn independently from a population, and the target is $\theta _ { k } = \mathbb { E } _ { X } p _ { k } ( X )$ . The midpoint of any valid width-2ε interval estimates $\theta _ { k }$ to within $\varepsilon ,$ so a valid audit yields estimates with uniform error at most ε with probability $1 - \delta$ . Caps are deterministic, and an audit that reaches a cap without an answer has failed.

## B.1 A hard family

Work with uniform score percentiles and the coordinate $t = - \log u$ , in which the winner of k answers has density $k e ^ { - k t }$ . Put $J = 1 + \lfloor \log _ { 1 6 } K \rfloor , k _ { j } = 1 6 ^ { j }$ and $I _ { j } = [ 1 / k _ { j } , 2 / k _ { j } ]$ ; these intervals are disjoint. For signs $v \in \{ - 1 , 1 \} ^ { J }$ and $a = 8 \varepsilon \leq 1 / 4$ , let correctness at score coordinate t be Bernoulli $( 1 / 2 + a v _ { j } )$ when $t \in I _ { j }$ and Bernoull $. ( 1 / 2 )$ otherwise. At the budgets $k _ { i }$

$$
\begin{array} { r } { \theta ( v ) = \frac 1 2 \mathbf { 1 } + a A v , \qquad A _ { i j } = e ^ { - k _ { i } / k _ { j } } - e ^ { - 2 k _ { i } / k _ { j } } . } \end{array}\tag{12}
$$

The diagonal entries are $e ^ { - 1 } - e ^ { - 2 } > 0 . 2 3 2 5$ . Using $e ^ { - z } - e ^ { - 2 z } \leq z$ , the entries for finer bands sum to at most $\textstyle \sum _ { d > 1 } 1 6 ^ { - d } = 1 / 1 5$ in each row, and the entries for coarser bands to at most $e ^ { - 1 6 } / ( 1 - e ^ { - 1 6 } )$ . Every row is therefore diagonally dominant by more than 0.16, and $\| A ^ { - 1 } \| _ { \infty } \leq 6 . 2 5$ (Varah, 1975). An estimate of the curve with error at most ε recovers every sign by rounding $a ^ { - 1 } A ^ { - 1 } ( \widehat { \theta } - \mathbf { 1 } / 2 )$ , because the coordinate error is at most $6 . 2 5 \varepsilon / a < 1$

## B.2 Correctness queries

Give the signs a uniform prior and let the auditor choose, adaptively, a percentile at which to see a fresh correctness label, up to m times; this is at least as informative as querying labels of generated answers. Action probabilities given the past carry no information about the signs, and a label in band $I _ { j }$ depends only on $v _ { j }$ , so the posterior factorizes over coordinates. Let $L _ { j }$ be the posterior log-odds of $v _ { j } , e _ { j } = 1 / ( 1 + \stackrel {  } { e } ^ { | L _ { j } | } ) \geq \frac { 1 } { 2 } e ^ { - | L _ { j } | }$ and $\begin{array} { r } { p ( T ) = \prod _ { j } ( 1 - e _ { j } ) } \end{array}$ the largest posterior probability of a sign vector. A valid audit decodes all signs with Bayes probability at least $1 - \delta ,$ so $\mathbb { E } p ( T ) \geq 1 - \delta$ , and by Markov’s inequality $p ( T ) \geq 1 - 2 \delta$ with probability at least $1 / 2 .$ . On that event $\begin{array} { r } { \sum _ { j } e _ { j } \le - \log p ( T ) \le 4 \delta } \end{array}$ and $\begin{array} { r } { \sum _ { j } e ^ { - | L _ { j } | } \leq 8 \delta } \end{array}$ , and Jensen’s inequality gives

$$
\mathbb { E } \sum _ { j } | L _ { j } | \geq \frac { J } { 2 } \log \frac { J } { 8 \delta } .\tag{13}
$$

Let $N _ { j }$ count the labels seen in band $j , \ell = \log \{ ( 1 / 2 + a ) / ( 1 / 2 - a ) \} \le$ 8a and $d = 2 a \ell \leq 1 6 a ^ { 2 }$ . Under the true signs, $v _ { j } L _ { j } = d N _ { j } + M _ { j }$ with $M _ { j }$ a martingale and $\mathbb { E } M _ { j } ^ { 2 } \le \ell ^ { 2 } \mathbb { E } N _ { j } ,$ , and $\textstyle \sum _ { j } N _ { j } \leq m$ . Hence

$$
\mathbb { E } \sum _ { j } | L _ { j } | \leq 1 6 a ^ { 2 } m + 8 a \sum _ { j } { \sqrt { \mathbb { E } N _ { j } } } \leq 1 6 a ^ { 2 } m + 8 a { \sqrt { J m } } .\tag{14}
$$

With $b = \log \{ J / ( 8 \delta ) \} \ge \log 2$ and $z = a ^ { 2 } m / J ,$ , the two displays give $1 6 z + 8 \sqrt { z } \geq b / 2$ , which fails when $z < b / 1 0 2 4$ Therefore

$$
m \geq \frac { J } { 6 5 5 3 6 \varepsilon ^ { 2 } } \log \frac { J } { 8 \delta } .
$$

The argument allows adaptive allocation and adaptive stopping under a deterministic label cap.

## B.3 Question visits

Now let questions difer. Under signs $v ,$ question i carries independent bits $X _ { i j }$ ∼ Bernoul $\mathrm { i } ( 1 / 2 + a v _ { j } ) , j < J ;$ an answer whose score coordinate lies in $I _ { j }$ has correctness $X _ { i j }$ , and every other answer an independent fair bit. Answer pairs stay independent and identically distributed given the question, and averaging over questions gives exactly (12). Reveal every bit of every visited question for free. The data then split over coordinates, and each coordinate is a test between Bernoull $( 1 / 2 + a )$ and Bernoul $\mathrm { i } ( 1 / 2 - a )$ from n draws. By the Bretagnolle–Huber inequality (Tsybakov, 2009) and kl{Bernoulli $( 1 / 2 + a )$ , Bernoulli $( 1 / 2 - a ) \} = 2 a \ell \leq 1 6 a ^ { 2 }$ , the Bayes error of one coordinate is at least $\begin{array} { r } { e _ { n } = \frac { 1 } { 4 } e ^ { - 1 6 a ^ { 2 } n } } \end{array}$ . Decoding all coordinates succeeds with probability at most $( 1 - e _ { n } ) ^ { J } ;$ requiring at least $1 - \delta$ gives $e _ { n } \leq 1 - ( 1 - \delta ) ^ { 1 / J } \leq 2 \delta / J$ , hence

$$
n \geq { \frac { 1 } { 1 0 2 4 \varepsilon ^ { 2 } } } \log { \frac { J } { 8 \delta } } .
$$

An actual audit sees no more per question, so the bound holds for it.

## B.4 Generated answers

Two laws that difer only for score coordinates in the band $B = [ 1 / K , 2 / K ]$ , where correctness is $\mathrm { B e r n o u l l i } ( 1 / 2 \pm a )$ have best-of-K targets that difer by 2a $\textstyle \int _ { 1 / K } ^ { 2 / K } K e ^ { - K t } d t = 2 a ( e ^ { - 1 } - e ^ { - 2 } ) > 2 \varepsilon$ when $a = 8 \varepsilon$ . A generated answer lands in B with probability at most $1 / K$ , and its label then has divergence at most $1 6 a ^ { 2 }$ . Reveal every label for free. With identical questions, the next answer has the same law whatever the audit does, so over a deterministic cap of N answers, padded with uninformative draws after stopping, the transcript divergence is at most $1 6 N a ^ { 2 } / K$ , and the Bretagnolle–Huber inequality gives $N \ge c K \varepsilon ^ { - 2 } \log ( 1 / \delta )$ . This bound uses one hard bit. A union over the J bands does not raise it, because a single stream of answers visits all bands at once, and we claim no extra log J for generated answers.

## B.5 The record audit

Records. Order the K answers of a path by generation time and let $R _ { k }$ be the rank of answer k among the first k. The map from orderings to $( R _ { 1 } , \ldots , R _ { K } )$ is a bijection onto $\textstyle \prod _ { k } \{ 1 , \dots , k \}$ , so when scores do not tie the ranks are independent and uniform, and the record indicators $\mathbf { 1 } \{ R _ { k } = k \}$ are independent Bernoulli $( 1 / k )$ , whatever the question (Rényi, 1962; Stepanov, 2021). With ties, the strict records are a subset of the records of the order broken by independent uniform keys, which have this law, so every bound below on the number of queries still holds. Querying the first answer and every later record, and carrying the last queried label forward, gives $W _ { k }$ for every k. For n paths the average $\widehat { \theta } _ { k }$ of $W _ { k }$ is unbiased with variance at most $1 / ( 4 n )$ , and the number of queries $Q$ has mean n ${ \cal { H } } _ { K }$ and variance at most $n H _ { K } ;$ by Bernstein’s inequality $Q \leq n H _ { K } + \sqrt { 2 n H _ { K } x } + 2 x / 3$ with probability $1 - e ^ { - x }$

Uniform accuracy. For $k \leq \ell$ the winners at k and ℓ coincide when the best of the first ℓ answers lies among the first $k ,$ which has probability at least $k / \ell ;$ so $\mathbb { E } ( W _ { k } - W _ { \ell } ) ^ { 2 } \le 1 - k / \ell \le \log ( \ell / k )$ . Cut [0, log K] into pieces of length at most $r ^ { 2 }$ and group the budgets whose logarithms fall in the same piece. For a group, the pointwise minimum and maximum of its $W _ { k }$ bracket every member, and $\mathbb { E } ( \operatorname* { m a x } - \operatorname* { m i n } ) ^ { 2 } \leq r ^ { 2 } \colon$ if the earliest and the latest budget of the group select the same answer, every budget between them selects it too, so the bracket is nonzero only if the winner changes inside the group, which has probability at most $1 - k _ { \operatorname* { m i n } } / k _ { \operatorname* { m a x } } \le r ^ { 2 }$ . So the class $\{ W _ { k } : k \le K \}$ has $L _ { 2 }$ bracketing number at most $2 + \lceil \log K / r ^ { 2 } \rceil$ . The bracketing maximal inequality (van der Vaart and Wellner, 1996, Theorem 2.14.2) with envelope 1 gives E sup<sub>k</sub> $| \widehat { \theta } _ { k } - \theta _ { k } | \leq C \sqrt { \log ( 2 + \log K ) / n }$ , since $\begin{array} { r } { \int _ { 0 } ^ { 1 } \sqrt { \log ( 2 + \log K / r ^ { 2 } ) } d r \leq \sqrt { \log ( 2 + \log K ) } + \int _ { 0 } ^ { 1 } \sqrt { 2 \log ( 1 / r ) } } \end{array}$ dr. Changing one path moves the supremum by at most $1 / n$ , so McDiarmid’s inequality adds $\sqrt { \log ( 1 / \delta ) / ( 2 n ) }$ with probability $1 - \delta$

Nested paths. Let $D = \lceil \log _ { 2 } K \rceil , b _ { 0 } = 1 , b _ { j } = \operatorname* { m i n } ( 2 ^ { j } , K )$ , and

$$
\delta _ { j } = \frac { \delta } { 4 ( D + 1 ) } + \delta 2 ^ { j - D - 2 } , \qquad n _ { j } = \bigl \lceil C \varepsilon ^ { - 2 } \log ( 2 / \delta _ { j } ) \bigr \rceil , \qquad 0 \le j \le D .
$$

The first $n _ { j }$ paths are extended to length $b _ { j }$ . The budgets in $( b _ { j - 1 } , b _ { j } ]$ span a logarithmic range of at most log 2, so their bracketing integral is a constant, and the argument above bounds their uniform error by ε except with probability $\delta _ { j }$ for a suitable C. The blocks share paths, but a union bound needs no independence, and $\textstyle \sum _ { i } \delta _ { j } = \delta ( 3 / 4 - 2 ^ { - D - 2 } ) < { \bar { 3 } } \delta / 4$ Because the $\delta _ { j }$ increase, the $n _ { j }$ decrease, and the number of paths is $n _ { 0 } \le 1 + C \varepsilon ^ { - 2 } \operatorname { l o g } \{ 8 ( D + 1 ) / \delta \}$ . With $d _ { 0 } = 1$ and $d _ { j } = b _ { j } - b _ { j - 1 }$ , the number of answers is $\begin{array} { r } { N _ { 0 } = \sum _ { j } d _ { j } n _ { j } } \end{array}$ . Since log $\cdot ( 2 / \delta _ { j } ) \le \log ( 8 / \delta ) + ( D - j )$ log 2 and $\begin{array} { r } { \sum _ { j } d _ { j } ( D - j ) = K - 1 } \end{array}$ when $K = 2 ^ { D }$ (and less than 2K otherwise, the last block having coeficient zero),

$$
N _ { 0 } \leq K + C \varepsilon ^ { - 2 } \{ K \log ( 8 / \delta ) + 2 K \log 2 \} \leq K + C K \varepsilon ^ { - 2 } \log ( 3 2 / \delta ) ,
$$

with no hidden log log K. Path lengths are fixed in advance, so record indicators stay independent; the expected number of queries is $\begin{array} { r } { \mu = n _ { 0 } + \sum _ { j > 1 } n _ { j } ( H _ { b _ { j } } - H _ { b _ { j - 1 } } ) \le n _ { 0 } H _ { K } } \end{array}$ . The audit refuses any query beyond $m _ { 0 } = \lceil \mu + \sqrt { 2 \mu \log ( 4 / \delta ) } +$ ${ \frac { 2 } { 3 } } \log ( 4 / \delta ) ]$ , which Bernstein’s inequality exceeds with probability at most $\delta / 4 ;$ without overflow the capped and uncapped audits agree. The total failure probability is below $\delta ,$ and the three caps are $n _ { 0 } = O \{ \varepsilon ^ { - 2 } \log ( J / \delta ) \}$ ， $m _ { 0 } = { \cal O } ( n _ { 0 } H _ { K } ) = { \cal O } \{ J \varepsilon ^ { - 2 } \log ( J / \delta ) \}$ } and $N _ { 0 } = O \{ K \varepsilon ^ { - 2 } \log ( 1 / \delta ) \}$ }. At $K = 1$ everything reduces to estimating a binomial mean.

## C Proof of Theorem 4

## C.1 The upper bound

Let $K \geq 2 , J = \lceil \log _ { 2 } K \rceil , a _ { \ell } = 2 ^ { \ell - 1 }$ and $b _ { \ell } = \operatorname* { m i n } ( 2 ^ { \ell } , K )$ for $\ell = 1 , \ldots , J$ . The audit spends $\delta / 2$ on an estimate of $\theta _ { 1 }$ to half-width $\varepsilon / 2$ and $\delta / 2$ on corrections for $\theta _ { k } - \theta _ { 1 }$ to total half-width $\varepsilon / 2$

The first budget. Two valid procedures run side by side, each with half of the error budget, and the audit stops with whichever reaches half-width $\varepsilon / 2$ first. The first visits the questions in independent random orders, one answer per question, and uses a fixed-look Chernof–Kullback–Leibler interval. At any fixed number $n = a M + b$ of answers, the moment generating function of the success count is at most that of $\mathrm { B i n } ( n , \theta _ { 1 } )$ : for the a complete orders by the arithmetic–geometric mean inequality applied to $\begin{array} { r } { \prod _ { x } \{ 1 + p _ { 1 } ( x ) ( e ^ { \lambda } - 1 ) \} } \end{array}$ , and for the b answers of an incomplete random order by Maclaurin’s inequality for elementary symmetric polynomials. Chernof intervals are therefore valid before the first complete round, and a fixed Hoefding horizon caps this procedure at $O ( \varepsilon ^ { - 2 } \log ( 1 / \delta ) )$ answers. Domination of moment generating functions alone does not give exact binomial tails, which is why this procedure uses Chernof intervals. The second procedure runs the paired sequence of Section 6 on complete pairs of rounds with a geometric grid of bets. With $v _ { 1 } = \bar { v } _ { 1 }$ and Q logarithmic in $1 / ( \varepsilon \delta )$ , a grid bet between $\varepsilon / \{ 8 ( v _ { 1 } + \varepsilon ) \}$ and $\varepsilon / \{ 4 ( v _ { 1 } + \varepsilon ) \}$ reaches half-width $\varepsilon / 2$ within $2 M + Q ( \varepsilon ^ { - 1 } + v _ { 1 } \varepsilon ^ { - 2 } )$ rows on a high-probability event: over n rows the disagreement count D has mean $n v _ { 1 }$ and ${ \mathbb P } \{ D > 2 n v _ { 1 } + 2 u \} \le e ^ { - u }$ , and substituting into the radius with $u = \log ( 1 / \eta )$ gives the count. The deterministic cap bounds the contribution of the complementary event by $\eta O ( \varepsilon ^ { - 2 } \log ( 1 / \delta ) )$ ; take $\eta = \varepsilon$ The cost of the first budget is at most twice the smaller of the two, $\widetilde { O } ( M _ { \varepsilon } + 1 / \varepsilon + \bar { v } _ { 1 } / \varepsilon ^ { 2 } )$ answers and correctness queries, without an additive M when $M \gg \varepsilon ^ { - 2 }$

Corrections. An observation at level ℓ draws a question uniformly at random and a fresh path of length $b _ { \ell } .$ , and records, for every budget k,

$$
D _ { \ell , k } = \left\{ \begin{array} { l l } { 0 , } & { k \le a _ { \ell } , } \\ { W _ { k } - W _ { a _ { \ell } } , } & { a _ { \ell } < k \le b _ { \ell } , } \\ { W _ { b _ { \ell } } - W _ { a _ { \ell } } , } & { k > b _ { \ell } . } \end{array} \right.
$$

For $k \in \left( a _ { L } , b _ { L } \right]$ the levels $\ell < L$ contribute $W _ { 2 ^ { \ell } } - W _ { 2 ^ { \ell - 1 } }$ and level L contributes $W _ { k } - W _ { a _ { L } }$ , so $\begin{array} { r } { \sum _ { \ell } \mathbb { E } D _ { \ell , k } = \theta _ { k } - \theta _ { 1 } } \end{array}$ Since $b _ { \ell } \leq 2 a _ { \ell }$ , Lemma 3 averaged over questions gives $\mathbb { E } D _ { \ell , k } ^ { 2 } \le \bar { v } _ { a _ { \ell } }$ for every k. Each level runs its own confidence sequence for all of its coordinates and stops when every coordinate has half-width at most $e = \varepsilon / ( 2 J )$ ; a union over levels, coordinates and sides spends $\delta / 2$ . Adding the level intervals to the first-budget interval gives half-width at most ε at every budget, powers of two or not.

One explicit sequence uses the bets $\lambda _ { h } = 2 ^ { - h - 2 } { \mathrm { ~ f o r ~ } } h = 0 , \ldots , H , \ H = \left\lceil \log _ { 2 } \{ ( 1 + e ) / e \} \right\rceil$ . For $Z \in [ - 1 , 1 ]$ and $0 \leq \lambda < 1 , \mathbb { E } e ^ { \lambda ( Z - \mathbb { E } Z ) - \psi ( \lambda ) Z ^ { 2 } } \leq 1$ , because $e ^ { \lambda z - \psi ( \lambda ) z ^ { 2 } } \leq 1 + \lambda z \ \mathrm { o n } \ [ - 1 , 1 ]$ and $e ^ { - \lambda \mathbb { E } Z } ( 1 + \lambda \mathbb { E } Z ) \leq 1$ . Ville’s inequality then gives, for each bet and each side, an interval valid at all times, and a union over the grid adds $\log ( H + 1 )$ to the logarithmic factor L. The radius after T observations with bet λ is $\begin{array} { r } { \{ \psi ( \lambda ) \sum _ { i < T } Z _ { i } ^ { 2 } + L \} / ( \lambda T ) } \end{array}$ Write $m _ { 2 } = \mathbb { E } Z ^ { 2 } \leq \bar { v } _ { a _ { \ell } }$ . Every interval $[ x , 2 x ]$ with $e / \{ 8 ( 1 + e ) \} \le x \le 1 / 8$ contains a grid bet, so the grid holds a $\lambda \in [ e / \{ 8 ( m _ { 2 } + e ) \} , e / \{ 4 ( m _ { 2 } + e ) \} ]$ ; also $\psi ( \lambda ) \leq \lambda ^ { 2 }$ for $\lambda \leq 1 / 2$ . On the event $\begin{array} { r } { \sum _ { i \leq T } Z _ { i } ^ { 2 } \leq 2 T m _ { 2 } + 2 L } \end{array}$ , which Bernstein’s inequality gives with probability at least $1 - e ^ { - L }$ at any fixed $T ,$ , the radius is at most

$$
2 \lambda m _ { 2 } + \frac { 2 \lambda ^ { 2 } L + L } { \lambda T } \leq \frac { e } { 2 } + \frac { L } { 2 T } + \frac { 8 L ( m _ { 2 } + e ) } { e T } ,
$$

which is at most e once $T \geq 3 4 L ( m _ { 2 } / e ^ { 2 } + 1 / e )$ . Outside that event the smallest bet gives a deterministic cap of $O ( L / e ^ { 2 } )$ observations, and choosing the failure probability of order e makes its contribution $O ( L / e )$ . Hence

$$
{ \mathbb E } T _ { \ell } = O \big \{ L \big ( J ^ { 2 } \bar { v } _ { a _ { \ell } } / \varepsilon ^ { 2 } + J / \varepsilon \big ) \big \} .
$$

An observation at level ℓ costs $b _ { \ell }$ answers and at most $H _ { b _ { \ell } }$ record queries in expectation.

Summing the levels. For $\ell \geq 3$ the $a _ { \ell } / 2$ indices $j \in ( a _ { \ell } / 2 , a _ { \ell } ]$ satisfy max $k \ge j \bar { v } _ { k } \ge \bar { v } _ { a _ { \ell } } ;$ for $\ell = 1 , 2$ the single index $j = a \ell$ does. These index blocks are disjoint, so $\textstyle \sum _ { \ell } \operatorname* { m a x } ( 1 , a _ { \ell } / 2 ) { \bar { v } } _ { a _ { \ell } } \leq \sum _ { \mathrm { w } }$ , and $b _ { \ell } \leq 4 \operatorname* { m a x } ( 1 , a _ { \ell } / 2 )$ gives

$$
\sum _ { \ell } b _ { \ell } \bar { v } _ { a \ell } \leq 4 \Sigma _ { \mathrm { w } } , \qquad \sum _ { \ell } b _ { \ell } < 3 K .
$$

The corrections therefore use $\widetilde O ( K / \varepsilon + s / \varepsilon ^ { 2 } )$ answers. For queries, each $\bar { v } _ { a , \ell }$ is at most $a _ { * } = \operatorname* { m i n } ( v , s )$ , and the same blocks give $\bar { v } _ { a _ { \ell } } \leq 4 s / 2 ^ { \ell }$ . Summing the smaller bound, at most $2 + \log _ { 2 } ( s / a _ { * } )$ levels contribute $a _ { * }$ each and the res form a geometric series below $2 a _ { * }$ , so

$$
\sum _ { \ell } \bar { v } _ { a _ { \ell } } \leq a _ { * } \{ 4 + \log _ { 2 } ( s / a _ { * } ) \} .
$$

The corrections thus use at most a polylogarithmic factor times $H _ { K } / \varepsilon + a _ { * } \{ 1 + \log ( s / a _ { * } ) \} / \varepsilon ^ { 2 }$ queries, and the first budget adds $\widetilde { O } ( M _ { \varepsilon } + 1 / \varepsilon + a _ { * } / \varepsilon ^ { 2 } )$ . The audit is never told v or s, and it attains both upper bounds at once.

## C.2 Lower bounds on generated answers

The alternatives below may lie outside $\mathcal { C } ( M , K , v , s )$ , which is allowed because a valid audit must cover every law.   
Each bound is proved by testing, as described in Appendix A.

Calibration, at every law. Let P be any law, $a = \mathrm { m a x } ( \theta _ { K } , 1 - \theta _ { K } ) \geq 1 / 2$ , and $y = 0$ if $\theta _ { K } \geq 1 / 2$ and $y = 1$ otherwise. If the scores of every $P _ { x }$ are bounded above, the alternative $Q _ { x } = ( 1 - \beta ) P _ { x } + \beta \delta _ { ( s _ { x } ^ { + } , y ) }$ adds an answer with correctness y and a score $s _ { x } ^ { + }$ above the support of $P _ { x }$ . Then $p _ { K } ^ { Q } ( x ) = ( 1 - \beta ) ^ { K } p _ { K } ( x ) + \{ 1 - ( 1 - \beta ) ^ { K } \} y ,$ so the target moves by $\{ 1 - ( 1 - \beta ) ^ { K } \} a .$ , which exceeds 2ε once $K \{ - \log ( 1 - \beta ) \} > - \log ( 1 - 2 \varepsilon / a )$ . Since $d P _ { x } / d Q _ { x } \leq 1 / ( 1 - \beta )$ , one generated answer has divergence at most $- \log ( 1 - \beta )$ , and questions and the audit’s choices are otherwise unchanged, so testing gives $\mathbb { E } _ { P } N \ge \kappa _ { \delta } / \{ - \log ( 1 - \beta ) \}$ . Letting $\beta$ decrease to the threshold,

$$
\mathbb { E } _ { P } N \ge \frac { \kappa _ { \delta } K } { - \log ( 1 - 2 \varepsilon / a ) } \ge \frac { \kappa _ { \delta } ( 1 - 4 \varepsilon ) K } { 4 \varepsilon } \ge \frac { 7 \kappa _ { \delta } K } { 3 2 \varepsilon } ,
$$

by $- \log ( 1 - z ) \leq z / ( 1 - z )$ and $a \ge 1 / 2$ . If the scores are not bounded above, place the new answer at a score $c _ { x }$ that is not an atom of $P _ { x }$ and has $P _ { x } ( S > c _ { x } ) \leq \gamma / K$ . The new answer then wins whenever it is among the K draws, except on an event of probability at most $\gamma ,$ the target moves by at least $\{ 1 - ( 1 - \beta ) ^ { K } \} a - \gamma ,$ , the divergence is still at most $- \log ( 1 - \beta )$ , and letting $\gamma  0$ gives the same bound. The bound holds at every law, and in particular at every law of $\mathcal { C } ( M , K , v , s )$

Within-question noise. Take identical questions and two separated score bands. An answer lands in the upper band with probability q and is then correct; otherwise it is incorrect. Choose q so that $p _ { K } = t \leq 1 / 2$ and $K t ( 1 - t ) = s$ Then $\bar { v } _ { k } = p _ { k } ( 1 - p _ { k } )$ increases in $k ,$ so $\Sigma _ { \mathrm { w } } = s$ and max $\AA _ { \cdot k } \bar { v } _ { k } = s / K \leq v . \mathrm { ~ I f ~ } s \geq 8 K \varepsilon$ , then $t \geq 8 \varepsilon$ ; lowering q so that $p _ { K }$ falls by 3ε keeps q within a constant factor, and the band indicator of one answer has divergence $O \{ \varepsilon ^ { 2 } / ( K t ) \}$ Testing gives $\mathbb { E } N \overset { - } { = } \bar { \Omega _ { \delta } } ( K t / \varepsilon ^ { 2 } ) = \Omega _ { \delta } ( s / \varepsilon ^ { 2 } )$ . If $s < 8 K \varepsilon$ , then $s / \varepsilon ^ { 2 } < 8 K / \varepsilon$ and the calibration bound already covers it.

Telling questions apart. Give each question a deterministic correctness bit, with continuous scores independent of the bits. Let $\varepsilon \le 1 / 1 2 8$ and $2 1 \leq M \leq \pi / ( 2 5 6 \varepsilon ^ { 2 } )$ , and draw the bits independently and uniformly. If at least $M / 2$ bits are unrevealed, the conditional probability that any interval of width 2ε contains $\theta _ { 1 }$ is at most

$$
( 2 \varepsilon M + 1 ) \operatorname* { m a x } _ { i } { \mathbb { P } } \{ \mathrm { B i n } ( u , 1 / 2 ) = j \} \leq 4 \varepsilon { \sqrt { M / \pi } } + 2 / { \sqrt { \pi M } } \leq { \frac { 1 } { 2 } } .
$$

Averaging the coverage requirement over the prior, a valid audit reveals more than half the bits with probability at least $1 - 2 \delta \geq 1 / 2$ , so $\mathbb { E } N \geq M / 4$ . For $M < 2 1$ we have $2 \varepsilon M < 1$ , a single unrevealed bit keeps coverage at most $1 / 2$ and $\mathbb { E } N \geq M / 2$ . For $M > m = \lfloor \pi / ( 1 0 2 4 \varepsilon ^ { 2 } ) _ { - }$ ⌋, split rm of the questions into m known groups of $r = \lfloor M / m \rfloor$ , give each group one unknown bit, and make the remaining questions incorrect. The groups carry mass $w = r m / M \geq 1 / 2$ so a width-2ε interval for $\theta _ { 1 }$ gives a width-4ε interval for the average of the m bits, and $m \leq \pi / \{ 2 5 6 ( 2 \varepsilon ) ^ { 2 } \}$ ; the same argument at precision $2 \varepsilon$ gives $\mathbb { E } N \ge m / 4 = \Omega ( \varepsilon ^ { - 2 } )$ . One generated answer reveals at most one group’s bit, and scores reveal none. When $1 / 1 2 8 < \varepsilon \leq 1 / 3 2 , M _ { \varepsilon } \leq 1 2 8 / \varepsilon$ and the calibration bound covers this term. Every revealed bit also needs a correctness query.

The largest of the three bounds is at least a third of their sum, which proves the lower half of (4).

## C.3 Lower bounds on correctness queries

Every continuous-score law. Let P be any law with continuous conditional score distributions $F _ { x }$ . For $j ~ =$ $0 , \ldots , \lfloor \log _ { 1 6 } K \rfloor$ put $k _ { j } = 1 6 ^ { j }$ and

$$
B _ { j } ( x ) = \big \{ s : ( 4 k _ { j } ) ^ { - 1 } \leq - \log F _ { x } ( s ) < 4 / k _ { j } \big \} .
$$

These bands are disjoint for each $x ,$ and the winner at budget $k _ { j }$ falls in $B _ { j } ( x )$ with probability $c _ { 0 } = e ^ { - 1 / 4 } - e ^ { - 4 } > 3 / 4$ Averaged over questions, the winner mass of $B _ { j }$ splits into a part carrying label 1 and a part carrying label 0, and one of them, say the part $q _ { y }$ with label $y ,$ is at least $c _ { 0 } / 2$ . The alternative $Q _ { j }$ changes only labels inside the bands $B _ { j } ( x )$ : independently, each answer there with label $y$ has its label replaced by $1 - y$ with probability $\alpha _ { j } = 3 \varepsilon / q _ { y }$ . The winner at budget $k _ { j }$ then changes label with probability exactly $\alpha _ { j } q _ { y } = 3 \varepsilon$ , so $\theta _ { k _ { j } }$ moves by $3 \varepsilon .$ , and $\alpha _ { j } \leq 6 \varepsilon / c _ { 0 } < 8 \varepsilon$ Question and score observations are unchanged, a queried label in $B _ { j }$ has divergence at most $- \log ( 1 - \alpha _ { j } ) \leq 3 2 \varepsilon / 3$ and every other label has divergence zero. If $m _ { j }$ counts queries in $B _ { j }$ , testing gives

$$
\mathbb { E } _ { P } m _ { j } \geq \frac { 3 \textup { k l } ( 1 - \delta , \delta ) } { 3 2 \varepsilon } ,
$$

and summing over the disjoint bands gives $\mathbb { E } _ { P } m = \Omega _ { \delta } ( H _ { K } / \varepsilon )$ at the same, arbitrary, law P. The bound counts queries to answer occurrences; if a known deterministic grader lets one grade serve every repeat of an answer, distinct grader calls are a diferent resource.

Telling questions apart. The deterministic bits above also force $\Omega _ { \delta } ( M _ { \varepsilon } )$ queries.

The multiscale term. Let $a _ { * } > 0 , r = a _ { * } / 4 , F = s / ( s + r )$ and $h \ = \ - \log F$ . At every question let the score percentile be uniform, with correctness Bernoulli $( 1 - r )$ below $F$ and certain above it. Then $p _ { k } = 1 - r F ^ { k }$ $\bar { v } _ { k } = r F ^ { k } ( 1 - r F ^ { k } ) \leq v$ , and $\begin{array} { r } { \Sigma _ { \mathrm { w } } \leq \sum _ { k > 1 } r F ^ { k } = r F / ( 1 - F ) = s } \end{array}$ . Let $J _ { * }$ be the number of integers $j \geq 0$ with $k _ { j } = 1 6 ^ { j } \leq$ min $\{ K , 1 / ( 4 h ) \}$ . Since $a _ { * } / ( \overline { { 5 } } s ) \leq h \leq a _ { * } / ( 4 s )$ and $s / a _ { * } \le K , J _ { * }$ lies between constant multiples of $1 + \log ( s / a _ { * } )$ . For each such j the band $\{ u : ( 4 k _ { j } ) ^ { - 1 } \leq - \log u < 4 / k _ { j } \}$ lies below $F _ { ; }$ , the bands are disjoint, and each holds the winner at budget $k _ { j }$ with probability $c _ { 0 }$ . The alternative lowers the success probability inside band $j$ from $1 - r \mathrm { ~ t o ~ } 1 - r - \Delta$ with $\Delta = 3 \varepsilon / c _ { 0 } < 4 \varepsilon$ , which moves the target by 3ε. The alternative success probability is at least 13/16 and its failure probability at least $r , \mathrm { s o }$ one queried label in the band has divergence at most $2 5 6 \varepsilon ^ { 2 } / ( 1 3 r )$ , and testing gives $\mathbb { E } _ { P } m _ { j } \geq 1 3 ~ \mathrm { k l } ( 1 - \delta , \delta ) ~ r / ( 2 5 6 \varepsilon ^ { 2 } )$ . Summing over the $J _ { * }$ bands,

$$
\mathbb { E } _ { P } m \ge \frac { 1 3 \mathrm { ~ k l } ( 1 - \delta , \delta ) } { 1 0 2 4 } \frac { a _ { * } J _ { * } } { \varepsilon ^ { 2 } } ,
$$

which is the logarithmic structure in (5). This lower bound does not identify the logarithmic factors that the confidence union adds to the upper bound, which is why Theorem 4(iii) holds up to polylogarithmic factors.

The same law forces generated answers. Let $k _ { \mathrm { l a s t } }$ be the largest selected $k _ { j }$ and $p _ { \mathrm { l a s t } } = e ^ { - 1 / ( 4 k _ { \mathrm { l a s t } } ) } - e ^ { - 4 / k _ { \mathrm { l a s t } } }$ the probability that one answer’s percentile lies in its band. Every generated answer lands there with probability $p _ { \mathrm { l a s t } } .$ independently of the past, and a query in the band needs an answer there, so Wald’s identity gives $p _ { \mathrm { l a s t } } \mathbb { E } _ { P } N \geq$ $\mathbb { E } _ { P } m _ { \mathrm { l a s t } } \geq 1 3 \kappa _ { \delta } r / ( 2 5 6 \varepsilon ^ { 2 } )$ . Since $p _ { \mathrm { l a s t } } \leq 1 5 / ( 4 k _ { \mathrm { l a s t } } )$ and $k _ { \mathrm { l a s t } } > \mathrm { m i n } \{ K , 1 / ( 4 h ) \} / 1 6 \geq s / ( 1 6 a _ { * } )$

$$
\mathbb { E } _ { P } N \ge \frac { 1 3 \kappa _ { \delta } s } { 6 1 4 4 0 \varepsilon ^ { 2 } }
$$

at the same law P. One law in the class thus forces both the label term and the $s / \varepsilon ^ { 2 }$ answer term.

The first budget alone. Let $r = \operatorname* { m i n } ( 2 v , s , 1 / 4 )$ . At every question let the score percentile U be uniform, with correctness 1 when $U \ge 1 / 2$ and $\operatorname { B e r n o u l l i } ( 1 - r )$ otherwise. The budget-k winner lies in the lower half with probability $2 ^ { - k }$ , so $1 - p _ { k } = 2 ^ { - k } r , { \bar { v } } _ { k } = 2 ^ { - k } r ( 1 - 2 ^ { - k } r )$ decreases in $k ,$ max<sub>k</sub> $\bar { v } _ { k } = \bar { v } _ { 1 } \leq r / 2 \leq \upsilon$ and $\Sigma _ { \mathrm { w } } < r \leq s ;$ the law is in $\mathcal { C } ( M , K , v , s )$ . The alternative lowers the success probability in the lower half to $1 - r - 4 \varepsilon ( 1 + \gamma )$ for a small $\gamma > 0$ , which moves $\theta _ { 1 }$ by $2 \varepsilon ( 1 + \gamma ) > 2 \varepsilon$ and leaves scores and upper-half labels unchanged. Its success probability is at least 0.62 and its failure probability at least $r ,$ so a lower-half label has divergence at most $1 6 \varepsilon ^ { 2 } ( 1 + \gamma ) ^ { 2 } / ( 0 . 6 2 r ) \le 2 6 \varepsilon ^ { 2 } ( 1 + \gamma ) ^ { 2 } / r$ Testing and $\gamma  0$ give

$$
\mathbb { E } _ { P } m \ge \frac { \kappa _ { \delta } r } { 2 6 \varepsilon ^ { 2 } } \ge \frac { \kappa _ { \delta } a _ { * } } { 2 6 \varepsilon ^ { 2 } } ,
$$

since $r \geq \operatorname* { m i n } ( v , s )$ when $v \leq 1 / 4$ . This law moves only $\theta _ { 1 }$ . With label calibration at $K = 1$ and the deterministic bits, certifying $\theta _ { 1 }$ alone already takes $\Omega _ { \delta } ( 1 / \varepsilon + M _ { \varepsilon } + a _ { * } / \varepsilon ^ { 2 } )$ queries over the class, and by (iii) the whole curve costs at most polylogarithmic factors more.

Noisy labels. If in addition $\eta \le \mathbb { P } ( Y = 1 \mid x , S ) \le 1 - \eta$ almost surely for some $\eta \in ( 0 , 1 / 2 ]$ , and $\varepsilon \leq \eta c _ { 0 } / 6 .$ , raise the success probability by $\Delta = 3 \varepsilon / c _ { 0 } \le \eta / 2$ inside one band $B _ { j } ( x )$ at a time. This moves $\theta _ { k _ { i } }$ by exactly $3 \varepsilon$ , keeps the success probability in $[ \eta , 1 - \eta / 2 ]$ , and costs at most $\Delta ^ { 2 } / ( \eta / 4 ) \overset { \cdot } { = } 3 6 \varepsilon ^ { 2 } / ( \eta c _ { 0 } ^ { 2 } )$ per queried label in that band. Summing over bands, every valid audit has $\mathbb { E } _ { P } m \ge ( 1 + \lfloor \log _ { 1 6 } K \rfloor ) \kappa _ { \delta } \eta c _ { 0 } ^ { 2 } / ( 3 6 \varepsilon ^ { 2 } )$ at every continuous-score law with this margin.

## D Proof of Theorem 6

Throughout, the list $\{ 1 , \dots , M \}$ and its equal weights are known, $P _ { x }$ is the law of one answer $Z = ( S , Y )$ at question ${ \underline { { x } } } ,$ and a valid audit covers every $\theta _ { k }$ at once with probability at least $1 - \delta$ at every law. For k answers $z _ { 1 } , \ldots , z _ { k }$ $\bar { W } _ { k } ( z _ { 1 } , \ldots , z _ { k } )$ is the average correctness of the answers that attain the highest score. Given the scores, the tied maxima share one conditional success probability, so the first maximizer and a tie-averaged one have the same expected correctness and ${ \mathbb E } \{ \widetilde { W } _ { k } ( Z _ { 1 } , . . . , Z _ { k } ) \mid x \} = p _ { k } ( x )$ . Write

$$
g _ { k , x } ( z ) = k \bigl [ \mathbb { E } \{ \widetilde { W } _ { k } ( z , Z _ { 2 } , \ldots , Z _ { k } ) \mid x \} - p _ { k } ( x ) \bigr ] , \qquad v _ { k } ( x ) = \mathbb { E } \{ g _ { k , x } ( Z ) ^ { 2 } \mid x \} ,
$$

so that $\mathbb { E } g _ { k , x } ( Z ) = 0$ and $| g _ { k , x } | \leq k$ . With $w _ { x } = M a _ { x } ,$ the definition (6) of Γ reads $\begin{array} { r } { \Gamma = \operatorname* { i n f } _ { w } \operatorname* { m a x } _ { k } M ^ { - 1 } \sum _ { x } v _ { k } ( x ) / w _ { x } } \end{array}$ over $w \geq 0$ with $\begin{array} { r } { M ^ { - 1 } \sum _ { x } w _ { x } = 1 } \end{array}$ , using $0 / 0 = 0$ . Lower bounds are proved by testing $( { \mathrm { A p p e n d i x ~ A } } )$

## D.1 Part (i)

Fix a valid audit with $\bar { N } = \mathbb { E } _ { P } N < \infty ;$ otherwise there is nothing to prove. Let $C ( x )$ be the expected number of answers at x and $w _ { x } = M C ( x ) / { \bar { N } } ,$ , so that $\begin{array} { r } { M ^ { - 1 } \sum _ { x } w _ { x } = 1 } \end{array}$ . For each k put $\begin{array} { r } { B _ { k } = M ^ { - 1 } \sum _ { x } v _ { k } ( x ) / ( 1 + w _ { x } ) } \end{array}$ and let k maximize it, with $B = B _ { k }$ . Since $( 1 + w ) / 2$ is an admissible allocation, $B \geq \Gamma / 2$

Suppose first that $\Gamma \geq 1 6 \varepsilon K$ , so $B \geq 8 \varepsilon K$ . Put $h _ { x } = g _ { k , x } / ( 1 + w _ { x } ) , t = 4 \varepsilon / B$ and $Q _ { x } ( d z ) = \{ 1 + t h _ { x } ( z ) \} P _ { x } ( d z )$ Since $| t h _ { x } | \leq t k \leq 1 / 2$ and $\mathbb { E } h _ { x } = 0$ , each $Q _ { x }$ is a law. Expanding $\begin{array} { r } { \prod _ { i \leq k } \{ 1 + t h _ { x } ( Z _ { i } ) \} } \end{array}$ over subsets of $\{ 1 , \ldots , k \}$

$$
p _ { k } ^ { Q } ( x ) - p _ { k } ( x ) = t \sum _ { i \leq k } \mathbb { E } \{ \widetilde { W } _ { k } h _ { x } ( Z _ { i } ) \} + \mathbb { E } ( \widetilde { W } _ { k } G ) , \qquad G = \sum _ { | A | \geq 2 } t ^ { | A | } \prod _ { i \in A } h _ { x } ( Z _ { i } ) .
$$

By symmetry the linear term is $t k \mathbb { E } \{ \mathbb { E } ( \widetilde { W } _ { k } \mid Z _ { 1 } ) h _ { x } ( Z _ { 1 } ) \} = t \mathbb { E } ( g _ { k , x } h _ { x } ) = t v _ { k } ( x ) / ( 1 + w _ { x } )$ . Products over distinct subsets are orthogonal, so with $r _ { x } = \mathbb { E } h _ { x } ^ { 2 } = v _ { k } ( x ) / ( 1 + w _ { x } ) ^ { 2 }$ we get $\mathbb { E } G ^ { 2 } = ( 1 + t ^ { 2 } r _ { x } ) ^ { k } - 1 - k t ^ { 2 } r _ { x }$ , and since $\mathbb { E } G = 0$ and $| \widetilde { W } _ { k } - 1 / 2 | \le 1 / 2$

$$
| \mathbb { E } ( \widetilde { W } _ { k } G ) | \le \frac { 1 } { 2 } \big ( e ^ { z _ { x } } - 1 - z _ { x } \big ) ^ { 1 / 2 } \le \frac { z _ { x } e ^ { z _ { x } / 2 } } { 2 \sqrt { 2 } } \le \frac { z _ { x } } { 2 } , \qquad z _ { x } = k t ^ { 2 } r _ { x } \le \frac { ( t k ) ^ { 2 } } { 4 } \le \frac { 1 } { 1 6 } ,
$$

using $v _ { k } ( x ) \leq k / 4$ from part (iii). Averaging over questions and using $r _ { x } \le v _ { k } ( x ) / ( 1 + w _ { x } )$

$$
\theta _ { k } ( Q ) - \theta _ { k } ( P ) \geq t B - \frac { k t ^ { 2 } } { 2 } B = t B \Big ( 1 - \frac { k t } { 2 } \Big ) \geq 3 \varepsilon .
$$

Because $- \log ( 1 + y ) \leq - y + y ^ { 2 }$ for $y \ge - 1 / 2 , \mathrm { K L } ( P _ { x } \| Q _ { x } ) \le t ^ { 2 } r _ { x }$ , and since $w _ { x } r _ { x } \le v _ { k } ( x ) / ( 1 + w _ { x } )$ the chain rule gives a transcript divergence at most $\begin{array} { r } { \bar { N } t ^ { 2 } M ^ { - 1 } \sum _ { x } w _ { x } r _ { x } \le \bar { N } t ^ { 2 } B } \end{array}$ . Separation forces this to be at least $\kappa _ { \delta }$ , so $\bar { N } \ge \kappa _ { \delta } B / ( 1 6 \varepsilon ^ { 2 } ) \ge \kappa _ { \delta } \Gamma / ( 3 2 \varepsilon ^ { 2 } )$ . As $K / \varepsilon + \Gamma / \varepsilon ^ { 2 } \leq 1 \overline { { 7 \Gamma } } \widetilde { / } ( 1 6 \varepsilon ^ { 2 } )$ in this case, the claim follows. If instead $\Gamma < 1 6 \varepsilon K$ then $K / \varepsilon + \Gamma / \varepsilon ^ { 2 } < 1 7 K / \varepsilon$ , and the every-law calibration bound $\bar { N } \ge 7 \kappa _ { \delta } K / ( 3 2 \varepsilon )$ of Appendix C.2 is larger than $( \kappa _ { \delta } / 1 2 8 ) \cdot 1 7 K / \varepsilon$

For the limit, fix $\eta , \zeta > 0 .$ , put $\begin{array} { r } { B _ { k } = M ^ { - 1 } \sum _ { x } v _ { k } ( x ) / ( \eta + w _ { x } ) } \end{array}$ and $B = \operatorname* { m a x } _ { k } B _ { k } \geq \Gamma / ( 1 + \eta )$ , because $( \eta + w ) / ( 1 + \eta )$ is admissible, and use $h _ { x } = g _ { k , x } / ( \eta + w _ { x } )$ and $t = 2 ( 1 + \zeta ) \varepsilon / B$ . Then $\begin{array} { r } { | h _ { x } | \le K / \eta , r _ { x } \le K / ( 4 \eta ^ { 2 } ) , M ^ { - 1 } \sum _ { x } r _ { x } \le B / \eta } \end{array}$ and $\begin{array} { r } { M ^ { - 1 } \sum _ { x } w _ { x } r _ { x } \le B } \end{array}$ , and $t K / \eta \le 2 ( 1 + \zeta ) ( 1 + \eta ) \varepsilon K / ( \eta \Gamma )$ tends to zero uniformly over audits $( \mathrm { i f ~ } \Gamma = 0$ there is nothing to prove). The remainder above, averaged over questions, is at most $\{ K t ^ { 2 } \dot { B } / ( 2 \sqrt 2 \eta ) \} \exp \{ K ^ { 2 } t ^ { 2 } / ( 8 \eta ^ { 2 } ) \} =$ $o ( t B )$ , so the target moves by $2 ( 1 + \zeta ) \varepsilon \{ 1 - o ( 1 ) \} > 2 \varepsilon$ for small ε. With $- \log ( 1 + y ) + y \leq y ^ { 2 } / \{ 2 ( 1 - | y | ) \}$ , $\mathrm { K L } ( P _ { x } \| Q _ { x } ) \leq t ^ { 2 } r _ { x } / \{ 2 ( 1 - t K / \eta ) \}$ , and the chain rule gives

$$
\bar { N } \ge \frac { 2 \kappa _ { \delta } ( 1 - t K / \eta ) } { t ^ { 2 } B } \ge \frac { \kappa _ { \delta } \Gamma \{ 1 - o ( 1 ) \} } { 2 ( 1 + \eta ) ( 1 + \zeta ) ^ { 2 } \varepsilon ^ { 2 } } .
$$

Taking the lower limit and then letting $\eta , \zeta \to 0$ proves the claim.

## D.2 Part (ii)

Blocks. A block at question x is $b \geq K$ fresh answers with all correctness values queried. For each $k , U _ { k }$ is the average of $\widetilde { W _ { k } }$ over all k-subsets of the block. After sorting the block by score it has a closed form: a tied group of d answers with c answers strictly below contributes its average correctness times $\{ \binom { c + d } { k } - \binom { c } { k } \} / \binom { b } { k }$ . Then $\mathbb { E } U _ { k } = p _ { k } ( x )$ and $U _ { k } \in [ 0 , 1 ]$ . Write $\sigma _ { k , x } ^ { 2 } = \mathrm { V a r } ( U _ { k } )$ . With the orthogonal decomposition $\begin{array} { r } { \widetilde { W } _ { k } - p _ { k } ( x ) = \sum _ { \emptyset \neq A } \phi _ { | A | } ( Z _ { A } ) } \end{array}$ of Hoefding (1948) and $\eta _ { j } = \mathbb { E } \phi _ { j } ^ { 2 }$ , one has $\begin{array} { r } { \mathrm { V a r } ( \widetilde { W } _ { k } ) = \sum _ { j < k } \binom { k } { j } \eta _ { j } \le 1 / 4 } \end{array}$ and Var $\begin{array} { r } { ( U _ { k } ) = \sum _ { j \le k } \binom { k } { j } ^ { 2 } \eta _ { j } / \binom { b } { j } } \end{array}$ . The term $j = 1$ gives $b \sigma _ { k , x } ^ { 2 } \geq k ^ { 2 } \eta _ { 1 } = v _ { k } ( x )$ , and for $\begin{array} { r } { j \ge 2 , b \binom { k } { j } / \binom { b } { j } = k \prod _ { i = 1 } ^ { j - 1 } ( k - i ) / ( b - i ) \le k ( k - 1 ) / ( b - 1 ) } \end{array}$ . Hence

$$
v _ { k } ( x ) \leq b \sigma _ { k , x } ^ { 2 } \leq v _ { k } ( x ) + \frac { k ( k - 1 ) } { 4 ( b - 1 ) } .
$$

Pilot. Draw m pairs of blocks at every question. Each pair gives $( U _ { k } - U _ { k } ^ { \prime } ) ^ { 2 } / 2 \in [ 0 , 1 / 2 ]$ with mean $\sigma _ { k , x } ^ { 2 }$ . With $r = \{ \log ( 2 M K / \alpha ) / ( 8 m ) \} ^ { 1 / 2 }$ , let $u _ { k , x }$ be the minimum of $1 / 4$ and the average of these values plus r. By Hoefding’s inequality and a union bound, on an event E of probability at least $1 - \alpha , \sigma _ { k , x } ^ { 2 } \leq u _ { k , x } \leq \sigma _ { k , x } ^ { 2 } + 2 r$ for all k and x. Put $U _ { k , x } = b u _ { k , x } ;$ on $E , v _ { k } ( x ) \leq U _ { k , x } \leq v _ { k } ( x ) + \eta _ { b }$ with $\eta _ { b } = K ( K - 1 ) / \{ 4 ( b - 1 ) \} + 2 b r$

Allocation. For a nonnegative array V let $\Gamma ( V ) ~ = ~ \mathrm { i n f } _ { a } \mathrm { m a x } _ { k } M ^ { - 2 } \sum _ { x } V _ { k , x } / a _ { x }$ over the simplex. For $\gamma \in$ $( 0 , 1 )$ , combining an optimal a for V with weight γ and the uniform allocation with weight $1 - \gamma$ and using $( \sqrt { V } + \sqrt { \eta } ) ^ { 2 } / ( \gamma a + ( 1 - \gamma ) / M ) \leq V / ( \gamma a ) + \eta M / ( 1 - \gamma )$ gives $\Gamma ( V + \eta ) \leq \Gamma ( V ) / \gamma + \eta / ( 1 - \gamma )$ , and optimizing over γ,

$$
\begin{array} { r } { \Gamma ( V + \eta ) \le \left\{ \Gamma ( V ) ^ { 1 / 2 } + \eta ^ { 1 / 2 } \right\} ^ { 2 } . } \end{array}
$$

Let $a ^ { * }$ be a pilot-measurable allocation with $\begin{array} { r } { G = \operatorname* { m a x } _ { k } M ^ { - 2 } \sum _ { x } U _ { k , x } / a _ { x } ^ { * } \le \Gamma ( U ) + \tau } \end{array}$ , and use $a = ( 1 - \rho ) a ^ { * } + \rho / M$ so $a _ { x } \geq ( 1 - \rho ) a _ { x } ^ { * }$ and $a _ { x } \geq \rho / M$ . On $E , G \leq \{ \Gamma ^ { 1 / 2 } + \eta _ { b } ^ { 1 / 2 } \} ^ { 2 } + \tau ;$ , because $\Gamma ( \cdot )$ is monotone. On every pilot outcome $G \leq b / 4 + \tau$ , by comparison with the uniform allocation.

Fresh blocks and the band. Given the pilot, choose N<sup>¯</sup> below and draw $n _ { x } = \lceil \bar { N } a _ { x } / b \rceil$ fresh blocks at each question. The estimate $\begin{array} { r } { \widehat { \theta } _ { k } = M ^ { - 1 } \sum _ { x } \bar { U } _ { k , x } , } \end{array}$ , with $\bar { U } _ { k , x }$ the average of the $n _ { x }$ fresh blocks, is a sum of independent terms, each within $1 / ( M n _ { x } ) \le b / ( \rho \bar { N } )$ of its mean, with total variance at most $\begin{array} { r } { V _ { \mathrm { u p } } = \operatorname* { m a x } _ { k } M ^ { - 2 } \sum _ { x } u _ { k , x } / n _ { x } } \end{array}$ on $E _ { : }$ , and $V _ { \mathrm { u p } } \leq G / \{ ( 1 - \rho ) \bar { N } \}$ on every pilot outcome. Bernstein’s inequality and a union over the 2K sides give radius $( 2 V _ { \mathrm { u p } } \ell ) ^ { 1 / 2 } + b \ell / ( 3 \rho \bar { N } )$ ) with $\ell = \log \{ 2 K / ( \delta - \alpha ) \}$ and conditional failure probability at most $\delta - \alpha$ on $E ,$ so the audit is valid at every law. The choice

$$
\bar { N } = \Big \lceil \Big \{ \frac { A ^ { 1 / 2 } + ( A + 4 \varepsilon B _ { 0 } ) ^ { 1 / 2 } } { 2 \varepsilon } \Big \} ^ { 2 } \Big \rceil , \qquad A = \frac { 2 G \ell } { 1 - \rho } , \quad B _ { 0 } = \frac { b \ell } { 3 \rho } ,
$$

makes the radius at most ε, and the audit generates at most 2m $. M b + \bar { N } + M b$ answers.

Schedule. Take $b = \operatorname* { m a x } ( 2 , K , \lceil \varepsilon ^ { - 1 / 4 } \rceil ) , m = \lceil \varepsilon ^ { - 3 / 4 } \rceil , \rho = \operatorname* { m i n } ( 1 / 2 , \varepsilon ^ { 1 / 8 } ) , \alpha = \operatorname* { m i n } ( \delta / 2 , \varepsilon )$ and $\tau = \varepsilon ;$ none depends on the law. As $\varepsilon  0 \colon \eta _ { b }  0 .$ , the pilot 2m $M b = O ( M / \varepsilon )$ and Mb are $o ( \varepsilon ^ { - 2 } ) , \varepsilon B _ { 0 } \to 0$ and $\ell \to \log ( 2 K / \delta )$ . On $E ,$ $\varepsilon ^ { 2 } \bar { N } \le A \{ 1 + o ( 1 ) \} + \varepsilon ^ { 2 }$ and $A \le 2 \{ \Gamma + o ( 1 ) \} \log ( 2 K / \delta )$ . Of E, which has probability at most $\varepsilon , G \leq b / 4 + \tau$ gives $\varepsilon ^ { 2 } \bar { N } = O ( b )$ , a contribution $O ( \varepsilon b ) = o ( 1 )$ . Hence lim sup $\varepsilon ^ { 2 } \mathbb { E } N \le 2 \Gamma \log ( 2 K / \delta )$ . The allocation $a ^ { * }$ is the Neyman allocation for the least favorable mixture of budgets (Cochran, 1977); learning it from a pilot is the adaptive stratified sampling problem of Carpentier et al. (2015), and the audit charges every pilot answer and label.

## D.3 Part (iii)

In the decomposition above, $\operatorname { V a r } ( \widetilde { W } _ { k } \mid x ) \ge k \eta _ { 1 } = v _ { k } ( x ) / k$ , and $\operatorname { V a r } ( { \widetilde { W } } _ { k } \mid x ) \leq p _ { k } ( x ) \{ 1 - p _ { k } ( x ) \}$ because $\widetilde { W } _ { k } \in [ 0 , 1 ]$ has mean $p _ { k } ( x )$ . So $v _ { k } ( x ) \leq k p _ { k } ( x ) \{ 1 - p _ { k } ( x ) \} \leq k / 4$ . The uniform allocation gives $\begin{array} { r } { \Gamma \leq \operatorname* { m a x } _ { k } M ^ { - 1 } \sum _ { x } v _ { k } ( x ) \leq \operatorname* { m a x } _ { k } k \bar { v } _ { k } } \end{array}$ and for each $\begin{array} { r } { k , k \bar { v } _ { k } = \sum _ { j \leq k } \bar { v } _ { k } \leq \sum _ { j \leq k } \operatorname* { m a x } _ { i \geq j } \bar { v } _ { i } \leq \sum _ { \mathbf { w } } } \end{array}$

## D.4 Computing Γ

For a question whose answers fall in score groups $1 , \ldots , G$ in increasing order, with masses $\mu _ { g } ,$ success fractions $\eta _ { g }$ and cumulative masses $F _ { g } \ ( F _ { 0 } = 0 )$ , an answer in group g with correctness y has

$$
k \mathbb { E } \{ \widetilde { W } _ { k } ( z , Z _ { 2 } , \ldots , Z _ { k } ) \mid x \} = k \Big \{ \eta _ { g } F _ { g } ^ { k - 1 } + \sum _ { h > g } \eta _ { h } \big ( F _ { h } ^ { k - 1 } - F _ { h - 1 } ^ { k - 1 } \big ) \Big \} + ( y - \eta _ { g } ) \frac { F _ { g } ^ { k } - F _ { g - 1 } ^ { k } } { \mu _ { g } } ,
$$

so $v _ { k } ( x )$ is a finite sum. For every λ in the simplex of budgets, Cauchy–Schwarz gives the lower bound $\Gamma \geq D ( \lambda ) =$ $\{ M ^ { - 1 } \stackrel { \textstyle } { \sum } _ { x } ( \sum _ { k } \lambda _ { k } v _ { k } ( x ) ) ^ { 1 / 2 } \} ^ { 2 }$ , attained by the allocation proportional to the square roots when λ is optimal, and any allocation gives an upper bound. We maximize D by conditional-gradient steps and report the best lower and upper values. On the law of the first 4,000 reference answers per question of the MMLU-Pro study, at $K = 6 4$ , the bracket is [0.27961, 0.27961], against $\Sigma _ { \mathrm { w } } = 2 . 5 6 6$ on the same law; uniform allocation with all-subsets reuse gives 1.126, and max<sub>k</sub> $k \bar { v } _ { k } = 2 . 3 1 4$ . On the 185 held-out pools at $K = 6 4$ the median Γ is 0.34, the median $\Sigma _ { \mathrm { w } }$ is 3.88, and $\Sigma _ { \mathrm { w } } / \Gamma$ has median 12.0 and interquartile range 8.4 to 21.9. On these pools the median ratio of ma $\mathrm { x } _ { k } k { \bar { v } } _ { k }$ to the uniform-allocation value is 2.3, and the median ratio of that value to Γ is 3.7.

## E Proof of the cost bound of Theorem 9

We carry out the argument for the cap $\lambda _ { \mathrm { m a x } } = 0 . 9 5$ used in the stored-pool study and then explain the change for other caps. Fix a budget k and write $z _ { k } = \bar { v } _ { k } + \varepsilon , L = \log ( 2 K / \alpha _ { S } )$ and $t = \log \{ 4 K \operatorname* { m a x } ( 1 , R _ { F } / 2 ) / \varepsilon \}$ , where $R _ { F }$ is the final round. The variance estimate before pair j is half the average disagreement rate of the earlier pairs, with one pseudo-pair of variance $1 / 4$ and a floor of at most $0 . 0 1 \varepsilon$ . By Bernstein’s inequality and a union bound over pairs and budgets, outside an event of probability $\varepsilon / 2$ the unfloored estimate at pair j is within

$$
\sqrt { \frac { \bar { v } _ { k } t } { ( j - 1 ) M } } + \frac { t } { 3 ( j - 1 ) M } + \frac { 1 } { 4 ( j - 1 ) M }
$$

of $\bar { v } _ { k }$ . With $A = 1 0 0 { , } 0 0 0$ , this deviation is below $0 . 0 1 z _ { k }$ once $A t / z _ { k }$ question visits have been made, and the floor does not change that. Call later pairs good. On a good pair,

$$
\lambda _ { j } \geq 0 . 9 4 \varepsilon / z _ { k } , \qquad \psi ( \lambda _ { j } ) \bar { v } _ { k } \leq 0 . 5 7 \varepsilon \lambda _ { j } .
$$

The first follows from the bet rule. For the second, if $\bar { v } _ { k } \ge \varepsilon / 1 0$ the estimate is at least $0 . 8 9 \bar { v } _ { k }$ , the bet satisfies $\lambda / ( 1 - \lambda ) \leq \varepsilon / \hat { v }$ , and $\psi ( \lambda ) \leq \lambda ^ { 2 } / \{ 2 ( 1 - \lambda ) \}$ gives $\psi ( \lambda ) \bar { v } _ { k } \le \lambda \varepsilon \bar { v } _ { k } / ( 2 \hat { v } ) \le 0 . 5 7 \varepsilon \lambda ; \mathrm { ~ i f ~ } \bar { v } _ { k } < \varepsilon / 1 0$ , use $\psi ( \lambda ) / \lambda \leq$ $\psi ( 0 . 9 5 ) / 0 . 9 5 < 2 . 1 5 4 . \mathrm { ~ A ~ }$ cap on the bet can only help both comparisons.

Let $\begin{array} { r } { P _ { J } = \sum _ { j < J } \psi ( \lambda _ { j } ) D _ { j } , b = \psi ( 0 . 9 5 ) < 2 . 0 5 } \end{array}$ and $u = 1 / ( 4 b )$ . The disagreement indicators are independent given the past, with means $2 p _ { k } ( x ) \{ 1 - p _ { k } ( x ) \}$ , and $e ^ { q } - 1 \leq 1 . 2 q$ for $0 \leq q \leq 1 / 4$ , so

$$
\exp \Bigl \{ u \Bigl [ P _ { J } - 1 . 2 \sum _ { j \le J } 2 M \psi ( \lambda _ { j } ) \bar { v } _ { k } \Bigr ] \Bigr \}
$$

is a nonnegative supermartingale. Ville’s inequality, with a union over budgets, bounds $P _ { J }$ for all J by its compensator plus 4bt, outside an event of probability at most $\varepsilon / 2$ . The first pair adds at most $6 M \varepsilon ^ { 2 }$ to the compensator and all other bad pairs at most 2Abt. With the good-pair bounds,

$$
P _ { J } \leq 0 . 6 8 4 \varepsilon \mathsf { m a s s } _ { J } + 7 . 2 M \varepsilon ^ { 2 } + 5 0 0 , 0 0 0 t , \qquad \mathsf { m a s s } _ { J } = 2 M \sum _ { j \leq J } \lambda _ { j } .
$$

At most $2 M + 2 A t / z _ { k }$ rows lie in bad pairs. If $n = 2 M J \geq 4 M + 4 A t / z _ { k }$ , then mass $J \ge 0 . 4 7 n \varepsilon / z _ { k }$ , and substituting into the radius $( P _ { J } + L ) / \mathsf { m a s s } _ { J }$ shows it is at most ε as soon as $n \geq 2 0 M + 4 , 0 0 0 , 0 0 0 z _ { k } ( t + L ) / \varepsilon ^ { 2 }$ . Rounding up to a whole pair adds fewer than 2M rows. So, with probability at least $1 - \varepsilon .$ , simultaneously over budgets,

$$
n _ { k } \le 2 2 M + 4 , 0 0 0 , 0 0 0 \left( t + L \right) \frac { \bar { v } _ { k } + \varepsilon } { \varepsilon ^ { 2 } } .
$$

The constant is loose; it is a device of the proof and does not enter the intervals. An answer at position j is generated only while some budget $k \geq j$ is active, so $\begin{array} { r } { N \le \sum _ { j \le K } \operatorname* { m a x } _ { k \ge j } n _ { k } } \end{array}$ , and $\begin{array} { r } { \sum _ { j } \operatorname* { m a x } _ { k \geq j } ( \bar { v } _ { k } + \varepsilon ) \leq \sum _ { \mathrm { w } } + K \varepsilon . } \end{array}$ On the complementary event the round cap gives $N \leq K R _ { F } \bar { M }$ , and inverting Hoefding’s inequality gives $R _ { F } M =$ ${ \cal O } \{ M + \varepsilon ^ { - 2 } \log ( K / \delta ) \}$ . Taking expectations proves the bound on EN in (9). Each row has a length fixed before it is generated, and a row of length ℓ has $H _ { \ell } \leq H _ { K }$ score records in expectation (Rényi, 1962); Wald’s identity gives $\mathbb { E } m \leq H _ { K } \mathbb { E } n$

Other caps. For ${ \lambda _ { \operatorname* { m a x } } } \le 0 . 9 5$ every comparison above holds as stated, except that the bet on a good pair is at least $\operatorname* { m i n } \{ \lambda _ { \operatorname* { m a x } } , 0 . 9 4 \varepsilon / z _ { k } \} \ge \lambda _ { \operatorname* { m a x } } \cdot 0 . 9 4 \varepsilon / z _ { k } .$ , so the constant in the bound on $n _ { k }$ grows by a factor at most $1 / \lambda _ { \mathrm { m a x } }$ . For $0 . 9 5 < \lambda _ { \operatorname* { m a x } } < 1$ , replace the threshold $\varepsilon / 1 0$ by a constant multiple of ε that depends on $\psi ( \lambda _ { \mathrm { m a x } } ) / \lambda _ { \mathrm { m a x } }$ , and enlarge A accordingly. In every case the order in (9) is unchanged.

## F Proof of Theorem 13

Write U for the percentile of an answer (with an independent uniform tie-breaker), $w _ { k } ( u ) = k u ^ { k - 1 }$ and $f ( u ) = \mathbb { P } ( Y =$ $1 \mid U = u )$ , so that $\begin{array} { r } { \theta _ { k } = \int _ { 0 } ^ { 1 } w _ { k } ( u ) f ( u ) } \end{array}$ du. Let $L = \log K . \ \mathrm { A l l } \ o ( 1 )$ terms hold along arbitrary joint sequences with $T / L \to \infty$ and $C / K  \infty$

Labels: a lower bound for any adaptive design. Strengthen the oracle further: it may request a fresh label at any percentile it chooses, and answers are free. In the coordinate $t = - \log u .$ the winner of k answers has density $k e ^ { - k t }$ . Fix $A \geq 2$ and a mesh $h > 0$ , let $d = \left\lfloor ( L - 2 \log A ) / h \right\rfloor$ and $t _ { j } = ( A / K ) e ^ { j h }$ for $j = 0 , \ldots , d ,$ and use the cells $[ t _ { j - 1 } , t _ { j } )$ plus two outer cells. Put an independent smooth cosine-squared prior of radius $\rho$ around the Bernoulli mean $1 / 2$ in each cell. With $p _ { k j }$ the winner probability of cell $j$ at budget k and $q _ { j }$ the prior-averaged expected label count in cell $j$ divided by $T _ { i }$ , the van Trees inequality (Gill and Levit, 1995), which holds for adaptive designs because action probabilities carry no parameter score and label score increments are martingale diferences, gives

$$
R \geq \frac { 1 - 4 \rho ^ { 2 } } { 4 T } \operatorname* { m a x } _ { k \leq K } \sum _ { j } \frac { p _ { k j } ^ { 2 } } { q _ { j } + b } , \qquad b = \frac { \pi ^ { 2 } ( 1 - 4 \rho ^ { 2 } ) } { 4 T \rho ^ { 2 } } , \qquad \sum _ { j } q _ { j } \leq 1 .
$$

Truncating a stopped transcript and passing to the limit in $L _ { 2 }$ extends it to expected caps; it also covers biased estimators. Completing $q$ to a probability vector and normalizing $q _ { j } + b ,$ with $R _ { 0 } = d + 2$ cells,

$$
R \geq \frac { 1 - 4 \rho ^ { 2 } } { 4 T ( 1 + R _ { 0 } b ) } \operatorname* { i n f } _ { q \in \Delta } \operatorname* { m a x } _ { k } \sum _ { j } \frac { p _ { k j } ^ { 2 } } { q _ { j } } .
$$

For weights $\lambda _ { k } \geq 0$ summing to one, Cauchy–Schwarz bounds the infimum below by

$$
\Big [ \sum _ { j } \Big ( \sum _ { k } \lambda _ { k } p _ { k j } ^ { 2 } \Big ) ^ { 1 / 2 } \Big ] ^ { 2 } .
$$

Take $\lambda _ { k } = 1 / ( k H _ { K } )$ . On each internal cell, direct integration and

$$
\sum _ { k = 1 } ^ { K } k e ^ { - 2 k t } = \frac { 1 - e ^ { - 2 K t } \{ 1 + K ( 1 - e ^ { - 2 t } ) \} } { 4 \sinh ^ { 2 } t }
$$

give at least $c _ { A } ( 1 - e ^ { - h } ) / ( 2 \sqrt { H _ { K } } )$ inside the sum, where $c _ { A } = e ^ { - 1 / A } \sqrt { 1 - ( 1 + 2 A ) e ^ { - 2 A } }$ . Therefore

$$
R \geq \frac { 1 - 4 \rho ^ { 2 } } { 4 T ( 1 + R _ { 0 } b ) } \frac { d ^ { 2 } ( 1 - e ^ { - h } ) ^ { 2 } c _ { A } ^ { 2 } } { 4 H _ { K } } .
$$

Choose $A = L , h = ( L / T ) ^ { 1 / 3 }$ and $\rho = \operatorname* { m i n } \{ 1 / 8 , ( R _ { 0 } / T ) ^ { 1 / 4 } \}$ . Then $R _ { 0 } / T = O \{ ( L / T ) ^ { 2 / 3 } \}$ , so $h , \rho$ and $R _ { 0 } b$ tend to zero, while $d h / L \to 1$ and $H _ { K } / L \to 1$ , and $R \ge ( 1 - o ( 1 ) ) L / ( 1 6 T )$ along the actual sequence, with no need to fix $K$ first.

Answers: lower bounds. Give every label for free. In the oracle experiment take $f _ { a } ( u ) = 1 / 2 + a u ^ { K - 1 }$ under a shrinking smooth prior around $a = 0 ,$ . The derivative of $\theta _ { K }$ in a is $K / ( 2 K - 1 )$ , and one answer carries Fisher information at most $4 / \{ ( 1 - 4 \rho ^ { 2 } ) ( 2 K - 1 ) \}$ ; with $\rho = \operatorname* { m i n } \{ 1 / 8 , ( K / C ) ^ { 1 / 4 } \}$ the van Trees inequality gives $R _ { \mathrm { o r a c l e } } \geq ( 1 - o ( 1 ) ) K / ( 8 C )$ In the unknown experiment, let an answer score uniformly on [0, 1] and be incorrect with probabilit $1 - \lambda / K$ , or score uniformly on [2, 3] and be correct otherwise. An answer then reveals one Bernoulli bit, and $\theta _ { K } ( \lambda ) = 1 - ( 1 - \lambda / K ) ^ { K }$ A smooth prior around $\lambda = 1 / 2$ with radius min $\{ 1 / 4 , ( K / C ) ^ { 1 / 3 } \}$ gives a squared derivative tending to $e ^ { - 1 }$ and information $( 2 + o ( 1 ) ) / K$ per answer, so $R _ { \mathrm { u n k n o w n } } \geq ( 1 - o ( 1 ) ) K / ( 2 e C )$ . Acquisition actions add no likelihood term under adaptive stopping, and both bounds also hold under deterministic caps.

The complete-pool estimator. For $n \geq K$ answers, sort the scores and set

$$
p _ { k , n } ( r ) = { \binom { r - 1 } { k - 1 } } { \Big / } { \binom { n } { k } } , \qquad U _ { n , k } = \sum _ { r = 1 } ^ { n } p _ { k , n } ( r ) Y _ { ( r ) } .
$$

This is the complete U-statistic that averages the selected label over all subsets of size $k ,$ so it is unbiased for $\theta _ { k }$ and it satisfies the variance bound (11) uniformly over score–correctness laws. To see this, let R be the overlap of two random subsets of size $k , Z = R / k$ , and $\zeta _ { r }$ the covariance of their selected labels when r answers are shared. Conditioning on the shared maximum’s rank and label, expanding the conditional winner mean and integrating by parts gives

$$
\zeta _ { r } = \gamma \Big [ \theta ( 1 - \theta ) - ( 1 - \gamma ) \int _ { 0 } ^ { 1 } z ^ { - 1 - \gamma } G ( z ) \{ z - G ( z ) \} d z \Big ] , \qquad \gamma = r / k ,
$$

with $\begin{array} { r } { G ( z ) = \int _ { 0 } ^ { z } f ( v ^ { 1 / k } ) d v } \end{array}$ , so that $0 \leq G ^ { \prime } \leq 1$ and $G ( 1 ) = \theta$ . The conditional mean used is $g _ { r } ( u , y ) = u ^ { k - r } y + ( k -$ $\begin{array} { r } { r ) \int _ { u } ^ { 1 } v ^ { k - r - 1 } f ( v ) d v } \end{array}$ , and the boundary terms vanish because $0 \leq G ( z ) \leq z$ . After complementing labels so that $\theta \leq 1 / 2$ , admissibility gives max $\{ 0 , z - ( 1 - \theta ) \} \leq G ( z ) \leq \operatorname* { m i n } \{ z , \theta \}$ . The product $G ( z ) \{ z - G ( z ) \}$ is concave in $G ,$ so its minimum over this range is at an endpoint; with $s = 1 - \theta$ , for $z > s$ the upper endpoint’s value exceeds the lower one’s by $\theta ( z - \theta ) - s ( z - s ) = ( 1 - 2 \theta ) ( 1 - z ) \geq 0$ , and for $z \leq s$ the lower value is zero. The lower endpoint is therefore the minimizer, which is the threshold mechanism, and it gives $\zeta _ { r } \leq s ^ { 2 - r / k } - s ^ { 2 }$ . With $x = - \log s \in [ 0 , \log 2 ]$ the overlap decomposition becomes

$$
\mathrm { V a r } ( U _ { n , k } ) \leq e ^ { - 2 x } \mathbb { E } ( e ^ { x Z } - 1 ) \leq \frac { 1 } { 2 e } \mathbb { E } Z + \mathbb { E } Z ^ { 2 } ,
$$

and the hypergeometric overlap has $\mathbb { E } Z = k / n$ and $\mathbb { E } Z ^ { 2 } \leq ( k / n ) ^ { 2 } + 1 / n .$ , which proves (11). All overlap orders are kept; a first-projection approximation at fixed k would not sufice as K grows.

From percentiles to observed ranks. Let $b _ { k , n } ( r ) = ( r / n ) ^ { k } - \{ ( r - 1 ) / n \} ^ { k }$ and $D _ { K , n } = \{ 1 - ( K - 1 ) / ( 2 n ) \} ^ { - 1 }$ For $k \geq 2$ and $r \geq k , b _ { k , n } ( r + 1 ) / b _ { k , n } ( r ) \leq \{ r / ( r - 1 ) \} ^ { k - 1 } \leq r / ( r - k + 1 ) = p _ { k , n } ( r + 1 ) / p _ { k , n } ( r )$ , so $p _ { k , n } ( r ) / b _ { k , n } ( r )$ increases over its support (the case $k = 1$ is immediate). Its value at $r = n \mathrm { { i s } } ( k / n ) / \{ 1 - ( 1 - 1 / n ) ^ { k } \} \le D _ { K , n }$ by the quadratic lower bound on $1 - ( 1 - 1 / n ) ^ { k }$ , hence

$$
p _ { k , n } ( r ) \leq D _ { K , n } b _ { k , n } ( r ) , \qquad 1 \leq k \leq K .\tag{15}
$$

For a density q with $0 < q \leq n / t$ , let $Q _ { i }$ <sub>r</sub> be its mass on rank bin r and include each sorted answer independently with probability $a _ { r } = t Q _ { r } \leq 1$ . Query the included labels and return

$$
\widehat { \theta } _ { k } = \frac { 1 } { 2 } + \sum _ { r } p _ { k , n } ( r ) \frac { I _ { r } } { a _ { r } } \Big ( Y _ { ( r ) } - \frac { 1 } { 2 } \Big ) .
$$

The expected number of labels is t. Given the pool, the estimator has mean $U _ { n , k }$ and variance $\begin{array} { r l } { \frac { 1 } { 4 } \sum _ { r } p _ { k , n } ( r ) ^ { 2 } ( 1 / a _ { r } - 1 ) } & { { } } \end{array}$ Cauchy–Schwarz within rank bins and (15) give, for any union B of bins, $\begin{array} { r } { \sum _ { r \in B } p _ { k , n } ( r ) ^ { 2 } / Q _ { r } \leq \hat { D } _ { K , n } ^ { 2 } \int _ { B } w _ { k } ( u ) ^ { 2 } / q ( u ) d u } \end{array}$ On a tail where $q = n / t$ every bin is included and contributes no sampling variance, so before choosing q the coordinate error satisfies

$$
R _ { k } \leq \frac { k } { 2 e n } + \left( \frac { k } { n } \right) ^ { 2 } + \frac { 1 } { n } + \frac { D _ { K , n } ^ { 2 } } { 4 t } \int _ { \mathrm { o u t s i d e ~ t h e ~ s a t u r a t e d ~ t a i l } } \frac { w _ { k } ( u ) ^ { 2 } } { q ( u ) } d u .
$$

An allocation and its risk. Work first where $c = n L / ( K t )$ lies in a compact subset of [1, 4]. Put $Y = { \sqrt { L } } , J _ { 0 } = \lceil Y \rceil$ ， and a tail of width $h = \lceil n Y / K \rceil / n$ aligned with the rank bins. Let $\begin{array} { r } { g ( u ) = \operatorname* { m a x } _ { j \leq J _ { 0 } } j u ^ { j - 1 } , C _ { 0 } = \int g = O ( 1 + \log J _ { 0 } ) } \end{array}$ and $E = g / C _ { 0 }$ . Use

$$
q ( u ) = \left\{ \begin{array} { l l } { n / t , } & { u \geq 1 - h , } \\ { ( 1 - a ) \big [ ( 1 - d ) \{ ( 1 - u ) \log ( 1 / h ) \} ^ { - 1 } + d E ( u ) / Z \big ] , } & { u < 1 - h , } \end{array} \right.
$$

with $a = ( n / t ) h , d = 8 C _ { 0 } / L$ and $\begin{array} { r } { Z = \int _ { 0 } ^ { 1 - h } E } \end{array}$ . Both components below the tail integrate to one there, so $q$ is a density. Also $\begin{array} { r } { C _ { 0 } = 1 + \sum _ { j < J _ { 0 } } j ^ { j } / ( j + 1 ) ^ { j + 1 } = \breve { O } ( \log J _ { 0 } ) , Z \geq 1 - J _ { 0 } h } \end{array}$ , and

$$
\operatorname* { s u p } _ { u < 1 - h } \frac { q ( u ) } { n / t } \leq \frac { 1 - a } { n / t } \Big \{ \frac { 1 - d } { h \log ( 1 / h ) } + \frac { d J _ { 0 } } { C _ { 0 } ( 1 - J _ { 0 } h ) } \Big \} \longrightarrow 0 ,
$$

so $q \leq n / t$ once K is large. With $A _ { 0 } = \log ( 1 / h ) / \{ ( 1 - a ) ( 1 - d ) \}$ , integrating the logarithmic component bounds the second moment below the tail by

$$
A _ { 0 } \frac k { 2 ( 2 k - 1 ) } ( 1 - h ) ^ { 2 k - 1 } \{ 1 + ( 2 k - 1 ) h \} .
$$

For $k \leq J _ { 0 }$ the envelope gives the sharper $L / \{ 8 ( 1 - a ) \}$ . For $J _ { 0 } < k \le \lfloor K / \sqrt { Y } \rfloor$ the display is at most $( 1 + o ( 1 ) ) L / 4$ and for larger k it is $o ( L )$ uniformly because $( 2 k - 1 ) h \to \infty$ . The complete-pool variance is negligible against $L / t$ in the middle range and has leading term at most $K / ( 2 e n )$ in the upper range. With (11) the three ranges give

$$
\operatorname* { s u p } _ { P } \operatorname* { m a x } _ { k } \mathrm { M S E } ( \widehat \theta _ { k } ) \leq \operatorname* { m a x } \Big \{ \frac { L } { 1 6 t } , \frac K { 2 e n } \Big \} + o ( L / t ) .
$$

In the oracle experiment, include each answer independently with probability $q ( U _ { i } ) / ( n / t )$ and use $\widehat { \theta } _ { k } \ = \ \frac 1 2 \ +$ $\begin{array} { r } { \frac { 1 } { t } \sum _ { i \leq n } A _ { i } w _ { k } ( U _ { i } ) ( Y _ { i } - \frac { 1 } { 2 } ) / q ( U _ { i } ) } \end{array}$ . Its expected label count is t, its variance is $\begin{array} { r l r } { \frac { 1 } { 4 t } \int w _ { k } ^ { 2 } / q - ( \dot { \theta } _ { k } - 1 / 2 ) ^ { 2 } / n } \end{array}$ , and its second moment on the saturated tail is at most $k ^ { 2 } / \{ ( n / t ) ( 2 k - 1 ) \}$ }. The same three ranges give

$$
\operatorname* { s u p } _ { P } \operatorname* { m a x } _ { k } \mathrm { M S E } ( \widehat \theta _ { k } ^ { \mathrm { o r a c l e } } ) \leq \operatorname* { m a x } \Big \{ \frac { L } { 1 6 t } , \frac { K } { 8 n } \Big \} + o ( L / t ) .
$$

Both estimators use ordinary answers, and the rank estimator uses only their observed order. The bounds are uniform for c in the compact range.

For general C and $T ,$ , take $n = \operatorname* { m i n } \{ \lfloor C \rfloor , \lfloor 4 K T / L \rfloor \}$ and $t = \mathrm { m i n } \{ T , n L / K \}$ . Then $n / K  \infty , t / L  \infty$ and $c \in [ 1 , 4 + o ( 1 ) ]$ ; below this range the answer term dominates and above it the label term does, so the upper and lower bounds match along oscillating as well as convergent ratios of resources.

Deterministic label caps. To impose a cap $B = \lfloor T \rfloor$ on every run, recompute either inclusion design with expected count $( 1 - \xi ) B$ , where $\xi = \sqrt { 2 4 \log B / B }$ , draw the whole inclusion mask before requesting any label, and return $1 / 2$ if the mask exceeds B; otherwise use the clipped estimator. Coupling gives $\mathrm { M S E _ { c a p p e d } \leq M S E _ { u n c a p p e d , c l i p p e d } + I P ( o v e r f l o w ) } / 4$ and Bernstein’s inequality bounds the overflow probability by $B ^ { - 8 }$ , which is $o ( L / T )$ . Answer counts are deterministic already. The leading constants are therefore the same under both kinds of cap. The oracle has at least the information of the unknown experiment, so $R _ { \mathrm { u n k n o w n } } \geq R _ { \mathrm { o r a c l e } }$ , with equality to first order exactly in the label-limited region.

## G Experimental details

## G.1 Stored score pools

Every stored pool is a finite list of questions with a finite set of scored and graded answers per question. An audit draws answers uniformly with replacement from a question’s set, so the pool defines an exact answer law whose curve we compute in closed form, ties included.

• Weaver (Saad-Falcon et al., 2025), released under the MIT license according to its dataset cards. Eight answer sets: GPQA, MATH500, MMLU and MMLU-Pro questions answered by Llama-3.1-8B-Instruct and Llama-3.1- 70B-Instruct, 100 answers per question, each set scored by 19 to 27 reward models and verifiers. The questions of each benchmark are split into two halves once, by question, and the same split is used for every scorer. The 185 held-out pools use the evaluation half (250 to 360 questions). Seven further pools use Weaver’s combined score on the full question sets (500 to 719 questions).

• CodeRM (Ma et al., 2025): the execution results released with the CodeRM code repository. Twenty-four pools: HumanEval+ and MBPP+ (Liu et al., 2023) and LiveCodeBench tasks, solutions from Llama-3-8B, Llama-3-70B, GPT-3.5-turbo and GPT-4o-mini, and unit tests generated by CodeRM-8B or Llama-3.1-70B. The score is the number of 100 generated unit tests a solution passes, the correctness bit is the benchmark’s hidden tests, and each task has 100 solutions (164 to 378 tasks per pool).

• CodeContests (Li et al., 2022): DeepMind-provided non-code materials are CC BY 4.0; third-party materials may have separate terms. One pool of 104 test tasks with 16 human-written Python solutions each, graded by the dataset, and scored by negative length in bytes, so the shortest program is selected.

Only the Weaver evaluation halves are held out. The other 32 pools come from sources seen during development and are reported separately.

## G.2 Protocol

The paired audit’s settings were chosen among 26 configurations of stratified betting audits (3 paired, 23 per-row) on the Weaver development halves, by the lowest mean ratio of generated answers to the per-pool cheapest competitor at $K = 2 5 6$ and 1024, and then frozen: $\alpha _ { S } = 0 . 9 5 \delta _ { \mathrm { { \small f } } }$ , bet cap $\lambda _ { \mathrm { m a x } } = 0 . 9 5$ , one pseudo-pair, the bet $\varepsilon / ( \varepsilon + \hat { v } )$ , and 0.05 δ for the final exact-binomial look. Evaluation ran five independent audits per pool and horizon at $\varepsilon = 1 / 3 2$ and $\delta = 0 . 0 5$ . During development, provisional evaluation runs of an earlier configuration were launched and one completed; its output was set aside unread until the final method had been fixed and evaluated. The per-pool minimum over three competing designs is an optimistic denominator. The rank-based audit’s calibration size of 1,200 paths was chosen after its outcomes on the Weaver pools were known, the nested design uses the fixed look ratio 1.25, and the exact-binomial designs have no tuning parameter.

## G.3 Competing certified audits

All competitors meet the same requirement: half-width ε at every budget with simultaneous coverage 1 − δ for every answer law. The four designs in the first three items draw questions uniformly at random.

• Exact binomial with completion. A predetermined number of paths is sized so that an exact-binomial interval has half-width at most ε at every possible count. For each budget in turn, paths are revealed in order until the set of intervals consistent with every value of the unrevealed paths has width at most 2ε.

• Nested exact binomial. Looks grow geometrically by a factor 1.25, the error is split over budget–look pairs in advance, each new path is generated only up to the largest unresolved budget, and a budget retires when its interval is narrow enough. The last look resolves every budget.

• Rank-based stopping. Complete calibration paths (1,200 of them) learn a score threshold above which a correct incumbent can be frozen; a distribution-free rank bound limits later threshold exceedances, and a widened exact-binomial interval covers the stopped paths. A nested version adds early looks with error spending.

• Fixed designs. Every question receives the same number of full paths, with a Hoefding or an exact-binomia interval and a union over budgets. These designs are balanced but cannot see the within-question variance.

• Record design (Fitas, 2026). Independent full paths with labels at score records and simultaneous bands, sized for the same (ε, δ).

Table 4 lists every design on the held-out pools. The first three form the prespecified comparator of Table $2 ;$ at K = 64 and 256 the cheapest of all six is the cheapest of those three on every held-out pool.

Table 4. Median ratio of each certified design’s cost to the paired audit’s on the 185 held-out pools (answers / correctness queries). The fixed designs and the record design were run at K = 64 and 256.
<table><tr><td>design</td><td>K = 64</td><td>K = 256</td><td>K = 1024</td></tr><tr><td>exact binomial with completion</td><td>1.49 / 1.37</td><td>1.98 / 1.47</td><td>2.74 / 1.66</td></tr><tr><td>nested exact binomial</td><td>1.80 / 1.70</td><td>2.13 / 1.75</td><td>2.73 / 1.78</td></tr><tr><td>rank-based stopping</td><td>1.37 / 1.57</td><td>1.54 / 1.68</td><td>1.90 / 1.88</td></tr><tr><td>nested rank-based stopping</td><td>1.36 / 1.63</td><td>1.55 / 1.73</td><td>1.87 / 1.94</td></tr><tr><td>fixed Hoeffding</td><td>2.15 / 1.99</td><td>2.69 / 2.00</td><td></td></tr><tr><td>fixed exact binomial</td><td>1.65 / 1.53</td><td>2.12 / 1.60</td><td></td></tr><tr><td>record design (Fitas, 2026)</td><td>2.06 / 2.25</td><td>2.62 / 2.73</td><td></td></tr></table>

Uncertified practice. A within-question bootstrap band (199 replicates) at the budget of the fixed Hoefding design missed the exact curve in 58 of 925 held-out runs at K = 64 (6.3%), and in 70 of 1,085 runs over all pools. The fixed Hoefding design missed in none of its 370 held-out runs at the same budget.

## G.4 Ablation

All arms run on the stored pools with the random-number streams of the main study (the balanced exact-binomial arm draws its extra balanced rounds after them), and the paired and per-row costs of every held-out run are reproduced exactly. Balanced exact binomial runs the nested exact-binomial looks on balanced rounds, each look rounded up to a whole round and each tail inverted at the corrected level of Lemma 8. Per-row betting is a stratified confidence sequence that centers each question’s outcome at its running mean with a quarter pseudo-count, estimates the variance from the running squared residuals, bets min $\{ 0 . 6 , \varepsilon / \hat { \sigma } ^ { 2 } \}$ and checks after every round; its settings were frozen during development. A matched version uses the paired audit’s bet $\varepsilon / ( \varepsilon + \hat { \sigma } ^ { 2 } )$ and cap 0.95. Table 5 gives the medians.

## G.5 The all-subsets audit

The implementation uses bet cap 0.9, a prior of one pseudo-row for the predictions $^ { c _ { x , k } , }$ a first round with zero bet that only sets the predictions, and the error split 0.95 δ for the sequence and 0.05 δ for a fixed-round exact-binomial look on the binary outcome of each full path, inverted at the corrected level of Lemma 8. Its unbiasedness requires the answers within a path to be independent given the question.

Table 5. Ablation on the 185 held-out pools: median ratio to the cheapest competing certified audit (five-run means).
<table><tr><td>design</td><td> $K = 6 4$ </td><td> $K = 2 5 6$ </td><td> $K = 1 0 2 4$ </td></tr><tr><td>nested exact binomial, random questions</td><td>1.42</td><td>1.45</td><td>1.45</td></tr><tr><td>nested exact binomial, balanced rounds</td><td>1.45</td><td>1.45</td><td>1.47</td></tr><tr><td>per-row betting, balanced rounds</td><td>0.86</td><td>0.75</td><td>0.63</td></tr><tr><td>per-row betting, matched bet and cap</td><td>0.91</td><td>0.81</td><td>0.70</td></tr><tr><td>paired audit</td><td>0.74</td><td>0.66</td><td>0.53</td></tr><tr><td>balanced / random nested exact binomial</td><td>1.01</td><td>1.00</td><td>1.03</td></tr><tr><td>paired / better per-row variant</td><td>0.91</td><td>0.89</td><td>0.83</td></tr><tr><td>pools where paired is cheaper than both per-row variants</td><td>185/185</td><td>185/185</td><td>185/185</td></tr></table>

## G.6 The cost law

The law across settings is fitted to the five-run mean costs at $\varepsilon = 1 / 3 2$ and $K \in \{ 6 4 , 2 5 6 , 1 0 2 4 \}$ and the two-run mean costs at $K = 6 4$ and $\varepsilon \in \{ 1 / 1 6 , 1 / 6 4 \}$ , with $\Sigma _ { \mathrm { w } }$ computed exactly from each pool; its 90th-percentile relative error is 12.0%. The law used for the MMLU-Pro prediction was fitted earlier, to two paired audits per pool at $K = 6 4$ and $\varepsilon = 1 / 3 2$ on all 217 pools, run with bet cap 0.6. It was fitted before any answer of the study, pilot answers included, was generated. The paired audits of the MMLU-Pro study used cap 0.6 as well. Replaying the three stored audit streams at caps 0.6 and 0.95 gives identical best-of-k costs (78,800, 80,400 and 78,200 answers), because the bet $\varepsilon / ( \varepsilon + \hat { v } )$ never exceeded 0.58; for pass@k and majority voting the lower cap binds on 66 to 143 of about 290 bets and moves the cost by a few percent in either direction.

## G.7 The MMLU-Pro study

The protocol was written before any evaluation answer was generated. It fixed a subset of 100 MMLU-Pro questions from mathematics, physics, chemistry and engineering, with 50 further questions for pilots; the model gpt-6-luna (the interface reports no dated version; answers were generated on 27–28 September 2026) at temperature 1, without reasoning efort, with token log-probabilities, at most 400 completion tokens and eight answers per request; a prompt asking for a brief reason followed by a final line “Answer: $\mathrm { X } '$ ; the verifier score, which is the mean token log-probability of a response with a valid extracted letter, invalid responses receiving the lowest score; correctness, which is agreement of the extracted letter with the published answer key; three audit streams consumed in generation order with common random numbers across audits; a separate reference stream; and the comparison designs.

Audit streams were generated through the batch interface, and part of the reference stream through the flexiblepriority interface with the same parameters. Answers from diferent generation windows agree: across the 100 questions, single-answer accuracy in the audit streams difers from the reference stream by 0.0022 on average (paired $t = 1 . 3 0 )$ . Answers within a request behave as independent draws: given the stream and the question, the intraclass correlation of correctness among the eight answers of a request is 0.005. Of the 716,384 answers, 5.2% had no valid letter and 12 were truncated.

Reference curves come from the 409,448 reference answers: exact finite-sample formulas for best-of-k and pass@k, and an exact multinomial computation for majority voting. The all-subsets audits for pass@k and majority voting stopped at the same round in each stream (8, 8 and 7), and their paths were never shortened, so their costs equal the number of rounds times $1 0 0 \times 6 4$

## G.8 Computation

Prefix winners along a generated path are updated in time linear in its length. Updating the active budgets after a pair of rounds takes $O ( M K )$ operations, so a paired audit of R rounds runs in $O ( N + R M K )$ time with $O ( M K )$ working memory. The multilevel audit processes an observation of level ℓ in $O ( b _ { \ell } )$ time with $O ( K )$ accumulators. All audits ran on one CPU with eight worker processes; the 3,255 pool–horizon–seed jobs of the stored-pool study, each running all six audits, took 936 seconds in total.