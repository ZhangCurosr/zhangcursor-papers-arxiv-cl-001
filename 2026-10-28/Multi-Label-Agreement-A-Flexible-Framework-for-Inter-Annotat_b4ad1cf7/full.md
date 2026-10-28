# Multi-Label Agreement: A Flexible Framework for Inter-Annotator Reliability

Tim Menzner<sup>1</sup>, Jochen L. Leidner<sup>2</sup>

<sup>1</sup> Center for Responsible Artificial Intelligence (CRAI), Coburg University of Applied Sciences

tim.menzner@hs-coburg.de

<sup>2</sup> Center for Responsible Artificial Intelligence (CRAI), Coburg University of Applied Sciences

leidner@acm.org

Reliable evaluation of inter-annotator agreement (IAA) is importantfor assessing the consistency of annotation processes. While established IAA metrics such as Cohen’s kappa, Fleiss’ kappa, and Krippendorff’s alpha are commonly applied in single-label settings, multi-label tasks require additional decisions about how full label sets, joint selections, and joint omissions should be compared.

In this work, we present Multi-Label Agreement (MLA), a flexible framework for evaluating agreement in multi-label annotation tasks. MLA preserves thefull binary label vectors assigned to each item and uses average pairwise Hamming agreement, thereby accounting for both joint selections and joint omissions. Its pooled-marginal form is not a mathematically independent coefficient: it is a chance-disagreement-weighted average of label-wise Fleiss/Scott-type coefficients and is closely related to Krippendorff’s alpha with normalized Hamming distance. The framework supports pooled-marginal, Gwet AC1, and fixed-baseline normalizations, yields item-level, labellevel, and corpus-level results, and includes a conditional minimum-agreement diagnostic. Through a worked example and Monte Carlo simulations, we show how the resulting values depend on label prevalence, the declared label inventory, and the selected model of expected agreement. We also provide a Python implementation. MLA thus offers a transparent and interpretable framework for multi-label agreement analysis without claiming that one normalization is universally preferable.

## 1. Introduction

To assess the reproducibility of an annotation process, annotation studies commonly report inter-annotator agreement, also known as inter-coder or inter-rater agreement (Carletta 1996; Artstein and Poesio 2008; Leidner 2022). Agreement does not by itself prove that the annotations are valid or that they form an upper bound for a downstream classifier, but it can reveal unclear guidelines, inconsistent use of categories, and systematic differences between annotators.

This is done by having a set of human subjects annotate a shared set of data items so their individual decisions can be compared. After all annotators annotate the same piece of data and inter-annotator agreement (IAA) is established, annotators may each work on their own data subset without overlap to save time and money.

Factors that hurt IAA include unspecified, ambiguous, or differently interpreted annotation guidelines as well as human performance effects like lapses of concentration. A good annotation guideline codebook describes each class clearly and provides examples and counterexamples to differentiate it from other classes. Typically, writing the codebook requires several iterations, with multiple updates following each round of pre-trial annotations (Artstein 2017; Leidner 2022). At the same time, disagreement is not always annotation noise. In subjective tasks it may reflect genuine differences in interpretation and can itself be useful information (Plank, Hovy, and Søgaard 2014; Uma et al. 2021).

Tasks may be inherently ambiguous given a particular set of categories, or different annotators may display divergent annotation behavior even when using the same codebook, due, for example, to concentration issues.

Depending on the annotation setup, several established metrics can be used to assess IAA. With two annotators and one categorical label for each item, Cohen’s κ (Cohen 1960) uses annotator-specific marginal distributions. Scott’s $\pi$ and Fleiss’ κ instead use pooled marginals, with Fleiss’ formulation accommodating more than two annotations per item (Fleiss 1971; Di Eugenio and Glass 2004; Artstein and Poesio 2008). Krippendorff’s α (Krippendorff 2011, 2013) supports any number of annotators, missing data, and userdefined distance functions. This last feature also permits set-valued or vector-valued annotations.

For a formal description of a multi-label annotation task, let N be the number of data items i to be annotated by n annotators, each of whom assigns a total of k category labels (assigning in this case also includes not assigning, as the absence of a label is also information). For each item and annotator, we therefore do not have a single decision but a set of decisions L with $L \in \{ 0 , 1 \} ^ { k }$

Most common IAA metrics, however, such as the ones mentioned above, assume that when there are k possible categories for each item $i ,$ each annotator must choose one of the labels. Thus, for each item and annotator, you end up with $\ell _ { i , c } \in \{ 1 , 2 , \ldots , k \}$ where $\ell _ { i , c }$ denotes the single category label selected by annotator (coder) c for item i.

There are at least two natural ways to proceed. One can treat every label as a separate binary annotation problem and combine the resulting statistics, or treat the entire label set as one structured value and select a distance between two sets. These approaches are not necessarily different: normalized Hamming distance is additive over labels, and therefore reduces the vector comparison to a particular weighting of label-wise comparisons.

This paper makes that relationship explicit. We call the resulting framework Multi-Label Agreement (MLA), retaining the terminology used in the accompanying software. MLA is not a mathematically independent coefficient family; instead, our contributions are (i) an exact decomposition into label-wise coefficients, (ii) the connection to Krippendorff’s α with Hamming distance, (iii) a conditional lower bound showing how far agreement can fall for the observed prevalences, and (iv) a practical comparison of alternative chance terms and set similarities.

## 2. Related Work

## 2.1 Past Work on Multi-Label IAA

Multi-label agreement has been approached both through dedicated coefficients and through general weighted agreement frameworks. The latter perspective is particularly important because Krippendorff’s α can treat a label set as a structured value and use a distance such as Hamming, Jaccard, or MASI (Artstein and Poesio 2008; Passonneau 2006). Recent annotation systems consequently use α with MASI distance for multi-label monitoring (Juki´c et al. 2023).

Rosenberg and Binkowski (2004) proposed an augmentation of Cohen’s kappa, treating multiple labels not as a legitimate outcome of a task but rather a problem arising from annotator uncertainty, following the assumption that there has to be one true label and annotators just struggle to agree which one it is. The way it is calculated essentially boils down to the kappa for a scenario where coders were only allowed to give one label and had randomly selected one of multiple provided labels. Thus, this metric has been designed to deflate with the number of labels used, a higher number of labels automatically leads to a lower score. This is not really suitable for an annotation task where an item could have several true labels.

Marchal et al. (2022) propose a computational approach that uses bootstrapping to estimate chance agreement in multi-label data, while assessing agreement only over the intersection of annotators’ label sets and deliberately ignoring labels that were never assigned to an item. While this is a defensible design choice, the simulated (rather than analytic) chance term can be more difficult to interpret, and discarding the information conveyed by joint non-selections means that agreement on a label’s absence is treated as if the label had never been available for that item in the first place.

MASI was introduced by Passonneau (2006) and combines the Jaccard index with a penalty factor based on the degree of overlap, ranging from no penalty for an exact match, to a small penalty for a subset match, to a higher penalty for non-subset overlap between label sets. MASI itself is a distance or similarity, not a chance-corrected coefficient; it can be reported directly or supplied as the distance function in a coefficient such as α.

Bhowmick, Basu, and Mitra (2008) introduce the Am coefficient, a κ-style statistic that extends pairwise agreement to the multi-label setting by treating every unordered pair of categories as a separate nominal variable. For each item, annotators’ three possible decisions for a category pair, (0,0), (1,1), or $( 0 , 1 ) / ( 1 , 0 )$ , are compared; the observed overlap is then normalized with a chance term derived from the product of the individual annotators’ marginal probabilities. Because it averages over all pairs, it implicitly weights errors between frequently co-occurring labels more heavily, yields only a single corpuswide score, and inherits $\kappa \mathrm { ' s }$ sensitivity to skewed prevalence while further scaling quadratically in the number of labels.

Related work on learning from disagreement and soft evaluation asks a different but connected question: whether the distribution of annotator judgements should be preserved rather than reduced to a single gold label (Uma et al. 2021; Rizzi et al. 2024; Harju and Mesaros 2023). Multi-label confusion matrices and tensors likewise provide detailed classification diagnostics but do not themselves model chance agreement (Heydarian, Doyle, and Samavi 2022; Krstini´c et al. 2024). These lines of work reinforce a central point of the present paper: a single corpus-level coefficient should be accompanied by diagnostics that preserve where and how disagreement occurs.

## 2.2 The Ongoing Chance Correction Debate

Many agreement coefficients have the form ${ ( A _ { o } - A _ { e } ) } / { ( 1 - A _ { e } ) }$ , where $A _ { o }$ is observed agreement and $A _ { e }$ is a model-dependent baseline. The word “chance” can obscure that $A _ { e }$ is not observed independently of the data: it follows from a particular model of how annotations would be allocated in the absence of the agreement of interest (Di Eugenio and Glass 2004; Artstein and Poesio 2008).

For a binary label $j ,$ let $p _ { j }$ denote the pooled proportion of positive decisions across annotators and items. The corresponding baseline $\dot { p } _ { j } ^ { 2 } + ( 1 - \dot { p } _ { j } ) ^ { 2 }$ is Fleiss/Scott-type because it uses a pooled marginal. It should not be called Cohen’s $\kappa ,$ whose two-annotator model keeps the annotators’ marginal distributions separate.

Kappa-like coefficients can become small in situations with high raw agreement and skewed prevalence. Gwet’s AC1 (Gwet 2001) uses a different baseline and often remains higher in such situations. PABAK (Byrt, Bishop, and Carlin 1993), also known in this binary form as the Brennan–Prediger coefficient, fixes the baseline at 0.5.

Konstantinidis et al. (Konstantinidis, Le, and Gao 2022) provide simulation evidence that Gwet’s AC1 often attains lower sampling variance than several kappa estimators in multiple-rater, binary-outcome settings and therefore recommend AC1 when skewed prevalences and small sample sizes make kappa unstable.

Zec et al. (Zec et al. 2017) highlight the classic "kappa paradox", cases of high raw agreement but extreme category prevalence where Cohen’s κ can appear deceptively low, and show empirical examples in which AC1 (and related measures) avoid that particular interpretive pitfall. Conversely, Vach and Gerke (Vach and Gerke 2023) stress that AC1 is not a drop-in replacement: substituting AC1 for κ changes the underlying model rather than repairing the same estimand. We therefore treat these normalizations as sensitivity analyses. Their values answer different questions and should not be selected after looking at which one gives the most favourable result.

## 3. Multi-Label Agreement as a Decomposable Construction

## 3.1 Observed agreement

Let $L _ { c , i } = ( \ell _ { c , i , 1 } , \ldots , \ell _ { c , i , k } ) \in \{ 0 , 1 \} ^ { k }$ be the vector assigned to item i by annotator $c .$ The normalized Hamming distance and its corresponding agreement are

$$
d _ { H } ( L _ { c , i } , L _ { c ^ { \prime } , i } ) = \frac 1 k \sum _ { j = 1 } ^ { k } \mathbf { 1 } ( \ell _ { c , i , j } \neq \ell _ { c ^ { \prime } , i , j } ) , \qquad s _ { H } = 1 - d _ { H } .
$$

Thus, a joint selection (1, 1) and a joint omission (0, 0) both count as agreement. With $h = { \binom { n } { 2 } }$ annotator pairs, the item-level and corpus-level observed agreements are

$$
A _ { o , i } = \frac { 1 } { h } \sum _ { 1 \leq c < c ^ { \prime } \leq n } s _ { H } ( L _ { c , i } , L _ { c ^ { \prime } , i } ) ,\tag{1}
$$

$$
A _ { o } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } A _ { o , i } .\tag{2}
$$

For label j alone, define

$$
A _ { o , j } = \frac { 1 } { N h } \sum _ { i = 1 } ^ { N } \sum _ { c < c ^ { \prime } } \mathbf { 1 } ( \ell _ { c , i , j } = \ell _ { c ^ { \prime } , i , j } ) .
$$

Because Hamming agreement is additive, $\begin{array} { r } { A _ { o } = k ^ { - 1 } \sum _ { j } A _ { o , j } } \end{array}$ . This equality is useful but also marks a limitation: the construction compares the components of the vectors

independently and does not retain information about dependencies or co-occurrence patterns between labels.

## 3.2 Alternative baseline models

For binary label $j ,$ its pooled prevalence is

$$
p _ { j } = \frac { \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { n } \ell _ { c , i , j } } { n N } .
$$

The pooled-marginal (Fleiss/Scott-type) agreement baseline is

$$
A _ { e , j } ^ { ( F ) } = p _ { j } ^ { 2 } + ( 1 - p _ { j } ) ^ { 2 } , \qquad A _ { e } ^ { ( F ) } = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } A _ { e , j } ^ { ( F ) } .
$$

For comparison, the binary AC1 chance term and the fixed PABAK chance term are

$$
A _ { e , j } ^ { ( A C 1 ) } = 2 p _ { j } ( 1 - p _ { j } ) ,
$$

$$
A _ { e } ^ { ( A C 1 ) } = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } A _ { e , j } ^ { ( A C 1 ) } ,\tag{3}
$$

$$
A _ { e , j } ^ { ( P B ) } = 0 . 5 ,
$$

$$
A _ { e } ^ { ( P B ) } = 0 . 5 .\tag{4}
$$

For binary decisions $A _ { e , j } ^ { ( A C 1 ) } = 1 - A _ { e , j } ^ { ( F ) }$ , but the two quantities are used as baselines in different coefficient models and must not be interpreted as interchangeable estimates of the same null process.

For $m \in \{ F , A C 1 , P B \}$ , the aggregate coefficient is

$$
A _ { m l } ^ { ( m ) } = \frac { A _ { o } - A _ { e } ^ { ( m ) } } { 1 - A _ { e } ^ { ( m ) } } .\tag{5}
$$

It equals one for perfect observed agreement. A value of zero means that $A _ { o }$ equals the selected baseline; what that statement means depends on m. Item-level values can be obtained by replacing $A _ { o }$ with $A _ { o , i }$ while keeping the corpus-level baseline, but these are diagnostics relative to the corpus and not independent reliability estimates.

The raw prevalence

$$
p _ { r } = { \frac { 1 } { k } } \sum _ { j = 1 } ^ { k } p _ { j }
$$

is useful as a compact summary, but it does not replace the label-wise prevalences because very different vectors $( p _ { 1 } , \ldots , p _ { k } )$ can have the same $p _ { r }$

## 3.3 Exact relation to label-wise coefficients

Let

$$
A _ { j } ^ { ( m ) } = \frac { A _ { o , j } - A _ { e , j } ^ { ( m ) } } { 1 - A _ { e , j } ^ { ( m ) } }
$$

be the coefficient for label $j ,$ whenever its denominator is non-zero, and let $\mathcal { I } _ { m } = \{ j$ $1 - A _ { e , j } ^ { ( m ) } > 0 \}$ . Labels outside $\mathcal { I } _ { m }$ contribute zero to both the numerator and denominator below. Substituting $\begin{array} { r } { A _ { o } = k ^ { - 1 } \sum _ { j } A _ { o , j } } \end{array}$ and $\begin{array} { r } { A _ { e } ^ { ( m ) } = k ^ { - 1 } \sum _ { j } A _ { e , j } ^ { ( m ) } } \end{array}$ into Equation 5 gives, provided that the aggregate denominator is non-zero,

$$
A _ { m l } ^ { ( m ) } = \frac { \sum _ { j \in \mathcal { I } _ { m } } \left( 1 - A _ { e , j } ^ { ( m ) } \right) A _ { j } ^ { ( m ) } } { \sum _ { j \in \mathcal { I } _ { m } } \left( 1 - A _ { e , j } ^ { ( m ) } \right) } .\tag{6}
$$

The aggregate is therefore a weighted average of label-wise coefficients, not a new functional family. Under the pooled-marginal baseline, the weights are $2 p _ { j } ( 1 - p _ { j } )$ , so labels with extreme prevalence receive less weight. Under AC1 they are $p _ { j } ^ { 2 } + ( 1 - p _ { j } ) ^ { 2 } .$ which gives relatively more weight to extreme labels. Under PABAK all weights are equal. These implicit weights should be reported because they determine what the corpus-level number summarizes.

## 3.4 Relation to Krippendorff’s alpha

The same construction can be written as a distance-based coefficient. Let $D _ { o } = 1 - A _ { o }$ be observed normalized Hamming disagreement. Drawing two label vectors independently from the pooled empirical distribution gives

$$
D _ { e } ^ { ( \infty ) } = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } 2 p _ { j } ( 1 - p _ { j } ) = 1 - A _ { e } ^ { ( F ) } .
$$

Consequently,

$$
A _ { m l } ^ { ( F ) } = 1 - \frac { D _ { o } } { D _ { e } ^ { ( \infty ) } } .
$$

This has the form of Krippendorff’s α with normalized Hamming distance. Standard α draws two distinct values without replacement from the $M = n N$ observed vectors and, for complete balanced data, uses

$$
D _ { e } ^ { ( \alpha ) } = \frac { M } { M - 1 } D _ { e } ^ { ( \infty ) } , \qquad \alpha _ { H } = 1 - \frac { D _ { o } } { D _ { e } ^ { ( \alpha ) } } .
$$

Thus the pooled-marginal MLA coefficient and Hamming α differ only by this finitesample convention and converge as M grows. Krippendorff’s framework is more general because it also provides explicit handling of missing data and permits other distances.

For example, Jaccard or MASI distance deemphasizes joint omissions and instead focuses on overlap between selected labels.

## 3.5 Minimum feasible agreement conditional on prevalence

With more than two annotators, zero pairwise agreement is generally impossible for a binary decision. Without fixing the prevalence, the smallest possible pairwise observed agreement for a label is

$$
A _ { o , \mathrm { m i n } } = { \left\{ \begin{array} { l l } { \displaystyle { \frac { n - 2 } { 2 ( n - 1 ) } } , } & { n { \mathrm { ~ e v e n } } , } \\ { \displaystyle { \frac { n - 1 } { 2 n } } , } & { n { \mathrm { ~ o d d } } . } \end{array} \right. }
$$

A more informative diagnostic keeps the observed number of positive decisions fixed. Let

$$
T _ { j } = \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { n } \ell _ { c , i , j } , \qquad x _ { i , j } = \sum _ { c = 1 } ^ { n } \ell _ { c , i , j } .
$$

For an item with x positive and $n - x$ negative decisions, the number of agreeing annotator pairs is

$$
f ( x ) = { \binom { x } { 2 } } + { \binom { n - x } { 2 } } .
$$

The function f is convex on the integers. If two item counts differ by at least two, moving one positive decision from the larger to the smaller count cannot increase their combined pair agreement. Repeating this exchange shows that, subject to $\textstyle \sum _ { i } x _ { i , j } = T _ { j }$ , agreement is minimized when all item counts differ by at most one.

Set

$$
m _ { j } = \left\lfloor { \frac { T _ { j } } { N } } \right\rfloor , \qquad t _ { j } = T _ { j } - m _ { j } N .
$$

Then $N - t _ { j }$ items receive $m _ { j }$ positives and $t _ { j }$ items receive $m _ { j } + 1$ positives. The conditional lower bound is therefore

$$
A _ { o , \operatorname* { m i n } , j } ^ { ( p _ { j } ) } = \frac { ( N - t _ { j } ) f ( m _ { j } ) + t _ { j } f ( m _ { j } + 1 ) } { N { \binom { n } { 2 } } } ,\tag{7}
$$

where the second term is omitted when $t _ { j } = 0$ . Averaging over labels and applying the same normalization yields

$$
A _ { o , \operatorname* { m i n } } ^ { ( \mathbf { p } ) } = \frac { 1 } { k } \sum _ { j = 1 } ^ { k } A _ { o , \operatorname* { m i n } , j } ^ { ( p _ { j } ) } , \qquad A _ { m l , \operatorname* { m i n } } ^ { ( m ) } = \frac { A _ { o , \operatorname* { m i n } } ^ { ( \mathbf { p } ) } - A _ { e } ^ { ( m ) } } { 1 - A _ { e } ^ { ( m ) } } .
$$

This is a feasible-range diagnostic, not another chance model and not part of the observed agreement. It explains why a coefficient may be unable to attain strongly negative values once the number of annotators and empirical marginals are fixed.

## 3.6 Interpreting Prevalence

When the raw prevalence $p _ { r }$ is near 0, almost every coder assigns 0 to almost every label. Because a (0, 0) comparison counts as agreement, $A _ { o }$ can become high even when coders rarely coincide on the scarce positive cases. This matters in tasks where recognizing an occasional positive label is the actual challenge. Take the rabbit–duck illusion: coders may dispute whether a rabbit, a duck, or both are present, yet they will unanimously agree that other animals are absent. Expanding the inventory with irrelevant animals therefore inflates Hamming agreement without improving reliability on the labels of interest.

A low $p _ { r }$ is not automatically a problem. In some applications positives are genuinely rare and certifying their absence is an important part of the task. Think of aircraftmaintenance engineers who tick possible defects on an inspection checklist: almost all boxes will, and should, remain unticked. Here, joint omissions may be meaningful. The relevant question is therefore not whether negative matches should always count, but whether all labels were genuinely considered and whether confirming their absence has comparable value to confirming their presence.

The mirror image occurs when $p _ { r }$ is close to 1: $A _ { o }$ is then dominated by $( 1 , 1 )$ matches. Whether this is informative again depends on the task. A single global prevalence is insufficient for making this judgement, so the label-wise $p _ { j } , A _ { o , j } ,$ positive agreement, and negative agreement should be inspected alongside a corpus-level coefficient.

Natural ambiguity also has to be separated from annotation error. Genre annotation, for example, may legitimately produce different but defensible label sets. Such disagreement should often be preserved for later analysis rather than hidden behind a reliability threshold (Plank, Hovy, and Søgaard 2014; Uma et al. 2021).

No normalization resolves this domain decision automatically. The pooled-marginal coefficient asks how much observed Hamming agreement exceeds that implied by the empirical label marginals. AC1 uses a different baseline that remains less punitive at extreme prevalence, but can consequently be positive even when annotators draw independently from the same skewed Bernoulli distribution. PABAK fixes a balanced baseline and has the same issue under skewed independent decisions. These are properties of the respective models, not evidence that one coefficient has recovered the uniquely correct amount of reliability. We therefore recommend pre-specifying the primary model and reporting $A _ { o } ,$ , label-wise prevalences, and at least one sensitivity analysis.

## 3.7 Example

To demonstrate the calculations, consider the following subsample from an experiment where three annotators were asked to annotate bias in sentences from news articles using a pre-specified taxonomy of 31 types. Table 1 shows the selected types and the corresponding vectors. This sample is far too small for a substantive reliability conclusion; it is only a worked example.

We take $N = 4$ sentences, k = 31 categories, n = 3 annotators $\textstyle ( h = { \binom { 3 } { 2 } } = 3 )$

<table><tr><td rowspan=1 colspan=1>Sentence</td><td rowspan=1 colspan=1>Annotator 1</td><td rowspan=1 colspan=1>Annotator 2</td><td rowspan=1 colspan=1>Annotator 3</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Faulty Analogy Bias $[ 0 , 0 , \dot { 0 , } 0 , 0 , 0 , 0 , \dot { 0 , } 0 , 0 , 0 ,$  $\bar { 0 } , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 ,$  $0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]$ </td><td rowspan=1 colspan=1>Faulty Analogy Bias $[ 0 , 0 , \dot { 0 , } 0 , 0 , 0 , 0 , \dot { 0 , } 0 , 0 , 0 ,$  $\bar { 0 } , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 ,$  $0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]$ </td><td rowspan=1 colspan=1>Faulty Analogy Bias $[ 0 , 0 , \dot { 0 , } 0 , 0 , 0 , 0 , \dot { 0 , } 0 , 0 , 0 ,$  $\bar { 0 } , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 ,$  $0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]$ </td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Political Bias, Projec-tion Bias $\begin{array} { r } { [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $\begin{array} { r } { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $0 , 1 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]$ </td><td rowspan=1 colspan=1>Speculation Bias, Un-substantiated ClaimsBias, Word ChoiceBias $\begin{array} { r } { [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $\begin{array} { r } { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 1 , 0 , 1 ]$ </td><td rowspan=1 colspan=1>UnsubstantiatedClaims Bias, WordChoice Bias $\begin{array} { r } { [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $\bar { 0 } , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,$  $0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 1 ]$ </td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Ambiguous Attribu-tion Bias, Cherry Pick-ing Bias, Political Bias $\begin{array} { r } { [ 0 , 1 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $\bar { 0 } , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,$  $0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]$ </td><td rowspan=1 colspan=1>Anecdotal EvidenceBias $\begin{array} { r } { [ 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $\begin{array} { r } { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]$ </td><td rowspan=1 colspan=1>Ambiguous Attribu-tion Bias $\begin{array} { r } { [ 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $\begin{array} { r } { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]$ </td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Intergroup Bias, Opin-ionated Bias, Politi-cal Bias, Word ChoiceBias $\begin{array} { r } { [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $\begin{array} { r } { 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , } \end{array}$  $1 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 ]$ </td><td rowspan=1 colspan=1>Ambiguous Attribu-tion Bias, IntergroupBias, Political Bias,Projection Bias, Un-substantiated ClaimsBias $\begin{array} { r } { [ 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $\begin{array} { r } { 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , } \end{array}$  $0 , 1 , 1 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 ]$ </td><td rowspan=1 colspan=1>Ambiguous Attribu-tion Bias, Opinion-ated Bias, ProjectionBias, Word ChoiceBias $\begin{array} { r } { [ 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $\begin{array} { r } { 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , } \end{array}$  $1 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 ]$ </td></tr></table>

Assigned bias labels and corresponding 31-bit vectors by sentence and annotator.

Table 1

For each sentence $i ,$ compute

$$
S _ { \mathcal { L } _ { i } } = \sum _ { c < c ^ { \prime } } I ( L _ { c , i } , L _ { c ^ { \prime } , i } ) , \quad A _ { o _ { i } } = \frac { S _ { \mathcal { L } _ { i } } } { 3 1 \cdot 3 }
$$

$$
i = 1 : \quad S = 3 1 + 3 1 + 3 1 = 9 3 \quad \Rightarrow \ A _ { o _ { 1 } } = 9 3 / 9 3 = 1 . 0 0 0 ,
$$

$$
i = 2 : \quad S = 2 6 + 2 7 + 3 0 = 8 3 \quad \Rightarrow \ A _ { o _ { 2 } } = 8 3 / 9 3 \approx 0 . 8 9 2 5 ,
$$

$$
i = 3 : S = 2 7 + 2 9 + 2 9 = 8 5 \quad \Rightarrow \ A _ { o _ { 3 } } = 8 5 / 9 3 \approx 0 . 9 1 4 0 ,
$$

$$
i = 4 : \quad S = 2 6 + 2 7 + 2 6 = 7 9 \quad \Rightarrow \ A _ { o 4 } = 7 9 / 9 3 \approx 0 . 8 4 9 5 ,
$$

For the pooled-marginal and AC1 baseline terms we obtain

$$
A _ { e } ^ { ( F ) } = \frac { 2 7 . 5 2 7 8 } { 3 1 } \approx 0 . 8 8 8 0 , \qquad A _ { e } ^ { ( a c 1 ) } = 1 - A _ { e } ^ { ( F ) } \approx 0 . 1 1 2 0 .
$$

Hence the overall multi-label agreements under the three methods are

$$
\begin{array} { r l } & { A _ { m l } ^ { ( F ) } = \frac { \frac { 1 + 0 . 8 9 2 5 + 0 . 9 1 4 0 + 0 . 8 4 9 5 } { 4 } - 0 . 8 8 8 0 } { 1 - 0 . 8 8 8 0 } \approx 0 . 2 3 2 0 , } \\ & { A _ { m l } ^ { ( a c 1 ) } \approx 0 . 9 0 3 1 , } \end{array}
$$

$$
A _ { m l } ^ { ( p b ) } \approx 0 . 8 2 8 0 .
$$

As for the raw prevalence, across all $1 2 = 4 \times 3$ decisions per label there are 28 positive assignments, so

$$
p _ { r } = { \frac { 1 } { k } } \sum _ { j = 1 } ^ { k } p _ { j } = { \frac { 2 8 } { 1 2 \cdot 3 1 } } = { \frac { 7 } { 9 3 } } \approx 0 . 0 7 5 3 .
$$

The minimum observed agreements given the empirical prevalence based on the total number of 1s are

<table><tr><td>total  $" 1 " \mathrm { s }$  for a label</td><td>#labels</td><td> $\underline { { A _ { o _ { \mathrm { m i n } } , j } ^ { p _ { j } } } }$ </td></tr><tr><td>0</td><td>20</td><td> $\overline { { 1 . 0 0 0 0 } }$ </td></tr><tr><td>1</td><td>3</td><td> $5 / 6 = 0 . 8 3 3 3$ </td></tr><tr><td>2</td><td>2</td><td> $2 / 3 = 0 . 6 6 6 7$ </td></tr><tr><td>3</td><td>3</td><td> $1 / 2 = 0 . 5 0 0 0$ </td></tr><tr><td>4</td><td>3</td><td> $1 / 3 = 0 . 3 3 3 3$ </td></tr></table>

Averaging over all 31 labels gives

$$
A _ { o _ { \mathrm { m i n } } } ^ { p _ { j } } = { \frac { 2 0 \cdot 1 + 3 \cdot { \frac { 5 } { 6 } } + 2 \cdot { \frac { 2 } { 3 } } + 3 \cdot { \frac { 1 } { 2 } } + 3 \cdot { \frac { 1 } { 3 } } } { 3 1 } } = { \frac { 7 9 } { 9 3 } } \approx 0 . 8 4 9 5 .
$$

This sets the three normalized floors to

$$
A _ { m l , \mathrm { m i n } } ^ { ( F ) } = \frac { \frac { 7 9 } { 9 3 } - 0 . 8 8 8 0 } { 1 - 0 . 8 8 8 0 } \approx - 0 . 3 4 4 0 ,
$$

$$
A _ { m l , \mathrm { m i n } } ^ { ( a c 1 ) } = \frac { \frac { 7 9 } { 9 3 } - 0 . 1 1 2 0 } { 1 - 0 . 1 1 2 0 } \approx 0 . 8 3 0 5 ,\tag{8}
$$

$$
A _ { m l , \mathrm { { m i n } } } ^ { ( p b ) } = { \frac { { \frac { 7 9 } { 9 3 } } - 0 . 5 0 0 0 } { 0 . 5 0 0 0 } } \approx 0 . 6 9 8 9 .
$$

This is a scenario with very low prevalence. The pooled-marginal coefficient is low because most raw agreement is also implied by the empirical marginals. AC1 and PABAK are high, but their conditional floors are high as well: given the observed number of positive decisions, these coefficients cannot approach −1 and in the AC1 case cannot even approach zero. This is why a value without its baseline and feasible range is easy to overinterpret.

Twenty of the 31 pre-specified categories are never selected in the example. Table 2 shows the effect of removing them post hoc. The pooled-marginal coefficient and its floor remain unchanged because constant labels have weight $1 - \bar { A } _ { e , j } ^ { ( F ) } = 0$ in Equation 6. AC1 and PABAK change substantially because constant labels receive non-zero weight

under these models. Neither label universe is automatically correct: the full inventory is appropriate if all annotators genuinely considered all 31 labels, whereas post-hoc removal changes the estimand and can hide the effect of an unnecessarily broad label inventory.
<table><tr><td>Label universe</td><td> $A _ { o }$ </td><td> $A _ { m l } ^ { ( F ) }$ </td><td> $A _ { m l } ^ { ( a c 1 ) }$ </td><td> $A _ { m l } ^ { ( p b ) }$ </td><td> $A _ { o , \mathrm { m i n } } ^ { ( \mathbf { p } ) }$ </td></tr><tr><td>All 31 labels</td><td>0.914</td><td>0.232</td><td>0.903</td><td>0.828</td><td>0.849</td></tr><tr><td>11 selected labels</td><td>0.758</td><td>0.232</td><td>0.646</td><td>0.515</td><td>0.576</td></tr></table>

Table 2  
Effect of the declared label universe in the worked example. Removing never-selected labels leaves the pooled-marginal coefficient unchanged but changes the AC1 and fixed-baseline summaries.

Table 3 reports, for every category that received at least one positive assignment, its prevalence, observed agreement, and the three label-wise coefficients. Their variation highlights why a corpus-level number should not be reported alone.

To put these results into perspective, the same example was also used to calculate the measures mentioned in Section 2 (B gives the number of bootstrap runs):

$$
\begin{array} { r l r } { \mathrm { A u g m e n t e d ~ } \kappa = 0 . 2 5 3 , } & { } & { \qquad \mathrm { b o o t } \mathrm { - F } 1 \left( B = 1 0 0 0 \right) = 0 . 3 6 6 , } \\ { \mathrm { M A S I } = 0 . 5 0 9 , } & { } & { A _ { m } = 0 . 2 8 1 . } \end{array}
$$

These values are not estimates of one common quantity. Augmented κ and $A _ { m }$ apply their own chance models, boot-F1 ignores joint negative decisions, and the directly reported MASI similarity is not chance-corrected. Their disagreement is therefore informative: different summaries reward different aspects of the same annotations.

<table><tr><td rowspan=2 colspan=3>label j</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1> $p _ { j }$ </td><td rowspan=1 colspan=1> $A _ { o , j }$ </td><td rowspan=1 colspan=1> $\underline { { A _ { m l , j } ^ { ( F ) } } }$ </td><td rowspan=1 colspan=1> $\underline { { A _ { m l , j } ^ { ( a c 1 ) } } }$ </td><td rowspan=1 colspan=1> $A _ { m l , j } ^ { ( p b ) }$ </td></tr><tr><td rowspan=11 colspan=3>2 (Ambiguous Attribution Bias)3 (Anecdotal Evidence Bias)5 (Cherry Picking Bias)14 (Faulty Analogy Bias)17 (Intergroup Bias)21 (Opinionated Bias)22 (Political Bias)23 (Projection Bias)27 (Speculation Bias)29 (Unsubstantiated Claims Bias)31 (Word Choice Bias)</td><td rowspan=1 colspan=1>0.333</td><td rowspan=1 colspan=1>0.667</td><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>0.400</td><td rowspan=1 colspan=1>0.333</td></tr><tr><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>-0.091</td><td rowspan=1 colspan=1>0.803</td><td rowspan=2 colspan=1>0.6670.667</td></tr><tr><td rowspan=2 colspan=2>cking ias)</td><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>-0.091</td><td rowspan=1 colspan=1>0.803</td></tr><tr><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=3 colspan=1>1.0000.6670.667</td></tr><tr><td rowspan=1 colspan=1>0.167</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>0.400</td><td rowspan=1 colspan=1>0.769</td></tr><tr><td rowspan=1 colspan=1>0.167</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>0.400</td><td rowspan=1 colspan=1>0.769</td></tr><tr><td rowspan=1 colspan=1>0.333</td><td rowspan=1 colspan=1>0.500</td><td rowspan=1 colspan=1>-0.125</td><td rowspan=1 colspan=1>0.100</td><td rowspan=1 colspan=1>0.000</td></tr><tr><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>0.667</td><td rowspan=1 colspan=1>0.111</td><td rowspan=1 colspan=1>0.467</td><td rowspan=1 colspan=1>0.333</td></tr><tr><td rowspan=1 colspan=1>0.083</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>-0.091</td><td rowspan=1 colspan=1>0.803</td><td rowspan=1 colspan=1>0.667</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>0.667</td><td rowspan=1 colspan=1>0.111</td><td rowspan=1 colspan=1>0.467</td><td rowspan=1 colspan=1>0.333</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.333</td><td rowspan=1 colspan=1>0.667</td><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>0.400</td><td rowspan=1 colspan=1>0.333</td></tr></table>

Table 3  
Label-wise agreement statistics for the 11 labels assigned at least once.

## 4. Behaviour Under Controlled Data Generators

To study how the three normalizations respond to different data characteristics and how they compare to the measures presented in Section 2, we generated synthetic annotation matrices. The purpose is descriptive: the simulations show the consequences of the models and do not establish that one coefficient is generally more valid than another. Two orthogonal factors control the data:

Representative agreement metrics (averaged over $n \in \{ 3 , 5 , 1 0 \}$ , $N \in \{ 2 0 , 5 0 , 1 0 0 \}$ , $k \in \{ 5 , 1 0 , 2 5 \}$ 20 repeats, 100 bootstrap iterations for boot-F1)
<table><tr><td>Scenario</td><td> $A _ { m l } ^ { ( F ) }$ </td><td> $A _ { m l } ^ { ( a c 1 ) }$ </td><td> $A _ { m l } ^ { ( p b ) }$ </td><td>MASI</td><td>Aug. κ</td><td>boot-F1</td><td> $A _ { m }$ </td></tr><tr><td>Perf. Agr. + High Prev.</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.042</td><td>1.000</td><td>1.000</td></tr><tr><td>Perf. Agr. + Mod. Prev.</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.171</td><td>1.000</td><td>1.000</td></tr><tr><td>Perf. Agr. + Low Prev.</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.381</td><td>1.000</td><td>1.000</td></tr><tr><td>High Agr. + High Prev.</td><td>0.524</td><td>0.711</td><td>0.641</td><td>0.870</td><td>0.030</td><td>0.523</td><td>0.426</td></tr><tr><td>High Agr. + Mod. Prev.</td><td>0.634</td><td>0.647</td><td>0.641</td><td>0.817</td><td>0.101</td><td>0.619</td><td>0.495</td></tr><tr><td>High Agr. + Low Prev.</td><td>0.523</td><td>0.710</td><td>0.640</td><td>0.693</td><td>0.164</td><td>0.487</td><td>0.424</td></tr><tr><td>Indep. + High Prev.</td><td>-0.004</td><td>0.532</td><td>0.362</td><td>0.791</td><td>0.000</td><td>0.002</td><td>0.002</td></tr><tr><td>Indep. + Mod. Prev.</td><td>-0.005</td><td>0.006</td><td>0.001</td><td>0.547</td><td>-0.000</td><td>-0.000</td><td>-0.000</td></tr><tr><td>Indep. + Low Prev.</td><td>-0.007</td><td>0.529</td><td>0.359</td><td>0.354</td><td>-0.000</td><td>-0.001</td><td>-0.001</td></tr><tr><td>Min. Feas. + High Prev.</td><td>-0.287</td><td>0.230</td><td>0.052</td><td>0.687</td><td>-0.017</td><td>-0.274</td><td>-0.113</td></tr><tr><td>Min. Feas. + Mod. Prev.</td><td>-0.287</td><td>-0.155</td><td>-0.215</td><td>0.468</td><td>-0.033</td><td>-0.263</td><td>-0.076</td></tr><tr><td>Min. Feas. + Low Prev.</td><td>-0.287</td><td>0.230</td><td>0.052</td><td>0.155</td><td>-0.071</td><td>-0.236</td><td>-0.113</td></tr></table>

Prevalence: For every label the probability that it is set in the hidden ground-truth vector is fixed to high $\dot { ( p = 0 . 8 0 ) }$ , moderate $( p = 0 . 5 0 )$ or low $( p = 0 . 2 0 )$   
Agreement level: After the ground-truth vectors have been drawn, the individual annotator decisions are created as follows.

1. Perfect agreement – every annotator copies the ground truth, hence the observed agreement $\dot { A } _ { o } = 1 . 0 $

2. High agreement – each $0 / 1$ decision is independently flipped with a small error probability $\varepsilon = 0 . 1 0 .$

3. Independent draws – annotators ignore the ground truth and draw their own 0/1 decision from the same Bernoull $( p )$ distribution; any agreement that occurs is therefore implied by the marginal distribution.

Minimum feasible agreement – for every item–label pair we first compute $a = \operatorname* { m a x } \bigl ( 1 , \operatorname* { m i n } ( n - 1 , \lfloor p n + 0 . 5 \rfloor ) \bigr )$ and then choose a of the n annotators at random to assign the value 1, while the remaining $n - a$ assign 0. This guarantees that the realised prevalence $\hat { p } = a / n$ matches the requested $p$ (up to the unavoidable rounding of a) and at the same time forces every pair of annotators to disagree on as many decisions as possible given that prevalence.

For every combination $n \in \{ 3 , 5 , 1 0 \}$ annotators, $N \in \{ 2 0 , 5 0 , 1 0 0 \}$ items and $k \in$ {5, 10, 25} labels, the simulation was repeated 20 times; the table reports the averages of the resulting $3 \times 3 \times 3 \times 2 0 = 5 4 0$ metric values per scenario.

Table 4 reports averages over $n , N ,$ , and k. This compact summary makes the main prevalence interaction visible but should not be read as a study of sampling variance; a full assessment would keep these dimensions separate and report interval estimates.

For perfect agreement, all measures except augmented kappa attain 1.0. The different value of augmented kappa follows from its specific model for multiply labelled data and illustrates why results from different coefficients should not be placed on one interpretive scale without qualification.

Under independent Bernoulli draws, the pooled-marginal coefficient remains close to zero because its baseline matches the generator. AC1 and PABAK become positive at extreme prevalence because they use different baselines. This is not a simulation failure: it shows that zero has a different meaning under the three models. Direct MASI similarity also increases with the number of shared positive labels and is not centred at a chance value.

In the noisy-ground-truth condition, pooled-marginal agreement is smaller at extreme prevalence, AC1 is larger, and PABAK is constant. These patterns can also be obtained analytically. With flip probability ε, two annotators agree with probability $( 1 - \varepsilon ) ^ { 2 } + \varepsilon ^ { 2 }$ , while their marginal positive probability is $q = \varepsilon + p ( 1 - 2 \varepsilon )$ . Figure 1 plots the resulting coefficients continuously over p and also shows the independent-draw generator.

![](images/5e95e40c6784acc1888ce4a5089c1e43b26c3422d23c5413ee1864a6fb046936.jpg)  
Figure 1

![](images/9b9ba710d5767ca6578da47b2fe19fafc8ba5e38f7a5f0ee484624149972c3c8.jpg)  
Analytic prevalence sensitivity. Left: annotators draw independently from $\operatorname { B e r n o u l l i } ( p )$ . Right: annotators copy a latent label and independently flip it with probability $\varepsilon = 0 . 1$ . The pooled-marginal coefficient is calibrated to the independent generator; AC1 and PABAK use different baselines.

The minimum-feasible-agreement rows also confirm the lower-bound analysis in Equation 7: with more than two annotators and fixed binary marginals, strongly negative values may be unattainable. This is why universal verbal cut-offs such as “good” or “poor” agreement are particularly inappropriate here.

The code used to generate Figure 1 is included with the preprint source. The Monte Carlo implementation is available in the accompanying software repository.

## 5. Reporting Recommendations

The analysis suggests that a multi-label reliability report should make the following choices explicit:

1. Label universe. State whether the inventory was fixed before annotation and whether every annotator was expected to consider every label. Never-selected labels should not be removed only because they lower or raise a preferred summary.

2. Meaning of joint omissions. Use Hamming-based agreement when joint absence is informative. If overlap among positive selections is the scientific focus, additionally report a set distance such as Jaccard or MASI, preferably within a chance-corrected framework when such correction is needed.

3. Baseline model. Pre-specify the primary chance model. A pooled-marginal coefficient, AC1, and PABAK do not provide three interchangeable estimates of the same quantity.

4. Diagnostics. Report N, the number of annotators per item, $k , A _ { o } ,$ label-wise prevalences and agreements, and the conditional lower bound. Item-level values can identify difficult examples, but reliability conclusions require an adequate number of items.

5. Uncertainty. For substantive studies, accompany corpus-level and important label-level estimates with confidence intervals obtained through a design-appropriate bootstrap or an analytic method. Resampling must respect the item and annotator structure of the study.

If labels are hierarchical, mutually exclusive in groups, or otherwise dependent, the additive Hamming construction is often not sufficient. In that case, Krippendorff’s framework with a domain-specific distance, or an explicit probabilistic annotator model, is preferable to treating the dependencies as if they did not exist.

## 6. Implementation

A Python implementation is available in the project repository and from the Python Package Index as ml-agreement. The package implements corpus-, item-, and labellevel summaries, the three baseline choices discussed here, label prevalence, and the conditional minimum-agreement calculation. The pooled-marginal option is retained as the package’s legacy k/kappa option; in the paper we use F to avoid confusing pooled marginals with Cohen’s annotator-specific model.

## 7. Limitations

The present analysis is restricted to complete, balanced annotation designs with binary, non-hierarchical labels. The simple formulas do not by themselves handle missing annotations. Treating a missing decision as 0 would incorrectly turn missingness into an observed omission; instead, pair counts and marginal denominators must be computed from the available values, as done in the general formulation of Krippendorff’s α.

Normalized Hamming distance is additive and therefore insensitive to dependencies among labels. It also weights all label decisions equally before chance correction. Hierarchical relations, mutually exclusive subsets, and differences in the cost of false positive and false negative decisions require a different distance or a model tailored to the annotation design.

The worked example contains only four items and is illustrative. The simulation varies several design factors but averages over n, N, and k and does not provide inferential comparisons or confidence intervals. A more complete empirical study should analyse these factors separately, include annotator-specific marginals and correlated labels, and compare against Hamming α using the same data. Finally, tasks with genuine ambiguity may be better represented by confidence judgements or distributions over interpretations rather than binary decisions alone.

## 8. Conclusion

Multi-label agreement does not require an entirely new coefficient family. Average pairwise Hamming agreement, combined with a pooled-marginal chance term, is a chance-disagreement-weighted average of label-wise Fleiss/Scott-type coefficients and is closely related to Krippendorff’s α with Hamming distance. Stating this equivalence is useful because it reveals both the strengths and the limitations of the construction: it scales linearly in the number of labels and counts joint omissions, but it decomposes over labels and does not model their dependencies.

The choice of baseline remains consequential. The pooled-marginal, AC1, and fixed 0.5 baselines can give very different values on the same annotations, especially under extreme prevalence. The simulations show that these differences follow from the models rather than from numerical instability alone. Reporting several coefficients can be a useful sensitivity analysis, but it does not remove the need to select and justify a primary estimand.

The conditional minimum-agreement formula adds a complementary diagnostic. It shows how low observed and normalized agreement can feasibly fall once the number of annotators and empirical label prevalences are fixed. Together with raw agreement, label-wise results, and an explicit declaration of the label universe, this provides a more informative account than a single coefficient. The accompanying implementation packages these calculations so that the assumptions can be inspected rather than hidden behind one recommended score.

## References

Artstein, Ron. 2017. Inter-annotator agreement. In Nancy Ide and James Pustejovsky, editors, Handbook ofLinguistic Annotation. Springer, Dordrecht, The Netherlands, pages 297–313.

Artstein, Ron and Massimo Poesio. 2008. Inter-coder agreement for computational linguistics. Computational Linguistics, 34(4):555–596.

Bhowmick, Plaban Kumar, Anupam Basu, and Pabitra Mitra. 2008. An agreement measure for determining inter-annotator reliability of human judgements on affective text. In Coling 2008 Workshop on Human Judgements in Computational Linguistics, pages 58–65, Coling 2008 Organizing Committee, Manchester, UK.

Byrt, T., J. Bishop, and J. B. Carlin. 1993. Bias, prevalence and kappa. Journal of Clinical Epidemiology, 46(5):423–429.

Carletta, Jean. 1996. Assessing agreement on classification tasks: the kappa statistic. Comput. Linguist., 22(2):249–254.

Cohen, Jacob. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46.

Di Eugenio, Barbara and Michael Glass. 2004. The kappa statistic: A second look. Computational Linguistics, 30(1):95–101.

Fleiss, Joseph L. 1971. Measuring nominal scale agreement among many raters. Psychological Bulletin, 76(5):378–382.

Gwet, Kilem L. 2001. Handbook of Inter-Rater Reliability: How to Estimate the Level ofAgreement Between Two or Multiple Raters. STATAXIS Publishing Company, Gaithersburg, MD, USA.

Harju, Manu and Annamaria Mesaros. 2023. Evaluating classification systems against soft labels with fuzzy precision and recall. In Proceedings of the 8th Workshop on Detection and Classification of Acoustic Scenes and Events, Tampere, Finland.

Heydarian, Mohammadreza, Thomas E. Doyle, and Reza Samavi. 2022. MLCM: Multi-label confusion matrix. IEEE Access, 10:19083–19095.

Juki´c, Josip, Fran Jeleni´c, Miroslav Bi´cani´c, and Jan Šnajder. 2023. ALANNO: An active learning annotation system for mortals. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 228–235, Association for Computational Linguistics.

Konstantinidis, Menelaos, Lisa W. Le, and Xin Gao. 2022. An empirical comparative assessment of inter-rater agreement of binary outcomes and multiple raters. Symmetry, 14(2):262.

Krippendorff, Klaus. 2011. Computing krippendorff’s alpha-reliability. SAGE Open, 1(3):215824401141522.

Krippendorff, Klaus. 2013. Content Analysis: An Introduction to Its Methodology, 3rd edition. Sage, Thousand Oakes, CA, USA.

Krstini´c, Damir, Ana Kuzmani´c Skelin, Ivan Slapniˇcar, and Maja Braovi´c. 2024. Multi-label confusion tensor. IEEE Access, 12:9860–9870.

Leidner, Jochen L. 2022. Data-to-value: An ’evaluation-first’ methodology for natural language projects. In Natural Language Processing and Information Systems: Proceedings of the 27th International Conference on Applications of Natural Language to Information Systems (NLDB 2022), Valencia, Spain, June 15–17, 2022, pages 517–523, Springer-Verlag, Berlin, Heidelberg.

Marchal, Marian, Merel C.J. Scholman, Frances Yung, and Vera Demberg. 2022. Establishing annotation quality in multi-label annotations. In Proceedings of the 29th International Conference on Computational Linguistics (COLING 2022), pages 3659–3668, International Committee on Computational Linguistics, Gyeongju, Republic of Korea. Language Science and Technology / Saarland University.

Passonneau, Rebecca J. 2006. Measuring agreement on set-valued items (MASI) for semantic and pragmatic annotation. In Proceedings of the Fifth International Conference on Language Resources and Evaluation (LREC 2006), pages 831–836, European Language Resources Association (ELRA), Genoa, Italy.

Plank, Barbara, Dirk Hovy, and Anders Søgaard. 2014. Linguistically debatable or just plain wrong? In Proceedings of the 52nd Annual Meeting of the Associationfor Computational Linguistics (Volume 2: Short Papers), pages 507–511, Association for Computational Linguistics, Baltimore, MD, USA.

Rizzi, Giulia, Elisa Leonardelli, Massimo Poesio, Alexandra Uma, Maja Pavlovic, Silviu Paun, Paolo Rosso, and Elisabetta Fersini. 2024. Soft metrics for evaluation with disagreements: An assessment. In Proceedings of the 3rd Workshop on Perspectivist Approaches to NLP, pages 84–94, ELRA and ICCL.

Rosenberg, Andrew and Ed Binkowski. 2004. Augmenting the kappa statistic to determine interannotator reliability for multiply labeled data points. In Proceedings of HLT-NAACL 2004: Short Papers, pages 77–80, Association for Computational Linguistics, Boston, Massachusetts, USA.

Uma, Alexandra N., Tommaso Fornaciari, Dirk Hovy, Silviu Paun, Barbara Plank, and Massimo Poesio. 2021. Learning from disagreement: A survey. Journal ofArtificial Intelligence Research, 72:1385–1470.

Vach, Werner and Oke Gerke. 2023. Gwet’s AC1 is not a substitute for Cohen’s kappa–a comparison of basic properties. MethodsX, 10:102212.

Zec, Sabrina, Nicoletta Soriani, Roberto Comoretto, and Irene Baldi. 2017. High agreement and high prevalence: The paradox of Cohen’s Kappa. The Open Nursing Journal, 11:211–218.