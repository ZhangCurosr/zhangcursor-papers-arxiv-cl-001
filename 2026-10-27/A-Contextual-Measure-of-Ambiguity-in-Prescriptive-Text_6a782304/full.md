# A Contextual Measure of Ambiguity in Prescriptive Text

Sally Dong\*<sup>1</sup>, Luke Thorburn\*<sup>2,1</sup>, Andrew Konya<sup>1</sup>, Michiel Bakker<sup>1</sup>

<sup>1</sup>Massachusetts Institute of Technology,

<sup>2</sup>AI & Democracy Foundation

\* Equal contribution

Correspondence: sallyqd@mit.edu

## Abstract

Ambiguity in prescriptive text such as laws, contracts, rules, policies, or recommendations can be both desirable (e.g., if it enables a law to adapt to changing context, such as new technologies) and undesirable (e.g., if it leads to arbitrary discretion or conflict over the law’s interpretation). To help navigate these trade-offs, it would be valuable to have methods for quantifying ambiguity that are predictive of such outcomes, but existing measures capture how text is perceived in isolation rather than how it is interpreted in context. In this paper, we propose quantifying ambiguity as the degree of disagreement among a set of stakeholders (“raters”) on how alternative prescriptive texts (“policies”) apply in each of a set of relevant scenarios (“cases”). Using two policy domains, we present preliminary analysis suggesting that, even when using simulated raters, this approach (i) correlates with real human disagreement, and (ii) is robust to spurious text features. We then apply the measure to existing deliberation datasets to describe the association between policy ambiguity and support for those policies. A Python package for our framework and all data used in this paper are available on GitHub.

## 1 Introduction

Ambiguity or vagueness is a consequential consideration when drafting prescriptive text such as laws, contracts, or policies. Too much ambiguity, and it becomes difficult for citizens to understand the rules they must adhere to, while officials may have greater discretion to enforce rules selectively. For example, in the United States there is a “void-forvagueness” doctrine in constitutional law which holds that criminal statutes may be void if they lack sufficient specificity, motivated by a view that “men of common intelligence cannot be required to guess at the meaning of [an] enactment” (Congressional Research Service, 2022; Amsterdam, 1960). The stakes can be national in scale: the 1995 Quebec sovereignty referendum failed by barely a percentage point on a ballot question whose wording left genuine doubt about what a ‘Yes’ vote would have mandated, and Canada’s subsequent Clarity Act now empowers the House of Commons to judge whether the question in a proposed referendum is sufficiently clear before it is held (Parliament of Canada, 2000).

But ambiguity can also be useful. Some degree of interpretive flexibility can allow legal rules to adapt to unforeseen circumstances, including technological or social changes that could not have been fully specified in advance. Ambiguity can also preserve discretion for judges, agencies, or other decision-makers tasked with applying general principles to heterogeneous cases. In political settings, ambiguity can facilitate compromise by allowing actors who agree on broad goals to defer disagreement over implementation details.

As large language models (LLMs) are increasingly used to write and interpret policies, and have further been proposed as tools to augment democratic processes (Tessler et al., 2024; Konya et al., 2025), questions about how policy texts are interpreted and how ambiguity should be operationalized become ever more salient. Despite this, ambiguity remains difficult to formalize and measure. There are early signs that the problem is both tractable and consequential: Hall and Paschal (2026) find that scoring prediction-market contracts against a hand-built clarity rubric is enough to predict which contracts will later end up in dispute. But such rubrics must be engineered for each domain, and so far, a general, principled operationalization of ambiguity does not exist.

In our paper, we introduce DELTA, a framework for measuring ambiguity as interpretive divergence over consequences. DELTA evaluates a policy by pairing it with realistic application cases and asking a diverse set of raters whether the policy permits or prohibits each case. We incorporate LLMs to generate difficult evaluation cases, construct rater personas, and simulate judgments, producing a scalable approximation we call LLM-DELTA.

![](images/6e3ce146c76f80ed6fa02d2b02a94816bc342b7627c5d189c055388d26a48c60.jpg)  
Figure 1: Visual intuition for the DELTA framework. Given context on a policy domain, a set of alternative policies, a set of cases to which the policies might apply, and a set of raters, each rater judges whether each policy permits and whether it prohibits each case, and the two answers are averaged into a single judgment. For each policy, disagreement is computed across raters for each case and then averaged across cases, yielding a score that captures the ambiguity of the policy relative to the chosen cases and population of raters. Cases and raters may be human-provided or, as in LLM-DELTA, generated and simulated with LLMs.

Across four datasets, we present preliminary validation of DELTA as a methodological framework and use it as a tool for substantive analysis. We first benchmark LLM-DELTA scores against DELTA scores obtained from a human-subject experiment, finding a substantial positive correlation in a single domain. In an adversarially constructed dataset, we show that DELTA recovers the intended distinction between consensus-inducing and disagreement-inducing policies, while direct LLM-as-a-judge ratings fail. We then apply the measure to two large-scale public discussions to test whether ambiguous statements are more likely to bridge disagreement. Contrary to the intuition that ambiguity facilitates compromise, we find null relationships: more ambiguous statements do not attract broader support, and they appear to soften strong agreement rather than convert opposition.

## 2 Related work

Our approach to conceptualising and measuring ambiguity has connection to several large bodies of work from different disciplines. In this short survey, we aim to be reasonably comprehensive with respect to prior work on the narrow task of algorithmically quantifying degrees of ambiguity. We note also that use of terminology is not consistent in the literature; the general concept we refer to as “ambiguity” is sometimes also called “vagueness” (Amsterdam, 1960; Lebanoff and Liu, 2018), “specificity” (Li and Nenkova, 2015) or “concreteness” (Yeomans, 2021), sometimes refers to different concepts (e.g., (Ellsberg, 1961)), and is sometimes decomposed in various ways, such as by distinguishing between subjective uncertainty (where the interpreter finds a policy text unclear) and intersubjective disagreement (where different interpreters disagree about how a policy applies to cases) (Farnsworth et al., 2010; Chen and Zhang, 2023).

Several approaches have been taken to quantifying ambiguity. One cluster of papers approaches the task as a binary classification, labelling texts as ambiguous/unambiguous or general/specific (Massey et al., 2014; Li and Nenkova, 2015; Zhang and Ma, 2023), grounded in binary labels from human annotators. Slightly more granular approaches have humans score text on Likert scales capturing degrees of perceived ambiguity, and train models to predict those ratings (Yeomans, 2021). One notable paper introduced a dataset of human-labelled “concreteness” scores for about 40,000 English words and n-grams, rated on a 5- point Likert scale from abstract to concrete (Brysbaert et al., 2014).

A second line of work builds disagreement between interpreters into the conception of ambiguity itself, rather than treating ambiguity as a property of the text as perceived by an individual reader. In natural language inference (NLI), annotators judge whether a hypothesis follows from a premise, a task closely analogous to judging whether a policy permits a case. Pavlick and Kwiatkowski (2019) showed that for many NLI examples, disagreement persists as more judgments are collected, and does not disappear when annotators are given additional context, indicating genuinely multiple plausible interpretations. Jiang and de Marneffe (2022) taxonomized the causes of such disagreement into semantic uncertainty, annotator-specific assumptions, and artifacts of the task itself, distinctions that loosely parallel our decomposition of disagreement into intra-rater indeterminacy and inter-rater disagreement. Chen and Zhang (2023) made a similar distinction operational in their Judgment Sieve workflow, separately measuring ambiguity (an individual’s uncertainty about an item, often due to limited context) and disagreement (divergence between individuals with different perspectives), and showing that targeting each with a matched intervention (adding context or deliberation, respectively) reduces both, whereas uniform interventions tend to trade one off against the other. This literature supplies the conceptual core of DELTA.

Closest to our setting is work on rules written to govern AI systems. He et al. (2026) formalized the interpretive ambiguity of a rule as the entropy of compliance judgments across a panel of reasonable interpreters (instantiated as LLM judges), and adapted two mechanisms from the legal tradition to reduce it: iterative rule refinement at the rulecreation step, analogous to agency rulemaking, and prompt-based interpretive constraints at the ruleapplication step, analogous to canons of statutory construction. Approaching from the opposite direction, Clifford et al. (2026) studied the reconstruction of natural-language principles from preference data, and found that applying the resulting “constitutions” is itself ambiguous: different executors of the same constitution agree only 73% of the time, leading them to conclude that principles can only be evaluated jointly with their interpreters.

Most of the measures above are validated against judgments elicited for the purpose rather than against disagreements arising in practice. An exception is Hall and Paschal (2026), who found that LLM-assigned scores on a clarity rubric predicted which prediction-market contracts were later disputed (out-of-sample AUC of roughly 0.75). But such rubrics must be engineered for each domain, and so far, a general, principled operationalization of ambiguity does not exist.

The more automated versions of the algorithmic pipeline we introduce below include tasks of generating cases, generating personas (of raters), and for each policy-case pair, predicting whether a rater with a given persona would judge the case consistent with the policy. Versions of these tasks have been studied by others, for example work on case generation (Perez et al., 2022; Feng et al., 2023; Kuo et al., 2026), persona generation (Park et al., 2022; Moon et al., 2024; Castricato et al., 2025), case-based reasoning (Feng et al., 2023; Chen and Zhang, 2025), and predicting the judgments of individual raters, both in annotation tasks with inherent disagreement (Zhang and de Marneffe, 2021; Davani et al., 2022; Lee et al., 2023) and in inferring opinions or votes on political and social issues (Argyle et al., 2023; Bakker et al., 2022; Small et al., 2023; Fish et al., 2024; Gudiño et al., 2024). We do not aim to advance state-of-the-art on these tasks with this paper, but expect that as performance on these tasks improves, those insights can be incorporated to improve our proposed ambiguity metric.

## 3 Formalization

We begin with an abstraction: evaluating the consequences of a policy is equivalent to performing binary classification on its applicability to concrete scenarios. That is, given a case, one must decide whether it is consistent with the policy (e.g., permitted by it, or compatible with the position it expresses). Ambiguity arises when this mapping from cases to binary decisions is ill-defined or contested. In the same spirit as Farnsworth et al. (2010)’s notion of intersubjective disagreement, we define the ambiguity of a policy as the expected disagreement among a diverse pool of raters regarding its consequences.

## 3.1 The DELTA score

We operationalize the above view through the DELTA (Divergence Evaluation ofLinguistic Text Ambiguity) framework. Abstractly, given a policy, we construct hypothetical cases, and poll raters on whether they believe the case is consistent with, permitted by, or compatible with the policy. We then compute a disagreement measure over these judgments, where a value of zero indicates perfect consensus and higher values indicate increasing division among raters. The DELTA score of the policy is defined as an average of disagreement scores across the set of cases, with higher scores indicating greater ambiguity.

DELTA is an intentionally flexible framework: users can tailor the ambiguity measurement through their choice of cases and stakeholder distributions. Consequently, DELTA scores are relative and are meaningful only for comparing policies within a shared evaluation setup. Cases should be realistic yet engender debate regarding whether they are permitted or prohibited by the policy. Raters should be a diverse and broadly representative sample of domain stakeholders, so that their judgments capture a wide spectrum of viewpoints. This construction process reflects how policies are developed in practice: policymakers anticipate edge cases and potential conflicts that might arise during enforcement, so the policy language can be refined to preempt them.

Formally, suppose we have n policies $p _ { 1 } , \ldots , p _ { n }$ in the same domain and wish to compare their ambiguity level. Assume we have already constructed cases $c _ { 1 } , \ldots , c _ { m }$ and found raters $r _ { 1 } , \ldots , r _ { \ell }$ . Given policy $p _ { i } ,$ case $c _ { j }$ , we ask each rater $r _ { k }$ “does $p _ { i }$ permit $c _ { j } ? ^ { \dag }$ and receive a binary answer $q _ { i j k }$ . Theoretically, if we then ask the rater “does $p _ { i }$ prohibit $c _ { j } ? ^ { \dag }$ , the answer $q _ { i j k } ^ { \prime }$ should be the complement, but in practice, this is not the case if $p _ { i }$ is only tangentially related to $c _ { j }$ . As an example, consider the classroom AI policy “students cannot use AI for their homework” and case “a student uses AI to quiz herself on course material”. The policy clearly does not prohibit the case, but it also does not affirmatively address $\mathrm { i t } ;$ raters may therefore feel unable to confidently say either that the policy permits the case or that it prohibits it. To capture such asymmetry, we average the answers to the positive framing and the complement of the negative framing to determine the revealed judgment: $x _ { i j k } \ { \stackrel { \mathrm { d e f } } { = } } \ ( q _ { i j k } + ( 1 - q _ { i j k } ^ { \prime } ) ) / 2$ . Each $x _ { i j k }$ is interpreted as the probability that rater $r _ { k }$ would issue a positive verdict.

Next, let $\bar { x } _ { i j }$ denote the mean of the revealed judgments across all raters. Observe that $2 \bar { x } _ { i j } \left( 1 - \right.$ $\bar { x } _ { i j } )$ is the probability that the verdicts of two raters drawn independently from the pool conflict (the Gini impurity of the pooled verdict), which is largest $( 1 / 2 )$ when the pooled verdict is an even coin flip. We define the case disagreement score of policy $p _ { i }$ on case $c _ { j }$ as this conflict probability normalized by its maximum:

$$
d _ { i j } \ { \stackrel { \mathrm { d e f } } { = } } \ 4 { \bar { x } } _ { i j } \ ( 1 - { \bar { x } } _ { i j } ) ,
$$

which is twice the Gini impurity, giving a range of [0,1], where the value is 0 under full consensus and 1 when the pooled verdict is an even coin flip.

By the law of total variance, the score decomposes exactly into the two sources of interpretive conflict:

$$
d _ { i j } = \underbrace { \frac { 4 } { \ell } \sum _ { k = 1 } ^ { \ell } x _ { i j k } ( 1 - x _ { i j k } ) } _ { \mathrm { i n t r a - r a t e r } } + \underbrace { 4 s _ { i j } ^ { 2 } } _ { \mathrm { i n t e r - r a t e r } } ,
$$

where $s _ { i j } ^ { 2 }$ is the population variance of the revealed judgments across raters. The first term captures intra-rater indeterminacy: a rater with $x _ { i j k } = 0 . 5$ $( \mathrm { e . g . }$ ., because they cannot decide whether or how $p _ { i }$ applies to $c _ { j } )$ conflicts even with an independent redraw of their own verdict. The second is inter-rater disagreement: the additional conflict attributable to raters differing from one another.

Both are symptoms of ambiguity, since a policy can fail to determine consequences either because no individual reader can tell what it implies, or because different readers reach incompatible conclusions, and the score credits both on a common probability scale. In particular, unanimous indeterminate judgments $( x _ { i j k } = 0 . 5$ for every rater) yield the maximal $d _ { i j } = 1$ through the intra-rater term alone.

Finally, we average across all m cases to arrive at the DELTA score for policy $p _ { i } \mathbf { : }$

$$
\mathrm { D E L T A } ( p _ { i } ) \stackrel { \mathrm { d e f } } { = } \frac { 1 } { m } \sum _ { j = 1 } ^ { m } 4 \bar { x } _ { i j } ( 1 - \bar { x } _ { i j } ) .
$$

The overall score takes value in [0, 1], with 0 indicating no and 1 indicating maximum ambiguity with respect to the chosen set of cases and raters.

## 3.2 Approximation via language models

To make our framework tractable at scale, we develop a systematic method for LLMs to perform the two core tasks that would otherwise require inordinate human effort: (1) generating a diverse set of challenging synthetic evaluation scenarios for a given collection of policies, and (2) generating a diverse pool of synthetic raters and simulating their judgments. Full details of our prompts and model selections are presented in Appendix E.

Cases. To generate challenging evaluation cases, we repeatedly sample pairs of policies and prompt a reasoning LLM (GPT-5) to produce diverse cases, using the pair as inspiration. The model is instructed to craft cases that comply with one policy in the pair while violating the other if possible. Since policies typically express similar goals, the LLM must exploit the nuanced differences between the pair to fulfill the request; as a result, the output highlights precisely the details that give rise to disagreements in interpretation, especially with respect to other texts besides the input pair. Asking for a batch of cases induces a minimum amount of contextual variety; on the other hand, to avoid redundancy in the underlying cause of disagreement, we sample per batch to keep in the final set.

Rater personas. To construct a diverse pool of raters, we prompt a reasoning LLM (GPT-5) with the policy topic and request short descriptions of stakeholder identities in batches. To encourage viewpoint diversity, identities are generated in opposed pairs (e.g., on the policy topic of campus free-speech, a pair might be a university administrator versus a student activist).

In preliminary testing, we found directly asking an LLM to adopt a rater identity and issue a binary judgment yields a narrow distribution of responses, especially in non-reasoning models. Prior work has shown that persona-conditioned LLMs can approximate some human response patterns but exhibit demographic misalignment and prompt sensitivity (Argyle et al., 2023; Aher et al., 2023; Santurkar et al., 2023), and that richer agent state— e.g., natural-language reflection over memories and goals—can improve simulation fidelity (Park et al., 2023). To elicit more differentiated perspectives, we introduce an intermediate step: the model first expands each identity into a brief paragraph describing that stakeholder’s values and concerns regarding the policy topic, serving as a lightweight reasoning trace that anchors subsequent judgments in a more coherent viewpoint.

The final prompt provided to the LLM therefore includes the rater identity, their generated reflections on the topic, and the judgment question in both positive and negative framing. We use a relatively lightweight LLM, Gemini-2.5-Flash at temperature 1, to carry out the judgments.

## 3.3 Score convergence

![](images/c2155e6efb25e38ecba27436ff3f286fb7d38e209698995184a25e445942962c.jpg)  
Figure 2: Each cell in the heatmap reports the average Spearman correlation between LLM-DELTA scores computed from a random subset of i cases and $j$ raters and reference scores computed using the full evaluation design.

Repeated prompting of LLMs is known to yield diminishing returns in semantic diversity: newly generated cases or raters tend to resemble those already produced. Consequently, the marginal impact of additional samples on DELTA scores decreases over time, suggesting that a practical stopping criterion can be based on empirical convergence of the scores.

We illustrate this using the Polis–Uber dataset introduced formally in Section 4.2. For each policy in this dataset, we first compute a reference LLM-DELTA score using the full evaluation design, with 60 generated cases and 30 LLM raters. We then estimate how quickly these scores converge by recomputing LLM-DELTA scores on smaller random subsets of the full design: For each pair of subset sizes (i, j), we repeat the subsampling procedure 100 times, each time sampling i cases and j raters, compute LLM-DELTA scores on this subset, and calculate the Spearman correlation between the resulting policy rankings and the full-sample reference rankings. The heatmap reports the average correlation across the 100 iterations.

The results show rapid convergence (fig. 2). Correlations with the full-sample scores increase quickly as the number of sampled cases and raters grows, approaching high values well before reaching the full design of 60 cases and 30 raters. We note that this diagnostic establishes within-pool rank stability—that subsampling from a fixed pool of generated cases and personas yields similar rankings to the full pool—rather than convergence to the broader distribution of cases and personas that further generations could produce. With this caveat, we use the diagnostic as a practical stopping criterion across datasets, typically with 30 raters and at most 50–60 cases as a balance between score stability and computational cost. In future applications, researchers can use the same diagnostic to select an appropriate number of cases and raters empirically. Appendix A.4 reports the same diagnostic for our human subject study with similar results.

## 4 Methods

In this section, we describe our hypotheses and approach to evaluating them.

## 4.1 Hypotheses

We had two hypotheses about the validity of the LLM-DELTA measure of ambiguity. First, LLM-DELTA scores would better correlate with human-DELTA scores than would other baseline ambiguity measures (including LLM-as-a-judge, and a measure based on word-level concreteness scores). When comparing ambiguity measures on a set of policies, we focus on rank correlation because the DELTA score is a measure of relative rather than absolute ambiguity. Second, LLM-DELTA scores would be more robust than the baseline methods to “adversarial” examples: policies that appear unambiguous (because they are worded in a way that sounds decisive), but that are actually ambiguous (because they lead to significant disagreement among a given population of raters when applied to a given set of cases), or vice versa.

We had one substantive hypothesis, tested by applying the measure, that ambiguity in prescriptive text can help with consensus or agreement at the policy level by effectively postponing disagreement to future disputes about how the policy should be applied to individual cases. We hypothesize that this effect will be visible in real datasets as a positive relationship between ambiguity and various measures of consensus or bridging agreement (Ovadya and Thorburn, 2023).

## 4.2 Datasets

We worked with four disparate datasets, corresponding to multiple policy domains. The complete datasets can be found in our project repository.

Dataset 1: University speech. For benchmarking LLM-DELTA and other measures against a human-DELTA baseline, we used the domain of university policies related to freedom of expression, particularly pertaining to speech about harm or violence. We (ourselves) wrote 20 alternate policies, designed to span from the highly ambiguous (e.g., platitudes about maintaining a “welcoming environment”) to highly unambiguous (e.g., modeled on the widely-used legal test for incitement). For cases, we paraphrased 20 real campus speech controversies from databases maintained by the Foundation for Individual Rights and Expression (FIRE) (Foundation for Individual Rights and Expression, 2026a,b,c). Both policies and cases can be found in Appendix A.

Dataset 2: Adversarial. For testing the robustness hypothesis, we developed a procedure for generating “adversarial” configurations of policies, cases, and rater personas (described in full in Appendix B). In brief, it begins with a minimal example of two policies, two raters, and four cases such that the raters agree on all cases under one policy but disagree on all cases under the other, due to where the cases sit on underlying axes of consensus and conflict. For the results reported below, we generated one adversarial setup modeled on the 1999 Australian republic referendum and scaled it up to 20 policies, 32 cases, and 30 rater personas.

Dataset 3: Pol.is. Pol.is is an online platform for large-scale deliberation and opinion mapping conducted in partnership with governments or civic organizations interested in broad public participation. Given a policy topic, participants submit short position statements and vote on statements submitted by others.

Although the statements in Pol.is conversations are not formal policies, they are policy-relevant texts where participants express what they believe governments ought to do, and we therefore treat them as directional policy proposals. In the DELTA framework, cases can be understood as policy implementations or decisions that might be evaluated against the proposal. Under this interpretation, it is still meaningful to ask whether a statement is ambiguous; that is, whether different raters disagree about the concrete actions supported and ruled out by the statement.

We use four publicly available Pol.is datasets (The Computational Democracy Project, 2026). The first concerns how Taiwan should regulate app-based ride-hailing services, collected as part of the vTaiwan participatory process that contributed to subsequent legislation (vTaiwan, 2023). The remaining three are sub-topics drawn from Austria’s Klimarat process, a national citizens’ assembly on climate policy, covering production and consumption, housing, and energy (Paice and Rausch, 2022). Full details are provided in Appendix C.

Dataset 4: Habermas experiment. Tessler et al. (2024) developed an AI system, the Habermas Machine, to help small groups of citizens identify common ground on social and political issues. The system takes as input individual opinion statements on a select topic and generates collective statements intended to reflect positions that the group could broadly endorse.

The system was evaluated in a human experiment in the UK. Participants were assigned to groups of four or five and asked to deliberate on one of eight political topics (e.g., “should the minimum wage be raised in the UK?”). To begin, participants reported their personal agreement with the topic statement on a 1–7 scale and wrote a private opinion statement. The Habermas Machine then used these individual statements to generate collective statements representing possible common ground within the group. Group members then rated each collective statement according to their own level of agreement on a 1–7 scale, and a refinement iteration followed.

For each discussion topic, we aggregated Habermas-generated collective statements across groups for ambiguity evaluation. This yields eight topic-specific sets, each with approximately 70 entries; full details of the experiment and our dataset construction are provided in Appendix D.

## 4.3 Baselines

For eliciting a human-DELTA ground truth, we conducted a human subject study where a politicallybalanced sample of U.S. adults, recruited via Prolific, was asked to rate (policy, case) tuples from Dataset 1 described above. The consent form, instructions, and a sample of the rating interface are included in Appendix A. Each participant rated all 20 cases against two policies and was paid \$6; the median completion time was 18 minutes. Each policy was shown to 30 raters in total, distributed evenly across self-identified Republicans, Independents, and Democrats, allowing us to estimate disagreement across a politically diverse group. This setup differs slightly from the LLM-DELTA setup used elsewhere in the paper where the same rater evaluates all policies against all cases, due to practical limitations of human-subject experiments.

As a lexical baseline, we use a dictionary-based measure from Brysbaert et al. (2014). The dictionary contains approximately 40,000 English words, each assigned a human-annotated concreteness score on a 1–5 scale. For each policy, we identify words that appear in the dictionary (consistently above 85% of the full text) and compute the average concreteness score across those words. Since more concrete language is typically less ambiguous, we negate the resulting score so that higher values correspond to greater ambiguity.

We further establish two direct LLM-as-a-judge measures of ambiguity (Zheng et al., 2023). In these baselines, an LLM is given a policy to be rated on a 1–7 scale. We use two prompt variants. The first prompt defines ambiguity in standard dictionary-style terms, with higher values indicating more perceived ambiguity, while the second is designed to more closely mirror the DELTA mechanism: the model is asked to imagine plausible scenarios and assess the degree to which people with diverse viewpoints might disagree about the applicability of policy in those settings. Full prompts and model choices are reported in Appendix E.

## 5 Results

## 5.1 Human-DELTA benchmark

![](images/c668b5532cc046f1dafb4447cc417ef73f90fd855f969881b95701e1ce9a04d8.jpg)  
Figure 3: Spearman correlations between four computational ambiguity measures and the human-DELTA scores for the first dataset of 20 university free-speech policies, with bootstrapped 95% confidence intervals over 2000 iterations.

![](images/42110475222c330b08ccad399dc9fd00295d1eae24437d4810094ca0698e0c77.jpg)  
Figure 4: Correlations between LLM-DELTA and human-DELTA on dataset 1, with raw scores shown on the left and rankings shown on the right. Each point represents one policy.

Figure 3 reports Spearman correlations between human-DELTA scores and four alternative computational measures from the first dataset. Confidence intervals come from a stratified rater bootstrap: for each policy we resample raters with replacement separately within each party (Democrat, Independent, Republican) while preserving the original per-policy party distribution, and recompute the human-DELTA on each resample while holding the other variable fixed.

Human-DELTA and LLM-DELTA scores are positively associated in accordance with our hypothesis (fig. 4), suggesting that the LLM-based procedure captures a similar form of interpretive disagreement to that observed in human judgments. This holds despite our not manually specifying the demographic or partisan characteristics of simulated raters, whereas the human experiment used raters balanced across Republicans, Independents, and Democrats.

By contrast, the dictionary-based measure shows little relationship with human-DELTA, suggesting that DELTA does not simply capture surface-level word choice. The direct LLM-as-a-judge ambiguity rating fares somewhat better than the dictionary measure, but its association remains weak and far below LLM-DELTA’s, likely because direct ratings are more sensitive to apparent vagueness or rhetorical style than to disagreement over downstream application.

Finally, the LLM-as-a-judge disagreement measure performs poorly relative to LLM-DELTA, despite being prompted with a description of the DELTA mechanism. This indicates that merely asking a model to reason abstractly about possible disagreement is not sufficient to reproduce DELTA scores, as it lacks the scaffolding through concrete cases and diverse judgments.

## 5.2 Adversarial robustness

![](images/0afef5797def2f588de432b743d36730c5e1c082b37e05eb86d653c4987c12dd.jpg)  
Figure 5: Each point represents one policy in the adversarial dataset. Both LLM-as-a-judge ambiguity ratings and disagreement ratings rank the policies in nearly the reverse order of LLM-DELTA.

Recall that the adversarial policy set begins from a minimal example with two policy types: one that should elicit consensus among two raters, and is therefore unambiguous under DELTA, and one that should elicit disagreement, and is therefore ambiguous under DELTA. We then scale this construction to a full set of 20 policies, with ten policies of each type (and similarly scale up the two types of raters and four types of cases). DELTA successfully recovers this intended distinction in the richer evaluation design: it ranks the ten consensus-inducing policies as the ten least ambiguous policies.

As shown in fig. 5, both versions of the LLMas-a-judge baselines rank the adversarial policies in nearly the reverse order of DELTA, revealing

Correlation between bridging and DELTA their lack of robustness. The adversarial setup was designed to decouple apparent ambiguity from realized disagreement along a latent axis; the reversal thus demonstrates that direct LLM-as-a-judge ratings fail to discover the pitfall, even when explicitly prompted to reason about disagreement.

## 5.3 Relationship with bridging

Pol.is uses participant voting patterns to identify preference clusters. In the datasets we use, Pol.is identifies two primary opinion groups: participants within each group vote similarly on statements, while the two groups differ systematically. We use these algorithmically identified groups to examine the relationship between ambiguity and bridging.

Bridging is the ability of a statement to garner support from two opposing sides on a topic. Let $p _ { A }$ and $p _ { B }$ denote the agreement rates for a statement among participants in groups A and B respectively. We define the bridging score as min $\left( p _ { A } , p _ { B } \right)$ . This measure is conservative: a statement scores highly only if both groups agree with it at high rates. Statements that are strongly supported by one group but rejected by the other receive low bridging scores. Empirically, our definition closely tracks alternative cross-group alignment scores, such as p<sub>A</sub>p<sub>B</sub>, used by Pol.is, and $p _ { A } p _ { B } + ( 1 - p _ { A } ) ( 1 - p _ { B } )$ , which is high when both groups tend to give the same response, whether agreement or disagreement. Our definition more clearly reflects shared endorsement: statements that can serve as common ground by attracting support from both opinion groups, rather than merely produce common rejection.

We find that ambiguity is not significantly associated with bridging in any of the four datasets at the 20% engagement threshold shown in fig. 6. Across the full range of thresholds examined in Appendix C.1, no dataset shows a stable positive association—the direction the ambiguity-ascompromise hypothesis predicts.

One interpretation of this absence of a positive association is that in public deliberation, ambiguity does not necessarily create common ground by permitting interpretive latitude: For a statement to attract support from both opinion groups, it may need to articulate a position that is thoughtfully responsive to the underlying question. Ambiguous statements, by contrast, may fail to reassure either side about what the proposal would imply in practice and be seen as counterproductive. In this sense, bridging may depend less on allowing for multiple interpretations than on identifying concrete solutions that both groups can endorse.

![](images/815a63aee88c6c2a59f12b642d4d558b42ddb39229523e4d22ac2715a729d963.jpg)  
Figure 6: LLM-DELTA and bridging show null or negative Pearson correlation across four Pol.is datasets, contrary to the intuition that ambiguity fosters broader support.

![](images/1531e73038af183ed69e4d62ecb924352847bf15a369c543c730632d68c83472.jpg)  
Figure 7: Collective statements are ranked from least to most ambiguous according to DELTA scores. Lines show smoothed shares of participant agreement ratings on the 1–7 scale. 95% confidence bands are obtained using 300 bootstrapping iterations.

We complement this finding with an analysis of how ambiguity is associated with the degree of individual agreement using the Habermas dataset. Here, we consider only collective statements from groups with at least one participant initially supporting the discussion topic and at least one opposed, since those with no initial disagreement cannot meaningfully test opinion change. Within each topic, we rank statements by their DELTA score and normalize to [0, 1], then examine how the distribution of participants’ agreement level varies as ambiguity increases. This within-question ranking allows us to compare patterns across topics without assuming that DELTA scores are similarly distributed.

Figure 7 depicts the distributional shifts in participant agreement as a function of DELTA. Most notably, as DELTA increases, the proportion of “strongly agree” drops significantly, while “slightly agree” and neutral responses gain by a comparable combined amount; the entire disagree side changes only modestly. This pattern reinforces the Pol.is finding: Highly ambiguous statements do not appeal to the initially opposed to produce a bridging effect, but rather have the additional downside of softening the enthusiasm of the initially supportive.

## 6 Conclusion

We put forth a definition of ambiguity in prescriptive text as a divergence in interpretations of consequences. The DELTA framework operationalizes this view by making explicit the cases and stakeholder perspectives under which ambiguity is evaluated, enabling comparisons of policies beyond lexical properties alone. Our results further complicate a common view of ambiguity as politically useful: while ambiguity may sometimes enable elite bargaining or defer conflict, in the public deliberation settings we examined, it does not reliably produce common ground. Future work should improve rater simulation, refine case generation, benchmark the framework across additional domains and languages, and investigate whether DELTA can predict downstream implementation conflict in real policy settings.

## 7 Limitations

Our framework and findings come with several limitations. First, LLM-DELTA depends on the capabilities and biases of the underlying language models used to generate cases, construct rater personas, and produce judgments. Although we observe a positive correlation between LLM-DELTA and human-DELTA on our university speech dataset (Spearman $\rho \ : = \ : 0 . 4 8 )$ , this is a moderate rather than strong association, and it is established on a single domain with 20 policies. The extent to which LLM-simulated raters reproduce real interpretive disagreement in other policy domains, particularly those underrepresented in pretraining data, those involving non-English text, or those where stakeholders hold views that are systematically misrepresented in training corpora, remains an open empirical question. Prior work has documented that persona-conditioned LLMs exhibit demographic misalignment and prompt sensitivity, and our framework inherits these issues.

Second, DELTA scores are inherently relative: they are only meaningful for comparisons within a shared evaluation setup (pool of cases and raters). Scores cannot be directly compared across studies that use different case sets or rater distributions, and the choice of cases and personas embeds assumptions about which scenarios and viewpoints are worth representing. Our case generation procedure samples policy pairs to surface points of nuanced disagreement, but it may systematically miss forms of ambiguity that only become apparent across larger sets of policies or in interaction with cases that no policy author would have anticipated.

Third, our human benchmark is limited in scope. We recruited 30 raters per policy in a single domain (U.S. university speech), balanced across selfidentified Democrats, Independents, and Republicans, but this captures only one axis of viewpoint diversity in one context. The human study also differs from the LLM setup in that each human rater evaluated only two policies rather than all 20, which may affect comparability. The confidence intervals in fig. 3 hold LLM-DELTA scores fixed; so, for example, the relationship between LLM- and human-DELTA scores reflects variance in human judgments but not in LLM case generation, persona generation, or judgment sampling. We did not benchmark against human-DELTA in the Pol.is or Habermas datasets, so the validity of LLM-DELTA in those settings is supported only indirectly.

Fourth, the adversarial robustness result demonstrates that LLM-as-a-judge baselines fail on a setup we constructed precisely to expose their weaknesses. We do not claim that LLM-DELTA would prove similarly robust to adversarial constructions designed against it; such constructions are plausible and worth investigating in future work.

Finally, our bridging analysis is correlational and observational. We cannot identify whether the negative or null relationships between ambiguity and bridging reflect a causal property of ambiguous statements, selection effects in what kinds of statements get submitted to Pol.is, or features specific to online deliberation that may not generalize to formal policy drafting contexts. Our analysis also does not account for plausible confounds such as statement length, extremist rhetoric, or LLMtranslation artifacts.

Risks. We see limited direct risks from this work. DELTA is a measurement tool rather than a generative one, and it does not produce policy text, rank stakeholders, or make recommendations about which policies should be adopted. The most plausible misuse would be treating DELTA scores as objective measures of policy quality or legitimacy when they are in fact relative measures dependent on the chosen evaluation design; we have tried to be explicit about this throughout. A secondary concern is that using LLM-simulated raters in place of human deliberation, if applied uncritically in real policymaking contexts, could lend a veneer of stakeholder consultation to processes that have not actually consulted stakeholders. We view LLM-DELTA as a tool to support human judgment in early-stage policy drafting and analysis, not a substitute for genuine democratic participation.

## 8 Acknowledgments

We thank Emma Cabalé, Nan Clement, Manuel Wüthrich, and Ariel Procaccia for helpful discussions. We used AI in this project for coding, data analysis, and drafting and editing the paper.

## References

Gati V. Aher, Rosa I. Arriaga, and Adam Tauman Kalai. 2023. Using large language models to simulate multiple humans and replicate human subject studies. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202, pages 337–371. PMLR.

Anthony G. Amsterdam. 1960. The void-for-vagueness doctrine in the Supreme Court. University of Pennsylvania Law Review, 109:67–116.

Lisa P. Argyle, Ethan C. Busby, Nancy Fulda, Joshua R. Gubler, Christopher Rytting, and David Wingate. 2023. Out of one, many: Using language models to simulate human samples. Political Analysis, 31(3):337–351.

Michiel A. Bakker, Martin J. Chadwick, Hannah R. Sheahan, Michael Henry Tessler, Lucy Campbell-Gillingham, Jan Balaguer, Nat McAleese, Amelia Glaese, John Aslanides, Matthew M. Botvinick, and Christopher Summerfield. 2022. Fine-tuning language models to find agreement among humans with diverse preferences. In Advances in Neural Information Processing Systems, volume 35, pages 38176– 38189.

Marc Brysbaert, Amy Beth Warriner, and Victor Kuperman. 2014. Concreteness ratings for 40 thousand generally known English word lemmas. Behavior Research Methods, 46(3):904–911.

Louis Castricato, Nathan Lile, Rafael Rafailov, Jan-Philipp Fränken, and Chelsea Finn. 2025. PER-SONA: A reproducible testbed for pluralistic alignment. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 11348– 11368. Association for Computational Linguistics.

Quan Ze Chen and Amy X. Zhang. 2023. Judgment sieve: Reducing uncertainty in group judgments through interventions targeting ambiguity versus disagreement. Proceedings of the ACM on Human-Computer Interaction, 7(CSCW2):1–26.

Quan Ze Chen and Amy X. Zhang. 2025. Case law grounding: Using precedents to align decisionmaking for humans and AI. In Proceedings of the ACM Collective Intelligence Conference. Association for Computing Machinery.

Eleanor Clifford, Michael Amir, Arduin Findeis, Aaron Zhao, and Robert Mullins. 2026. Open problems in constitutional preference reconstruction. Preprint, arXiv:2606.30116.

Congressional Research Service. 2022. Overview of the void-for-vagueness doctrine. Constitution Annotated.

Aida Mostafazadeh Davani, Mark Díaz, and Vinodkumar Prabhakaran. 2022. Dealing with disagreements: Looking beyond the majority vote in subjective annotations. Transactions ofthe Associationfor Computational Linguistics, 10:92–110.

Daniel Ellsberg. 1961. Risk, ambiguity, and the Savage axioms. The Quarterly Journal of Economics, 75(4):643–669.

Ward Farnsworth, Dustin F. Guzior, and Anup Malani. 2010. Ambiguity about ambiguity: An empirical inquiry into legal interpretation. Journal of Legal Analysis, 2(1):257–300.

K. J. Kevin Feng, Quan Ze Chen, Inyoung Cheong, King Xia, and Amy X. Zhang. 2023. Case repositories: Towards case-based reasoning for AI alignment. Preprint, arXiv:2311.10934.

Sara Fish, Paul Gölz, David C. Parkes, Ariel D. Procaccia, Gili Rusak, Itai Shapira, and Manuel Wüthrich. 2024. Generative social choice. In Proceedings of the 25th ACM Conference on Economics and Computation. Association for Computing Machinery.

Foundation for Individual Rights and Expression. 2026a. Campus deplatforming database. Online database. Formerly the Campus Disinvitation Database; documents attempts to deplatform speakers on U.S. campuses, 1998–present.

Foundation for Individual Rights and Expression. 2026b. Scholars under fire database. Online database. Documents campus controversies involving attempts to investigate, penalize, or otherwise sanction scholars for constitutionally protected speech, 2000–present.

Foundation for Individual Rights and Expression. 2026c. Students under fire database. Online database. Documents efforts since 2020 to investigate, censor, or otherwise punish students and student groups at U.S. colleges and universities for protected expression.

Jairo F. Gudiño, Umberto Grandi, and César Hidalgo. 2024. Large language models (LLMs) as agents for augmented democracy. Philosophical Transactions ofthe Royal Society A: Mathematical, Physical and Engineering Sciences, 382(2285):20240100.

Andy Hall and Elliot Paschal. 2026. Can AI fix bad rules? Free Systems (Substack newsletter).

Luxi He, Nimra Nadeem, Michel Liao, Howard Chen, Danqi Chen, Mariano-Florentino Cuéllar, and Peter Henderson. 2026. Statutory construction and interpretation for AI. Proceedings of the National Academy ofSciences, 123(30):e2509766123.

Nan-Jiang Jiang and Marie-Catherine de Marneffe. 2022. Investigating reasons for disagreement in natural language inference. Transactions of the Associationfor Computational Linguistics, 10:1357–1374.

Andrew Konya, Luke Thorburn, Wasim Almasri, Oded Adomi Leshem, Ariel D. Procaccia, Lisa Schirch, and Michiel A. Bakker. 2025. Using collective dialogues and AI to find common ground between Israeli and Palestinian peacebuilders. In Proceedings ofthe 2025 ACM Conference on Fairness, Accountability, and Transparency. Association for Computing Machinery.

Tzu-Sheng Kuo, Sophia Liu, Quan Ze Chen, Joseph Seering, Amy X. Zhang, Haiyi Zhu, and Kenneth Holstein. 2026. Botender: Supporting communities in collaboratively designing AI agents through casebased provocations. In Proceedings ofthe 2026 CHI Conference on Human Factors in Computing Systems. Association for Computing Machinery.

Logan Lebanoff and Fei Liu. 2018. Automatic detection of vague words and sentences in privacy policies. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3508–3517. Association for Computational Linguistics.

Noah Lee, Na Min An, and James Thorne. 2023. Can large language models capture dissenting human voices? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4569–4585.

Junyi Li and Ani Nenkova. 2015. Fast and accurate prediction of sentence specificity. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29, pages 2281–2287. AAAI Press.

Aaron K. Massey, Richard L. Rutledge, Annie I. Antón, and Peter P. Swire. 2014. Identifying and classifying ambiguity for regulatory requirements. In Proceedings of the 22nd IEEE International Requirements Engineering Conference (RE), pages 83–92. IEEE.

Suhong Moon, Marwa Abdulhai, Minwoo Kang, Joseph Suh, Widyadewi Soedarmadji, Eran Kohen Behar, and David M. Chan. 2024. Virtual personas for language models via an anthology of backstories. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 19864–19897. Association for Computational Linguistics.

Aviv Ovadya and Luke Thorburn. 2023. Bridging systems: Open problems for countering destructive divisiveness across ranking, recommenders, and governance. Knight First Amendment Institute at Columbia University.

Andy Paice and Martin Rausch. 2022. Creating an online conversation between a nation and a mini-public: A case study on Polis and the Austrian Citizens’ Climate Council. Co-Intelligence Institute case study.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology, pages 1–22. Association for Computing Machinery.

Joon Sung Park, Lindsay Popowski, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2022. Social simulacra: Creating populated prototypes for social computing systems. In Proceedings ofthe 35th Annual ACM Symposium on User Interface Software and Technology, pages 1–18. Association for Computing Machinery.

Parliament of Canada. 2000. Clarity Act, S.C. 2000, c. 26. Statute. An Act to give effect to the requirement for clarity as set out in the opinion of the Supreme Court of Canada in the Quebec Secession Reference.

Ellie Pavlick and Tom Kwiatkowski. 2019. Inherent disagreements in human textual inferences. Transactions ofthe Associationfor Computational Linguistics, 7:677–694.

Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. 2022. Red teaming language models with language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3419–3448. Association for Computational Linguistics.

Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. 2023. Whose opinions do language models reflect? In Proceedings of the 40th International Conference on Machine Learning, volume 202, pages 29971–30004. PMLR.

Christopher T. Small, Ivan Vendrov, Esin Durmus, Hadjar Homaei, Elizabeth Barry, Julien Cornebise, Ted Suzman, Deep Ganguli, and Colin Megill. 2023. Opportunities and risks of LLMs for scalable deliberation with Polis. Preprint, arXiv:2306.11932.

Michael Henry Tessler, Michiel A. Bakker, Daniel Jarrett, Hannah Sheahan, Martin J. Chadwick, Raphael Koster, Georgina Evans, Lucy Campbell-Gillingham, Tantum Collins, David C. Parkes, Matthew Botvinick, and Christopher Summerfield. 2024. AI can help humans find common ground in democratic deliberation. Science, 386(6719):eadq2852.

The Computational Democracy Project. 2026. Open Polis data. Open data repository. Data exports from open-data Pol.is conversations. Licensed under CC BY 4.0.

vTaiwan. 2023. vTaiwan: Rethinking democracy. Last updated December 2023.

Michael Yeomans. 2021. A concrete example of construct construction in natural language. Organizational Behavior and Human Decision Processes, 162:81–94.

Xinliang Frederick Zhang and Marie-Catherine de Marneffe. 2021. Identifying inherent disagreement in natural language inference. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4908–4915.

Zijing Zhang and Ling Ma. 2023. Using machine learning for automated detection of ambiguity in building requirements. In Proceedings of the 2023 European Conference on Computing in Construction (EC3), volume 4. European Council on Computing in Construction.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623.

## A Human subject study

We wrote 20 policies on the topic of campus free speech, designed to span from the highly ambiguous to highly unambiguous. For cases, we paraphrased 20 real campus speech controversies from databases maintained by the Foundation for Individual Rights and Expression (FIRE) (Foundation for Individual Rights and Expression, 2026a,b,c).

## A.1 Policies

1. The university prohibits acts of physical violence or force on campus. Expression alone, in the absence of physical conduct or a true threat, does not violate this policy.

2. The university prohibits expression that constitutes a true threat: a serious expression of intent to commit unlawful violence against a specific person or group.

3. The university prohibits expression that is both (a) directed at inciting imminent lawless action and (b) likely to produce such action.

4. The university prohibits faculty and staff from advocating violence against identifiable persons while acting in their professional capacity. Expression outside that capacity is not covered.

5. Instructors may not present, as established fact in their teaching, claims contradicted by the consensus of the relevant field. Contested interpretations and opinions remain permitted.

6. The university prohibits expression that celebrates violence committed against a specific person. Expression about ideas, groups, or abstractions is not covered.

7. Instructors may not express personal hostility toward identifiable individuals in required classes. The same expression in voluntary venues is not covered.

8. The university prohibits expression that disparages an identifiable group on the basis of race, sex, religion, nationality, sexual orientation, or political affiliation.

9. The university prohibits expression that a reasonable person would interpret as encouraging others to commit acts of violence.

10. The university prohibits expression that creates a hostile environment for members of the campus community.

11. The university is committed to maintaining a diverse, inclusive, and welcoming learning environment for all members of its community.

12. The university permits criticism of public figures, including criticism delivered in harsh or intemperate terms.

13. Members of the campus community are expected to engage with one another in a spirit of mutual respect and civility.

14. The university affirms the principles of academic freedom and the free exchange of ideas as essential to its educational mission.

15. The university prohibits expression that undermines the dignity of members of the campus community.

16. The university supports the right of every student to learn in an environment free from intimidation, harassment, and discrimination.

17. The university values robust debate and the open contest of ideas, including ideas that members of the community may find objectionable.

18. The university values courage, including the courage to speak and the courage to listen.

19. The university affirms its commitment to truth.

20. Members of the campus community are expected to conduct themselves in a manner befitting the institution.

## A.2 Cases

1. An English instructor at the university shared an article on Instagram about the assassination attempt on Donald Trump, adding the comment: “If you’re gonna shoot, man, don’t miss.” The post was re-shared by the “Libs of TikTok” account on X (formerly Twitter), and gained widespread attention.

2. A group of students, posting under a social media account called “CanesCritters,” uploaded a video for a Digital Marketing class assignment in which a Black student is compared to a gorilla.

3. A professor in the Department of Asian and Middle Eastern Studies at the university, criticized the idea of free speech on campus, associating it with the rise of fascism, and mocked those who promote debate over resistance, saying the moral high ground is “for those who wish to drown in blood.”

4. A student, in a private group chat, described the January 6 Capitol breach as “a peaceful occupation of the capitol and our people were provoked with tear gas and flash bangs!”

5. A part-time math instructor modified the state’s mandatory Oath of Allegiance to align with her pacifist beliefs. The standard oath requires employees to “support and defend” the U.S. and California Constitutions “against all enemies, foreign and domestic.” She inserted the word “nonviolently” before “support and defend,” crossed out “swear,” and circled “affirm.” Previously, similar modifications had been accepted without issue during her employment with public schools.

6. An electrical engineering and computer science professor, author of a 1976 book called ‘The Hoax of the Twentieth Century’ which denies the systematic extermination of six million Jews during World War II, publicly endorsed Iranian President Mahmoud Ahmadinejad’s Holocaust denial statements, saying: “I congratulate him. . . It was about time an Islamic leader stated this publicly.”

7. An emeritus professor of medicine publicly supported physician-assisted suicide and the removal of Terri Schiavo’s feeding tube. (As context, Terri Schiava was a Florida woman who suffered severe brain damage in 1990 and was diagnosed as being in a persistent vegetative state. Her husband sought to have her feeding tube removed, arguing she would not have wanted to be kept alive in that condition; her parents fought to keep it in. The dispute went through years of court battles and became a national flashpoint in 2005, involving the Florida legislature, Governor Jeb Bush, Congress, and President George W. Bush, who signed emergency legislation aimed at keeping her alive. The feeding tube was ultimately removed and she died in March 2005.)

8. A student government denied Turning Point USA’s application for campus recognition, citing the group’s “misalignment” with university values, “confrontational language throughout the TPUSA website,” the national organization’s “Professor Watchlist,” and the group’s “actions of encouraging a growth in conflict rather than fostering communication.”

9. A psychology professor posted on Facebook in response to Charlie Kirk’s assassination: “Me thinks the word ‘karma’ is appropriate.

Sad all the way around.” In a second post, she wrote: “But when we tell y’all that statistically . . . White American men are the most dangerous animals on the planet we’re wrong. Let’s not be hasty they say . . . ” She then deleted the posts.

10. A psychology professor, when asked during a social psychology class about her views on NFL quarterback Colin Kaepernick’s national anthem protests against police violence, said she “would kill him.”

11. A student produced a “visual thesis” that focuses on Islamic extremism and criticized violence and bigotry against Israel.

12. A philosophy professor posted on Facebook: “I want the entire world to burn until the last cop is strangled with the intestines of the last capitalist, who is strangled in turn with the intestines of the last politician.”

13. Nine students, including two student journalists, participated in an unapproved “die-in” protest on campus to object to what they saw as the university administration’s failure to acknowledge faculty and staff safety concerns regarding its COVID-19 reopening plan.

14. A video circulated on Snapchat and Twitter showing an incoming student and two others laughing as one of them knelt on the chest of another who was saying “I can’t breathe.”

15. A math instructor wrote in a Facebook exchange that there were “no safe spaces in my classes” for students in the country illegally, and said he would report such students to Immigration and Customs Enforcement (ICE).

16. A professor of industrial and systems engineering, public policy, and economics displayed a “Thin Blue Line” flag, commonly associated with the “Blue Lives Matter” movement, on his office door (“blue lives” refers to the lives of police officers). He said he aimed to present an alternative viewpoint and support law enforcement, asserting that “blue lives protect black lives.”

17. A professor of politics and global studies posted tweets about a Las Vegas mass shooting suggesting that “white male entitlement” was a contributing factor to mass shootings.

18. A part-time anthropology lecturer was involved in a confrontation with members of the campus College Republicans during a counterprotest to a pro-Palestinian event. The College Republicans accused the lecturer of attempting to grab a student’s sign and striking another student who intervened. The lecturer denied these allegations, asserting his commitment to nonviolence.

19. The School of Public Affairs invited Condoleezza Rice to speak on civil rights. Rice supported the war in Iraq, the use of torture on detainees, and helped implement other Bush administration policies.

20. Salman Rushdie, author of the 1988 novel The Satanic Verses, was invited by a college to deliver a commencement address. Rushdie had recently made comments equating pro-Palestinian campus protests to support for a “fascist terrorist group.”

## A.3 Study

The study was determined to be exempt from ethics review by the university review board. We recruited participants using Prolific and served the survey on Qualtrics. Each participant rated all 20 cases against two policies and was paid \$6; the median completion time was 18 minutes. Each policy was shown to 30 raters in total, distributed evenly across self-identified Republicans, Independents, and Democrats, allowing us to estimate disagreement across a politically diverse group.

We used two attention check questions in the survey. They are presented as another case to rate per policy, and the order of cases is randomized. Seven percent of respondents failed at least one attention check; subsequently, we excluded their responses and recruited replacements.

## A.4 Score convergence

Figure 2 in the main text reports the case-and-rater convergence diagnostic for the Polis–Uber dataset, whose cases are LLM-generated. Figure A.5 repeats the diagnostic on the human subject study dataset with hand-crafted policies and cases. We compute reference LLM-DELTA scores for the 20 policies using the full design of 20 cases and 30 LLM raters, and each cell reports the mean Spearman correlation between the reference ranking and rankings recomputed on 100 random subsamples of i cases and j raters.

The pattern matches the main text. Rankings stabilize well before the full design is reached: the mean correlation is 0.87 with half the cases and a third of the raters (10 of each), and exceeds 0.9 from roughly 8 raters and 15 cases.

![](images/173e30e2a631b0997e6f324125c391846f009f60ab5071d2a3dfdaa48b50c615.jpg)  
Figure A.5: Each cell reports the average Spearman correlation between LLM-DELTA scores computed from a random subset of i cases and j raters and reference scores computed using the full design of 20 manually curated cases and 30 LLM raters, on the campus speech dataset (100 subsamples per cell).

## B Procedure for generating adversarial examples

In this section, we describe the procedure used to generate the adversarial dataset referenced in section 4.2. The goal of the procedure is to produce small, structured configurations of policies, cases, and rater personas in which two conceptions of ambiguity will lead to different assessments of the policies: ‘surface ambiguity’, the impression that a policy is vague, underspecified, or open-textured when read in isolation (as measured by, e.g., LLMas-a-judge), and ‘operational ambiguity’: the disagreement that arises between interpretive communities when the policy is applied to concrete cases (measured, e.g., by LLM-DELTA).

## B.1 Overview

Each adversarial example consists of two policies, four cases, and two rater personas representing two interpretive groups. The two policies are designed to invert the relationship between surface and operational ambiguity:

• Policy A (the consensus policy) is worded so as to sound ambiguous—it uses open-textured, evaluative, or rhetorical language—but yields agreement between the two groups on every sampled case.

• Policy B (the conflict policy) is worded so as to sound concrete—it uses explicit, operational, or administrative language—but yields disagreement between the two groups, because its central terms encode a contested political, moral, institutional, or interpretive cleavage.

A surface-level ambiguity judge—whether an LLM-as-a-judge prompted to rate vagueness on a Likert scale, or a lexical measure based on concreteness—is expected to rate Policy A as more ambiguous than Policy B. An operational measure such as LLM-DELTA, which estimates ambiguity from disagreement among diverse raters about how the policy applies to cases, should rank the two policies in the opposite order. The adversarial dataset therefore functions as a discriminating test between the two families of measures.

## B.2 Latent dimensions

The construction relies on identifying two latent dimensions within a chosen policy domain.

Consensus axis C(·). A feature of cases that both groups classify in the same way, even though they may disagree on adjacent normative questions. Policy A is written to key onto this axis: although its language sounds vague, both groups read it as picking out the same feature of the cases.

Conflict axis H(·). A feature of cases whose meaning is contested between the groups— typically a term that is operationally clear on its face but interpreted differently across interpretive frameworks. Policy B is written to key onto this axis: although its language sounds concrete, the two groups disagree about whether particular cases satisfy it.

The four cases are then drawn as a cross-product of the two axes. Each case takes one of two values along C and one of two values along H, yielding the target judgment matrix shown in table B.1. Policy A produces uniform Y/Y or N/N judgments across the two groups (driven by C); Policy B produces split judgments (driven by H).

## B.3 Construction procedure

To generate an example in a target domain, we follow nine steps.

1. Choose a domain with both (i) recognisable implementation distinctions that different stakeholders agree on, and (ii) contested concepts that different stakeholders interpret differently.

<table><tr><td></td><td colspan="2">Policy A surface-ambig, operationally unambig</td><td colspan="2">Policy B surface-unambig, operationally ambig</td></tr><tr><td>Case</td><td>Group 1</td><td>Group 2</td><td>Group 1</td><td>Group 2</td></tr><tr><td>Case A (C=Y, H favours G1)</td><td>Y</td><td>Y</td><td>Y</td><td>N</td></tr><tr><td>Case B (C=N, H favours G1)</td><td>N</td><td>N</td><td>Y</td><td>N</td></tr><tr><td>Case C (C=Y, H favours G2)</td><td>Y</td><td>Y</td><td>N</td><td>Y</td></tr><tr><td>Case D (C=N, H favours G2)</td><td>N</td><td>N</td><td>N</td><td>Y</td></tr></table>

Table B.1: Target judgment matrix. Each case takes a value on the consensus axis C (shared by both groups) and on the conflict axis H (interpreted differently by the two groups); the notation “H favours G1” (resp. “G2”) denotes the value of H on which Group 1’s (resp. Group 2’s) reading of Policy B leads to Y. Y denotes that a group judges the case to be permitted by, or consistent with, the policy; N denotes the opposite.

2. Define two groups that plausibly partition the relevant population along an interpretive cleavage rather than along an outcome preference. Each group is specified by what it values, how it interprets key terms in the domain, and what kinds of implementations it sees as legitimate or illegitimate. Group labels should reflect interpretive stance rather than partisanship or desired outcome.

3. Identify a consensus axis C on which the two groups predictably agree. The axis should be a relatively concrete feature of cases even though the policy describing it will be worded vaguely.

4. Identify a conflict axis H on which the two groups predictably disagree, corresponding to a real political, moral, institutional, or interpretive cleavage in the domain. Each group should have a principled reading of the contested term.

5. Write Policy A so that it tracks C but uses surface-ambiguity cues such as open-textured evaluative adjectives, rhetorical contrasts, and value-laden but underspecified language.

6. Write Policy B so that it tracks H but uses surface-concreteness cues such as enumerated criteria, named harms, explicit institutional objectives, rights, or measurable-sounding outcomes.

7. Generate the four cases as the cross-product of C and H. Cases should be ecologically realistic—drawn from the kinds of proposals, programs, or incidents that actually arise in the domain—rather than artificial edge cases or technical loopholes.

8. Explain each judgment. For every cell in the matrix, give a one- or two-sentence rationale for why each group reaches its Y or N verdict. The exercise serves as a verification step: if a cell’s rationale is forced or implausible, the policy wording, group definition, or case is revised.

9. Verify the matrix against table B.1. Examples that intentionally depart from the standard matrix are flagged explicitly and the departure is justified.

## B.4 Rater personas

Each group is operationalised as one or more rater personas: third-person descriptions of how a person within that interpretive community thinks about the domain, what they value, and how they interpret the contested terms in Policy B. Personas are written without self-reference to their group and use plausible role labels rather than outcome labels. At the minimal scale, one persona per group is produced. When more personas are added (see Appendix B.6), the role, vocabulary, and specific concerns are varied while the underlying interpretive stance is preserved, so that within-group stylistic variation can be distinguished from between-group disagreement.

When real respondents are used in place of LLM personas, groups are derived empirically from a small set of anchor items diagnostic of the interpretive commitments, using rule-based scoring, latentclass analysis, or clustering. Groups are validated by checking that within-group agreement on anchor items is high, between-group disagreement is high, and assignments are stable on held-out items.

## B.5 Quality criteria

A candidate example is accepted only if it satisfies the following criteria.

• Surface plausibility. Policy A reads as vague or open-textured to an unprimed reader, and Policy B reads as concrete or operational.

• Operational behaviour. On the sampled cases, Policy A produces inter-group agreement and Policy B produces inter-group disagreement.

• Charitable groups. Both groups have principled, internally consistent reasons for their judgments. Neither is portrayed as irrational, malicious, or caricatured.

• Ecological realism. The cases are recognisable as plausible members of the domain— close to real policy instruments, programs, or disputes — and not dependent on obscure technical loopholes, semantic tricks, or unrealistic hypotheticals.

• Boundary distance. The Y/Y and N/N cases under Policy A are far enough from the consensus boundary that accidental within-policy-A disagreement is unlikely.

## B.6 Scaling within a domain

The default output of the procedure is a minimal set of two policies, four cases, and two rater personas. Three orthogonal directions of scaling are available within a single domain:

1. Paraphrase variation. Multiple wordings of Policy A and Policy B are generated, preserving the consensus or conflict axis but varying surface vocabulary. Paraphrases test whether the operational pattern is robust to wording. They are not treated as independent conceptual examples.

2. Cell variation. Additional realistic cases are generated within each cell of the matrix. This produces a larger case set while preserving the four-cell structure. Cell counts need not be uniform; we favour ecologically realistic frequencies over balanced quotas.

3. Rater paraphrase. Additional personas are generated within each group, varying role and vocabulary while preserving the interpretive stance. This allows within-group stylistic variation to be distinguished from between-group disagreement.

## \*Study Information & Consent Form

This is a study conducted by researchers at MIT Sloan School of Management about how people interpret university speech policies. You will be shown 2 policies and 20 examples of speech for each, and asked whether each example is permitted or prohibited under the policy. The data we collect will be used as part of a larger project on developing methods for quantifying ambiguity in policy language.

• The study takes about 20-25 minutes.

• Participation is voluntary. You may skip any question or stop at any time without penalty.

• The risks of participating are minimal. There is no direct benefit to you beyond contributing to research.

• Your responses are anonymous. We do not collect your name or other identifying information. De-identified data may be shared with other researchers or used in future research.

• For questions about the study, contact A/Prof Michiel Bakker via bakker@mit.edu

• You are not waiving any legal claims, rights or remedies because of your participation in this research study. If you feel you have been treated unfairly, or you have questions regarding your rights as a study participant you may contact the Chairman of the Committee on the Use of Humans as Experimental Subjects, M.I.T., Room E25-143B, 77 Massachusetts Ave, Cambridge, MA 02139, phone 1-617-253 6787.

We emphasize that you may find some of the examples of speech offensive, upsetting, or objectionable. Some of the speech described or quoted in the examples could be reasonably interpreted as controversial, prejudiced, discriminatory, or as celebrating or endorsing violence against specific people or groups. If you do not want to read this kind of language, please exit this page and do not participate in the study.

Do you consent to partipate in this study?

![](images/d04ccfeef751def1d1ae0d9571fc71cba9cb0aee8caf53012ac9a5305cd0c1d9.jpg)  
Figure A.2: Instructions given to participants in the human subject study at the beginning.

![](images/be95414462da154e20b976d2c5d561f5aaf428f1de451c553448d35122bd341f.jpg)  
Figure A.3: An example of the rating task assigned to participants in the human subject study.

\*Policy:

The university affirms its commitment to truth.

## Case:

A post-doc on the research team, as a check to make sure you are paying attention, is asking you to select both "Yes, permitted" and "Yes, prohibited" to this question.

Is the case permitted under the policy?

Choose the best answer, and assume this policy is the only policy in effect.

Yes, permitted

No, not permitted

\*Is the case prohibited under the policy?

Choose the best answer, and assume this policy is the only policy in effect

Yes, prohibited

No, not prohibited

## C Pol.is dataset details

We use four datasets from the Computational Democracy Project’s open Pol.is data repository. Pol.is is a platform for large-scale online deliberation in which participants submit short statements and vote on statements submitted by others. The repository describes these datasets as anonymized human expressions in political contexts, gathered through facilitated conversations using the open-source Pol.is software. The repository also provides statement-level metadata, vote data, participant–vote matrices, conversation summaries, and aggregate statistics for each conversation. The data are distributed under a Creative Commons Attribution 4.0 license, with attribution to The Computational Democracy Project. Our research use is consistent with their intention.

Our analysis uses the statement texts from the provided comments.csv files. Each row corresponds to a submitted statement and includes a timestamp, comment identifier, author identifier, numbers of agrees and disagrees, a moderation field, and the statement text. The bridging analysis additionally uses the per-group vote tables included in the same data releases, which report, for each statement, the aggregate agree, disagree, and pass counts within each of the two opinion groups that Pol.is identifies by clustering participants’ voting patterns. We use these published aggregates directly, rather than recomputing group assignments from the participant–vote matrices. We treat these statements as policy-relevant directional proposals: while they are not formal policy texts, they express claims about what governments or collective decision-makers should do. In the DELTA setup, cases are therefore interpreted as possible policy implementations or downstream decisions that could be evaluated against the proposal.

We use one conversation from the vTaiwan process and three conversations from Austria’s Klimarat process. The vTaiwan conversation concerns the regulation of UberX and app-based ride-hailing services in Taiwan, held in 2015. The Computational Democracy Project describes this conversation as an engagement with the government of Taiwan as part of the vTaiwan participatory process that contributed to the regulation of Uber in Taiwan. The conversation summary describes the prompt as asking whether UberX, as an innovative ridehailing service, should be required to follow the same laws as Taiwan’s taxi or rental-car industries.

In our analysis, we use the submitted statements from this conversation as the Taiwan ride-hailing subset.

The remaining three conversations are drawn from the Austrian Klimarat, a national citizens’ assembly on climate policy, held in 2022. The Klimarat began after a petition concerning Austria’s lack of a climate protection law received roughly 380,000 signatures, leading Parliament to fund a national citizens’ assembly. The assembly selected 100 Austrians through stratified lottery, of whom 84 participated over six weekends to develop climatepolicy recommendations. During the process, assembly members sought broader public feedback on draft recommendations through a project called “The Climate Council asks Austria,” which used Pol.is to collect and structure input from more than 5,000 Austrians.

From the Klimarat open data, we use three topicspecific Pol.is conversations: Production and Consumption, Housing, and Energy. The Production and Consumption conversation asked participants for opinions and ideas on achieving climate neutrality by 2040, with discussion areas including product use and production, cost transparency, social and technological innovation, infrastructure and financing, and awareness-building and labeling. The Housing conversation asked for views on housing and climate neutrality, including the use of existing housing stock, renovation, adaptation needs, and emissions from construction and operation. The Energy conversation asked for views on reducing social energy consumption, shifting to renewable energy, increasing energy sufficiency, and financing the energy system, including carbon pricing. We use the statements from each topic as a separate subset in the DELTA analysis.

For each Pol.is conversation, we preprocess the data: First, we retain only statements whose moderation field indicates inclusion. This is the platform’s own moderation flag; it removes roughly half of each export, including off-topic or inappropriate statements and statements that were not voted on. Next, we translate the statements (originally in Traditional Chinese for vTaiwan and German for the Klimarat conversations) into English using GPT-5, prompted to preserve hedging, sentiment, and policy terminology and to keep institutional names such as Klimarat as they are, so that remaining analyses are exclusively in English. Every statement DELTA scores is therefore a machine translation rather than the participant’s original wording. In the vTaiwan dataset, we additionally remove two kinds of statements by hand: factual self-disclosures, which report a fact about the speaker rather than making a proposal (“I have used Uber to hail a ride”, “I have a driver’s license for small cars”; 5 statements), and descriptive comparisons of Uber and taxis as they already operate (“I think UberX, on average, has better quality than taxis”; 22 statements). DELTA scores a statement by whether raters agree on which concrete actions it permits and rules out, so a purely descriptive claim gives a rater nothing to apply to a case. For the Klimarat conversations, we performed manual curation only to remove link spam and comments about the consultation process itself (two statements in Production and Consumption, one in Housing, none in Energy).

<table><tr><td>Dataset</td><td>n</td><td>Cleaned</td><td>High-eng.</td><td>A</td><td>B</td></tr><tr><td>Uber</td><td>199</td><td>72</td><td>63</td><td>410</td><td>398</td></tr><tr><td>Consump.</td><td>522</td><td>275</td><td>126</td><td>540</td><td>134</td></tr><tr><td>Housing</td><td>613</td><td>314</td><td>91</td><td>335</td><td>705</td></tr><tr><td>Energy</td><td>1,041</td><td>515</td><td>133</td><td>206</td><td>984</td></tr></table>

Table C.2: Summary statistics by dataset. n denotes the number of statements in the raw export. “Cleaned” denotes the number remaining after the moderation filter and manual curation; these are the statements DELTA scores. “High-eng.” denotes the number of high-engagement statements used in the bridging analysis, defined as those on which at least 20% of the participants in each of groups A and B registered a response (agree, disagree, or pass). “A” and “B” denote the sizes of the two opinion groups algorithmically identified by Pol.is, estimated as the largest number of votes the group cast on any single statement.

We apply the DELTA pipeline separately within each topic-specific subset, scoring every statement that survives cleaning. Case generation is seeded from 30 statements of each subset, favoring longer statements, and every statement is evaluated against 60 sampled cases for vTaiwan and 40 for each Klimarat conversation. The bridging analysis additionally restricts to statements with sufficient engagement from both opinion groups; Appendix C.1 details this threshold and shows that the results are not sensitive to it. The summary statistics are presented in table C.2.

## C.1 Engagement threshold

The group agreement rates $p _ { A }$ and $p _ { B }$ underlying the bridging analysis in Figure 6 are estimated from observed votes, Laplace-smoothed as $p = ( a + 1 ) / ( a + d + 2 )$ , where a and d are the group’s agree and disagree counts on the statement; passes are not counted as votes. Some engagement filter is necessary when using these estimates: Pol.is shows different statements to different participants, so statements submitted late in a conversation are seen by few voters, and rates estimated from a handful of votes are noisy. We therefore retain only statements on which at least 20% of the participants in each opinion group registered a response (agree, disagree, or pass), taking as each group’s size the largest number of votes it cast on any single statement (columns A and B of table C.2). The value of 20% itself, however, is a judgment call. Figure C.6 therefore recomputes the Spearman rank correlation between bridging and DELTA at every engagement cutoff from 0 (no filter) to 0.9, with each line truncated once fewer than 20 statements remain; shaded bands are Fisher-z 95% confidence intervals, which widen as the filter shrinks the sample.

The conclusions do not depend on the cutoff. Most importantly, at no cutoff does any dataset show a stable positive correlation—the direction the ambiguity-as-compromise hypothesis predicts. The vTaiwan–Uber and consumption conversations hover around zero throughout (ρ between −0.30 and +0.16), with intervals that always include zero. The housing conversation is weakly negative only before any engagement filtering is applied $( \rho \approx - 0 . 1 5 )$ , consistent with noise from low-engagement statements; its one positive interval, at a single isolated high cutoff, is no more than chance would produce across the roughly 120 intervals plotted. The energy conversation is the exception: its correlation is negative at every cutoff, and the interval excludes zero across the contiguous band of cutoffs from 0.1 to 0.4 (ρ between −0.14 and −0.30), which includes the 0.2 cutoff used in the main text. We treat this as suggestive rather than conclusive: it appears in one dataset of four, and the corresponding Pearson correlations are weaker (fig. 6), indicating an association carried by ordering rather than by a linear trend in raw scores.

![](images/08a75496067bcc4e469c69c39570401d9b2f79ade58b4f7b755bd3022417c381.jpg)  
Figure C.6: Spearman rank correlation between bridging, min(p<sub>A</sub>, p<sub>B</sub>), and DELTA as a function of the minimum engagement cutoff: statements are retained only if at least the given fraction of each opinion group’s participants responded to them. Shaded bands are Fisher-z 95% confidence intervals; lines truncate where fewer than 20 statements remain. The dashed vertical line marks the 0.2 cutoff used in Figure 6. The climate– energy correlation is negative at every cutoff and significantly so across a band of intermediate cutoffs; no dataset shows a stable positive correlation at any cutoff.

## D Habermas dataset details

The Virtual Citizens’ Assembly in the Habermas study was a controlled deliberation experiment involving a demographically representative sample of the United Kingdom population (Tessler et al., 2024). Participants were recruited through randomized stratified sampling (implemented by the Sortition Foundation) to ensure representativeness across age, gender identity, ethnicity, socioeconomic status, and geographic region. The assembly took place over three weekly one-hour sessions in April and May 2023. Our research use is consistent with their intention.

Unlike the convenience-sampled cohorts in the broader study, the Virtual Citizens’ Assembly was specifically designed to simulate the deliberation phase of a real citizens’ assembly, albeit without the typical fact-finding or expert testimony phases. All participant groups deliberated over the same nine questions, enabling cross-group comparison of deliberative outcomes. The nine assembly questions addressed potentially divisive issues in UK public policy. They were selected from a larger corpus of approximately 5,500 deliberation questions based on participants’ ratings of topic importance during earlier phases of data collection:

1. Should we be trying to reduce the number of people in prison?

2. Should we raise the minimum wage in the UK?

3. Should we raise the retirement age from 66 to 68?

4. Should the UK change its asylum seeking policy to make it easier to enter the country?

5. Should the UK apply to rejoin the European Union?

6. Should we be doing more to encourage a greater degree of national pride?

7. Should the government provide universal free childcare from birth?

8. Has the Internet been a net positive for humanity?

9. Should the UK set a goal to be carbon neutral (reach net zero) by 2050?

One deliberation round on a question refers to a group of 4-5 participants following a structured deliberation protocol:

1. Opinion Elicitation: Participants were given five minutes to write a short paragraph (10– 200 words) describing their personal opinion on the question. Groups consisted of five participants each.

2. Initial Statement Generation: The Habermas Machine, an AI system based on large language models, generated four candidate “group statements” designed to capture common ground among the participants’ opinions.

3. Rating and Ranking: Participants independently rated each candidate statement on two dimensions: personal agreement (7-point scale from “Strongly Disagree” to “Strongly Agree”) and statement quality (7-point scale from “Terrible Quality” to “Excellent Quality”). They also provided preference rankings of all the statements.

4. Critique: The top-ranked initial statement (selected via the Schulze method) was returned to participants, who wrote critiques identifying areas for improvement.

5. Revised Statement Generation: The AI generated four revised candidate statements incorporating the critiques along with the original opinions.

6. Final Evaluation: Participants rated, ranked, and compared the revised statements, then indicated their preference between the winning initial and revised statements.

Each deliberation round lasted approximately 15 minutes, with participants completing three questions per session.

## D.1 Extraction and cleaning

The original dataset from Tessler et al. (2024) is available on Github. We use two of its tables. The candidate-comparisons table records, for every participant, round, and iteration, the candidate statements shown, the participant’s agreement and quality rating of each candidate, and their preference ranking over the candidates. The positionstatement-ratings table records each participant’s agreement with a statement affirming the question, elicited before and after deliberation. In both tables a participant’s judgments are stored as parallel arrays—candidate identifiers alongside numerical ranks and ratings—which we unnest jointly into one record per participant, round, iteration, and candidate. Categorical rating labels are mapped to the 1–7 scale; rankings a participant did not submit are stored by the release as −1, which we treat as missing, and rows from the release’s MOCK control condition carry no agreement rating and are excluded from the agreement analysis.

We clean the extracted data as follows. First, we exclude question 8 about the Internet: it asks for an evaluation rather than a prescription, so its statements have no policy content whose interpretation could diverge. Second, we keep only deliberation rounds that completed both the initial and the revised iteration, leaving 309 rounds. Third, we recompute each round’s winning statement per iteration. The top-candidate field in the release is a placeholder, so we identify the winner as the candidate with the lowest mean numerical rank across the round’s participants; candidates with no text or no rankings are ineligible, and ties fall to the earlier candidate, matching the order in which participants saw them. Six round-iterations contain no eligible candidate, and two rounds thereby lose both of their iterations, leaving 307 rounds and 612 winning statements. Finally, we truncate 29 collective statements where it was clear that the raw text contained additional misidentified content, for example, the statement

Yes, we should be trying to reduce the number of people in prison. There is much evidence to suggest that prison can have a negative effect on prisoners, with no benefit to the wider community.

Increasing emphasis should be placed on community-based punishment, rehabilitation and reintegration, with a reduction in the number of prison sentences given. There may be a need to build more prisons to reduce overcrowding, however we believe this should not be done at the expense of rehabilitation and reintegration. Prison should be used as a last resort for those who are a danger to society.

Final Comments from the group participants: The group worked really well together, sharing opinions and working to come to a consensus. We did not have anyone who strongly disagreed with the opinion of the majority of the group so it made the discussion and consensus easier. How did this tool help participants to overcome their fear of working in groups? This tool helped participants to overcome their fear of working in groups as it helped them to get to know others in the group and their opinions, and to feel comfortable sharing their opinions.

## was truncated to the first paragraph only.

The 612 winning statements—the highestranked draft and revised statements across all deliberation rounds—form the policy sets for the DELTA analysis: one set per question (72–82 statements each), scored in one run per question so that scores are only ever compared within a question. Each run used 30 raters and 60 sampled cases, with the generation and judgment settings of Appendix E. Alongside the winning statements we extract two participant-level tables: each participant’s agreement rating of their round’s winning statements (3,060 ratings, of which 2,350 carry an agreement value; the remainder are MOCK rows), and each participant’s pre- and post-deliberation agreement with the question’s affirming statement, which the figure below uses to identify initially divided rounds. The position ratings carry no round identifier in the release, so they are joined to rounds through the participant–question pair. Table D.3 presents summary statistics before and after cleaning.

## D.2 The agreement-distribution figure

Figure 7 relates each participant’s agreement with their round’s winning statement to that statement’s DELTA score.

<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Raw release (nine questions)</td><td></td></tr><tr><td>Deliberation questions Unique deliberation rounds</td><td>9 344</td></tr><tr><td>Total participants</td><td>1,720</td></tr><tr><td>Total collective statements Draft statements</td><td>2,693</td></tr><tr><td>Revised statements</td><td>1,367</td></tr><tr><td>After cleaning (eight questions) Rds completing both iterations</td><td>1,326</td></tr><tr><td>Rds with a winning statement</td><td>309 307</td></tr><tr><td>Winning stmts (draft + revised)</td><td>612</td></tr><tr><td>Ratings of winning statements with an agreement value</td><td>3,060</td></tr><tr><td>Cleaned rounds / winning stmts per question</td><td>2,350</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Prison population</td><td>41 / 82</td></tr><tr><td>Minimum wage</td><td>41/81</td></tr><tr><td>Asylum policy</td><td></td></tr><tr><td></td><td>38/76</td></tr><tr><td>National pride</td><td>38/76</td></tr><tr><td>Universal childcare</td><td>38 /75</td></tr><tr><td>Retirement age</td><td>38 /76</td></tr><tr><td></td><td></td></tr><tr><td>European Union</td><td>37/74</td></tr><tr><td>Carbon neutrality</td><td>36/72</td></tr></table>

Table D.3: Summary statistics for the raw release and the cleaned dataset. The raw block spans all nine questions; cleaning excludes the Internet question, rounds that did not complete both iterations, and rounditerations with no rateable winning statement. Where a question’s statement count is below twice its round count, one iteration of one round produced no rateable winner.

First, as described in the main text, we restrict to rounds whose participants were initially divided: at least one member strictly above the neutral midpoint (positions 5–7) on the pre-deliberation position rating, and at least one strictly below (positions 1–3); participants exactly at the midpoint count toward neither side. A round that began in agreement cannot show ambiguity failing to bridge opposed positions. This restriction keeps 203 of the 307 rounds and 1,706 of the agreement ratings.

Second, DELTA scores are comparable only within a single scoring run, so the horizontal axis is the percentile rank of the statement’s DELTA score within its question. This pools the eight questions onto a common [0, 1] axis without assuming that DELTA scores are similarly distributed across questions.

Third, for each level k of the 1–7 agreement scale, we form the indicator that a rating equals k and smooth its share as a function of DELTA rank with LOWESS (bandwidth fraction 0.45, evaluated on a common grid). The 95% confidence bands come from a cluster bootstrap with 300 resamples that resamples whole deliberation rounds with replacement, rather than individual ratings: all ratings in a round refer to the same winning statement and share its DELTA value, so their errors are correlated, and resampling individual ratings would understate the uncertainty.

## E Model and prompts

## E.1 Case and rater generation

To generate cases and raters, we use GPT-5 with medium thinking effort, temperature 1, and a maximum of 10,000 output tokens. The rater-expansion step, which appends one short sentence per call, instead runs with thinking disabled and a 200-token cap.

Cases are generated from pairs of policies: the prompt in fig. E.7 presents two policies as alternative versions of the same rule and asks for three short fictional scenarios that comply with the first while violating the second. Because policies in a set typically express similar goals, this forces generation onto precisely the distinctions between them. For each dataset we sample 30 ordered pairs uniformly at random (fixed seed) from the dataset’s seed policies, yielding a pool of 90 scenarios, from which the final case set is randomly subsampled: 60 cases for the vTaiwan and Habermas runs, 40 for the Klimarat runs.

Raters are generated in pairs with opposing interests using the top prompt in fig. E.8, ten stakeholders (five pairs) per call; we run the query three times to obtain 30 rater identities per dataset. Generating in several smaller calls rather than one large one is deliberate, as a single call tends to produce a more homogeneous list. Each identity is then expanded using the bottom prompt in fig. E.8, which takes the current description and adds one sentence to it; four applications yield the final five-sentence rater description.

To illustrate the outputs of this pipeline, we give a worked example drawn from the Habermas topic series (Appendix D), for the discussion question “Should we be trying to reduce the number of people in prison?” The policies in this run are the collective statements generated for that question, and the experiment configuration describes a case as a specific criminal justice policy or related action— for instance, a change to sentencing guidelines, rehabilitation programs, prison funding, or parole decisions—that raters can judge each statement against. One case generated from a pair of collective statements reads:

Presumptive Non-Custodial Sentencing for Low-Level, Non-Violent Offences: Amend the Sentencing Code to create a statutory presumption against custodial sentences under 12 months, replacing them with structured community orders combining unpaid work, fines, curfews, and mandated treatment; judges may depart only with written reasons; cap custody for technical probation breaches at 7 days; increase probation staffing and electronic monitoring capacity; require quarterly compliance reporting to victims and courts; expected to reduce shortterm prison receptions. Tradeoffs include higher probation caseloads and risk of non-compliance.

An example rater identity generated for the same question is:

This person is a suburban homeowner and taxpayer in Oakland County, Michigan, age 67 and on a fixed income, who supports decarceration only if a statutory budget lockbox dedicates documented savings to property tax relief, local victim services, and neighborhood patrols, with quarterly public dashboards on recidivism and costs and a five-year sunset requiring reauthorization; they care about fiscal accountability and visible community benefits.

The four expansion passes then append one sentence each, grounding the identity in concrete experiences and increasingly specific positions on the question; for this rater they add:

After her car was stolen from her driveway in Rochester Hills last summer, she’ll only support decarceration if verified savings show up as a line on her property tax bill and fund extra neighborhood patrols and victim services. After paying a \$500 insurance deductible and replacing her house locks when the thief took her spare key, she wants savings earmarked for victim reimbursements like deductibles and locksmith fees. Since her car was never recovered and no one was charged, she wants an audited county fund that reimburses deductibles and locksmith costs within 30 days, itemized as a credit on her property tax bill. She wants a hard trigger that pauses any decarceration plan if auto thefts or recidivism rise in her ZIP code for two consecutive quarters.

Note that although the underlying deliberation took place in the UK, the generation prompts carry only the discussion question and the policy texts, so the model was free to situate stakeholders elsewhere.

## E.2 Judgment simulation

Figure E.9 contains the prompt used to simulate a rater’s judgment of a policy against a case in the Pol.is and Habermas datasets. The university speech and adversarial datasets use the same structure with domain-specific wording (fig. E.10): raters judge whether a speech incident is permitted or prohibited under a university policy, and whether a proposed constitutional-reform implementation is consistent or inconsistent with a stated aspiration. Each judgment is elicited twice, under a positive and a negative framing of the final question, in separate queries; the two answers are collapsed into the revealed probability described in section 3.

We use Gemini-2.5-Flash for this task, with temperature 1 and thinking disabled, under a system prompt restricting the answer to yes or no. Each rater × policy × case × framing query is issued once, and the sampled answer is recorded as a binary value. A lightweight non-reasoning model is a deliberate choice beyond its low cost, which addresses the combinatorially large number of queries: the DELTA score is built on the spread of answers across raters, which reasoning tends to collapse.

## E.3 LLM-as-a-judge baselines

The two direct LLM-as-a-judge baselines rate each policy on a 1–7 scale from its text alone, with no cases and no rater personas. The system prompt carries the rubric, and the user prompt supplies the discussion question and the statement to be rated. The first rubric (fig. E.11) defines ambiguity in dictionary-style terms of clarity and specificity.

![](images/c4c36ad04f23b41572452767abed78994a7a50afbb78e0b43cfe6bf09f8d1f53.jpg)  
Figure E.7: Prompt to generate cases, three at a time, from an ordered pair of policies. Placeholders in braces are filled from the dataset’s experiment configuration.

The second rubric (figs. E.12 to E.14) mirrors the DELTA mechanism: it defines consequential ambiguity as the disagreement that would arise among diverse stakeholders classifying concrete actions as consistent or inconsistent with the statement, provides guiding patterns and three worked examples, and instructs the model to mentally generate boundary-testing scenarios before rating.

Three judge models rate every statement under both rubrics: GPT-5 (medium thinking effort, temperature 1), GPT-5.2 (thinking disabled, temperature 0), and Gemini-3-Pro Preview (medium thinking effort, temperature 1). Each judge produces one rating per statement, and a statement’s baseline score is the mean of the three judges’ ratings. The baselines are applied to the campus speech policies (Appendix A) and to the adversarial policies (Appendix B).

![](images/996a9ab7ba4f510524d04fa74bd8710ee9ee0fb1e57c79ebe84e5081b445a606.jpg)

![](images/e721b27ecbf5ab509879db850c6788cf51ec03fa457bda3c490dbffe2058a8ed.jpg)  
Figure E.8: Top: prompt to generate rater identities, ten at a time in opposing pairs. Bottom: prompt to expand a rater description, which takes the current description and adds one sentence to it; it is applied four times, starting from the generated identity, to reach the final five-sentence description.

![](images/075be461fa1bc8e07a6cc914e10ba485b72ddf613160879d13af4eb07fec7715.jpg)  
Figure E.9: The judgment prompt for the Pol.is and Habermas datasets. The final question takes one of two framings, asked in separate queries: the positive framing “Does the intention of the policy proposal support/is compatible with the principles and beliefs described in the opinion statement?” and the negative framing “Does the intention of the policy proposal contradict/stand in opposition to the principles and beliefs described in the opinion statement?”

![](images/d54966ef725698e364f1bbb5ba1668685c205bb1638b185102624eee6b95bd39.jpg)

![](images/057ccac590e48ff4c44df41995c942f3c06e7ac0724ebd22cf0c79c64e8b15aa.jpg)  
Figure E.10: The judgment prompts for the university speech dataset (top) and the adversarial dataset (bottom). As in fig. E.9, each judgment is asked under two framings in separate queries. University speech: “Would this person say the speech in the incident is permitted under the polic $I ? ^ { \dag }$ and “Would this person say the speech in the incident is prohibited under the policy?” Adversarial: “Would this person say the proposed implementation is consistent with the constitutional aspiration?” and “Would this person say the proposed implementation is inconsistent with the constitutional aspiration?”

![](images/544734a78e663bc8d9bda92d44e41b8f756b14a24a21c43fc2aae9744d556bb7.jpg)  
Figure E.11: Prompt for the LLM-as-a-judge ambiguity baseline.

![](images/0f7183aa83f4319df70c28766c536564f3438b5b882b8a58e2dae9f60e36e526.jpg)  
Figure E.12: Prompt for the LLM-as-a-judge disagreement baseline, which mirrors the DELTA mechanism (part 1 of 3).

![](images/90977d7d6d4fa84e7568de0c4834aa7ba7fc63962922356d4dcce4f7bb003f38.jpg)  
Figure E.13: Prompt for the LLM-as-a-judge disagreement baseline, which mirrors the DELTA mechanism (part 2 of 3).

![](images/92bb0ff30056bd0572e72d17be2b6f9a473398a59da9e08931fc191d54e4c8a5.jpg)  
Figure E.14: Prompt for the LLM-as-a-judge disagreement baseline, which mirrors the DELTA mechanism (part 3 of 3).