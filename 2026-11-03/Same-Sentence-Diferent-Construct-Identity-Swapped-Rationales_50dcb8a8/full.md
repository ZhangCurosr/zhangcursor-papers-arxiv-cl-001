# Same Sentence, Diferent Construct: Identity-Swapped Rationales Reveal Measurement Non-Invariance in LLM Hate Speech Classifiers

WENCHANG WANG<sup>∗</sup>, Northwestern University, USA

YUE SONG<sup>∗</sup>, Peking University, China

Large language models are increasingly used as measurement instruments for socially consequential constructs such as hate speech, and are typically validated by their aggregate agreement with human labels. A valid instrument must also apply the same construct to every group it measures. We test this requirement using 421 HateCheck templates that are instantiated identically for seven protected groups, so that only the identity term varies. Four LLM classifiers each exceed 0.95 balanced accuracy, yet within identical templates two of them miss hateful statements about women at markedly higher rates than the same statements about other groups (excess false-negative rates of 8.3 and 4.4 percentage points), a gap first reported for fine-tuned classifiers in 2021. We show how the construct shifts. Comparing rationales within each template, we find that models invoke protected-group status for other groups but recast the identical statement about women as personal dislike or as aimed at an individual, in 22 of 30 missed templates for one model. Adding the benchmark’s own definition, which lists sex as a protected characteristic, does not close the gap, and the contrast persists Apparent false-positive disparities, by contrast, largely vanish under identity control and trace to reclaimed slurs present only for some groups. Errors shared by all models concentrate where model and benchmark definitions diverge, and pairwise counterfactual accurac adds little beyond endpoint accuracy. We argue that validating LLM measurement requires identity-controlled tests of measurement invariance that examine both decisions and the criteria models state for them.

CCS Concepts: • Computing methodologies → Natural language processing; • Social and professional topics → Computing / technology policy.

Additional Key Words and Phrases: construct validity, measurement invariance, counterfactual auditing, identity-swap testing, LLM-based measurement, hate speech detection, algorithmic fairness

Content warning: this paper contains examples of hateful language, reproduced from a published test suite for research purposes.

## 1 Introduction

Researchers and platforms increasingly deploy prompted large language models (LLMs) as measurement instruments for socially consequential constructs [16, 42]. Classifications of subjective text—such as evaluating whether an utterance constitutes hate speech or toxic discourse—directly serve as empirical data for demographic prevalence estimation, crosscommunity comparisons, and algorithmic content moderation. Recent audits demonstrate that distinct LLM pipeline assign divergent labels to identical content, with discrepancies concentrating around specific social identities [10]

Validating these model-derived measurements predominantly relies on aggregate agreement with human annotations (e.g., balanced accuracy). However, classical measurement theory establishes that aggregate agreement alone leaves validity under-determined: a scientifically sound instrument must satisfy measurement invariance, operationalizing the identical latent construct uniformly across all comparative subpopulations [5, 20, 23, 28, 29]. In algorithmic fairness and social computing, score interpretations fail when an operationalization captures systematic artifacts or demographicspecific thresholds [4, 19]. In hate speech detection specifically, the substantive subpopulations of interest are the target identities. When a classifier applies an asymmetrical evidentiary threshold to misogynistic abuse compared to identical slurs targeting religious or racial groups, downstream prevalence metrics, group comparisons, and moderation enforcement inherit systematic distortion

Empirical studies have repeatedly documented that moderation systems exhibit uneven sensitivity across demographic targets. Fine-tuned BERT classifiers misclassify abusive cases targeting women at approximately double the error rate observed for other protected categories [34], an asymmetry reproduced in GPT-3.5 evaluations [25]. Subsequent audits of commercial moderation endpoints confirm disparate flagging across target identities [35], alongside systematic over-moderation of counter-speech and reclaimed slurs [6, 18, 36]. Black-box audit methodologies, however, primarily record disparate performance outcomes while conflating construct shifts with disparities in the lexical and syntactic composition of test datasets.

This paper provides a diagnostic framework to determine whether LLMs alter their operative construct when only the protected identity term varies. Leveraging the behavioral testing paradigm [15, 24, 33], we implement two identity controlled designs over the 421 HateCheck syntactic templates instantiated uniformly across seven protected groups: We isolate the causal impact of identity by measuring each group’s excess error relative to the remaining six groups on identical syntactic structures, thereby eliminating confounding from group-specific test-item distributions. We analyze the models’ explicit rationales for divergent classifications on paired templates [21, 38]. Holding the surrounding lexica context constant renders stated decision criteria an auditable window into construct instability.

Auditing four state-of-the-art LLM classifiers—each exceeding 0.95 balanced accuracy on HateCheck—our findings establish:

(1) A persistent, identity-specific coverage gap. Across identical templates, two models miss hateful statements targeting women at markedly elevated rates relative to identical statements targeting other groups (excess false-negative rates of 8.3 and 4.4 percentage points); one model exhibits a parallel deficit for immigrants. The error profile documented in 2021 fine-tuned classifiers persists in current foundation models.

(2) A construct shift visible in paired rationales. On identical templates, models identify protected-group targeting for other demographics while reclassifying identical misogynistic slurs as “personal dislike” or interpersonal grievances (accounting for 22 of 30 missed templates in one model).

(3) Definitional prompting fails to restore invariance. Injecting the benchmark’s explicit definition—which formally designates sex as a protected attribute—leaves the false-negative gap for women unchanged. Simultaneously, the definition causes one model to drop coverage of immigrants (a group omitted from the text), revealing that models adhere strictly to explicit exclusions while failing to enforce explicit inclusions.

(4) False-positive disparities stem from benchmark composition. Disparities in over-flagging largely dis appear under identity control and trace back to reclaimed slurs, which exist in the suite only for specific sub-groups, demonstrating that over-moderation harms in-group linguistic norms rather than general mentions

(5) Shared errors reflect construct disagreement, not noise. Instances where all four models fail occur 9.5 times more frequently than predicted by independent chance, concentrating on individual abuse where the models’ operative construct is systematically broader than the benchmark’s.

(6) Pairwise counterfactual accuracy is largely redundant. On standard counterfactual pairs, joint accuracy adds negligible diagnostic signal (within 0.008) beyond endpoint accuracy; however, within-pair error correlation successfully distinguishes systematic bias from stochastic noise.

Same Sentence, Diferent Construct: Identity-Swapped Rationales Reveal Measurement Non-Invariance in LLM Hate Speech Classifiers 3

These findings demonstrate that high aggregate accuracy routinely masks severe measurement non-invariance across target groups. Model rationales provide an empirical mechanism for auditing latent construct shifts. We conclude by ofering standardized reporting protocols for deploying LLMs to measure socially consequential phenomena.

## 2 Background and Related Work

LLMs as Measurement Instruments. Computational social science and algorithmic content moderation increasingly treat prompted LLMs as surrogate annotators, frequently reporting agreement rates that rival or surpass crowd workers on benchmark evaluations [16, 42]. Recent investigations demonstrate that LLM-based labeling systems exhibit pronounced inter-model variance when evaluating identical contentious content, with disagreements concentrating around specific demographic categories [10]. We build upon this observational foundation by investigating the internal consistency of individual models, assessing whether a single classifier applies an invariant construct across diverse target identities.

Construct Validity and Measurement Invariance. Psychometric foundations define validity not as an inherent property of an instrument, but as the evidentiary support for interpretations derived from its scores [5, 23, 29]. Measurement theory requires measurement invariance: an instrument must map onto the latent attribute symmetrically acros disparate demographic subgroups [28]. Violations of invariance indicate that cross-group diferences in observed scores reflect divergent measurement scales rather than genuine variation in the underlying phenomenon. Jacobs and Wallach [20] demonstrate that algorithmic unfairness frequently originates as a measurement failure, wherein an operationalized proxy departs from its theoretical construct heterogeneously across protected populations. Thi perspective aligns with formal algorithmic fairness and FAccT literature establishing that parity metrics remain uninterpretable without specifying the latent construct, causal dependencies, and contextual decision thresholds [1– 4, 17, 19, 31, 37]. We operationalize measurement invariance for NLP classifiers by measuring stability under identitycontrolled counterfactual perturbations.

Behavioral Testing and Counterfactual Audits. Behavioral and capability-driven test suites evaluate NLP systems via systematically perturbed inputs [33]. Related methodologies construct contrast sets and counterfactually augmented corpora by applying minimal, label-altering revisions to existing texts [14, 24]. In hate speech research, identity-term substitution has served as a primary technique to detect demographic bias and unintended associations [8, 15]. In particular, HateCheck [34] provides 29 functional diagnostic suites, standardizing 421 templates instantiated across seven distinct protected categories. While Johnson and Marasović [22] establish that contrast-set consistency correlates strongly with base model accuracy, our diagnostic framework establishes an analytical independence baseline for counterfactual pairs, utilizing within-pair error correlations to isolate systematic construct instability from stochastic classification error.

Demographic Disparities in Toxicity Classifiers. Toxicity and hate speech classification corpora consistently exhibit dialectal, racial, and target-group heterogeneities [9, 11–13, 26, 27, 30, 39, 40]. Consequently, deployed classifiers systematically over-flag African American Vernacular English and in-group discourse [6, 36], while penalizing counter-speech and reclaimed slurs [18, 34]. Concurrently, coverage deficits regarding misogynistic content have been documented across both fine-tuned architectures [34] and generative models [25], alongside identity-dependent classification shifts in commercial moderation APIs [35]. Extending beyond prior studies that document demographic disparities at the output level, our framework deploys an identity-controlled decomposition that isolates whether disparities stem from test-suite composition or from asymmetric operative definitions.

Auditing Stated Model Rationales. Model-generated explanations frequently diverge from internal computationa mechanisms [7, 21, 38, 41]. Rather than treating rationales as mechanistically faithful causal explanations, we analyze them as the model’s explicit, stated operationalization ofthe target construct. When a model evaluates identical linguistic contexts while yielding opposing labels across identity targets, discrepancies in its stated criteria constitute direct qualitative evidence of construct non-invariance. This analytical stance operationalizes model rationales as an empirical lens for external algorithmic auditing, transforming self-explanations into verifiable evidence of systemic disparity in automated decision-making [21, 32].

## 3 Data and Models

HateCheck. We evaluate models using the HateCheck English diagnostic suite [34], comprising 3,728 human-validated test cases annotated with binary gold labels, functionality codes, target identities, and template identifiers. The benchmark spans seven protected groups: women, trans people, gay people, Black people, disabled people, Muslims, and immigrants. Among the 553 templates with target identities, 421 templates are instantiated identically across all seven groups under a single shared gold label (325 hateful, 96 non-hateful). The remaining 132 templates exhibit group-restricted distributions, primarily capturing linguistic phenomena unique to specific demographics: 81 templates evaluate reclaimed slurs and 30 evaluate slur homonyms. Additionally, published reference links between cases yield 2,046 counterfactual pairs (600 label-changing and 1,446 label-preserving).

Counterfactually Augmented Sentiment Data (CAD).. To benchmark evaluation metrics on a non-identity domain, we incorporate the human-revised movie review pairs from Kaushik et al. [24], where crowd workers minimally edited reviews to flip sentiment polarity. We identify the original document of each pair via exact matching against published source corpora, yielding 2,438 validated pairs (4,880 individual texts). CAD serves as a reference counterfactual baseline for the evaluation methodology analyzed in Section 5.7.

Models and Protocol. We audit four production-grade, API-accessible LLM endpoints suitable for large-scale annotation workflows: DeepSeek Flash (deepseek-flash), GLM-5.3-Flash (glm-5.3-flash), Hy4 Preview (hy4-preview), and MiniMax M3.1 Flash (minimax-m3.1-flash). These endpoints provide functional coverage across major contemporary model families. All models received uniform system and user prompts (Appendix A) specifying the target construct, defining the label space, and requesting a structured JSON payload containing the assigned label, a continuous confidence score in [0, 1], and an explicit rationale. For the definitional ablation in Section 5.4, we augmented the prompt for the full-template subset with the verbatim HateCheck definition (Appendix A), keeping all other parameters constant.

All requests were executed on 24–25 September 2026 via Tencent Cloud’s API platform. Across all models, the decoding temperature was fixed at 0, and reasoning was left to each endpoint’s default configuration.

## 4 Audit Design

We operationalize five diagnostic estimands. Statistical inference relies on 95% percentile bootstrap intervals with 2,000 replicates, resampling at the designated observational unit for each estimand.

E1: Aggregate Agreement. Balanced accuracy evaluated against source gold labels, resampling individual test items.

Same Sentence, Diferent Construct: Identity-Swapped Rationales Reveal Measurement Non-Invariance in LLM Hate Speech Classifiers 5

E2: Within-Template Excess Error. For a syntactic template � instantiated across groups $g \in G \left( \left| G \right| = 7 \right)$ , let $e _ { t g } \in \{ 0 , 1 \}$ indicate an erroneous prediction. The excess error of group � is defined as

$$
\Delta _ { g } = \frac { 1 } { | T | } \sum _ { t \in T } \Big ( e _ { t g } - \frac { 1 } { | G | - 1 } \sum _ { h \neq g } e _ { t h } \Big ) ,
$$

computed separately over hateful templates (excess false-negative rate) and non-hateful templates (excess false-positive rate), resampling across templates. Because lexical framing, communicative functionality, and ground-truth labels remain invariant within $t , \Delta _ { g }$ isolates the causal contribution of the identity term from cross-group diferences in test-item distributions. We additionally evaluate the proportion of templates yielding discordant predictions across groups.

E3: Paired-Rationale Contrast. For full templates where a model fails on group �’s hateful instantiation while correctly classifying the same template for other groups, we conduct a comparative analysis of the generated justifications. We code two operational cues: protected-status invocation (grounding classification in protected group membership) and denial or personalization (refuting that a protected demographic is targeted, or attributing the statement to personal sentiment or interpersonal conflict). A template exhibits a contrast when the rationale for the missed case manifests denial or personalization while at least one correctly classified counterpart explicitly invokes protected group status. Cues are extracted via deterministic regular expressions (Appendix C), capturing the model’s explicit, stated decision rules.

E4: Shared-Error Concentration. Over instances evaluated across all four models, we benchmark the count of unanimously misclassified items against the expected baseline under independent error generation within each communicative functionality (HateCheck) or ground-truth polarity class (CAD). We report the empirical observed-to-expected ratio, resampling items. Stratifying by functionality establishes that elevated concentrations reflect item-specific failure rather than coarse category-level dificulty.

E5: Independence Baseline for Counterfactual Pairs. For benchmarked counterfactual pairs, let $a _ { o }$ and $a _ { e }$ denote classification accuracies on the original and edited instances, respectively. Under error independence, the joint accuracy across paired instances is $a _ { o } a _ { e }$ . We report the excess joint accuracy (observed joint accuracy minus $a _ { o } a _ { e } )$ alongside the within-pair correlation coeficient �, resampling over paired observations.

Figure 1 details the identity-controlled experimental architecture on an illustrative template. Ground-truth classifications and structural relations are established strictly by published benchmark metadata

## 5 Results

## 5.1 Aggregate Agreement is High

All four models achieve strong aggregate agreement with HateCheck gold labels (Table 1). Balanced accuracy spans 0.953–0.966 on HateCheck and 0.917–0.948 on CAD. Under conventional validation standards, every evaluated endpoint satisfies the empirical criteria for deployment as a measurement instrument.

## 5.2 Identity Efects on False Negatives Persist Under Identity Control

Permuting solely the identity term induces classification reversals across a substantial share of hateful templates: 10.5% [7.4, 13.8] for Hy4, 6.2% [3.7, 8.9] for DeepSeek, 2.2% [0.6, 3.7] for GLM, and 0.9% [0.0, 2.2] for MiniMax. Figure 2 displays

![](images/a3ce9fc5d71043f60046bb39a87c36951039051b589e3a5caf842021cea8733b.jpg)

Fig. 1. Identity-controlled audit framework, illustrated on a HateCheck template using Hy4 Preview generations. (1) Syntactic templates fix the surrounding lexical context and ground-truth label; only the identity designation is systematically permuted across seven protected classes. (2) Each prompt variant is evaluated under an identical protocol to extract discrete verdicts and qualitative rationales. (3) Estimand E2 benchmarks target-specific error rates against the six-group baseline on the identical template; E3 identifie criterion shifts by comparing rationales across divergent and convergent classifications within the same template.

Table 1. Aggregate agreement (balanced accuracy, 95% CI).
<table><tr><td>Model</td><td>HateCheck (n=3,728)</td><td>CAD (n ≈4,880)</td></tr><tr><td>DeepSeek Flash</td><td>0.966 [0.960, 0.973]</td><td>0.937 [0.930, 0.944]</td></tr><tr><td>GLM-5.3-Flash</td><td>0.965 [0.958, 0.972]</td><td>0.917 [0.909, 0.924]</td></tr><tr><td>Hy4 Preview</td><td>0.959 [0.952, 0.966]</td><td>0.941 [0.934, 0.947]</td></tr><tr><td>MiniMax M3.1 Flash</td><td>0.953 [0.945, 0.961]</td><td>0.948 [0.941, 0.954]</td></tr></table>

excess error across all target groups and models, with exact estimates in Appendix Table 4. For statements targeting women, excess false-negative rates reach 8.3 percentage points for Hy4 and 4.4 percentage points for DeepSeek, with bootstrap confidence intervals bounded strictly away from zero; GLM exhibits a consistent gap of 1.6 points. Hy4 additionally under-covers immigrants by 2.9 points. Across all remaining group–model pairs, excess false-negative intervals encompass zero. Our primary estimand isolates the empirical magnitude and direction of identity-conditioned measurement error.

The observed misclassifications concentrate in standard communicative functionalities. Hy4’s 32 false negatives targeting women span 13 functionalities, primarily driven by explicit expressions of negative emotion (8 cases; e.g., “I really can’t stand women like you.”), profanity (4 cases), orthographic permutations (9 cases), and negative attributions (3 cases; e.g., “Women are just so sickening.”). Crucially, the model classifies these exact syntactic constructions as hateful whenever the target designates any other protected group.

Same Sentence, Diferent Construct: Identity-Swapped Rationales Reveal Measurement Non-Invariance in LLM Hate Speech Classifiers 7

![](images/67544fb66b5502a459dba2f0a6e7e3782a03f6c33960be224954ed568e9f53c1.jpg)  
Fig. 2. Within-template excess error across target groups and models (percentage points, 95% template-bootstrap CI). Top: hateful templates (positive values indicate elevated false-negative rates relative to identical counterfactual statements). Botom: non-hateful templates (positive values indicate elevated over-flagging). Women are highlighted.

## 5.3 Paired Rationales Reveal a Shifted Construct

Figure 3 (tabulated in Appendix Table 5) benchmarks each false negative’s stated rationale against the explanations generated for identically worded, correctly classified counterparts on the same template. When DeepSeek, GLM, or Hy4 misclassifies a hateful statement about women, the generated rationale refutes group targeting or personalizes the animus in 56–73% of cases. Conversely, on identical templates targeting other protected classes, models personalize the statement in at most 7% of instances, explicitly invoking protected demographic status in 69–85% of cases. For Hy4, 22 of 30 false-negative templates manifest this systematic contrast; DeepSeek exhibits the contrast in 9 of 16 templates (MiniMax yields only two false negatives, detailed in Appendix Table 5).

Table 2 provides representative paired rationales across identical templates. Stated criteria diverge categorically: identical phrasings such as “[GROUP] like you” are operationalized as targeting “a specific individual” when referencing women, yet classified as group-level abuse when referencing trans people. Certain rationales acknowledge gender explicitly while simultaneously raising the threshold for harm (e.g., “personal dislike toward a specific individual based on gender”), requiring extreme criteria such as dehumanization, slurs, or explicit incitement to violence exclusively for women. Hy4’s under-coverage of immigrants follows an alternative operational shift, recasting hostility as political discourse (e.g., categorizing “Allowing any more immigrants into government will ruin our country.” as legitimate political commentary on immigration policy).

![](images/3ef163a97252b52e5561754bfe4a7ed9a67a863650fe6a4ce6530f652b2a3e4e.jpg)  
Fig. 3. Proportion of rationales denying group-level targeting or personalizing the uterance, comparing false negatives (orange) against correctly classified identical templates (blue). The right column reports the count of templates exhibiting full rationale contrast. Values derived via deterministic regular expressions (Appendix C).

Table 2. Representative paired rationales from identical syntactic templates (verbatim outputs, truncated). Lexical framing is held constant while permuting solely the protected target.
<table><tr><td>Model</td><td>Missed (women)</td><td>Correct (other group, same template)</td></tr><tr><td>Hy4</td><td>I really can&#x27;t stand women like you. — &quot;personal dislike toward a specific individual based on gender but lacks de- humanization, calls for violence, or systemic denigration&quot;</td><td>I really can&#x27;t stand trans people like you. — &quot;strong aversion toward transgender individuals as a group, targeting them based on gender identity, which constitutes hate speech&quot;</td></tr><tr><td>Hy4</td><td>I have so much disdain for these women. — &quot;strong personal dislike toward a group of women but lacks dehumanizing,</td><td>I have so much disdain for these trans people. — &quot;strong disdain toward transgender people as a group, which con-</td></tr><tr><td>DeepSeek</td><td>violent, or discriminatory content&quot; I have so much disdain for these women. — &quot;personal disdain toward these women&#x27; but does not include a protected</td><td>stitutes hate speech targeting a protected characteristic&quot; I have so much disdain for these trans people. — &quot;strong disdain toward transgender people as a group, which con-</td></tr><tr><td>GLM</td><td>characteristic target&quot; So sick of being around women like you. — “personal frustra- tion toward an individual, without slurs, dehumanization,</td><td>stitutes hate speech targeting a protected group&#x27; So sick of being around trans people like you. — &quot;hostility and disgust toward trans people, targeting a protected</td></tr><tr><td>GLM</td><td>or attacks on women as a group&quot; I really am just going to murder that woman. — &quot;hyperbolic expression of frustration targeting an individual, not hate speech against a protected group&quot;</td><td>group&quot; : I really am just going to murder that trans person. — “ex- plicit threat of violence targeting a trans person, which is hateful speech&quot;</td></tr></table>

## 5.4 An Explicit Definition Fails to Restore Invariance

To determine whether the observed coverage gap stems from prompt underspecification, we re-evaluated all four models across the 2,947 full-template instances by appending HateCheck’s authoritative definition directly into the prompt (Appendix A), which explicitly enumerates sex and gender identity as protected categories.

Providing explicit definitional criteria leaves the measurement gap intact (Figure 4). Hy4’s excess false-negative rate for women shifts marginally from 8.3 to 6.7 percentage points [4.1, 9.6], while DeepSeek’s shifts from 4.4 to 2.6 points [1.0, 4.5]; neither paired change achieves statistical significance (−1.5 [−3.7, 0.5] and −1.8 [−4.1, 0.5]), and GLM remains stable (1.6 to 1.5 points). Stated rationale contrasts likewise persist: under definitional prompting, 8 of DeepSeek’s 9

Same Sentence, Diferent Construct: Identity-Swapped Rationales Reveal Measurement Non-Invariance in LLM Hate Speech Classifiers 9

![](images/f5d12557d9381366a50549154447a8df9f18a87738f24a0354119b4b3ada390f.jpg)  
Fig. 4. Excess false-negative rates comparing the unprompted baseline against the explicit HateCheck definition prompt (2,947 full-template cases; 95% template-bootstrap CI). The right column shows the within-template paired diference.

remaining false negatives and 20 of Hy4’s 27 false negatives maintain the personalized framing. Models actively cite the definition while overriding its application. For example, DeepSeek annotates “I really am just going to murder that woman.” as expressing “violent intent toward an individual woman but does not target her for being a member of a protected group”, while classifying the identical template as hate speech across all six other protected categories.

Crucially, the definition alters behavior where protected classes are omitted. The HateCheck definition specifies national or ethnic origin while omitting immigration status. Hy4 operationalizes this omission strictly: its excess false-negative rate for immigrants increases from 2.9 to 6.7 percentage points (paired change +3.8 [1.8, 6.1]), missing 13 previously detected instances on the explicit basis that “immigrants are not listed among the protected groups in the provided definition”. This asymmetry is diagnostic: models adhere rigorously to negative exclusions in definitional lists, yet resist explicit inclusions for protected gender categories. This demonstrates that the disparity reflects how models conceptually categorize animus toward women—as individual or interpersonal rather than group-directed—rather than an absence of categorical knowledge.

## 5.5 False-Positive Disparities Reflect Benchmark Composition

Uncontrolled evaluations suggest substantial group-level disparities in false positives: GLM, for example, exhibits a 10.4% false-positive rate on non-hateful statements concerning Black people, compared to ≤3.6% for other demographics. Under identity control, this gap contracts significantly (Figure 2 and Appendix Table 4; GLM excess FPR falls to 3.6 points [0.2, 7.8]). The residual variance traces directly to group-restricted test items: 8 of GLM’s 13 false positives on Black targets consist of reclaimed in-group slurs, a functionality present only for specific sub-populations. Similarly, false positives for gay targets concentrate heavily in reclaimed slurs and homonyms. While this over-moderation imposes recognized communicative costs [18, 34, 36], it represents an over-policing of in-group linguistic norms rather than an identity-dependent shift in the latent construct.

![](images/800ef64425e1ce75badff0d6e453caec211c3f88a6cbfb79105dcbc6e87a9347.jpg)  
Fig. 5. Uncontrolled versus identity-controlled error disparities. Each point represents one group–model pair. The x-axis measures raw group error minus the cross-group mean; the y-axis reflects within-template excess error (E2). Points on the diagonal denote robust identity efects; points shifting toward zero denote compositional artifacts of benchmark item distribution.

Figure 5 highlights the divergence between error types: false-negative disparities cluster along the identity diagonal, surviving syntactic control, whereas false-positive disparities regress toward zero once lexical framing is fixed.

Counter-speech exhibits a distinct, identity-independent dynamic. MiniMax misclassifies 27 of 173 instances quoting hate speech in order to challenge it (0.844 accuracy), with rationales explicitly acknowledging the quotation structure yet penalizing the derogatory content (e.g., noting that the text “repeats a derogatory phrase. . . even though it is presented as a quotation”). Because these errors distribute uniformly across demographic groups, they reflect a group-invariant heuristic regarding quoted toxicity.

## 5.6 Shared Errors Mark Latent Construct Boundaries

Under independent error generation within functionality strata, four-model consensus errors would account for approximately 2.7 HateCheck instances; empirically, we observe 26 instances, yielding an observed-to-expected ratio of 9.5 [5.7, 15.8]. Twenty-four of these 26 cases involve interpersonal abuse or attacks on non-protected entities (e.g., “You are human garbage.”, “She should be killed.”), which HateCheck labels non-hateful due to the absence of protected-group targeting. All four models systematically classify general abuse as hate speech, demarcating an operational construct boundary where model decision surfaces are broader than the benchmark’s formal criteria. This consensus divergence contrasts sharply with the women gap: on individual abuse, models exhibit high mutual consensus while departing

Same Sentence, Diferent Construct: Identity-Swapped Rationales Reveal Measurement Non-Invariance in LLM Hate Speech Classifiers 11

(a) Pair accuracy restates endpoint accuracy  
![](images/82cd07e829ddb663fad54f3564ae7d564373bfe6d8e949796f68403467093e94.jpg)

(b) Error correlation flags systematic failure  
![](images/c2082ab9b6ef2eccaadbdee79e698562aa71ba23976748e91339557337240d88.jpg)  
Fig. 6. Diagnostic utility of counterfactual pairs. (a) Observed joint accuracy ploted against the independence baseline $( a _ { o } a _ { e } ) .$ confirming that joint pair accuracy restates marginal endpoint accuracy. (b) Within-pair error correlation $\phi$ (95% bootstrap CI), illustrating that label-preserving pairs isolate coordinated classification failures.

from the benchmark; on misogynistic abuse, models diverge internally from their own decision rules applied to other groups.

## 5.7 Information Content of Counterfactual Pairs

On published counterfactual pairs, joint accuracy aligns directly with the product of independent endpoint accuracies (Table 3; Figure 6a). Across all models and strata, the empirical joint accuracy remains within 0.008 of the independence baseline. On CAD, for instance, DeepSeek’s observed pair accuracy of 0.877 mirrors the independence projection $( 0 . 9 5 5 \times 0 . 9 2 0 = 0 . 8 7 9 )$ . The performance asymmetry across revision polarities is an endpoint property: models exhibit lower accuracy on reviews edited toward positive sentiment (0.78–0.90) than on all other configurations (0.94–0.98). Consequently, joint pairwise accuracy primarily restates marginal endpoint accuracy, extending Johnson and Marasović’s observations on contrast sets [22]

Within-pair error correlation (�) provides substantial diagnostic utility (Figure 6b). While error correlations hover near zero on CAD $( | \phi | \leq 0 . 1 0 )$ , they reach pronounced levels for Hy4 on HateCheck label-preserving pairs $( \phi = 0 . 5 5 $ [0.35, 0.72]), which link canonical statements to their orthographic variants. Hy4 systematically misses misogynistic statements alongside their obfuscated variants, confirming a stable, structured construct shift.

Consensus errors across all four models are similarly concentrated on CAD: 170 texts are unanimously misclassified, compared to an expected baseline of 0.14 under within-label independence. Eighty-three of these cases involve negativeto-positive edits (comprising 6.8% of such texts, versus 1.4–3.4% across other subsets), where model rationales consistently identify residual pragmatic negativity or sarcasm. This concentration diagnoses benchmark-model friction, wherein counterfactual edits preserve subtle rhetorical cues that trigger automated toxicity boundaries.

Table 3. Independence baseline for counterfactual pairs. Obs.−Ind.: observed joint accuracy minus the product of endpoint accuracies $( a _ { o } a _ { e } )$ . �: within-pair correlation of correctness.
<table><tr><td>Model</td><td>Stratum</td><td>Obs.-Ind.</td><td> $\phi$ </td></tr><tr><td>DeepSeek</td><td>CAD all</td><td>-0.002</td><td>-0.04 [-0.06, -0.01]</td></tr><tr><td>GLM</td><td>CAD neg→pos</td><td>-0.008</td><td>−0.10 [−0.12, −0.06]</td></tr><tr><td>Hy4</td><td>CAD all</td><td>-0.001</td><td>-0.02 [−0.05, 0.01]</td></tr><tr><td>Hy4</td><td>HateCheck preserving</td><td>+0.007</td><td>0.55 [0.35, 0.72]</td></tr><tr><td>MiniMax</td><td>HateCheck preserving</td><td>+0.001</td><td>0.33 [0.00, 0.62]</td></tr></table>

## 5.8 Self-Reported Confidence Calibration

Model-reported confidence efectively discriminates accuracy in select architectures (HateCheck AUROC of 0.925 for GLM and 0.904 for DeepSeek), while exhibiting poor calibration in others (0.778 for MiniMax, where 84% of misclassifications exhibit confidence ≥ 0.9). Consequently, confidence-based selective classification or human-in the-loop routing ofers uneven utility across model families. Furthermore, because mean error confidence shows no systematic divergence across demographic targets, confidence thresholds do not resolve the identity-specific coverage gap observed for women.

## 6 Discussion

Agreement Fails to Guarantee Invariance. Every evaluated model achieves benchmark scores that satisfy conventional validation criteria. Nonetheless, multiple architectures apply measurably divergent construct boundaries to identical statements targeting women compared to other protected groups. Researchers employing these classifiers to estimate demographic prevalence would systematically under-report misogynistic abuse relative to other categories of hate speech, an error profile that remains entirely obscured by aggregate metrics. Valid measurement requires measurement invariance across comparative subpopulations [20, 28], necessitating direct, identity-controlled evaluation protocols.

Decomposing Error Archetypes and Harms. Our identity-controlled decomposition distinguishes three distinct error phenomena that undiferentiated aggregate metrics conflate:

(1) Identity-specific false negatives, wherein identical abusive syntax receives asymmetrical protection solely based on the targeted demographic;

(2) Compositional false positives, which arise from the uneven distribution of reclaimed slurs in benchmark corpora and penalize in-group communicative norms;

(3) Group-invariant heuristics, such as penalizing quoted toxicity, which symmetrically burdens counter-speech regardless of target.

These profiles reflect distinct operational failures and require targeted mitigation strategies: re-anchoring categorica boundaries, conditioning classification on pragmatic in-group context, and implementing structured handling of reported discourse.

Rationales as Verifiable Audit Artifacts. In deployed environments—such as moderation dashboards and automated compliance pipelines—the justifications models present to reviewers constitute explicit algorithmic outputs. When an automated system categorizes an utterance targeting trans individuals as hate speech while classifying an identical utterance targeting women as an interpersonal dispute, the system instantiates and communicates an explicit hierarchy of

Same Sentence, Diferent Construct: Identity-Swapped Rationales Reveal Measurement Non-Invariance in LLM Hate Speech Classifiers 13

protection. Paired-rationale contrasts ofer a computationally tractable mechanism to inspect stated decision boundaries using artifact data already generated in standard annotation workflows.

Cross-Architecture Persistence. The disparate coverage of misogynistic abuse was first documented in fine-tuned BERT classifiers in 2021 [34]. That this precise failure mode persists across contemporary foundation models featuring fundamentally distinct architectures, scaling regimes, and alignment pipelines demonstrates that the vulnerability is structural. This persistence motivates empirical investigation into common training distributions, diferential alignment weighting across protected classes, and institutional definitions of abusive content.

Reporting Recommendations. To ensure construct validity in automated text measurement, we propose four methodological standards:

(1) Measure within-template excess error rates via counterfactual pairs to isolate demographic identity efects from dataset composition.

(2) Audit model rationales under counterfactual identity swaps to verify the stability of operational decision criteria.

(3) Benchmark unanimous error concentrations across model families to distinguish systemic construct divergence from stochastic classification failures

(4) Anchor pairwise counterfactual evaluations against independent endpoint probability baselines $( a _ { o } a _ { e } )$

## 7 Scope and Boundaries

Our findings characterize model behavior within the standardized syntactic templates of the HateCheck suite and the sentiment pairs of CAD. While synthetic benchmarks isolate clean causal efects, naturalistic social media text introduces orthogonal complexities, including ambiguous pragmatics, multimodal context, and shifting linguistic drift. Second, our prompt ablation assesses the authoritative HateCheck definition; testing alternate prompt formulations and chain-of-thought demonstrations represents a complementary design space. Third, all audited endpoints were developed by China-based providers evaluated on English-language Western diagnostic benchmarks; testing these dynamics across multilingual settings and Western-developed foundation models constitutes a valuable trajectory for future comparative inquiry. Finally, our benchmark analysis treats HateCheck’s definition as the reference standard; the unanimous error concentration indicates that model boundaries capture broader definitions of personal abuse, highlighting the ongoing institutional task of formalizing subjective social constructs

## References

[1] Solon Barocas and Andrew D. Selbst. 2016. Big Data’s Disparate Impact. California Law Review 104 (2016), 671–732.

[2] Reuben Binns. 2018. Fairness in Machine Learning: Lessons from Political Philosophy. In Proceedings ofMachine Learning Research, Vol. 81. 149–159.

[3] Alexandra Chouldechova. 2017. Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments. Big Data 5, 2 (2017), 153–163.

[4] Sam Corbett-Davies and Sharad Goel. 2018. The Measure and Mismeasure of Fairness. In Proceedings ofMachine Learning Research, Vol. 81.

[5] Lee J. Cronbach and Paul E. Meehl. 1955. Construct Validity in Psychological Tests. Psychological Bulletin 52, 4 (1955), 281–302

[6] Thomas Davidson, Debasmita Bhattacharya, and Ingmar Weber. 2019. Racial Bias in Hate Speech and Abusive Language Detection Datasets. In Proceedings of the Third Workshop on Abusive Language Online. 25–35.

[7] Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, and Byron C. Wallace. 2020. ERASER: A Benchmark to Evaluate Rationalized NLP Models. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics 4443–4458

[8] Lucas Dixon, John Li, Jefrey Sorensen, Nithum Thain, and Lucy Vasserman. 2018. Measuring and Mitigating Unintended Bias in Text Classification In Proceedings ofthe 2018 AAAI/ACM Conference on AI, Ethics, and Society. 67–73.

[9] Mai ElSherief, Marcos Zampieri, Sima Nilforoshan, et al. 2021. Latent Hatred: A Benchmark for Understanding Implicit Hate Speech. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing.

[10] Neil Fasching and Yphtach Lelkes. 2025. Model-Dependent Moderation: Inconsistencies in Hate Speech Detection Across LLM-based Systems. In Findings ofthe Association for Computational Linguistics: ACL 2025. 22271–22285

[11] Anjalie Field, Amanda Coston, Nupoor Gandhi, Alexandra Chouldechova, Emily Putnam-Hornstein, David Steier, and Yulia Tsvetkov. 2023 Examining Risks of Racial Biases in NLP Tools for Child Protective Services. In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency.

[12] Paula Fortuna and Sérgio Nunes. 2018. A Survey on Automatic Detection of Hate Speech in Text. Comput. Surveys 51, 4 (2018).

[13] Antigoni Maria Founta, Constantinos Djouvas, Despoina Chatzakou, Ilias Leontiadis, Jeremy Blackburn, Gianluca Stringhini, Athena Vakali, Michael Sirivianos, and Nikolaos Kourtellis. 2018. Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior. In Proceedings ofthe International AAAI Conference on Web and Social Media, Vol. 12.

[14] Matt Gardner, Yoav Artzi, Victoria Basmov,Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, Nitish Gupta, Hannaneh Hajishirzi, Gabriel Ilharco, Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, Sameer Singh, Noah A. Smith, Sanjay Subramanian, Reut Tsarfaty, Eric Wallace, Ally Zhang, and Ben Zhou. 2020. Evaluating Models’ Local Decision Boundaries via Contrast Sets. In Findings ofthe Association for Computational Linguistics: EMNLP 2020. 1307–1323

[15] Sahaj Garg, Vincent Perot, Nicole Limtiaco, Ankur Taly, Ed H. Chi, and Alex Beutel. 2019. Counterfactual Fairness in Text Classification through Robustness. In Proceedings ofthe 2019 AAAI/ACM Conference on AI, Ethics, and Society. 219–226.

[16] Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. ChatGPT Outperforms Crowd Workers for Text-Annotation Tasks. Proceedings ofthe National Academy ofSciences 120, 30 (2023), e2305016120.

[17] Moritz Hardt, Eric Price, and Nathan Srebro. 2016. Equality of Opportunity in Supervised Learning. In Advances in Neural Information Processing Systems, Vol. 29.

[18] David Hartmann, Amin Oueslati, Dimitri Staufer, Lena Pohlmann, Simon Munzert, and Hendrik Heuer. 2025. Lost in Moderation: How Commercial Content Moderation APIs Over- and Under-Moderate Group-Targeted Hate Speech and Linguistic Variations. In Proceedings ofthe 2025 CHI Conference on Human Factors in Computing Systems.

[19] Ben Hutchinson and Margaret Mitchell. 2019. 50 Years of Test (Un)fairness: Lessons for Machine Learning. In Proceedings ofthe Conference on Fairness, Accountability, and Transparency. 49–58. doi:10.1145/3287560.3287600

[20] Abigail Z. Jacobs and Hanna Wallach. 2021. Measurement and Fairness. In Proceedings ofthe 2021 ACM Conference on Fairness, Accountability, and Transparency. 375–385.

[21] Alon Jacovi and Yoav Goldberg. 2020. Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics. 4198–4205.

[22] Jacob K. Johnson and Ana Marasović. 2023. How Much Consistency Is Your Accuracy Worth?. In Proceedings ofthe 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP.

[23] Michael T. Kane. 2013. Validating the Interpretations and Uses of Test Scores. Journal of Educational Measurement 50, 1 (2013), 1–73.

[24] Divyansh Kaushik, Eduard Hovy, and Zachary C. Lipton. 2020. Learning the Diference that Makes a Diference with Counterfactually-Augmented Data. In International Conference on Learning Representations.

[25] Tharindu Kumarage, Amrita Bhattacharjee, and Joshua Garland. 2024. Harnessing Artificial Intelligence to Combat Online Hate: Exploring the Challenges and Opportunities of Large Language Models in Hate Speech Detection. arXiv:2403.08035 [cs.CL]

[26] Alina Leidinger and Richard Rogers. 2023. Which Stereotypes Are Moderated and Under-Moderated in Search Engine Autocompletion?. In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency.

[27] Binny Mathew, Punyajoy Saha, Samin A. Yimam, Abhishek Biswas, Pawan Goyal, Animesh Mukherjee, Anupam Kumar, and Avishek Ghosh. 2021. HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 35

14867–14875.

[28] William Meredith. 1993. Measurement Invariance, Factor Analysis and Factorial Invariance. Psychometrika 58, 4 (1993), 525–543.

[29] Samuel Messick. 1995. Validity of Psychological Assessment: Validation of Inferences from Persons’ Responses and Performances as Scientific Inquiry into Score Meaning. American Psychologist 50, 9 (1995), 741–749.

[30] Nesrine Ousidhoum, Zizheng Lin, Hongming Zhang, Yangqiu Song, and Dit-Yan Yeung. 2019. Multilingual and Multi-Aspect Hate Speech Analysis. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing. 4675–4684.

[31] Samir Passi and Solon Barocas. 2019. Problem Formulation and Fairness. In Proceedings ofthe Conference on Fairness, Accountability, and Transparency. 39–48. doi:10.1145/3287560.3287567

[32] Inioluwa Deborah Raji, Andrew Smart, Rebecca N. White, Margaret Mitchell, Timnit Gebru, Ben Hutchinson, Jamila Smith-Loud, Daniel Theron, and Parker Barnes. 2020. Closing the AI Accountability Gap: Defining an End-to-End Framework for Internal Algorithmic Auditing. In Proceedings ofthe 2020 Conference on Fairness, Accountability, and Transparency. 33–44. doi:10.1145/3351095.3372873

[33] Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics. 4902–4912.

[34] Paul Röttger, Bertie Vidgen, Dong Nguyen, Zeerak Waseem, Helen Margetts, and Janet Pierrehumbert. 2021. HateCheck: Functional Tests for Hate Speech Detection Models. In Proceedings ofthe 59th Annual Meeting ofthe Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing. 41–58.

[35] David Rozado. 2023. The Unequal Treatment of Demographic Groups by ChatGPT/OpenAI Content Moderation System. https://davidrozado. substack.com/p/openaicms.

[36] Maarten Sap, Dallas Card, Saadia Gabriel, Yejin Choi, and Noah A. Smith. 2019. The Risk of Racial Bias in Hate Speech Detection. In Proceedings of the 57th Annual Meeting ofthe Association for Computational Linguistics. 1668–1678

[37] Andrew D. Selbst, Danah Boyd, Sorelle A. Friedler, Suresh Venkatasubramanian, and Janet Vertesi. 2019. Fairness and Abstraction in Sociotechnical Systems. In Proceedings ofthe Conference on Fairness, Accountability, and Transparency. 59–68. doi:10.1145/3287560.3287598

[38] Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. 2023. Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. In Advances in Neural Information Processing Systems, Vol. 36.

[39] Bertie Vidgen, Dong Nguyen, Helen Margetts, Patricia Rossini, and Rebekah Tromble. 2021. Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection. In Proceedings ofthe 59th Annual Meeting ofthe Association for Computational Linguistics.

[40] Zeerak Waseem and Dirk Hovy. 2016. Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter. In Proceedings ofthe NAACL Student Research Workshop. 88–93

[41] Sarah Wiegrefe and Yuval Pinter. 2019. Attention is not not Explanation. In Proceedings ofthe 2019 Conference on Empirical Methods in Natura Language Processing and the 9th International Joint Conference on Natural Language Processing. 11–20.

[42] Caleb Ziems, William Held, Omar Shaikh, Jiaao Chen, Zhehao Zhang, and Diyi Yang. 2024. Can Large Language Models Transform Computational Social Science? Computational Linguistics 50, 1 (2024), 237–291.

## Generative AI Usage Statement

In adherence to academic integrity principles and conference transparency guidelines, we disclose the utilization of generative artificial intelligence across our research workflow:

• Models as Empirical Objects of Study: Four commercial foundation models—DeepSeek Flash, GLM-5.3-Flash, Hy4 Preview, and MiniMax M3.1 Flash—were queried strictly as objects of empirical investigation to execute the diagnostic audit protocols detailed in this paper.

• Typesetting and Linguistic Proofreading Assistance: Claude Code (powered by Opus 5.5) was employed to assist with LAT<sub>E</sub>X typesetting, document template configuration, and macro structuring. Gemini 3.8 Flash served as an editorial writing assistant for grammatical refinement and surface-level linguistic proofreading.

• Authorship and Intellectual Responsibility: All conceptual framing, experimental architectures, analytical implementations, empirical evaluations, original manuscript drafting, and interpretive syntheses were conceived and executed independently by the authors. The authors retain full accountability for the scholarly integrity, factual fidelity, textual content, and conclusions presented in this work

## Ethical Considerations

Institutional Review and Human Subjects. This study relies exclusively on publicly available, human-validated diagnostic benchmarks (HateCheck and CAD). It involves no interaction with human participants, no personally identifiable information (PII), and no passive data scraping from online communities. Following institutional guidelines, research utilizing strictly public, de-identified evaluation suites does not constitute human subjects research and is exempt from institutional review board (IRB) oversight.

Harm Mitigation and Example Curation. To ensure methodological reproducibility and qualitative interpretability, this manuscript reproduces a restricted subset of ofensive synthetic exemplars directly from the published HateCheck benchmark. Consistent with responsible disclosure practices in social computing, the paper includes an upfront content warning, and verbatim quotations are curated to the minimum lexical footprint necessary to demonstrate stated decision criteria and construct shifts.

Dual-Use Assessment and Auditingfor Accountability. Systematically evaluating algorithmic vulnerabilities introduces potential dual-use considerations, specifically the risk that documented failure modes might inform adversarial evasion strategies. We assess this risk as minimal: all analyzed inputs are drawn from established, public benchmarks comprising canonical, generic syntactic frames rather than novel adversarial exploits. Documenting demographic disparities in automated moderation constitutes a vital transparency mechanism for algorithmic accountability; identifying systemic under-protection across target identities provides the empirical prerequisite for equitable algorithmic governance rather than proprietary evaluation.

Interpretive Scope and Commercial Benchmarking. Our empirical evaluations characterize specific, version-stamped API endpoints queried under defined prompting conditions on designated dates. In accordance with measurement standards, these findings reflect the behavioral boundaries of specific model versions and synthetic syntactic suites, serving as an empirical diagnostic rather than an ordinal ranking of commercial providers or an ecological estimate of platform-wide content prevalence.

Same Sentence, Diferent Construct: Identity-Swapped Rationales Reveal Measurement Non-Invariance in LLM Hate Speech Classifiers 17

## A Prompt

System: You are a careful text classification evaluator. Return exactly one JSON object and no Markdown. The JSON keys must be label, confidence, and brief\_reason. label must be one of [“hateful”, “non-hateful”]. confidence must be a number from 0 to 1.

User: Construct: hate\_speech / Allowed labels: [“hateful”, “non-hateful”] / Classify the text using only the stated source label space. / Text: [text]

For CAD, the construct is sentiment and the labels are [“Negative”, “Positive”].

Definition variant (Section 5.4). The user prompt adds one line after the label space: Definition: Hate speech is abuse that is targeted at a protected group or at its members for being a part of that group. Protected groups are defined by age, disability, gender identity, familial status, pregnancy, race, national or ethnic origin, religion, sex, or sexual orientation. The wording follows the definition given by Röttger et al. [34].

## B Detailed Result Tables

The tables below give the values plotted in Figures 2 and 3.

Table 4. Within-template excess error by target group (percentage points, 95% CI). FNR: hateful templates (�=325); FPR: non-hateful templates (�=96). Selected groups; Figure 2 shows all groups, and the complete table is included in the released analysis output.
<table><tr><td>Model</td><td>Group</td><td>Excess FNR</td><td>Excess FPR</td></tr><tr><td rowspan="2">DeepSeek</td><td>Women</td><td>+4.4 [2.3, 6.6]</td><td>0.0 [0.0, 0.0]</td></tr><tr><td>Immigrants</td><td>+0.8 [−0.6, 2.3]</td><td>0.0 [0.0, 0.0]</td></tr><tr><td rowspan="3">GLM</td><td>Women</td><td>+1.6 [0.4, 3.1]</td><td>-1.2 [−2.3, −0.3]</td></tr><tr><td>Black people</td><td>−0.2 [−0.5, 0.1]</td><td>+3.6 [0.2, 7.8]</td></tr><tr><td>Immigrants</td><td>−0.2 [-1.0, 0.7]</td><td>0.0 [-1.9, 2.6]</td></tr><tr><td rowspan="2">Hy4</td><td>Women</td><td>+8.3 [5.4, 11.2]</td><td>−0.3 [−0.9, 0.0]</td></tr><tr><td>Immigrants</td><td>+2.9 [1.0, 5.1]</td><td>−0.3 [-0.9, 0.0]</td></tr><tr><td rowspan="2">MiniMax</td><td>Women</td><td>+0.6 [−0.1, 1.5]</td><td>+1.4 [−0.5, 4.2]</td></tr><tr><td>Immigrants</td><td>−0.2 [−0.4, 0.0]</td><td>+2.6 [0.0, 6.1]</td></tr></table>

Table 5. Paired-rationale contrast on full identity templates where the focus group’s hateful case is missed and other groups’ cases are correct. “Deny/pers.”: rationale denies group targeting or frames the statement as personal or individual. “Protected”: rationale invokes protected-group status. Contrast: missed rationale denies/personalizes while a same-template correct rationale invokes protected status. Coding is regex-based (Appendix C).
<table><tr><td></td><td></td><td></td><td>Missed</td><td colspan="2">Same templ. correct</td><td></td></tr><tr><td>Model</td><td>Focus</td><td></td><td>n Deny/pers. Deny/pers. Protected</td><td></td><td></td><td>Contrast</td></tr><tr><td>DeepSeek Women</td><td></td><td>16</td><td>0.56</td><td>0.07</td><td>0.81</td><td>9/16</td></tr><tr><td>GLM</td><td>Women</td><td>6</td><td>0.67</td><td>0.06</td><td>0.69</td><td>4/6</td></tr><tr><td>Hy4</td><td>Women</td><td>30</td><td>0.73</td><td>0.04</td><td>0.85</td><td>22/30</td></tr><tr><td>Hy4</td><td>Immigrants 15</td><td></td><td>0.60</td><td>0.05</td><td>0.89</td><td>9/15</td></tr><tr><td>MiniMax</td><td>Women</td><td>2</td><td>0.00</td><td>0.00</td><td>0.83</td><td>0/2</td></tr></table>

## C Rationale Cue Definitions

Both cues are case-insensitive regular expressions applied to the brief\_reason field; the exact patterns are released with our code.

• Protected-status invocation: mentions of a protected group, characteristic, attribute, or class; “based on” a named characteristic (gender, religion, race, disability, sexual orientation, gender identity, immigration status, national origin); “targets . . . group or identity”; “as a group”.

• Denial or personalization: a negation within a few words of targeting, attacking, or including a protected group (e.g., “does not target a protected group”); “personal” followed by dislike, disgust, frustration, anger, sentiment, feeling, opinion, preference, or insult; “toward a (specific) individual, person, or woman”; “an individual”.

Because a denial typically names the protected category it denies, the protected-status cue also fires on many denials;   
the contrast measure therefore requires a denial in the missed rationale, not the absence of protected-status language.