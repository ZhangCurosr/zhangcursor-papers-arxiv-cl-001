# EviCast: Forecasting Retrieval Gains Through Shared Response Calibration

Yibo Wang, Jisheng Dang, Bimei Wang, Jiongpei Hou, Jingwen Zhao, Wenjun Ma, Fei Shen, Juncheng Li, Meng Wang, Fellow, IEEE, Qi Tian, Fellow, IEEE, Bin Hu, Fellow, IEEE, and Tat-Seng Chua

Abstract—Retrieval-augmented generation (RAG) extends large language models with external evidence, yet relevant evidence does not ensure that a generator can use it. Indiscriminate retrieval can degrade task performance while adding retrieval and long-context inference costs. Existing methods neither determine before large-scale execution whether a generator, task distribution, and retriever configuration is worth deploying nor diagnose multiple candidate retrievers under a shared budget. We introduce EviCast, a framework that forecasts retrieval value through interventional evidence-state decomposition. EviCast measures a generator’s response to controlled evidence states once and combines this shared profile with the offline evidencestate distribution of each candidate retriever, yielding the endto-end net utility and its 95% lower confidence bound without executing every RAG configuration. A state-refinement analysis characterizes when its binary and topology-aware instantiations are exact and when shared calibration is statistically more efficient than per-retriever calibration. In frozen evaluations, Evi-Cast correctly decides all 14 HotpotQA retrieval configurations with a forecast error of 2.75 F1 points while reducing diagnostic generation calls by 80%, and it attains the lowest forecast error among seven predictors in the low-budget regime. Its topologyaware instantiation, EviCast-T, captures the complementarity between terminal and bridge evidence, correctly deciding all 16 configurations on MuSiQue and HotpotQA and 23 of 24 after prospective transfer to a third dataset. These results show that retrieval deployment depends on both evidence availability and the generator’s ability to convert evidence structure into stable net utility. Code is available at https://github.com/YiboWang2002/ EviCast.

Index Terms—Retrieval-augmented generation, utility forecasting, evidence calibration, multihop question answering, retrieval deployment.

## I. INTRODUCTION

ARGE language models (LLMs) excel at knowledgecomplex reasoning. However, their parametric knowledge is slow to update, incomplete, and difficult to trace to source material. Retrieval-augmented generation (RAG) introduces updatable and attributable external knowledge at inference time. It has therefore become a fundamental interface between LLMs and real-world knowledge bases [1].

![](images/e551189653c4168ba9f66e3559197462bd280a42eaaca0d6516911801c8aebf0.jpg)  
Fig. 1. Retrieval utility and diagnosis under a fixed task distribution and generator.

RAG does not benefit every model or task. Its value depends on both retrieval relevance and the generator’s ability to use the retrieved evidence. Irrelevant context can distract LLMs and induce incorrect answers [2], [3]. Indiscriminate retrieval may therefore harm performance while adding retrieval and longcontext inference costs, whereas overly conservative retrieval may forfeit substantial gains from external evidence. Figure 1 illustrates this deployment dilemma. We ask whether the effect of external evidence on a target model can be predicted before deployment and used to enable RAG only when beneficial. We further ask whether one diagnosis can serve multiple candidate retrieval schemes. These questions define retrieval necessity and its associated resource-efficiency problem.

Existing work studies retrieval decisions and utility at three levels. Forward-Looking Active Retrieval augmented generation (FLARE), Self-Reflective Retrieval-Augmented Generation (Self-RAG), and Adaptive-RAG trigger retrieval for an individual query or generation step [4]–[6]. Retrieval Augmented Generation Assessment (RAGAS) and RAGChecker evaluate RAG systems after execution [7], [8]. SePer, passage utility models, InfoGain-RAG, and retrieval performance prediction estimate the local utility of a query or document [9]–[12]. These methods do not determine before large-scale execution whether a complete generator, task distribution, and retriever configuration is worth deploying, nor do they diagnose multiple candidate retrievers under a shared generation budget. We formulate this problem as amortized retrieval necessity diagnosis at deployment time. Given a generator g, task distribution D, and candidate retriever r, the objective is to predict the expected gain $\Delta ( g , \mathcal { D } , r )$ over the no-retrieval baseline and make a deployment decision.

Diagnosing retrieval necessity presents two technical challenges. First, the evidence representation must reflect how the generator uses evidence because RAG utility depends jointly on retrieved evidence and the generator’s ability to exploit it. Relevance and correct-passage counts are insufficient. In multihop question answering, one top-2 result may contain a bridge passage and a distractor, while another contains a terminal answer passage and a distractor. Both contain one correct passage but may affect the final answer differently. Second, the generator response must transfer reliably across retrievers without executing every candidate RAG configuration, while accounting for uncertainty from small calibration samples. Directly comparing retrieval and no-retrieval performance for each retriever makes generation cost grow linearly with the number of candidates and yields variable small-sample gain estimates.

We address these challenges with EviCast, a retrieval necessity diagnosis framework based on interventional evidencestate calibration. EviCast decomposes candidate retriever utility into a generator response profile and a retriever-specific evidence-state distribution. Controlled evidence interventions estimate the response profile once on a small calibration set, while offline retriever outputs determine each state distribution. Their combination yields the expected net gain and its lower confidence bound (LCB). This decomposition reuses generator response calibration across retrievers, while topology-aware states capture complementary evidence roles in multihop reasoning.

Our frozen evaluation covers multiple RAG mechanisms, three multihop question-answering datasets, multimodal and text-only generators, and sparse, dense, and fusion retrievers. EviCast achieves 90% decision agreement across 10 task configurations without falsely enabling any configuration that lacks a significant gain. On HotpotQA, 96 shared generations yield 14 correct decisions across 14 retrieval configurations, a mean absolute error (MAE) of 2.75 F1 points, and an 80% reduction in diagnostic generation calls, and EviCast attains the lowest forecast error among seven predictors in the low-budget regime. Its topology-aware instantiation decides all 16 configurations on MuSiQue and HotpotQA correctly with an overall MAE of 2.81 points and 23 of 24 after prospective transfer to 2WikiMultiHopQA (2Wiki). Empirical checks further confirm each theoretical property. These results show that reusable calibration accurately forecasts retrieval value while substantially reducing the cost of diagnosing multiple configurations.

Our contributions are as follows:

• We formulate deployment-level, amortized retrieval necessity diagnosis to predict gains over the no-retrieval baseline across candidate retrievers under a shared calibration budget. Decisions combine evidence-state separability and the utility LCB.

• We develop EviCast, a framework that calibrates generator responses through controlled evidence interventions, maps offline retriever outputs into evidence-state distributions, and combines both components into expected retrieval utility. We instantiate it with binary and topologyaware evidence states, where the latter models nonlinear complementarity between terminal and bridge evidence, and we prove when each instantiation is exact through a state-refinement analysis.

• We validate EviCast across multiple RAG mechanisms, three multihop question-answering datasets, two generators, and sparse, dense, and fusion retrievers. EviCast achieves 14/14 correct decisions on HotpotQA retrieval configurations, 16/16 on MuSiQue and HotpotQA topology configurations, and 23/24 after prospective transfer to a third dataset, attains the lowest forecast error among seven predictors in the low-budget regime, and reduces diagnostic generation calls by up to 80%.

## II. RELATED WORK

## A. Retrieval-Augmented Generation And Evidence Use

Retrieval-augmented language models extend parametric knowledge with external memory. Early work established differentiable retrieval, dense passage retrieval, and joint retrieval and generation [1], [13], [14]. Later systems fuse multiple passages, retrieve from large corpora, or augment frozen models with in-context evidence [15]–[17]. Retrieval-Augmented Dual Instruction Tuning (RA-DIT) jointly adapts the generator and retriever, while RankRAG unifies context ranking with answer generation [18], [19]. Knowledge Intensive Language Tasks (KILT) provides a shared benchmark for knowledge-intensive tasks and provenance [20].

External knowledge also supports visual question answering. Wu et al. [21] integrate visual attributes and external knowledge, while FVQA [22] annotates questions with supporting facts. CRIC [23] studies compositional reasoning over visual and commonsense evidence and evaluates grounding alongside answer correctness. More recently, ReT-2 [24] develops universal multimodal retrieval and evaluates its use in retrieval-augmented visual question answering. These studies of knowledge integration, evidence grounding, and retrieval effectiveness complement EviCast’s focus on forecasting the gain of a candidate retrieval configuration for a fixed generator.

External evidence is not uniformly beneficial. Irrelevant passages can distract generators, motivating evidence filtering and robustness training [2], [3], [25]. Evidence position and document composition also alter model behavior [26], [27], while retrieval gains vary with the knowledge already stored by the model [28]. These findings motivate EviCast to model retrieval supply and generator response separately rather than equating relevance with end-to-end utility.

## B. Adaptive Retrieval And Selective Augmentation

Adaptive methods trigger retrieval from generation confidence, model self-knowledge, reflection, question complexity, or real-time information needs [4]–[6], [29], [30]. Selective augmentation instead compresses or filters retrieved context before generation [31], [32]. Unlike these query-level or steplevel methods, EviCast forecasts configuration-level deployment utility using one generator calibration shared across candidate retrievers.

Dynamic neural networks provide a broader perspective on allocating computation according to the input [33]. For example, Glance and Focus Networks [34] first inspect a coarse global view, then process selected image regions and stop when predictions are sufficiently confident. Such inference-time adaptation motivates efficient evidence use. EviCast makes a different resource decision before deployment: whether a retrieval configuration yields a reliable expected gain over the target task distribution.

## C. RAG Evaluation, Benchmarks, And Utility Estimation

RAGAS, the Automated RAG Evaluation System (ARES), and RAGChecker evaluate executed pipelines through context relevance, faithfulness, answer quality, and component-level diagnostics [7], [8], [35]. The Retrieval-Augmented Generation Benchmark (RGB), Automatic LLMs’ Citation Evaluation (ALCE), and RAGTruth assess robustness, information integration, citation quality, and hallucination [36]–[38]. These tools characterize RAG behavior after execution. Utility modeling is closest to our objective. Prior work estimates the value of a query or document through semantic perplexity reduction, learned passage utility, confidence change, and retrieval-quality features [9]–[12]. EviCast instead decomposes configuration-level net gain into interventional generator responses and retriever-specific evidence-state distributions, enabling amortized deployment diagnosis.

Confidence estimation also highlights the need to match reliability measures to the intended decision. Zhu et al. [39] show that improving confidence calibration need not improve failure prediction. In EviCast, calibration denotes measuring taskscore responses to controlled evidence states; the deployment criterion concerns a confidence bound on expected retrieval gain rather than the confidence of an individual answer.

## D. Multimodal Evidence And Reliable Fusion

Multimodal learning studies how heterogeneous sources can be represented, aligned, and fused [40], including crossmodal interactions implemented with Transformers [41]. Video models aggregate temporal evidence through spatio-temporal memory [42]–[47], decoupled segmentation tokens [48], and preference-optimized captioning [49], and omni-modal models ground affective reasoning in audiovisual and paralinguistic cues [50]–[52]. Trusted Multi-View Classification [53] models view-specific uncertainty through dynamic evidential fusion, underscoring the source reliability that also matters in our acoustic, visual, and language retrieval settings. EviCast evaluates a fixed generator and evidence interface: it measures how evidence conditions affect task utility and uses those responses to forecast retrieval gains, without learning a new fusion architecture.

## III. METHOD

EviCast forecasts end-to-end RAG utility before deployment. It measures a generator’s response to controlled evidence states, then estimates how often each candidate retriever produces these states in offline results. Combining the shared response profile with each retriever distribution yields an expected net gain, a confidence interval (CI), and a deployment decision. This decomposition avoids complete

RAG calibration for every retriever. Figure 2 shows the full data flow.

EviCast is a framework rather than a single estimator. It is specified by an evidence-state map, a set of controlled interventions that realize each state, and the shared estimation and decision rule of Sections III-C to III-E. We develop two instantiations that share every component except the state map. The binary instantiation, EviCast-B, labels each retrieved unit as supportive or misleading and suits mutually substitutable evidence such as retrieved cases, neighbor labels, or single supporting passages. The topology-aware instantiation, EviCast-T, distinguishes terminal and bridge roles and suits compositional evidence whose units contribute complementary information. Section III-G characterizes when each instantiation is exact.

## A. Problem Formulation

We define a fixed deployment environment as $\begin{array} { r l } { \mathcal { E } } & { { } = } \end{array}$ $( g , \mathcal { D } , \pi , \mathcal { K } , k )$ , where $g$ is the generator, D the target task distribution, π the prompt template, K the retrieval corpus, and k the number of returned passages. Given candidate retrievers $\mathcal { R } = \{ r _ { 1 } , . . . , r _ { J } \}$ , retriever $r _ { j }$ returns $E _ { j } ( x ) = r _ { j } ( x ; \mathcal { K } , k )$ Higher values of the task metric $M ( \hat { y } , y )$ indicate better predictions.

The true deployment utility of $r _ { j }$ relative to no retrieval is:

$$
\begin{array} { r } { \Delta _ { j } = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } } \left[ M ( g _ { \pi } ( x , E _ { j } ( x ) ) , y ) \right. } \\ { \left. - \ M ( g _ { \pi } ( x , \emptyset ) , y ) \right] . } \end{array}\tag{1}
$$

The ideal deployment action is:

$$
a _ { j } ^ { * } = \mathbf { 1 } [ \Delta _ { j } > 0 ] ,\tag{2}
$$

where one enables retrieval and zero retains no-retrieval inference.

Directly computing $\Delta _ { j }$ requires separate RAG generations for every retriever. EviCast instead uses a labeled calibration set $\mathcal { C } ~ = ~ \{ ( x _ { i } , y _ { i } , \mathcal { P } _ { i } , \lambda _ { i } ) \} _ { i = 1 } ^ { n }$ disjoint from the final test set. Here $\mathcal { P } _ { i }$ contains candidate evidence, and $\lambda _ { i }$ contains relevance or role annotations used only for calibration. EviCast returns $\widehat { \Delta } _ { j }$ , its CI, and $\widehat { \boldsymbol { a } } _ { j }$ . The generator, task distribution, prompt, corpus, retrieval depth, and evidence format remain fixed during diagnosis, while only the offline rankings vary. The objective is to recover the sign and magnitude of $\Delta _ { j }$ with generation cost independent of $J .$

## B. Binary And Topology-Aware Evidence States

EviCast uses an evidence state as its central intermediate variable rather than a relevance score or correct-passage count. For query x and evidence set $E _ { \mathrm { { : } } }$ , the state mapping is:

$$
s = \phi ( x , E ; \lambda ) \in \mathcal { S } ,\tag{3}
$$

where $s$ is a mutually exclusive and exhaustive state space. Each state should preserve the evidence properties that determine generator utility. Within task stratum $h ,$ the response should be approximately invariant to the retriever that produced the evidence. Let $\delta ( x , E ) ~ = ~ M ( g _ { \pi } ( x , E ) , y ) ~ - ~$ $M ( g _ { \pi } ( x , \emptyset ) , y )$ . The desired response-invariance condition is:

$$
\mathbb { E } [ \delta ( x , E ) \mid h , s , r _ { j } ] \approx \mathbb { E } [ \delta ( x , E ) \mid h , s ] .\tag{4}
$$

![](images/72bd07a08bacb4f236be0f0ee8598c2b319de63d2dcc4cced8c711b7b1309cb9.jpg)  
Fig. 2. EviCast framework for reusable retrieval utility forecasting and deployment.

Retrievers may change state probabilities but should not systematically change the response within a state. This condition allows one generator response calibration to be reused across retrievers.

Binary Instantiation. EviCast-B labels each of the k retrieved units as supportive or misleading. It intervenes only on the two endpoints, an all-supportive set and an all-misleading set, whose responses in stratum h are $G _ { h , + }$ and $G _ { h , - }$ . With $\rho _ { j , h }$ denoting the mean fraction of supportive units returned by retriever $r _ { j } .$ , its forecast is:

$$
\widehat { \Delta } _ { j } ^ { \mathrm { B } } = \sum _ { h \in \mathcal { H } } \widehat { w } _ { h } \left[ \widehat { \rho } _ { j , h } \widehat { G } _ { h , + } + ( 1 - \widehat { \rho } _ { j , h } ) \widehat { G } _ { h , - } \right] .\tag{5}
$$

This is the general forecast of Section III-E with the unit-level state space $\{ + , - \}$ , and it requires only 3n generator calls. Topology-Aware Instantiation. Multihop question answering requires evidence with distinct, complementary roles. Let $e ^ { \bar { A } }$ denote the terminal passage that directly contains answer information, and let B denote bridge passages that identify an intermediate entity or connect the reasoning chain. For a fixed top-2 evidence set E, the topology-aware state mapping is defined as follows:

$$
\phi ( E ) = \left\{ { \begin{array} { l l } { N , } & { e ^ { A } \notin E , } \\ { A , } & { e ^ { A } \in E , \ E \cap B = \emptyset , } \\ { A B , } & { e ^ { A } \in E , \ E \cap B \neq \emptyset . } \end{array} } \right.\tag{6}
$$

State N lacks terminal evidence, state A contains only terminal evidence, and state AB contains terminal evidence with at least one bridge passage. This topology distinguishes a bridge passage paired with a distractor from a terminal passage paired with a distractor, although both sets contain one annotated supporting passage.

For each calibration example, EviCast constructs lengthmatched interventions for all three states. The N intervention excludes terminal evidence, prioritizes a bridge passage, and fills the remaining position with a distractor. The A intervention contains terminal evidence and one distractor, while the AB intervention contains terminal and bridge evidence. A prespecified anchor retriever selects and orders candidates after their roles are fixed. Evidence states, task strata, and ordering rules are frozen before calibration generation.

## C. Reusable Interventional Response Calibration

Observed retrieval alone cannot distinguish missing evidence from a generator’s failure to use available evidence. EviCast therefore intervenes on evidence states to measure the generator response separately. For item i, the no-retrieval score, controlled score, and paired gain are:

$$
\begin{array} { c } { b _ { i } = M ( g _ { \pi } ( x _ { i } , \emptyset ) , y _ { i } ) , } \\ { q _ { i , s } = M ( g _ { \pi } ( x _ { i } , \widetilde { E } _ { i , s } ) , y _ { i } ) , } \\ { d _ { i , s } = q _ { i , s } - b _ { i } , } \end{array}\tag{7}
$$

where $\widetilde { E } _ { i , s }$ is the controlled evidence for state s. Pairing reduces question-difficulty variation.

Let ${ \mathcal { C } } _ { h } = \{ i : h _ { i } = h \}$ and $n _ { h } = | \mathcal { C } _ { h } |$ . The response of generator g in stratum h and state s is:

$$
\widehat { G } _ { g , h , s } = \frac { 1 } { n _ { h } } \sum _ { i \in \mathcal { C } _ { h } } d _ { i , s } .\tag{8}
$$

All retrievers share the baseline and controlled generations. Thus, $\widehat { G }$ is computed once for a fixed generator, prompt, and evidence format. Changing retrievers requires no new response generation.

## D. Retriever Evidence-State Characterization

Each candidate retriever runs offline on the shared calibration set. Let $E _ { i j } = r _ { j } ( x _ { i } ; K , k )$ and $s _ { i j } = \phi ( x _ { i } , E _ { i j } ; \lambda _ { i } )$ . The state probability for retriever $r _ { j }$ within stratum h is:

$$
\widehat { p } _ { j , h , s } = \frac { 1 } { n _ { h } } \sum _ { i \in \mathcal { C } _ { h } } \mathbf { 1 } [ s _ { i j } = s ] .\tag{9}
$$

This mapping places heterogeneous retrievers in a common state space and combines them with the shared response profile without additional generator calls.

EviCast also tests whether a retrieval representation separates task-relevant evidence from distractors. Let $A _ { j }$ be a separability statistic and $A _ { 0 }$ its uninformative baseline. The evidence separability margin is defined as follows:

$$
m _ { j } = A _ { j } - A _ { 0 } .\tag{10}
$$

For ranked question answering, $A _ { j }$ is the pairwise area under the receiver operating characteristic curve (AUC) for ranking supporting passages above distractors, and $A _ { 0 } ~ = ~ 0 . 5$ . For case retrieval, $A _ { j }$ is held-out prediction performance based on retrieval keys, and $A _ { 0 }$ is the majority baseline. This excess statistic provides a common separability gate across retrieval mechanisms.

## E. Utility Forecast And Deployment Decision

The response profile quantifies the value of each evidence state, while the retriever distribution gives its frequency. The law of total expectation yields:

$$
\begin{array} { l } { \displaystyle \widehat { \Delta } _ { j } = \sum _ { h \in \mathcal { H } } \widehat { w } _ { h } \sum _ { s \in \mathcal { S } } \widehat { p } _ { j , h , s } \widehat { G } _ { g , h , s } , } \\ { \displaystyle \widehat { w } _ { h } = \frac { n _ { h } } { n } . } \end{array}\tag{11}
$$

The utility forecast weights the generator response to each state by the probability that the retriever supplies that state. Under $\mathrm { E q . } \ ( 4 )$ , a new retriever requires only a new estimate of ${ \widehat { p } } ,$ not $\widehat { G }$ .

Both terms are uncertain when estimated from a small calibration set. EviCast propagates their joint uncertainty through a paired stratified bootstrap. Each replicate samples calibration items with replacement within h and recomputes the response profile, state probabilities, and utility. We use $B = 1 0 , 0 0 0$ replicates. The utility LCB is:

$$
{ \cal L } _ { j } = \mathrm { Q u a n t i l e } _ { 0 . 0 2 5 } \left( \left\{ \widehat { \Delta } _ { j } ^ { \ast ( b ) } \right\} _ { b = 1 } ^ { B } \right) .\tag{12}
$$

The final deployment rule combining evidence separability with utility confidence is:

$$
\widehat { a } _ { j } = { \bf 1 } [ m _ { j } > \tau ] { \bf 1 } [ L _ { j } > 0 ] .\tag{13}
$$

We fix $\tau = 0 . 0 5$ before evaluation. Retrieval is enabled only when the representation accesses task-relevant evidence and the expected gain has a positive 95% LCB. Forecast $\widehat { \Delta } _ { j }$ supports retriever ranking and resource planning.

## F. Diagnostic Algorithm And Amortized Cost

The diagnostic procedure, summarized in Appendix Algorithm S1, calibrates the shared generator response before evaluating any number of retrievers. Retrieved evidence is used only to map outputs to states and estimate their distribution. It is not provided to the generator during response calibration.

With n calibration items, |S| evidence states, and J candidate retrievers, the diagnostic generation costs are:

$$
\begin{array} { l } { C _ { \mathrm { E v i C a s t } } = ( \vert S \vert + 1 ) n , } \\ { C _ { \mathrm { D i r e c t } } = ( J + 1 ) n . } \end{array}\tag{14}
$$

For EviCast-T with top-2 retrieval, $| S | = 3$ , so EviCast uses 4n generator calls for a fixed generator regardless of J. Adding a retriever requires offline retrieval and state counting but no additional generation. The savings of shared calibration therefore grow linearly with the number of candidate retrievers. Appendix A-A provides the notation summary and additional intervention details.

## G. Theoretical Properties

The decomposition in Eq. (11) is exact under the following condition, which sharpens Eq. (4).

Assumption 1 (State Sufficiency): For every stratum $h ,$ state s, and retriever $r _ { j }$ with $p _ { j , h , s } > 0$ , the retrieved and controlled responses coincide:

$$
\begin{array} { r } { \mathbb { E } \big [ \delta ( x , E _ { j } ( x ) ) \mid h , \ \phi ( x , E _ { j } ( x ) ) = s \big ] } \\ { = \mathbb { E } \big [ \delta ( x , \widetilde { E } _ { s } ( x ) ) \mid h \big ] = : G _ { h , s } . } \end{array}
$$

Under Assumption $\begin{array} { r } { 1 , \Delta _ { j } = \sum _ { h } w _ { h } \sum _ { s } p _ { j , h , s } G _ { h , s } } \end{array}$ <sub>s</sub> for every retriever, and the plug-in forecast is consistent (Appendix Proposition S1). The remaining design question is how coarse a state map may be. Proposition 1 answers it.

Proposition 1 (State Refinement): Let $\phi ^ { \prime }$ refine $\phi ,$ so that each state s of $\phi$ is a union of states $f \in F ( s )$ of $\phi ^ { \prime }$ , and let Assumption 1 hold for $\phi ^ { \prime }$ with responses $G _ { h , f }$ . Suppose the intervention for s realizes the composition $\kappa _ { h } ( \cdot \mid s )$ over $F ( s )$ whereas retriever $r _ { j }$ produces $p _ { j , h } ( \cdot \mid s )$ . Then the population forecast $\widetilde { \Delta } _ { j } ^ { \phi }$ under $\phi$ satisfies:

$$
\begin{array} { r l r } {  { \widetilde { \Delta } _ { j } ^ { \phi } - \Delta _ { j } = \sum _ { h } w _ { h } \sum _ { s } p _ { j , h , s } \sum _ { f \in F ( s ) } \big ( \kappa _ { h } ( f \mid s ) } } \\ & { } & { \qquad - p _ { j , h } ( f \mid s ) \big ) G _ { h , f } , } \end{array}\tag{15}
$$

and hence $\begin{array} { r } { | \widetilde { \Delta } _ { j } ^ { \phi } - \Delta _ { j } | \leq \sum _ { h } w _ { h } \sum _ { s } p _ { j , h , s } \mathrm { T V } _ { j , h , s } \Gamma _ { h , s } , } \end{array}$ , where $\mathrm { T V } _ { j , h , s }$ is the total-variation distance between $\kappa _ { h } ( \cdot \mid s )$ and $p _ { j , h } ( \cdot \mid s )$ , and $\begin{array} { r } { \Gamma _ { h , s } = \operatorname* { m a x } _ { f \in { \cal F } ( s ) } G _ { h , f } - \operatorname* { m i n } _ { f \in { \cal F } ( s ) } G _ { h , f } . } \end{array}$

A coarse state is therefore exact whenever the responses it merges coincide or every retriever reproduces the composition realized by the intervention. Merging terminal-only and bridge-only evidence violates both conditions, because their responses differ by 23 to 49 F1 points (Appendix Table S6) and retrievers return them with different frequencies. The binary instantiation is the coarsest case and admits an exact characterization.

Corollary 1 (Exactness Of EviCast-B): For top-2 retrieval, let Assumption 1 hold for the number of supportive units $c \in \{ 0 , 1 , 2 \}$ with responses $G _ { h , c } ,$ , and let $\gamma _ { h } \ = \ G _ { h , 1 } \ -$ $\textstyle { \frac { 1 } { 2 } } ( G _ { h , 0 } + G _ { h , 2 } )$ denote the linearity gap. Then $\widetilde { \Delta } _ { j } ^ { \mathrm { B } } - \Delta _ { j } =$ $\begin{array} { r } { - \sum _ { h } w _ { h } p _ { j , h , 1 } \gamma _ { h } } \end{array}$ . EviCast-B is thus exact for every retriever whenever $\gamma _ { h } = 0$ , and $\gamma _ { h }$ is measurable before deployment with one additional partial-evidence intervention.

TABLE I  
EVICAST RETRIEVAL DECISIONS AND INDEPENDENT TEST GAINS ACROSS 10 TASK CONFIGURATIONS.
<table><tr><td>Mechanism</td><td>Dataset / setting</td><td>Separability margin (pp)</td><td>Utility LCB (pp)</td><td>Decision</td><td>Test gain [95% CI]</td><td></td><td>Agree</td></tr><tr><td>RAER</td><td>EMER</td><td>+5.56</td><td>+0.00</td><td>Disable</td><td></td><td>-0.25 [−0.76, 0.00]</td><td></td></tr><tr><td>RAER</td><td>MELD</td><td>-12.05</td><td>+0.00</td><td>Disable</td><td></td><td>0.00 [0.00, 0.00]</td><td></td></tr><tr><td>kNN</td><td>RAVDESS</td><td>+62.50</td><td>+18.46</td><td>Enable</td><td></td><td>+27.50 [+19.50, +35.50]</td><td></td></tr><tr><td>RAMER</td><td>EMER  / A</td><td>+0.00</td><td>-12.18</td><td>Disable</td><td>+3.69</td><td>[−2.01,+9.53]</td><td></td></tr><tr><td>RAMER</td><td>EMER / V</td><td>+2.78</td><td>-7.34</td><td>Disable</td><td>+1.32</td><td>[−2.36, +5.01]</td><td></td></tr><tr><td>RAMER</td><td>EMER / L</td><td>+48.61</td><td>-3.80</td><td>Disable</td><td></td><td>+6.71 [+1.54, +11.60]</td><td>×</td></tr><tr><td>RAMER</td><td>EMER /  AV</td><td>+2.78</td><td>+0.41</td><td>Disable</td><td>-0.13</td><td>[−2.86, +2.57]</td><td></td></tr><tr><td>RAMER</td><td>EMER / AL</td><td>+44.44</td><td>-4.34</td><td>Disable</td><td>-3.37</td><td>[−9.55, +2.74]</td><td></td></tr><tr><td>RAMER</td><td>EMER / VL</td><td>+16.67</td><td>-8.73</td><td>Disable</td><td>-2.85</td><td>[−5.94, +0.16]</td><td></td></tr><tr><td>Dense RAG</td><td>HotpotQA / BGE-M3</td><td>+34.38</td><td>+22.49</td><td>Enable</td><td></td><td>+41.15 [+34.96, +47.34]</td><td></td></tr><tr><td colspan="6">Overall</td><td colspan="2">9/10 agreement, 0 false enables</td></tr></table>

Shared calibration is also statistically efficient. Appendix Proposition S2 shows that, under a common budget of generator calls, EviCast attains lower asymptotic variance than per-retriever Direct calibration whenever $J > J _ { j } ^ { \star } = ( | S | +$ $\bar { 1 } ) \nu _ { j } ^ { 2 } / \sigma _ { j } ^ { 2 } \ - \ 1$ , where $\nu _ { j } ^ { 2 }$ and $\sigma _ { j } ^ { 2 }$ are the influence-function variances of the two estimators. Section IV-G verifies these properties empirically, and Appendix E gives all proofs.

## IV. EXPERIMENTS

We evaluate EviCast along six dimensions. First, we assess retrieval necessity decisions across tasks and RAG paradigms. The cross-task settings cover Retrieval-Augmented Emotion Reasoning (RAER), Retrieval Augment for Missing Modality Multimodal Emotion Recognition (RAMER), and k-nearestneighbor (kNN) label evidence on Explainable Multimodal Emotion Recognition (EMER), the Multimodal EmotionLines Dataset (MELD), and the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). Here A, V, and L denote acoustic, visual, and language modalities. Second, we test whether response calibration transfers across candidate retrievers and reduces diagnostic cost. Third, we compare its utility forecasts with recent performance predictors under matched generation budgets. Fourth, we evaluate generalization across datasets, generators, and model scales. Fifth, state-representation ablations and controlled-response analyses examine the role of evidence topology. Finally, we verify the theoretical properties of Section III-G empirically. All experiments use disjoint calibration and test data, and every deployment decision is frozen before observing the final test results.

## A. Evaluation Protocol

Scope And Protocol. The evaluation comprises 10 crosstask units spanning case-based emotion reasoning, missingmodality augmentation, label evidence, and dense passage RAG (Dense RAG) [1], [54]–[56]. We evaluate shared calibration across 14 HotpotQA retrieval configurations and topology-aware diagnosis on MuSiQue and HotpotQA with two generators and four dense, sparse, and fusion retrievers, followed by a prospective transfer to 2Wiki. A separately frozen transfer evaluates Qwen2.5 text-only generators at 3B and 7B scales. Every decision is fixed before independent testing, and calibration and test examples are disjoint. Within each comparison, all methods use the same queries, prompts, retrieval depth, evidence format, and answer parser. Appendices A and H provide complete splits, evidence-role rules, inference settings, and external-method adaptations.

Evaluation. We use mean example-level F1 for multi-label emotion recognition, accuracy for single-label classification, and normalized token F1 for open-ended question answering. We evaluate utility forecasts by their MAE against independent test gains and by Pearson and Spearman correlations. Gains, separability margins, utility bounds, and forecast errors are absolute differences reported in percentage points (pp) of the task metric, whereas percentages denote relative changes. A configuration has a significant positive gain only when the 95% LCB of its paired test utility exceeds zero. We report agreement with this criterion, false enabling decisions, and diagnostic generation calls. Call counts exclude final test inference common to all methods. Appendix A details baselines, state ablations, bootstrapping, and metrics.

Models And Retrieval. The multihop question-answering experiments use Qwen2.5-Omni-7B thinker (Qwen) and Phi-4-multimodal-instruct (Phi-4) in text-only mode. Both use deterministic greedy decoding, a batch size of 4, and at most 64 generated tokens. The four base retrievers return the top two full passages. BGE-M3 encodes the question and each title concatenated with its passage, then ranks candidates by cosine similarity. BM25 uses $k _ { 1 } = 1 . 5$ and $b = 0 . 7 5$ . Term frequency inverse document frequency (TF-IDF) uses cosine similarity over lowercased word unigrams and bigrams. Reciprocal Rank Fusion (RRF) combines BGE-M3 and BM25 rankings with c = 60. Ties follow the original candidate order. Title and leadsentence variants alter only the ranking text. The generator always receives the full passage.

Calibration And Data Isolation. The HotpotQA dense-RAG diagnosis and the 14-configuration reuse study share 32 calibration and 256 test queries. The topology-aware studies on MuSiQue and HotpotQA use 128 calibration and 512 test queries each, and the topology-aware HotpotQA split shares no example IDs with the reuse split. Stratification follows question hops on MuSiQue and question type and difficulty on HotpotQA. Prospective 2Wiki transfer and each text-only scale study also use 128 calibration and 512 disjoint test queries. All decisions are frozen before final evaluation; neither test answers nor test retrieval states are fed back into thresholds. Appendix A gives the complete split counts, and Appendix F records the isolation checks.

![](images/c0b1dc6b278231675f5443c24252d18a4f69f4a4a6c67581c376b9ffb6314ccf.jpg)  
Fig. 3. Two-gate decision plane for 10 cross-task configurations.

## B. Retrieval Necessity Across Tasks And RAG Paradigms

Table I reports the frozen EviCast decisions for 10 task configurations. EviCast agrees with the independent test conclusion in 9 of 10 cases, yielding 90% decision accuracy with no false enabling decisions. It correctly enables kNN label evidence on RAVDESS and Dense RAG on HotpotQA, which yield test gains of 27.50 and 41.15 points, respectively. The remaining eight configurations are disabled, which avoids every configuration without a stable positive gain. These results span different objectives, evidence forms, and injection mechanisms, showing that EviCast diagnoses the end-to-end gain from external evidence rather than a local property of one RAG implementation.

Figure 3 projects the 10 configurations onto the two-gate decision plane in Eq. (13). RAMER-AV has a slightly positive utility LCB but a separability margin below 5 points, so the separability gate prevents deployment based on this small estimate. RAMER-AL has a margin of 44.44 points but a utility LCB of −4.34 points, so the utility gate rejects a retrievable signal that does not yield actual gain. RAVDESS and HotpotQA satisfy both conditions and achieve large improvements on independent test sets. These cases show that evidence separability and model utility impose distinct deployment requirements whose conjunction defines an executable decision. Figure 4 shows an EMER case in which the configuration-level decision retains the correct Base answer instead of the extra emotion label produced by retrieval.

## C. Reusable Calibration Across Retrievers

The candidate set fixes the HotpotQA questions, generator, and prompt. BGE-M3, BM25, TF-IDF, and RRF form the base group. Additional dense, sparse, and fusion variants rank passages using titles, lead sentences, or text truncated to 128 tokens. The set contains 12 deployable configurations and two controls: Oracle prioritizes the two annotated supporting passages, and Anti-BGE reverses the BGE-M3 ranking. All 14 configurations share one EviCast-B calibration. Appendix Table S4 reports their forecasts, Direct estimates, independent test gains, absolute errors of both estimators, and deployment decisions.

TABLE II  
SHARED CALIBRATION ON 14 HOTPOTQA CONFIGURATIONS. MAE IS IN PP.
<table><tr><td>Method</td><td>Agree</td><td>Pearson</td><td>Spearman</td><td>MAE (pp) ↓</td><td>Calls↓</td></tr><tr><td>EviCast</td><td>14/14</td><td>0.983</td><td>0.890</td><td>2.75</td><td>96</td></tr><tr><td>Direct</td><td>14/14</td><td>0.818</td><td>0.332</td><td>9.59</td><td>480</td></tr></table>

For BGE-title, EviCast predicts a 31.21-point F1 gain against an observed 31.05-point gain, giving an absolute error of 0.16 points. The errors remain 0.65 and 2.47 points when ranking uses lead sentences and 128-token inputs, respectively. The same response profile yields errors of 1.26 points for BM25-lead and 4.12 points for BGE–TF-IDF fusion, compared with Direct’s 13.42 and 14.64 points.

Table II evaluates the accuracy and efficiency of shared calibration. Across 14 HotpotQA retrieval configurations, one EviCast calibration uses 96 generation calls and makes 14 of 14 correct deployment decisions, with an MAE of 2.75 points and Pearson and Spearman correlations of 0.983 and 0.890. Relative to Direct, EviCast reduces forecast error by 71.3% and generation calls by 80%. The 96 calls comprise Base, Supportive, and Misleading responses for 32 calibration queries (3 × 32). Direct evaluates Base and an Actual retrieval condition for each of the 14 configurations, requiring 15 × 32 = 480 calls.

Across 12 standard deployable configurations, EviCast obtains 12 of 12 correct decisions and an MAE of 2.22 points, compared with 10.20 points for Direct, while reducing generation calls by 76.9%. These results support the decomposition in Fig. 2, where one generator response profile serves dense, sparse, and fusion retrievers through their state distributions. Figure 5 shows that this shared profile places every configuration close to the ideal diagonal, and Appendix Fig. S1 shows how diagnostic savings scale with retriever count.

## D. Utility Forecasting Under Matched Budgets

We compare EviCast, Direct, retrieval performance prediction (RPP), unnormalized Weighted Information Gain (U-WIG), the score-magnitude-and-variance predictor U-SMV, BERT-based Query Performance Prediction (BERT-QPP), and QualT5 on HotpotQA using Qwen2.5-3B-Instruct. The study fixes 128 calibration queries, 512 disjoint test queries, and BGE-M3, BM25, TF-IDF, and RRF. Fifty label-independent calibration permutations produce paired subsamples shared by every method and budget; all estimators and budgets are fixed before evaluation.

For answer-generation budget C, EviCast uses ⌊C/4⌋ queries for Base and three topology states. Direct and comparison predictors use $\lfloor C / 5 \rfloor$ queries for the shared Base and four Actual conditions. EviCast recomputes stratum-conditioned responses and state frequencies on each subset, while Direct estimates the paired mean gain. We first average absolute forecast errors across the four retrievers and then average these MAEs across repetitions. The target is each retriever’s mean F1 gain on the fixed test set.

![](images/12073bfdb1631563693bf687d405d9bade04db624c8833ea96f4987704629231.jpg)

Fig. 4. EMER example with video frames, recorded generator reasoning excerpts, and the EviCast configuration-level decision.  
![](images/8fc1ddf590c18871462df94301a655b1079381a3a0bef7d0e1fe85d199a83dd4.jpg)  
Fig. 5. Forecast and independent test utility for 14 HotpotQA retrieval configurations, with EviCast on the left and Direct calibration on the right. Error bars show 95% CIs.

U-WIG averages the two retrieved-passage scores, while U-SMV combines their magnitude and dispersion [57], [58]. BERT-QPP supplies a supervised query-passage performance score [59], and QualT5 uses the maximum learned passagequality score among the same passages [60]. All scalar baselines use the same calibration-only linear map from their native scores to F1 gain.

RPP adapts the 30-feature pre-generation utility predictor of Tian et al. [12] to the 10-candidate HotpotQA pool and fixed top-2 generator evidence. Features include retrievalscore statistics, embedding geometry and coherence, BERT-

QPP, readability, QualT5, and teacher-forced generator context likelihood. Local Normalized Query Commitment (NQC) normalizes the top-two score deviation by the absolute mean candidate score, while coherence compares the top five and bottom five candidates. We use E5-small-v2, qt5-small, and the released mean average precision (MAP)-trained BERT-QPP checkpoint. Each retriever uses ordinary least squares with an intercept and calibration-only feature standardization. All 30 features enter the regression.

Budgets count answer generations. RPP additionally incurs teacher-forced likelihood scoring, embedding and qualitymodel inference, and feature computation, none of which EviCast requires. Appendix I specifies the sampling seeds.

Table III shows that EviCast is most accurate in the lowbudget regime that matters most for pre-deployment diagnosis.

TABLE III  
MATCHED ANSWER-GENERATION BUDGETS ON HOTPOTQA WITH QWEN2.5-3B-INSTRUCT AND FOUR RETRIEVERS. MAE IS IN PP, AND LOWER IS BETTER.
<table><tr><td>Method</td><td>160</td><td>320</td><td>480 512</td></tr><tr><td>EviCast</td><td>5.73</td><td>4.34</td><td>3.91 3.86</td></tr><tr><td>Direct</td><td>6.85</td><td>4.89</td><td>3.94 3.92</td></tr><tr><td>RPP [12]</td><td>15.25</td><td>5.81</td><td>4.88 4.77</td></tr><tr><td>U-WIG [58]</td><td>7.04</td><td>4.89</td><td>3.95 3.94</td></tr><tr><td>U-SMV [57]</td><td>6.99</td><td>4.83</td><td>3.84 3.79</td></tr><tr><td>BERT-QPP [59]</td><td>6.99</td><td>4.86</td><td>3.89 3.87</td></tr><tr><td>QualT5 [60]</td><td>6.77</td><td>4.68</td><td>3.703.70</td></tr></table>

TABLE IV

DECISION AGREEMENT AND MAE (PP) ACROSS DATASETS AND GENERATORS.
<table><tr><td>Setting</td><td></td><td>Agree MAE↓</td></tr><tr><td>MuSiQue / Ours + Qwen [61]</td><td>4/4</td><td>1.53</td></tr><tr><td>MuSiQue / Ours + Phi-4 [62]</td><td>4/4</td><td>1.55</td></tr><tr><td>HotpotQA / Ours + Qwen [61]</td><td>4/4</td><td>4.59</td></tr><tr><td>HotpotQA / Ours + Phi-4 [62]</td><td>4/4</td><td>3.58</td></tr><tr><td>2Wiki / Ours + Qwen [61]</td><td>3/4</td><td>4.70</td></tr><tr><td>2Wiki / Ours + Phi-4 [62]</td><td>4/4</td><td>1.15</td></tr><tr><td>Three-dataset total</td><td>23/24</td><td>2.85</td></tr><tr><td>Ours + Qwen2.5-Text [63] (3B+7B)</td><td>8/8</td><td>3.67</td></tr></table>

With 160 generations, EviCast attains the lowest MAE of 5.73 points, which is 15.4% lower than QualT5, the strongest alternative, 16.4% lower than Direct, and 62.4% lower than RPP. With 320 generations, EviCast again attains the lowest MAE of 4.34 points, compared with 4.68 for QualT5, 4.89 for Direct, and 5.81 for RPP. Its advantage over Direct, RPP, and U-WIG holds at every budget, including the full budget of 512 generations, where EviCast uses all 128 calibration queries and the comparison predictors use 102 queries and 510 generations. Spending a fixed generation budget on shared response states therefore yields more accurate forecasts than spending it on per-retriever Actual outputs, and the return is largest when calibration budgets are tight.

EviCast is also the most stable predictor across calibration subsamples and attains the lowest 95th-percentile MAE of all seven methods at every budget, as reported in Appendix Table S20.

## E. Generalization Across Datasets And Generators

Cross-Dataset And Cross-Generator Validation. Table IV evaluates topology-aware diagnosis on MuSiQue and HotpotQA with two generators and four retrievers. EviCast makes 16 of 16 decisions correctly with an overall MAE of 2.81 points. Each dataset–generator pair requires 512 shared calibration generations rather than Direct’s 640. Complete perconfiguration forecasts and independent test gains are provided in Appendix D.

Prospective Transfer To A New Dataset. We prospectively transfer the topology-aware diagnosis to 2Wiki under a protocol frozen before producing retrieval rankings or generator outputs. The study uses 128 shared calibration examples and 512 disjoint test examples. It evaluates two generators with BGE-M3, BM25, TF-IDF, and RRF under fixed top-2 retrieval. Across eight configurations, EviCast makes 7/8 correct decisions with no false enables. Forecast gains achieve Pearson and Spearman correlations of 0.955 and 0.905 with independent test gains. Direct makes 6/8 correct decisions and requires 640 rather than 512 generation calls per generator. Appendix Tables S10 and S11 report the aggregate and perconfiguration results. Across MuSiQue, HotpotQA, and 2Wiki, EviCast makes 23 of 24 frozen decisions correctly with an aggregate MAE of 2.85 points.

TABLE V  
FROZEN TRANSFER TO TEXT-ONLY GENERATORS AT THE 3B AND 7B PARAMETER SCALES.
<table><tr><td>Generator</td><td>Params.</td><td></td><td></td><td></td><td>False MAE Configs. Agree enable (pp) ↓</td></tr><tr><td>Qwen2.5-Instruct</td><td>3B</td><td>4</td><td>4/4</td><td>0</td><td>3.86</td></tr><tr><td>Qwen2.5-Instruct</td><td>7B</td><td>4</td><td>4/4</td><td>0</td><td>3.49</td></tr><tr><td>Overall</td><td>3B and 7B</td><td>8</td><td>8/8</td><td>0</td><td>3.67</td></tr></table>

TABLE VI

TWO-GATE ABLATION ON 10 TASK CONFIGURATIONS. RETRIEVAL IS ENABLED ONLY WHEN ALL INCLUDED GATES PASS. THE FINAL ROW IS EVICAST.
<table><tr><td colspan="3">Deployment decisions</td><td colspan="2">Gates included</td></tr><tr><td>Correct decisions ↑</td><td>False enables↓</td><td>Missed benefits↓</td><td>Relevant evidence</td><td>Positive gain</td></tr><tr><td>7/10</td><td>3</td><td>0</td><td>√</td><td></td></tr><tr><td>8/10</td><td>1</td><td>1</td><td></td><td>V</td></tr><tr><td>9/10</td><td>0</td><td>1</td><td></td><td></td></tr></table>

Transfer To Text-Only Generators Across Model Scales. We evaluate frozen topology-aware diagnosis on standard text-only generators at different scales. The study uses Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct on HotpotQA with BGE-M3, BM25, TF-IDF, and RRF. For each generator, EviCast shares 128 calibration examples and fixes the top-2 evidence states, anchor, separability threshold, and deployment rule. Each retriever is evaluated on 512 disjoint test queries. Table V reports the results. Across both scales, EviCast makes all eight deployment decisions correctly with no false enables or rejects. Its aggregate MAE is 3.67 points, compared with 6.28 for EviCast-B and 3.79 for Count. These results extend the shared intervention and evidence-state decomposition to text-only generators at two parameter scales.

## F. Ablation And Mechanism Analysis

Two-Gate Decision Rule. Table VI removes either gate across the same 10 configurations. Separability alone falsely enables RAER-EMER, RAMER-AL, and RAMER-VL. Utility alone falsely enables RAMER-AV and misses RAMER-L. The complete rule eliminates false enables while retaining both strongly beneficial configurations. The main experiments fix the separability threshold at $\tau \ : = \ : 0 . 0 5$ . Holding the utility gate $L _ { j } ~ > ~ 0$ fixed, a sweep over the same 10 configurations retains 9/10 agreement with no false enables throughout $0 . 0 5 \le \tau \le 0 . 3 0$ . Appendix Table S15 reports the complete sweep.

Evidence-State Representation. Table VII traces the refinement predicted by Proposition 1 while fixing datasets, retrievers, generators, and retrieval statistics. EviCast-B interpolates between the supportive and misleading endpoints and yields an overall MAE of 8.30 points. Count separates zero, one, and two correct passages and reduces the MAE to 3.87 points. EviCast-T further separates terminal and bridge roles and reduces the MAE to 2.81 points, lowering forecast error by 66.1% and 27.4%, respectively, while preserving 16 of 16 correct deployment decisions. EviCast-T matches the 2.77- point MAE of per-retriever Direct calibration within 0.04 points while requiring 20% fewer diagnostic calls, so its accuracy comes from structured, reusable response calibration rather than a larger generation budget.

TABLE VII  
DECISION AGREEMENT AND MAE (PP) ALONG THE STATE REFINEMENT OF PROPOSITION 1, WITH DIRECT CALIBRATION AS REFERENCE.
<table><tr><td>Method</td><td>MuSiQue agree</td><td>MuSiQue MAE</td><td>HotpotQA agree</td><td>HotpotQA MAE</td><td>Overall agree</td><td>Overall MAE</td><td>Gen. calls</td></tr><tr><td>EviCast-B</td><td>8/8</td><td>7.24</td><td>8/8</td><td>9.35</td><td>16/16</td><td>8.30</td><td>384</td></tr><tr><td>Count</td><td>8/8</td><td>3.49</td><td>8/8</td><td>4.25</td><td>16/16</td><td>3.87</td><td>512</td></tr><tr><td>EviCast-T</td><td>8/8</td><td>1.54</td><td>8/8</td><td>4.08</td><td>16/16</td><td>2.81</td><td>512</td></tr></table>

![](images/9825cfd22f7baa73b24fd00c2fead3c279a395540a4c2a48b92abe59146234d4.jpg)  
Fig. 6. Evidence-role effects on two-hop MuSiQue questions. Bars show F1 gains over no retrieval for Qwen and Phi-4, and every intervention supplies two passages.

Why Evidence Topology Matters. Figure 6 and Appendix Table S6 show why correct-passage counts are insufficient. States N and A each contain one correct supporting passage, but N provides only bridge information, whereas A directly contains terminal answer information. Their F1 gains are 4.55 and 50.95 points for Qwen and 2.58 and 45.65 points for Phi-4. Thus, one correct passage does not define a utilityhomogeneous state. Reusable profiles must distinguish terminal and bridge roles.

This contrast is exactly the refinement bias of Proposition 1. Because Count and EviCast-T use the same number of diagnostic calls, their accuracy difference in Table VII isolates the value of distinguishing evidence roles. Appendix Tables S6 and S7 report the responses for every stratum and the state distribution of every retriever.

TABLE VIII  
EMPIRICAL VERIFICATION OF THE THEORETICAL PROPERTIES ON MUSIQUE AND HOTPOTQA. DEVIATIONS ARE IN PP RELATIVE TO A FINER-STATE REFERENCE.
<table><tr><td>Property</td><td>Statistic</td><td>MuSiQue</td><td>HotpotQA</td></tr><tr><td>Refinement (Prop. 1)</td><td>Deviation of EviCast-B / Count / EviCast-T</td><td></td><td>7.55 / 3.80 / 0.31 5.95 / 1.58 / 0.27</td></tr><tr><td>Sufficiency (Assump. 1)</td><td>Smallest p</td><td>0.38</td><td>0.26</td></tr><tr><td>Efficiency (Prop. S2)</td><td>Largest Jj*</td><td>2.15</td><td>2.09</td></tr></table>

## G. Empirical Verification Of Theoretical Properties

Refinement. Table VIII confirms each property. Relative to a reference that additionally separates retrieved sets without any supporting passage from bridge-only sets, EviCast-T deviates by only 0.31 and 0.27 F1 points on MuSiQue and HotpotQA, compared with 7.55 and 5.95 points for EviCast-B. The topology-aware map is therefore already sufficient, and its remaining forecast error reflects calibration sampling rather than state representation. Corollary 1 also explains when EviCast-B suffices. Post hoc, the test-set linearity gap is −0.31 points on the 14-configuration reuse split, where EviCast-B is highly accurate, but −15.30 and −10.00 points for Qwen and Phi-4 on the topology-aware HotpotQA split.

Sufficiency. Stratified permutation tests detect no retriever dependence of the response within any state in all 15 tests, including the 12-retriever reuse study (all $p \geq 0 . 2 6 )$

Efficiency. The influence-function ratio $\nu _ { j } / \sigma _ { j }$ lies between 0.73 and 0.89, and $J _ { j } ^ { \star } ~ \leq ~ 2 . 1 5$ in all 16 configurations, so every candidate set in our experiments lies in the regime where shared calibration is the more efficient estimator. Appendix E-E reports the complete statistics.

## V. CONCLUSION

Retrieval deployment should depend on how a target generator uses evidence, not only on its availability. EviCast forecasts utility by combining shared interventional response calibration with retriever-specific evidence-state distributions. A state-refinement analysis characterizes when its binary and topology-aware instantiations are exact. Experiments show that response calibration transfers across retrievers and that terminal and bridge roles explain utility differences hidden by passage counts. Across three datasets, EviCast decides 23 of 24 topology configurations correctly, and in the lowbudget regime it forecasts utility more accurately than recent performance predictors. On HotpotQA, EviCast makes all 14 deployment decisions correctly while reducing diagnostic calls by 80%. These results shift RAG from default retrieval toward utility-guided deployment, with one calibration shared by multiple retrievers. Future work will extend evidence states to variable depths, iterative retrieval, and multimodal knowledge environments.

## ACKNOWLEDGMENTS

This work was supported in part by the National Natural Science Foundation of China under Grants 62606208, 62606209, 62227807, and U24B20186, by the National Science and Technology Major Project on Brain Science and Brain-like Intelligence Technology under Grants 2021ZD0200600 and 2021ZD0200408, and by the Talent Scientific Fund of Lanzhou University. We also acknowledge support from the WQ & UCAS Research Academy Intelligent Computing Center (WRA-ICC) and the Supercomputing Center of Lanzhou University.

## REFERENCES

[1] P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive nlp tasks,” Advances in neural information processing systems, vol. 33, pp. 9459–9474, 2020.

[2] F. Shi et al., “Large language models can be easily distracted by irrelevant context,” in International conference on machine learning. PMLR, 2023, pp. 31 210–31 227.

[3] C. Amiraz, F. Cuconasu, S. Filice, and Z. Karnin, “The distracting effect: Understanding irrelevant passages in rag,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 18 228–18 258.

[4] Z. Jiang et al., “Active retrieval augmented generation,” in Proceedings of the 2023 conference on empirical methods in natural language processing, 2023, pp. 7969–7992.

[5] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning to retrieve, generate, and critique through self-reflection,” in International conference on learning representations, vol. 2024, 2024, pp. 9112–9141.

[6] S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. C. Park, “Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity,” in Proceedings of the 2024 conference of the north american chapter of the association for computational linguistics: Human language technologies (volume 1: Long papers), 2024, pp. 7036– 7050.

[7] S. Es, J. James, L. E. Anke, and S. Schockaert, “Ragas: Automated evaluation of retrieval augmented generation,” in Proceedings ofthe 18th conference of the european chapter of the association for computational linguistics: system demonstrations, 2024, pp. 150–158.

[8] D. Ru et al., “Ragchecker: A fine-grained framework for diagnosing retrieval-augmented generation,” Advances in Neural Information Processing Systems, vol. 37, pp. 21 999–22 027, 2024.

[9] L. Dai, Y. Xu, J. Ye, H. Liu, and H. Xiong, “Seper: Measure retrieval utility through the lens of semantic perplexity reduction,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 78 977–79 007.

[10] L. Perez-Beltrachini and M. Lapata, “Uncertainty quantification in retrieval augmented question answering,” arXiv preprint arXiv:2502.18108, 2025.

[11] Z. Wang et al., “Infogain-rag: Boosting retrieval-augmented generation through document information gain-based reranking and filtering,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, p. 7190–7204.

[12] F. Tian, D. Ganguly, and C. Macdonald, “Predicting retrieval utility and answer quality in retrieval-augmented generation,” in European Conference on Information Retrieval. Springer, 2026, pp. 368–385.

[13] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval augmented language model pre-training,” in International conference on machine learning. PMLR, 2020, pp. 3929–3938.

[14] V. Karpukhin et al., “Dense passage retrieval for open-domain question answering,” in Proceedings ofthe 2020 conference on empirical methods in natural language processing (EMNLP), 2020, pp. 6769–6781.

[15] G. Izacard and E. Grave, “Leveraging passage retrieval with generative models for open domain question answering,” in Proceedings of the 16th conference of the european chapter of the association for computational linguistics: main volume, 2021, pp. 874–880.

[16] S. Borgeaud et al., “Improving language models by retrieving from trillions of tokens,” in International conference on machine learning. PMLR, 2022, pp. 2206–2240.

[17] O. Ram et al., “In-context retrieval-augmented language models,” Transactions of the Association for Computational Linguistics, vol. 11, pp. 1316–1331, 2023.

[18] V. Lin et al., “Ra-dit: Retrieval-augmented dual instruction tuning,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 19 138–19 162.

[19] Y. Yu et al., “Rankrag: Unifying context ranking with retrievalaugmented generation in llms,” Advances in neural information processing systems, vol. 37, pp. 121 156–121 184, 2024.

[20] F. Petroni et al., “Kilt: a benchmark for knowledge intensive language tasks,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2021, pp. 2523–2544.

[21] Q. Wu, C. Shen, P. Wang, A. Dick, and A. van den Hengel, “Image captioning and visual question answering based on attributes and external knowledge,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 6, pp. 1367–1381, 2018.

[22] P. Wang, Q. Wu, C. Shen, A. Dick, and A. van den Hengel, “FVQA: Fact-based visual question answering,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 10, pp. 2413–2427, 2018.

[23] D. Gao, R. Wang, S. Shan, and X. Chen, “CRIC: A VQA dataset for compositional reasoning on vision and commonsense,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 5, pp. 5561–5578, 2023.

[24] D. Caffagni, S. Sarto, M. Cornia, L. Baraldi, and R. Cucchiara, “Recurrence meets transformers for universal multimodal retrieval,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026, Early Access. [Online]. Available: https://doi.org/10.1109/TPAMI.2026. 3729293

[25] O. Yoran, T. Wolfson, O. Ram, and J. Berant, “Making retrievalaugmented language models robust to irrelevant context,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 29 862–29 883.

[26] N. F. Liu et al., “Lost in the middle: How language models use long contexts,” Transactions of the association for computational linguistics, vol. 12, pp. 157–173, 2024.

[27] F. Cuconasu et al., “The power of noise: Redefining retrieval for rag systems,” in Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval, 2024, pp. 719–729.

[28] A. Mallen, A. Asai, V. Zhong, R. Das, D. Khashabi, and H. Hajishirzi, “When not to trust language models: Investigating effectiveness of parametric and non-parametric memories,” in Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: Long papers), 2023, pp. 9802–9822.

[29] Y. Wang, P. Li, M. Sun, and Y. Liu, “Self-knowledge guided retrieval augmentation for large language models,” in Findings of the association for computational linguistics: EMNLP 2023, 2023, pp. 10 303–10 315.

[30] W. Su, Y. Tang, Q. Ai, Z. Wu, and Y. Liu, “Dragin: Dynamic retrieval augmented generation based on the real-time information needs of large language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 12 991–13 013.

[31] F. Xu, W. Shi, and E. Choi, “Recomp: Improving retrieval-augmented lms with context compression and selective augmentation,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 43 478–43 502.

[32] Z. Wang, J. Araki, Z. Jiang, M. R. Parvez, and G. Neubig, “Learning to filter context for retrieval-augmented generation,” arXiv preprint arXiv:2311.08377, 2023.

[33] Y. Han, G. Huang, S. Song, L. Yang, H. Wang, and Y. Wang, “Dynamic neural networks: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 7436–7456, 2022.

[34] G. Huang et al., “Glance and focus networks for dynamic visual recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 4, pp. 4605–4621, 2023.

[35] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, “Ares: An automated evaluation framework for retrieval-augmented generation systems,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2024, pp. 338–354.

[36] J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking large language models in retrieval-augmented generation,” in Proceedings of the AAAI

conference on artificial intelligence, vol. 38, no. 16, 2024, pp. 17 754– 17 762.

[37] T. Gao, H. Yen, J. Yu, and D. Chen, “Enabling large language models to generate text with citations,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 6465– 6488.

[38] C. Niu et al., “Ragtruth: A hallucination corpus for developing trustworthy retrieval-augmented language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 10 862–10 878.

[39] F. Zhu, X.-Y. Zhang, Z. Cheng, and C.-L. Liu, “Revisiting confidence estimation: Towards reliable failure prediction,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 5, pp. 3370– 3387, 2024.

[40] T. Baltrusaitis, C. Ahuja, and L.-P. Morency, “Multimodal machineˇ learning: A survey and taxonomy,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 2, pp. 423–443, 2019.

[41] P. Xu, X. Zhu, and D. A. Clifton, “Multimodal learning with transformers: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 10, pp. 12 113–12 132, 2023.

[42] J. Dang, H. Zheng, J. Lai, X. Yan, and Y. Guo, “Efficient and robust video object segmentation through isogenous memory sampling and frame relation mining,” IEEE Transactions on Image Processing, vol. 32, pp. 3924–3938, 2023.

[43] J. Dang, H. Zheng, X. Xu, L. Wang, and Y. Guo, “Beyond appearance: Multi-frame spatio-temporal context memory networks for efficient and robust video object segmentation,” IEEE Transactions on Image Processing, vol. 33, pp. 4853–4866, 2024.

[44] J. Dang, H. Zheng, B. Wang, L. Wang, and Y. Guo, “Temporo-spatial parallel sparse memory networks for efficient video object segmentation,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 11, pp. 17 291–17 304, 2024.

[45] J. Dang, H. Zheng, X. Xu, L. Wang, Q. Hu, and Y. Guo, “Adaptive sparse memory networks for efficient and robust video object segmentation,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 2, pp. 3820–3833, 2025.

[46] J. Dang, H. Zheng, Y. Guo, J. Lai, B. Hu, and T.-S. Chua, “Video decoupling networks for accurate, efficient, generalizable, and robust video object segmentation,” IEEE Transactions on Image Processing, vol. 35, pp. 1218–1230, 2026.

[47] J. Dang, H. Zheng, Z. Chen, Z. Li, Y. Guo, and T.-S. Chua, “Fast track anything with sparse spatio-temporal propagation for unified video segmentation,” IEEE Transactions on Image Processing, vol. 35, pp. 955–969, 2026.

[48] J. Dang et al., “Decoupled seg tokens make stronger reasoning video segmenter and grounder,” arXiv preprint arXiv:2506.22880, 2025.

[49] J. Dang et al., “SynPO: Synergizing descriptiveness and preference optimization for video detailed captioning,” arXiv preprint arXiv:2506.00835, 2025.

[50] Y. Wang et al., “AffectOmni: RL-verifiable people-centric grounded affective reasoning for social and art-related scenes,” IEEE Transactions on Affective Computing, pp. 1–12, 2026, early access.

[51] Y. Wang et al., “DualAffect: Dual-path paralinguistic integration for generative audiovisual emotion understanding,” arXiv preprint arXiv:2610.18073, 2026.

[52] Y. Wang et al., “Read-best is not steer-best: A probing–steering layer dissociation in omni-modal large language models,” arXiv preprint arXiv:2609.22135, 2026.

[53] Z. Han, C. Zhang, H. Fu, and J. T. Zhou, “Trusted multi-view classification with dynamic evidential fusion,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 2, pp. 2551–2566, 2023.

[54] Z. Wen et al., “Listen, watch, and learn to feel: Retrieval-augmented emotion reasoning for compound emotion generation,” in Findings of the Association for Computational Linguistics: ACL 2025, 2025, pp. 11 313–11 327.

[55] Q. Fan, H. Yuan, H. Zuo, R. Liu, and G. Gao, “Leveraging retrieval augment approach for multimodal emotion recognition under missing modalities,” arXiv preprint arXiv:2410.02804, 2024.

[56] Z. Ma et al., “emotion2vec: Self-supervised pre-training for speech emotion representation,” in Findings of the Association for Computational Linguistics: ACL 2024, 2024, pp. 15 747–15 760.

[57] Y. Tao and S. Wu, “Query performance prediction by considering score magnitude and variance together,” in Proceedings of the 23rd ACM international conference on conference on information and knowledge management, 2014, pp. 1891–1894.

[58] O. Dado, D. Carmel, and O. Kurland, “Rag performance prediction for question answering,” arXiv preprint arXiv:2604.07985, 2026.

[59] N. Arabzadeh, M. Khodabakhsh, and E. Bagheri, “Bert-qpp: contextualized pre-trained transformers for query performance prediction,” in Proceedings of the 30th ACM International Conference on Information & Knowledge Management, 2021, pp. 2857–2861.

[60] X. Chang, D. Mishra, C. Macdonald, and S. MacAvaney, “Neural passage quality estimation for static pruning,” in Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024, pp. 174–185.

[61] J. Xu et al., “Qwen2.5-Omni technical report,” arXiv preprint arXiv:2503.20215, 2025.

[62] A. Abouelenin et al., “Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras,” arXiv preprint arXiv:2503.01743, 2025.

[63] A. Yang, B. Yang, B. Zhang et al., “Qwen2.5 technical report,” arXiv preprint arXiv:2412.15115, 2024.

## AUTHOR BIOGRAPHIES

![](images/7701253fa57a8ca3448b282cb666635cd53fc6144109d19fc32dc6e80f8504f9.jpg)

Yibo Wang is currently pursuing the M.S. degree at the Gansu Provincial Key Laboratory of Wearable Computing, School of Information Science and Engineering, Lanzhou University, Lanzhou, China. He received the B.S. degree from Dalian University of Technology, Dalian, China. His research interests include multimodal large language models, affective computing, and reinforcement learning.

![](images/21807e6be5c3fa01b155b9b617fd86e9a15066edeb637883ffed9c3138f1e275.jpg)

Jisheng Dang received the Ph.D. degree from Sun Yat-sen University, China, advised by Prof. Jianhuang Lai and Prof. Huicheng Zheng. He worked as a research fellow at the NExT++ laboratory of the National University of Singapore, advised by Prof. Tat-Seng Chua. He is now a tenured associate professor at the School of Information Science and Engineering, Lanzhou University. His research interests include multimodal learning, video understanding, and embodied intelligence. He has published several papers as the first author or corresponding author in major journals and conferences including IEEE TPAMI/TIP/TNNLS/TITS/IJCAI/AAAI/ICLR/NeurIPS/PR.

![](images/5c5d3dec052a49bd3d82fcde05d0d102135bfed606bdfab31b704789471eabe8.jpg)

Meng Wang (Fellow, IEEE) received the B.E. and Ph.D. degrees (in the Special Class for the Gifted Young) from the Department of Electronic Engineering and Information Science, University of Science and Technology of China, Hefei, China, respectively. He is a Professor at the Hefei University of Technology, Hefei. His current research interests include multimedia content analysis, search, mining, recommendation, and large-scale computing. Dr. Wang was the recipient of the Best Paper Awards successively from the 17th and 18th ACM International Conference on Multimedia, the Best Paper Award from the 16th International Multimedia Modeling Conference, Best Paper Award from the 4th International Conference on Internet Multimedia Computing and Service, and Best Demo Award from the 20th ACM International Conference on Multimedia.

![](images/d8211a6ba58803ea7c011cbdb34555936298776d2f71e58877abf84fc12ef0fe.jpg)

Qi Tian (Fellow, IEEE) received the Ph.D. degree in ECE from the University of Illinois at Urbana-Champaign, Champaign, IL, USA, in 2002. He is currently the Chief Scientist of Huawei Terminal BG. He was the Chief Scientist in computer vision with Huawei Noah’s Ark Laboratory from 2018 to 2020. Before he joined Huawei, he was a Full Professor with the Department of Computer Science, The University of Texas at San Antonio, San Antonio, TX, USA, from 2002 to 2019. He was listed in the top ten of the 2016 Most Influential Scholars in Multimedia by Aminer.org. He is an IEEE Fellow (Class 2016) and an Academician of the International Eurasian Academy of Sciences (elected in 2021), with 117,426 citations on Google Scholar.

![](images/8758c2e68c78fbb393ffe6cc108e1d8258581af8e68c6b1045011feb7b1be484.jpg)

Bin Hu (Fellow, IEEE) received the Ph.D. degree in computer science from the Institute of Computing Technology, Chinese Academy of Sciences, China, in 1998. Since 2008, he has been a Professor and Dean of the School of Information Science and Engineering, Lanzhou University. He has also held a guest professorship at ETH Zurich. He serves as Editorin-Chief of IEEE Transactions on Computational Social Systems and is a Fellow of IET and AAIA. His research interests include pervasive computing, computational psychophysiology, data modeling, and

artificial intelligence.  
![](images/5fcda94cee9fbd8b3bba03e2f08ad4baa9669aa1cb89c55f15dec1e5636dc39c.jpg)

Tat-Seng Chua received the Ph.D. degree from the University of Leeds, U.K. He is the KITHCT chair professor with the School of Computing, National University of Singapore, where he was the acting and founding dean of the School from 1998 to 2000. He is the co-director of NExT, a joint center between NUS and Tsinghua University, to develop technologies for live social media search. He is the 2015 winner of the prestigious ACM SIGMM Award. He is the chair of the Steering Committee of the ACM International Conference on Multimedia Retrieval (ICMR) and the

Multimedia Modeling (MMM) conference series. He is also the general cochair of ACM Multimedia 2005, ACM CIVR (now ACM ICMR) 2005, ACM SIGIR 2008, and ACM Web Science 2015. He serves on the editorial boards of four international journals. He is the co-founder of two technology startups in Singapore and a Fellow of the Singapore Academy of Sciences, with 114,163 citations on Google Scholar.