# Th<sub>e</sub> L<sub>aws o</sub>f C<sub>on</sub>t<sub>ex</sub>t All<sub>oca</sub>ti<sub>on:</sub> C<sub>ausa</sub>l M<sub>easuremen</sub>t <sub>an</sub>d Cl<sub>ose</sub>d<sub>-</sub>L<sub>oop</sub> O<sub>rc</sub>h<sub>es</sub>t<sub>ra</sub>ti<sub>on</sub> i<sub>n</sub> G<sub>enera</sub>ti<sub>ve</sub> S<sub>earc</sub>h

PEIYANG LIU<sub>,</sub> National Engineering Research Center for Software Engineering, Peking University, China

XI WANG<sub>,</sub> Peking University, China

DI LIANG<sub>,</sub> Tencent, China

WEI YE<sup>∗</sup><sub>,</sub> National Engineering Research Center for Software Engineering, Peking University, China

As Retrieval-Augmented Generation (RAG) shifts toward diverse portfolio generation, it is stymied by two critical bottlenecks: flawed measurement of evidence utilization, and suboptimal context budget allocation. We resolve both sequentially.

To resolve measurement, we expose a pervasive “diagnostic illusion”: standard relevance proxies fail catastrophically on hard negatives. We replace them with an eficient causal leave-one-out probe that accurately isolates generative reliance and formally calibrates the structural dilution of LLM attention.

To resolve allocation, we deploy this causal probe in a deconfounded factorial grid. We prove that the prevailing strategy of monolithic context widening is an architectural trap penalized by relevance decay. Instead, allocating compute iteratively across multiple sequential generations drives transformative portfolio recall gains of 16.8–20.5 absolute percentage points, scaling robustly up to 32B models.

Finally, we unify these solutions into a deployable closed-loop submodular scheduler. Augmented by an attribution-steered contrastive decoder to override LLM attention inertia, our architecture systematically forces fresh evidence integration. By dominating classical open-loop baselines, we establish sequential, feedback-driven orchestration as the definitive paradigm for generative search. Our code, data, and causal measurement instruments are available at https://github.com/PeiYangLiu/ascp.

CCS Concepts: • Information systems  Retrieval models and ranking; Search results deduplication; Evaluation of retrieval results. Additional Key Words and Phrases: retrieval-augmented generation, context attribution, inference-time scaling, test-time compute, evaluation

## ACM Reference Format:

Peiyang Liu, Xi Wang, Di Liang, and Wei Ye. 2026. The Laws ofContext Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search. ACM Trans. Inf. Syst. 0, 0, Article 0 (July 2026), 37 pages. https://doi.org/XXXXXXX.XXXXXXX

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Classical information retrieval (IR) systems treat ambiguous or multi-faceted queries by returning a diversified ranked list, acknowledging a fundamental truth: a single document rarely satisfies all underlying user intents [10, 22, 92, 102, 112]. Retrieval-augmented generation (RAG) inherits this premise of underspecified queries but radically alters the delivery

Authors’ Contact Information: Peiyang Liu, National Engineering Research Center for Software Engineering, Peking University, Beijing, China, liupeiyang@pku.edu.cn; Xi Wang, Peking University, Beijing, China, wangxi5629@pku.edu.cn; Di Liang, Tencent, Beijing, China, liangd17@fudan.edu.cn; Wei Ye, National Engineering Research Center for Software Engineering, Peking University, Beijing, China, wye@pku.edu.cn.

mechanism [14, 61, 142]. Rather than providing a diversified list of documents for a human to browse, the prevailing RAG paradigm forces generative models to compress retrieved passages into a single context prompt, aiming to synthesize one monolithic response. However, for complex informational needs, this single-pass synthesis is structurally inadequate. To truly satisfy ambiguous queries in the generative era, a robust system must transition from extracting a single answer to generating a diverse portfolio of responses that collectively cover the space of evidence-supported truths.

This necessary paradigm shift from diversified ranking to generative portfolio construction introduces a critical system design dilemma: How should a fixed inference budget be optimally allocated? Given a retrieved pool of candidate documents and a constrained hardware budget, system architects face two divergent paths. Should they follow the current natural language processing (NLP) trend of feeding a massive, wide context into a single generation pass? Or should they adhere to IR diversification principles by dividing the budget iteratively, querying the model across multiple sequential rounds with narrower, focused contexts? Figure 1 visualizes this exact architectural dilemma and previews our core findings: despite consuming an identical physical evidence budget, iterative narrow contexts fundamentally eclipse monolithic wide contexts in answer space coverage. Crucially, standard relevance proxies completely mask this dynamic, necessitating a paradigm shift in generative evaluation.

Before we can empirically resolve this allocation question, we hit an epistemological wall: the RAG community lacks a rigorous mechanism to measure what evidence a Large Language Model (LLM) actually utilizes from its prompt. Standard proxies, such as embedding similarity and lexical overlap, sufer from severe methodological blind spots because they conflate genuine evidence utilization with mere topical relevance. Classical IR has long held that a measure must be diagnosed against the behaviour it claims to capture rather than trusted on face validity [27, 46]; we apply that same standard to context attribution. To overcome this, we formulate a causal measurement instrument based on counterfactual sensitivity, a leave-one-out (LOO) probe. Because the generated response is held fixed, our counterfactual evaluations act as highly eficient teacher-forced passes, allowing us to deeply audit the generative cognitive process without the prohibitive bottleneck of autoregressive decoding.

Armed with this causal probe, our first major contribution is uncovering a pervasive diagnostic illusion that plagues current RAG attribution literature. We demonstrate that the perceived efectiveness of existing attribution metrics is almost entirely a mirage constructed by flawed evaluation datasets. When evaluated on standard of-query distractor pools (passages retrieved for unrelated topics), naive similarity metrics appear near-perfect (AUCs approaching 1.000). However, when forced to distinguish challenging same-query hard negatives, documents that are topically dense but contain no actual answers, traditional proxies completely collapse to random chance. Only our causal probe maintains robust discrimination. By formally calibrating out target-shift artifacts, we establish a fundamental structural property of LLMs: the dilution of attribution across wider contexts is an inherent, inescapable generative behavior. Under strictly controlled diagnostic isolation, this yields a calibrated width elasticity of 0.68 0.02 , acting as an empirical baseline for attention decay.

Having secured a definitively validated measurement instrument, we systematically dismantle the budget allocation dilemma through a deconfounded � � factorial experiment. We discover that simply expanding context width, the current dominant scaling strategy, is an architectural trap fundamentally constrained by IR relevance decay. Wide contexts merely construct a slightly better single answer while leaving massive informational blind spots. Conversely, we establish a robust empirical law of context allocation: dedicating the computational budget to multiple, narrower sequential generations yields transformative absolute gains of 16.8 to 20.5 percentage points in comprehensive portfolio coverage. While this sequential approach inherently incurs higher autoregressive latency and probe overhead, it remains

Manuscript submitted to ACM

![](images/96bea0b83ad433e9ba76b7e1e2546b493319e5299c3d91db3adedd32be4c3eb8.jpg)  
more of the answer space for the same evidence budget

![](images/86a3fd9c76827192d34b3e7986bb56aa41b94ee8f96f7d93bd436ec25556f216.jpg)

(c) The measurement that makes it possible swap the distractors and the ranking inverts  
![](images/66e20d62da8d9ea8e1da48d26301ab3e08309434cc563012ec58f22f369f3a59.jpg)  
padding used to build the evaluation pool  
Fi . 1. The context-allocation roblem in one icture. (a) A fixed inference bud et of � � evidence slots can be acka ed <sub>as one w</sub>id<sub>e con</sub>t<sub>ex</sub>t (�=24<sub>,</sub> �=1) <sub>or as man narrow con</sub>t<sub>ex</sub>t<sub>s</sub> th<sub>a</sub>t <sub>ro</sub>t<sub>a</sub>t<sub>e</sub> th<sub>rou</sub> h f<sub>res</sub>h <sub>ev</sub>id<sub>ence</sub> (�=2<sub>,</sub> �=12)<sub>.</sub> B<sub>o</sub>th <sub>consume</sub> 24 1<sub>re</sub>t<sub>r</sub>i<sub>eve</sub>d d<sub>ocumen</sub>t<sub>s, ye</sub>t th<sub>e secon</sub>d <sub>pac</sub>k<sub>ag</sub>i<sub>ng covers</sub> 0.144 <sub>more o</sub>f th<sub>e answer space</sub> (T<sub>a</sub>bl<sub>e</sub> 4)<sub>.</sub> (b) T<sub>urn</sub>i<sub>ng</sub> th<sub>a</sub>t <sub>a</sub>ll<sub>oca</sub>ti<sub>on</sub> l<sub>aw</sub> i<sub>n</sub>t<sub>o</sub> <sub>a sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>er: on a</sub> f<sub>rozen</sub> h<sub>e</sub>ld<sub>-ou</sub>t f<sub>rame,</sub> Ascp b<sub>ea</sub>t<sub>s every se</sub>l<sub>ec</sub>ti<sub>on-s</sub>t<sub>y</sub>l<sub>e</sub> b<sub>ase</sub>li<sub>ne</sub> b<sub>y</sub> 0.033 t<sub>o</sub> 0.081 <sub>por</sub>tf<sub>o</sub>li<sub>o reca</sub>ll (<sub>a</sub>ll BH � < .001)<sub>;</sub> the two <sub>g</sub>re<sub>y</sub> arms are reference confi<sub>g</sub>urations whose remainin<sub>g</sub> <sub>g</sub>a<sub>p</sub>s are not si<sub>g</sub>nificant (Table 10). (c) None of this is measurable with relevance <sub>p</sub>roxies. Swa<sub>pp</sub>in<sub>g</sub> the <sub>p</sub>addin<sub>g</sub> of the evaluation <sub>p</sub>ool from of-<sub>q</sub>uer<sub>y</sub> distractors to same-<sub>q</sub>uer<sub>y</sub> hard ne<sub>g</sub>atives that <sub>en</sub>t<sub>a</sub>il <sub>no answer</sub> l<sub>eaves</sub> th<sub>e causa</sub>l l<sub>eave-one-ou</sub>t <sub>pro</sub>b<sub>es a</sub>l<sub>mos</sub>t <sub>unc</sub>h<sub>ange</sub>d ( 0.01 AUC)<sub>, w</sub>hil<sub>e</sub> BM25 <sub>an</sub>d <sub>query–</sub>d<sub>ocumen</sub>t <sub>cos</sub>i<sub>ne</sub> f<sub>a</sub>ll b<sub>y more</sub> th<sub>an</sub> 0.5 t<sub>o c</sub>h<sub>ance</sub> (Fi<sub>gure</sub> 3)<sub>.</sub>

the only mechanism capable of breaching the extraction ceiling of monolithic single-pass models, a structural supremacy verified up to the 32B scale.

Finally, we operationalize these conceptual insights into a deployable system architecture. Recognizing that traditional IR algorithms (e.g., MMR) operate as open-loop systems blind to actual generative consumption, we propose a feedbackdriven closed-loop submodular scheduler. By actively reading causal attribution feedback, our scheduler systematically outperforms all seven evaluated selection-style baselines. Furthermore, to combat the LLM’s inherent attention inertia, we augment our architecture with an orthogonal attribution-steered contrastive decoder. Acting as a cognitive override, this micro-level intervention forcefully shifts probability mass away from over-used evidence while maintaining strict plausibility guardrails, delivering mathematically guaranteed orthogonal gains. Ultimately, this framework pioneers the application of inference-time scaling to generative search: by deliberately investing test-time compute into iterative causal orchestration, we bridge the gap between classical search-result diversification and modern LLM pipelines.

Our main conceptual and empirical contributions are summarized as follows:

Exposing the Diagnostic Illusion in Generative Evaluation: We reveal that standard of-query distractor pools artificially inflate relevance proxies to apparent perfection. Utilizing rigorous same-query hard negatives, we prove these proxies fail catastrophically, establishing causal counterfactual probes as the indispensable standard for valid evidence attribution

• <sup>Formalizing</sup> <sup>the</sup> <sup>Dilution</sup> <sup>Law</sup> <sup>of</sup> <sup>Context</sup> <sup>Width:</sup> <sup>We</sup> <sup>resolve</sup> <sup>widespread</sup> <sup>measurement</sup> <sup>artifacts</sup> <sup>in</sup> <sup>RAG</sup> attribution to uncover a fundamental generative property: evidence utilization inevitably dilutes as context expands, yielding a strictly calibrated width elasticity of 0.68 0.02 .

Empirical Laws of Context Allocation and Inference-Time Scaling: Through a deconfounded factorial design, we prove that monolithic context-widening is an architectural trap that hits a rigid cognitive ceiling. Instead, we establish a new paradigm for inference-time scaling: deliberately investing test-time compute across multiple sequential generations drives massive absolute recall surges of 16.8 to 20.5 percentage points, a structural supremacy verified up to the 32B model scale.

A Closed-Loop Orchestration Architecture: Translating theory into practice, we introduce an attributionsteered submodular scheduler. Validated across rigorous cross-task evaluation frames, the architecture systematically dominates classical open-loop baselines. Augmented by a contrastive decoder that acts as a cognitive override against attention inertia, our framework proves that dynamic, feedback-driven context orchestration fundamentally surpasses static context maximization.

## 2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

## 2<sub>.</sub>1 F<sub>rom</sub> S<sub>earc</sub>h R<sub>esu</sub>lt Di<sub>vers</sub>ifi<sub>ca</sub>ti<sub>on</sub> t<sub>o</sub> Di<sub>verse</sub> RAG

Classical information retrieval treats a query as an underspecified expression of intent. To mitigate the risk of returning redundant near-duplicates, ranking models actively trade sheer relevance for novelty and subtopic coverage [10]. This foundational premise birthed a lineage of diversification machinery, including probabilistic, subtopic, axiomatic, and learned paradigms [1, 12, 20, 33, 51, 94, 102, 126]. The line remains actively developed: neural rankers encode diversity greedily with self-attention [92], model candidates at multiple granularities [21], resolve subtopics at passage rather than document level [112], pre-train diversification in a model-agnostic fashion [22], and extend coverage to streaming corpora [69]. Many modern coverage formulations exploit monotone submodularity to inherit rigorous greedy approximation guarantees [28, 70, 90], supported by scalable algorithms and determinantal point processes (DPP) [5, 57, 83, 84]. Consequently, classical metrics implicitly reward aspect coverage and penalize redundancy [16, 101, 135], and are themselves derived from explicit models of how a user consumes a ranking [86, 87]. That answer-bearing evidence is redundantly spread across a corpus—so that coverage, not any single passage, bounds what a system can answer—was also established well before RAG [71].

However, this classical lineage assumes a human consumes the ranked list. Retrieval-augmented generation (RAG) shifts the consumer from a human to a generative model. Recent diversity-aware RAG pipelines attempt to pack distinct information into a single limited prompt window to optimize a comprehensive single answer [98, 123]. Most notably, Carriage [41] explores recipe adaptations on cross-cultural benchmarks [40, 88] using MMR-like penalties and a sliding window. While these adaptations are valuable, our analysis suggests that the sliding window carries the primary efect because it intrinsically increases the distinct documents reaching the generator. Our work formalizes this transition: instead of hedging a single ranked list, we repeatedly query the generator, evaluating how distinct document exposure drives portfolio coverage across sequential readings.

Manuscript submitted to ACM

## 2<sub>.</sub>2 G<sub>enera</sub>ti<sub>ve</sub> C<sub>on</sub>t<sub>ex</sub>t Utili<sub>za</sub>ti<sub>on an</sub>d R<sub>esource</sub> All<sub>oca</sub>ti<sub>on</sub>

Since its inception [61], RAG research has rapidly expanded across dense retrieval, joint pre-training, adaptive orchestration, and graph-based structuring [4, 8, 23, 31, 35, 43, 52, 55, 56, 95, 103, 107, 116, 117, 122, 132, 137], with the retrieval and generation halves each now surveyed at length [14, 66, 91, 139, 142] and with classical first-stage devices such as pseudo-relevance feedback re-examined under dense encoders [62, 114]. To mitigate the burden of massive contexts, techniques such as distribution ensembling, fusion, and prompt compression have been proposed [43, 49, 50, 107, 127], alongside token-eficient agentic pipelines [136], joint optimization of knowledge selection with the reader [108], and explicit memory management for long-running agents [138]. Benchmarks have broadened accordingly, spanning create–read–update–delete task families [77], unified long-context needle-in-a-haystack probes [32], and deployed code assistance [67]. Crucially, almost all these systems optimize a single response rather than investigating what fraction of the retrieved evidence is actually consumed; the sequential regime in which evidence accumulates over successive turns is instead treated separately, as conversational search [85].

Emerging studies reveal that language models do not process retrieved context as perfect pipelines. Accuracy degrades based on document position, irrelevant context induces hallucinations, and efective context windows remain significantly shorter than advertised limits [3, 6, 19, 39, 53, 60, 65, 72, 105, 128, 130, 131]. While prior studies analogize RAG optimization to neural scaling laws [36, 54], they primarily treat context width as the sole scaling variable. In contrast, our deconfounded factorial experiment explicitly asks whether the same retrieved pool should fund a wider single context or be allocated across multiple narrower contexts to expose fresh evidence.

## 2<sub>.</sub>3 At<sub>r</sub>ib<sub>u</sub>ti<sub>on,</sub> F<sub>a</sub>ithf<sub>u</sub>l<sub>ness,</sub> <sub>an</sub>d C<sub>ausa</sub>l M<sub>easuremen</sub>t

Evaluating whether a RAG response is genuinely grounded necessitates rigorous source attribution. The explanation literature relies heavily on local approximations, Shapley values, or input erasure [18, 64, 76, 99, 113], repeatedly cautioning that mere attention scores are fundamentally unreliable proxies for evidence use [38, 45, 104, 125]. Within IR proper, ranking models have been rebuilt to emit extractive rationales precisely so that the evidence a scorer relied on becomes inspectable rather than inferred [59]. Consequently, evaluating generative search encompasses citation frameworks, generative automatic judgements, and fact-checking protocols [7, 30, 42, 47, 73, 78, 79, 81, 82, 89, 96, 133]. ContextCite [17] similarly ablates context to generate sparse linear surrogates, which we explicitly benchmark against in our study.

Simultaneously, reference-free RAG evaluation frameworks [24, 100] rely heavily on LLM-as-a-judge paradigms, despite their documented biases [15, 74, 140]. The IR community has begun mapping where such generated assessment hold: as relevance judgements driving query performance prediction [80], as the basis of entire test collections [118], and as simulators of user behaviour [121]. By utilizing sentence embeddings [97] as a similarity baseline, we demonstrate that standard proxies and judges saturate and fail catastrophically under hard same-query negative populations. Thus, establishing a causal attribution probe is a strict prerequisite for asserting portfolio coverage claims in diverse RAG

Our reliance on constructed ground truth, paired contrasts, and replicated seeds follows a long methodological tradition in IR evaluation. Efectiveness diferences are unstable unless replicate runs and topic-set variation are modelled explicitly [120]; ofline estimates can invert an online verdict when the logging policy confounds the comparison [11, 44]; ofline and online judgements of the same component need not agree [115]; and meta-evaluating the measure itself, rather than only the systems it scores, is the accepted obligation when a new metric is introduced [46, 75].

T<sub>a</sub>bl<sub>e</sub> 1<sub>.</sub> Withi<sub>n-query</sub> <sub>corre</sub>l<sub>a</sub>ti<sub>ons</sub> <sub>across</sub> <sub>sys</sub>t<sub>ems</sub> (�=2395 t<sub>as</sub>k<sub>–mo</sub>d<sub>e</sub>l<sub>–query</sub> <sub>groups,</sub> 23682 <sub>o</sub>b<sub>serva</sub>ti<sub>ons</sub>)<sub>,</sub> <sub>cen</sub>t<sub>ere</sub>d <sub>w</sub>ithi<sub>n</sub> <sub>group.</sub> Lower <sub>p</sub>anel: <sub>p</sub>artial correlations between <sub>p</sub>ortfolio recall and one measure after residualisin<sub>g</sub> a<sub>g</sub>ainst <sub>p</sub>aired covera<sub>g</sub>e or out<sub>p</sub>utdi<sub>vers</sub>it<sub>y. mar</sub>k<sub>s a measure w</sub>h<sub>ose s</sub>i<sub>gn</sub> i<sub>s no</sub>t <sub>cons</sub>t<sub>an</sub>t <sub>across</sub> t<sub>as</sub>k<sub>s, so</sub> it<sub>s poo</sub>l<sub>e</sub>d <sub>va</sub>l<sub>ue s</sub>h<sub>ou</sub>ld <sub>no</sub>t b<sub>e rea</sub>d <sub>a</sub>l<sub>one.</sub>
<table><tr><td colspan="2"></td><td colspan="3">vs. PR@T</td><td colspan="2">vs. groundedness</td></tr><tr><td>Family</td><td>Measure</td><td>n</td><td>r</td><td>p</td><td>r</td><td>p</td></tr><tr><td>coverage</td><td>Evidence coverage rate</td><td>23682</td><td>+0.097</td><td>2.8e-50</td><td>-0.167</td><td>9.7e-148</td></tr><tr><td></td><td>Utilisation concentration</td><td>23682</td><td>-0.023</td><td>3.0e-04</td><td>-0.146</td><td>2.7e-112</td></tr><tr><td>diversity</td><td>Semantic diversity</td><td>23682</td><td>+0.089</td><td>6.1e-43</td><td>-0.111</td><td>9.3e-66</td></tr><tr><td></td><td>Distinct-1</td><td>23682</td><td>+0.106</td><td>3.7e-60</td><td>-0.100</td><td>1.7e-53</td></tr><tr><td></td><td>Distinct-2</td><td>23682</td><td>+0.128</td><td>2.4e-87</td><td>-0.134</td><td>9.4e-96</td></tr><tr><td></td><td>Distinct-3</td><td>23682</td><td>+0.128</td><td>5.2e-87</td><td>-0.131</td><td>7.9e-91</td></tr><tr><td></td><td>Self-BLEU</td><td>23682</td><td>-0.102</td><td>5.1e-56</td><td>+0.161</td><td>1.8e-136</td></tr><tr><td colspan="7">Per-task correlation with PR@T: ASQA / QAMPARI / ELI5 / recipes</td></tr><tr><td></td><td>Evidence coverage rate</td><td></td><td></td><td></td><td>+0.161 / +0.146 / +0.041 / +0.076</td><td></td></tr><tr><td></td><td>Utilisation concentration‡</td><td></td><td></td><td></td><td>-0.014 / +0.014 / +0.038 / -0.209</td><td></td></tr><tr><td></td><td>Semantic diversity</td><td></td><td></td><td></td><td>+0.121 / +0.125 / +0.028 / +0.280</td><td></td></tr><tr><td></td><td>Distinct-1</td><td></td><td></td><td></td><td>+0.160 / +0.139 / +0.022 / +0.284</td><td></td></tr><tr><td></td><td>Distinct-2</td><td></td><td></td><td>+0.179 / +0.144 / +0.041 / +0.295</td><td></td><td></td></tr><tr><td></td><td>Distinct-3</td><td></td><td></td><td>+0.177  / +0.141 / +0.045 / +0.295</td><td></td><td></td></tr><tr><td></td><td>Self-BLEU</td><td></td><td></td><td>-0.155 / -0.044 / -0.054 / -0.284</td><td></td><td></td></tr><tr><td colspan="7">Partial correlations with PR@T</td></tr><tr><td>partial</td><td>Evidence coverage | distinct-2</td><td>23682</td><td>+0.060</td><td>1.6e-20</td><td></td><td></td></tr><tr><td>partial</td><td>Distinct-2 | evidence coverage</td><td>23682</td><td>+0.104</td><td>1.0e-57</td><td></td><td></td></tr></table>

## 2<sub>.</sub>4 I<sub>n</sub>f<sub>erence-</sub>Ti<sub>me</sub> S<sub>ca</sub>li<sub>ng an</sub>d Di<sub>verse</sub> D<sub>eco</sub>di<sub>ng</sub>

Our work inherently connects to mechanisms that control generation diversity and scale test-time compute. Traditional diversity controls operate on the token level, such as temperature, top-�, diverse beam search, generic-response penalties, contrastive representation, and DPP sampling [26, 29, 37, 57, 63, 111, 119, 141]. These are orthogonal to evidence availability; they manipulate textual variety while leaving evidence utilization largely static.

Simultaneously, inference-time scaling studies observe that task coverage (e.g., pass@�) scales smoothly with sample count [9, 13, 109], and allocating test-time compute efectively can often rival model upscaling [58, 124, 134]. While existing techniques contrast expert and amateur distributions [68] or context-aware formulations [106], they uniformly optimize for a single, definitive answer. Our scheduling and decoding interventions contrast over-used versus underused evidence, redistributing grounding across a multi-round portfolio to actively scale evidence coverage, rather than simply sampling identical contexts repeatedly

## 3 P<sub>ro</sub>bl<sub>em</sub> F<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>an</sub>d E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> M<sub>e</sub>t<sub>r</sub>i<sub>cs</sub>

To systematically optimally allocate a retrieval-augmented generation (RAG) system’s inference budget, we must first formalize what the system is trying to achieve and define rigorous metrics to measure its internal behavior. In this section, we define our ultimate end-to-end goal (Portfolio Recall) and our intermediate diagnostic metric (Evidence Coverage Rate), while exposing a critical pitfall in how the NLP community traditionally evaluates diversity.

## 3<sub>.</sub>1 Th<sub>e</sub> E<sub>n</sub>d<sub>-</sub>t<sub>o-</sub>E<sub>n</sub>d G<sub>oa</sub>l<sub>:</sub> F<sub>rom</sub> Si<sub>ng</sub>l<sub>e</sub> A<sub>nswers</sub> t<sub>o</sub> G<sub>enera</sub>ti<sub>ve</sub> P<sub>or</sub>tf<sub>o</sub>li<sub>os</sub>

Classical search engines handle ambiguous queries by returning a diversified list of documents, hedging their bets to cover various possible user intents [92, 112]. Modern RAG systems, however, typically force the generative model Manuscript submitted to ACM

to compress all retrieved evidence into a single monolithic response. For complex queries, this single-pass synthesis inevitably drops critical information.

To resolve this, we formalize the concept of response-portfolio generation. Given a query � and a pool of retrieved documents ${ \mathcal { P } } ,$ the RAG system is allowed a constrained computation budget to produce � distinct sequential responses, denoted as $y _ { 1 } , \ldots , y _ { T }$ . The user views these � responses as a unified portfolio.

The portfolio is considered optimal if the union of these responses covers as much of the underlying truth as possible. We quantify this end-to-end success using Portfolio Recall (PR@�). Let � be the set of gold-standard answer units for the query. PR@� is defined as the fraction of these gold units successfully recovered by any response in the portfolio:

$$
\mathrm { P R } @ T = \frac { 1 } { | A | } \left| \left\{ a \in A : \ \exists t \ \mathrm { s . t . } \ a \ \mathrm { i s \ a s s e r t e d \ i n } \ y _ { t } \right\} \right| .\tag{1}
$$

The mathematical gap between PR@� and the recall of the single best response precisely quantifies the value of sequential diversification.

## 3.2 The Diagnostic Signal: Quantifying Genuine Evidence Coverage

While PR@� measures the final success of the system, it is a “black-box” metric. To actually design an intelligent scheduling algorithm that decides which documents to feed the LLM in the next round, we need to look inside the box: we must measure what fraction of the ofered documents the LLM actually consumed.

Assume for a moment that we possess an attribution matrix $A \in [ 0 , 1 ] ^ { T \times N }$ (we will detail exactly how to construct this matrix using our causal probe in Section 4). In this matrix, each entry $A _ { t d }$ represents the causal utilization score of document � during generation round �.

Using this matrix, we define our ultimate operational metric: the Evidence Coverage Rate (ECR), the generative counterpart of the subtopic recall that classical diversification evaluates under an explicit model of how far a user reads [86, 87], and of the corpus redundancy that bounds what a question-answering system can recover at all [71]. ECR measures extraction eficiency—specifically, what percentage of the unique documents exposed to the model were actually utilized to generate the text. Let $O _ { T } = \cup _ { t = 1 } ^ { T } C _ { t }$ define the unique footprint of documents shown to the generator across all � rounds. We formalize ECR as:

$$
\mathrm { E C R } @ T = \frac { 1 } { | O _ { T } | } \big | \{ d : \exists t , A _ { t d } \geq \theta \operatorname* { m a x } _ { d ^ { \prime } } A _ { t d ^ { \prime } } \} \big | ,\tag{2}
$$

where � is a utilization threshold $( \mathbf { e . g . , } \theta = 0 . 1 )$ . Crucially, by putting the dynamically orchestrated set $\vert O _ { T } \vert$ in the denominator rather than an arbitrary static number, ECR strictly penalizes “lazy” scheduling policies that blindly inject ignored documents into the prompt. It isolates the system’s true scheduling precision.

## 3<sub>.</sub>3 Th<sub>e</sub> Ev<sub>a</sub>l<sub>ua</sub>ti<sub>o</sub>n Tr<sub>ap</sub>: T<sub>e</sub>xt<sub>ua</sub>l Div<sub>e</sub>r<sub>s</sub>it<sub>y</sub> v<sub>s.</sub> Evid<sub>e</sub>n<sub>ce</sub> Div<sub>e</sub>r<sub>s</sub>it<sub>y</sub>

At this point, a natural question arises: why build complex attribution matrices to measure evidence consumption? Why not simply measure how “diferent” the generated texts $y _ { 1 } , \ldots , y _ { T }$ are from each other using standard NLP textual diversity metrics (e.g., Distinct-2 or Semantic Diversity)?

This leads us to a critical methodological trap, which we term the textual diversity confound: the dangerous conflation of how diferently a model speaks with what diferentfacts it actually uses.

If an LLM generates three responses with vastly diferent phrasing but relies on the exact same underlying document, standard NLP metrics will heavily reward the system, creating a fake illusion of knowledge diversity. Our empirical

Manuscript submitted to ACM

![](images/34500f964cc3d561487d733d0ba52b9e8a0a0c3ee8b558922b88ec9451e87754.jpg)  
Fi <sub>.</sub> 2<sub>.</sub> Th<sub>e c</sub>l<sub>ose</sub>d<sub>-</sub>l<sub>oo measuremen</sub>t <sub>an</sub>d <sub>orc</sub>h<sub>es</sub>t<sub>ra</sub>ti<sub>on</sub> i <sub>e</sub>li<sub>ne.</sub> Th<sub>e sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>er</sub> t<sub>rans</sub>f<sub>orms a re</sub>t<sub>r</sub>i<sub>eve</sub>d <sub>oo</sub>l i<sub>n</sub>t<sub>o a cura</sub>t<sub>e</sub>d <sub>con</sub>t<sub>ex</sub>t <sub>er</sub> <sub>genera</sub>ti<sub>on roun</sub>d<sub>.</sub> S<sub>u</sub>b<sub>sequen</sub>tl<sub>y, our causa</sub>l <sub>pro</sub>b<sub>e</sub> i<sub>so</sub>l<sub>a</sub>t<sub>es</sub> t<sub>rue ev</sub>id<sub>ence u</sub>tili<sub>za</sub>ti<sub>on v</sub>i<sub>a</sub> � 1 <sub>para</sub>ll<sub>e</sub>li<sub>za</sub>bl<sub>e</sub> t<sub>eac</sub>h<sub>er-</sub>f<sub>orce</sub>d f<sub>orwar</sub>d 1<sub>passes, completely bypassing autoregressive decoding overhead. These causal signals dynamically populate the utilization matrix,</sub> f<sub>orm</sub>i<sub>ng</sub> th<sub>e exac</sub>t f<sub>ee</sub>db<sub>ac</sub>k l<sub>oop</sub> th<sub>a</sub>t <sub>governs su</sub>b<sub>sequen</sub>t <sub>su</sub>b<sub>mo</sub>d<sub>u</sub>l<sub>ar sc</sub>h<sub>e</sub>d<sub>u</sub>li<sub>ng an</sub>d <sub>a</sub>t<sub>r</sub>ib<sub>u</sub>ti<sub>on-s</sub>t<sub>eere</sub>d <sub>cogn</sub>iti<sub>ve</sub> d<sub>eco</sub>di<sub>ng.</sub>

analysis of 23,682 paired observations (Table 1) confirms this trap. While surface-level textual diversity naturally correlates with final task success (� = 0.128), partial correlation analysis proves it operates almost entirely independently from actual evidence coverage (� = 0.097).

More dangerously, blindly optimizing for this textual variety actively harms system reliability. Our data reveals a severe trade-of: systems that aggressively push for novel wording systematically lose their anchor in the source texts, exhibiting a massive negative correlation with lexical groundedness (� = 0.167). To build a trustworthy diverse RAG system, we cannot optimize for stochastic word-shufling; we must optimize a manipulable IR target.

Meta-Evaluation. To validate that our ECR metric captures genuine, human-aligned evidence utilization rather than statistical noise—the meta-evaluation any newly proposed measure owes its readers [75]—we benchmarked our metric against an independent, identity-blinded LLM-as-a-judge over 858 document-level judgments (Figure 7). The results were decisive: our ECR tracked the judge’s assessment of true informational coverage exceptionally well (� = 0.654). Conversely—and even though generated assessments are otherwise serviceable as relevance labels [80, 118]—when a judge was instructed to rate pure “textual novelty,” it proved fundamentally blind to whether new evidence was actually introduced (� = 0.425).

This dictates a strict prerequisite: without actively parsing the retrieved pool via a rigorous causal instrument, distinguishing genuinely fresh evidence extraction from stochastic paraphrasing is mathematically impossible. This justifies the necessity of our causal attribution probe, which we introduce next.

Manuscript submitted to ACM

## 4 Th<sub>e</sub> C<sub>ausa</sub>l At<sub>r</sub>ib<sub>u</sub>ti<sub>on</sub> P<sub>ro</sub>b<sub>e an</sub>d It<sub>s</sub> V<sub>a</sub>lid<sub>a</sub>ti<sub>on</sub>

Having established in Section 3 that diverse textual output does not guarantee diverse evidence utilization, we face a fundamental methodological bottleneck: we must operationalize a reliable measurement instrument before any context allocation rules can be evaluated. Because standard embedding similarity and lexical overlap are inherently confounded by query relevance, we construct an intervention-based causal probe. In this section, we formalize this instrument, rigorously validate its discriminative limits, and ultimately deconstruct a systemic evaluation flaw in current generative attribution literature.

## 4<sub>.</sub>1 Th<sub>e</sub> C<sub>ausa</sub>l I<sub>ns</sub>t<sub>rumen</sub>t<sub>:</sub> C<sub>oun</sub>t<sub>er</sub>f<sub>ac</sub>t<sub>ua</sub>l S<sub>ens</sub>iti<sub>v</sub>it<sub>y</sub>

Traditional attribution metrics operate observationally, measuring superficial semantic overlap between the prompt and the response. We argue that genuine utilization can only be isolated via intervention. For an already-produced generation $y _ { t }$ derived from context $C _ { t }$ at round $t ,$ we pose a strict counterfactual: how much less likely would the exact realized generation become ifdocument � were ablatedfrom the context?

We quantify this counterfactual sensitivity via the per-token drop in log-likelihood:

$$
a _ { t } ^ { \mathrm { r a w } } ( d ) = \frac { 1 } { | y _ { t } | } \biggl [ \log \phi _ { \theta } \bigl ( y _ { t } \mid q , C _ { t } \bigr ) - \log \phi _ { \theta } \bigl ( y _ { t } \mid q , C _ { t } \setminus \{ d \} \bigr ) \biggr ] ,\tag{3}
$$

normalized over the context as $a _ { t } ( d ) \propto [ a _ { t } ^ { \operatorname { r a w } } ( d ) ] _ { + } . \mathrm { ~ A ~ }$ positive value dictates that deleting document � causally reduces the likelihood of the generated text, establishing structural reliance. To discretize this continuous utilization into binary counts for our live scheduling matrix, we apply an operational, free-generation threshold $\tau _ { \mathrm { f r e e } } ( k ) = 0 . 5 5 5 k ^ { - 0 . 6 3 3 }$ . This specific decay scaling mathematically accounts for the natural text-lengthening and hedging behaviors LLMs exhibit when fed wider contexts (a confound we explicitly deconstruct in Section 4.4).

A hallmark of a practical IR measurement framework is computational feasibility, as visually mapped in our end-toend pipeline (Figure 2). Because our probe operates on a fixed response �<sub>�</sub>, the ablation process bypasses the prohibitive autoregressive decoding bottleneck entirely. Each counterfactual evaluation constitutes a single, highly parallelizable teacher-forced forward pass. Furthermore, documents maintain immutable semantic identifiers across all counterfactual passes, ensuring that deletion does not artificially conflate lost evidence with the mere renumbering of surface-form citations.

## 4<sub>.</sub>2 D<sub>econs</sub>t<sub>ruc</sub>ti<sub>ng</sub> th<sub>e</sub> Di<sub>agnos</sub>ti<sub>c</sub> Ill<sub>us</sub>i<sub>on v</sub>i<sub>a</sub> C<sub>on</sub>t<sub>ro</sub>ll<sub>e</sub>d P<sub>oo</sub>l<sub>s</sub>

Evaluating whether a model utilized specific evidence is notoriously confounded by the model’s internal parametric knowledge. To establish absolute causal ground truth—in the same spirit as diagnostic test collections built to isolate individual retrieval heuristics rather than score systems end to end [27]—we construct controlled retrieval pools preserving exactly $m = 2$ documents that attest to diferent gold answers (the necessary documents), padded to various context widths.

Testing against these controlled pools exposes a profound systemic flaw in current RAG evaluations (Figure 3(a,b)). Initially, under standard of-query padding (distractors retrieved for completely unrelated intents), traditional IR proxies appear scientifically flawless. At context width $k = 2 4$ , query–document cosine and BM25 achieve near-perfect AUCs approaching 1.000, while our deletion leave-one-out (LOO) probe trails at 0.829.

However, this apparent precision is a diagnostic illusion driven entirely by construction leakage. Of-query padding provides trivial negative examples, exactly the type of topically disjoint documents that query relevance algorithms

![](images/4cddb38d55efbabd6bb482b2b675047e97a30a27ad7376934312f62b53ad5faf.jpg)

![](images/6d6c46220ebd843e3db134f40591e3dc2bdabb2689aa3ab527782296c1a7f532.jpg)

![](images/b7911b126834ba538840e72599a58a68a79a7d174cae52900b703a266ae08ea2.jpg)

![](images/241dc8347f4d8a41acf5e0f0d9506ea743937b714f9679070c5da5ff797f7897.jpg)

![](images/baed9ab43a52f9eb84c0119602bcf0c97e51f85bb5888117b2e702238e62c736.jpg)  
panels (a)–(b) deletion LOO in-place LOO output–document cosine query–document cosine BM25 output overlap in (c) and (e): circles are deletion, squares in-place replacement

Fi<sub>g</sub>. 3. Probe validation and controls on constructed <sub>p</sub>ools. (a)–(b) AUC for recoverin<sub>g</sub> the desi<sub>g</sub>ned answer-bearin<sub>g</sub> documents, b<sub>y</sub> <sub>con</sub>t<sub>ex</sub>t <sub>w</sub>idth<sub>,</sub> <sub>un</sub>d<sub>er</sub> <sub>o</sub>f<sub>-query</sub> <sub>an</sub>d <sub>same-query</sub> <sub>pa</sub>ddi<sub>ng;</sub> th<sub>e</sub> <sub>or</sub>d<sub>er</sub>i<sub>ng</sub> <sub>o</sub>f <sub>causa</sub>l <sub>es</sub>ti<sub>ma</sub>t<sub>ors</sub> <sub>an</sub>d <sub>re</sub>l<sub>evance</sub> <sub>prox</sub>i<sub>es</sub> <sub>reverses</sub> <sub>w</sub>ith th<sub>e</sub> 1<sub>negative population. (c) Leave-one-out against ContextCite by padding regime, fixed protocol. (d) cover@� gain over a random</sub> <sub>ran</sub>ki<sub>ng w</sub>ith 95% <sub>c</sub>l<sub>us</sub>t<sub>er-</sub>b<sub>oo</sub>t<sub>s</sub>t<sub>rap</sub> i<sub>n</sub>t<sub>erva</sub>l<sub>s, a</sub> t<sub>arge</sub>t i<sub>nvar</sub>i<sub>an</sub>t t<sub>o w</sub>hi<sub>c</sub>h <sub>answer-equ</sub>i<sub>va</sub>l<sub>en</sub>t d<sub>ocumen</sub>t <sub>was</sub> l<sub>a</sub>b<sub>e</sub>ll<sub>e</sub>d <sub>necessary.</sub> (<sub>e</sub>) T<sub>rans</sub>f<sub>er o</sub>f th<sub>e o</sub>f<sub>-query</sub> 95th<sub>-percen</sub>til<sub>e opera</sub>ti<sub>ng po</sub>i<sub>n</sub>t t<sub>o same-query passages</sub> th<sub>a</sub>t <sub>con</sub>t<sub>a</sub>i<sub>n an</sub>d <sub>en</sub>t<sub>a</sub>il <sub>no go</sub>ld <sub>answer; on</sub>l<sub>y</sub> fixed-tar<sub>g</sub>et curves are false-<sub>p</sub>ositive rates<sub>,</sub> since under free <sub>g</sub>eneration the <sub>p</sub>addin<sub>g</sub> ma<sub>y</sub> le<sub>g</sub>itimatel<sub>y</sub> sha<sub>p</sub>e the res<sub>p</sub>onse.

rank last by definition. When we switch the padding to same-query hard negatives (distractors that are topically dense and highly relevant to the query, but entail no actual gold answer aliases), the discriminative hierarchy completely collapses.

Faced with these hard negatives, traditional metrics sufer a systemic failure. BM25 and query–document cosine degrade to random chance on narrow contexts (AUCs of 0.444 and 0.484 at � = 3, respectively). In stark contrast, our deletion LOO probe remains highly robust, resisting the topical confusion to maintain an AUC of 0.876 at � = 3 and 0.824 at � = 24. This definitive reversal establishes an absolute methodological rule: of-query distractor pools are fundamentally inadequate for validating context-attribution methods. Causal probes are uniquely equipped to isolate generative utilization amidst dense topical relevance.

## 4<sub>.</sub>3 Suficient Set Covera<sub>g</sub>e A<sub>g</sub>ainst Advanced Estimators

Having dismissed basic relevance proxies as unreliable under topical density, we benchmark our causal probe against advanced attribution algorithms on redundant padding pools, where additive methods historically struggle to divide credit among interchangeable substitutes. To evaluate strict informational utility, we adopt a permutation-invariant cover@� target: what fraction of the designed answer set do the top-� ranked documents collectively attest?

As illustrated in Figure 3(d), under a strict diagnostic protocol spanning 1, 200 conditions, relevance proxies fail entirely to identify the suficient set (e.g., BM25 marginally degrades random ranking by 0.014, � = 0.48). Conversely, Manuscript submitted to ACM

![](images/e226abd70312cb226d7186ec0b2e9ff47df27d61570a21138daf79e8ad5fa0b6.jpg)

(b) by redundancy stratum  
![](images/f172cd2ce8a77d5ea3baf0506fb554e30360cd24747c31e51d0494bd8806001f.jpg)  
protocol-clean sensitivity variant protocol-contaminated  
Fi<sub>g.</sub> 4<sub>.</sub> Width <sub>e</sub>l<sub>as</sub>ti<sub>c</sub>it<sub>y o</sub>f th<sub>e</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>e</sub>d <sub>u</sub>tili<sub>sa</sub>ti<sub>on s</sub>t<sub>a</sub>ti<sub>s</sub>ti<sub>c</sub> �<sub>, w</sub>ith 95% i<sub>n</sub>t<sub>erva</sub>l<sub>s.</sub> (<sub>a</sub>) B<sub>y ca</sub>lib<sub>ra</sub>ti<sub>on cr</sub>it<sub>er</sub>i<sub>on:</sub> th<sub>res</sub>h<sub>o</sub>ld<sub>s</sub> th<sub>a</sub>t d<sub>o</sub> <sub>no</sub>t th<sub>emse</sub>l<sub>ves</sub> <sub>s</sub>h<sub>r</sub>i<sub>n</sub>k <sub>w</sub>ith <sub>w</sub>idth <sub>agree</sub> <sub>on</sub> th<sub>e</sub> <sub>pro</sub>t<sub>oco</sub>l<sub>-c</sub>l<sub>ean</sub> <sub>va</sub>l<sub>ue,</sub> <sub>w</sub>h<sub>ereas</sub> f<sub>ree-genera</sub>ti<sub>on</sub> <sub>an</sub>d th<sub>res</sub>h<sub>o</sub>ld<sub>-</sub>f<sub>ree</sub> <sub>cr</sub>it<sub>er</sub>i<sub>a</sub> <sub>es</sub>ti<sub>ma</sub>t<sub>e</sub> <sub>a</sub> 1<sub>di</sub>f<sub>erent quantity. (b) By natural-pool redundancy stratum: the decline survives in every low-redundancy stratum, so redundanc</sub> <sub>mo</sub>d<sub>u</sub>l<sub>a</sub>t<sub>es</sub> b<sub>u</sub>t d<sub>oes no</sub>t <sub>exp</sub>l<sub>a</sub>i<sub>n</sub> it<sub>.</sub>

counterfactual estimators successfully isolate the underlying drivers of the generation. Deletion LOO covers 0.792 of the answer set, yielding a massive 0.144 absolute gain over a random ranking (� < 0.001). In-place LOO replacement and the learned surrogate ContextCite [17] similarly yield robust gains of 0.134 and 0.105. Deletion LOO’s coverage gain remains globally positive and significant across all context widths, proving that counterfactual scoring uniquely identifies a compact suficient set even under heavy informational redundancy.

## 4<sub>.</sub>4 I<sub>so</sub>l<sub>a</sub>ti<sub>ng</sub> C<sub>on</sub>f<sub>oun</sub>d<sub>ers:</sub> C<sub>a</sub>lib<sub>ra</sub>ti<sub>on an</sub>d th<sub>e</sub> Dil<sub>u</sub>ti<sub>on</sub> L<sub>aw</sub>

Before deploying this instrument to audit system budgets, we must systematically eliminate potential mechanical confounders that could pollute the sensitivity signal. First, we investigated whether simply removing a document artificially inflates score drops via subsequent text positional shifting. By comparing pure deletion against in-place replacement using length-matched neutral text, we observed that the dynamic elasticities remain nearly identical ( 0.104 versus 0.112), confirming that positional displacement does not artificially drive the causal signal.

Second, we observed that apparent limitations in LOO calibration are actually artifacts of protocol shifts rather than probe failure. Under a free-generation protocol, feeding wider contexts to an LLM systematically induces longer, more hedged text. This behavioral shift shrinks per-token likelihood diferences for reasons entirely disjoint from actual evidence utilization, driving apparent false-positive rates up to 37%. However, when we enforce a strict fixed-target protocol where the generated text is held constant, true false-positive rates organically converge to a nominal, highly stable margin (explicitly mapped in Figure 3).

This rigorous protocol isolation allows us to extract a pure, structural signal from the noise. We establish a strictly calibrated width elasticity of 0.68 0.02 for generative attribution (Figure 4(a)). This verifies that the dilution of attention across wider contexts is a fundamental generative property rather than a measurement error. Equipped with a rigorously validated, confounding-free measurement instrument, we are now scientifically positioned to empirically evaluate the precise laws governing context budget allocation.

T<sub>a</sub>bl<sub>e</sub> 2<sub>.</sub> P<sub>o</sub>li<sub>cy</sub> d<sub>epen</sub>d<sub>ence w</sub>ith <sub>one or</sub>di<sub>nary</sub> d<sub>eco</sub>d<sub>er.</sub> All <sub>arms use</sub> th<sub>e same exp</sub>li<sub>c</sub>it <sub>samp</sub>l<sub>er;</sub> f<sub>u</sub>ll <sub>cus</sub>t<sub>om</sub> d<sub>eco</sub>d<sub>er exc</sub>l<sub>u</sub>d<sub>e</sub>d<sub>.</sub> ECR <sub>uses</sub> fi<sub>xe</sub>d<sub>-response</sub> th<sub>res</sub>h<sub>o</sub>ld <sub>an</sub>d <sub>equa</sub>l t<sub>as</sub>k <sub>we</sub>i<sub>g</sub>ht<sub>s.</sub> A<sub>cross</sub> th<sub>e</sub> 8 <sub>eva</sub>l<sub>ua</sub>t<sub>e</sub>d <sub>po</sub>li<sub>c</sub>i<sub>es,</sub> ECR <sub>spans a range o</sub>f 0<sub>.</sub>251<sub>, w</sub>ith th<sub>e vas</sub>t majority o<sup>f</sup> paired ECR contrasts surviving strict BH-FDR correction. O<sup>f</sup>ered is a treatment, not an outcome.
<table><tr><td>policy</td><td>ECR</td><td>used</td><td>offered</td><td>PR@T</td></tr><tr><td>AsCP scheduler</td><td>0.626</td><td>6.76</td><td>10.55</td><td>0.300</td></tr><tr><td>deep rotation</td><td>0.375</td><td>9.37</td><td>25.00</td><td>0.303</td></tr><tr><td>vanilla RAG</td><td>0.578</td><td>2.89</td><td>5.00</td><td>0.237</td></tr><tr><td>MMR</td><td>0.587</td><td>3.13</td><td>5.28</td><td>0.239</td></tr><tr><td>DPP-RAG</td><td>0.526</td><td>4.71</td><td>8.83</td><td>0.274</td></tr><tr><td>xQuAD</td><td>0.571</td><td>2.85</td><td>5.00</td><td>0.238</td></tr><tr><td>PM-2-RAG</td><td>0.571</td><td>2.85</td><td>5.00</td><td>0.228</td></tr><tr><td>CARRIAGE</td><td>0.466</td><td>4.75</td><td>10.17</td><td>0.276</td></tr></table>

## 5 Th<sub>eore</sub>ti<sub>ca</sub>l B<sub>oun</sub>d<sub>s an</sub>d E<sub>mp</sub>i<sub>r</sub>i<sub>ca</sub>l L<sub>aws o</sub>f E<sub>v</sub>id<sub>ence</sub> C<sub>onsump</sub>ti<sub>on</sub>

Before architecting complex scheduling algorithms, we must understand the fundamental rules governing how a generative model consumes evidence across multiple rounds. By establishing an idealized mathematical baseline and contrasting it against the rigorous empirical laws extracted via our causal probe, we expose the exact generative bottlenecks that mandate intelligent context allocation.

## 5<sub>.</sub>1 Th<sub>e</sub> Id<sub>ea</sub>li<sub>ze</sub>d B<sub>ase</sub>li<sub>ne</sub> <sub>vs.</sub> G<sub>enera</sub>ti<sub>ve</sub> R<sub>ea</sub>lit<sub>y</sub>

To systematically model expected evidence coverage, we can first construct a simplified “toy model.” Imagine an idealized LLM that acts as a perfect, uniform consumer of information. Suppose it utilizes any given document independently with a fixed probability � every time it sees it in the prompt.

Under a constrained budget of context width � and generation rounds � , the total number of “document slots” available is ��. Mathematically (proven formally in Appendix A), the expected size of the utilized evidence set <sub>�</sub> is strictly bounded:

$$
\mathbb { E } | \mathcal { U } _ { T } | \leq q k T .\tag{4}
$$

According to this probabilistic baseline, the absolute maximum coverage is achieved if and only if no document is ever repeated across rounds. This suggests a naive conclusion: a simple “open-loop” policy that blindly rotates fresh documents into the prompt every round should be perfectly optimal, rendering complex feedback mechanisms unnecessary.

The Reality Check: However, deploying our causal probe on real-world generative data completely shatters this idealized assumption. LLMs absolutely do not operate as passive, uniform consumers.

Empirical analysis reveals that actual generative utilization is highly clustered and sufers from severe “attention inertia”: if an LLM fixates on a specific concept in round 1, it becomes structurally biased to ignore new, conflicting information in round 2 (diverging from the independence assumption with $\ t p = 6 \times 1 0 ^ { - 3 1 } )$ . Furthermore, pure arithmetic slot-counting ignores relevance density. If a naive rotation policy blindly pushes deeper into the retrieved ranking, it ends up feeding the LLM low-quality “trash” documents, actively degrading task performance.

These systematic divergences dictate that optimal context allocation cannot rely on abstract, blind rotation. It must be governed by the empirical laws of actual generative behavior.

Manuscript submitted to ACM

## 5<sub>.</sub>2 E<sub>mp</sub>i<sub>r</sub>i<sub>ca</sub>l L<sub>aw</sub> I<sub>:</sub> Th<sub>e</sub> P<sub>ower o</sub>f A<sub>c</sub>ti<sub>ve</sub> O<sub>rc</sub>h<sub>es</sub>t<sub>ra</sub>ti<sub>on</sub>

Because real LLM consumption is stubborn and context-dependent, we must establish exactly how front-end document scheduling alters downstream generative reliance. Testing distinct scheduling policies under an identical generation budget (� = 5), we observe profound variance in actual utilization.

As detailed in Table 2, the Evidence Coverage Rate (ECR) spans a massive range, proving that document exposure is a highly manipulable treatment, not a static outcome. For instance, the naive “deep rotation” policy—which blindly pushes fresh documents without tracking if they are useful—passively achieves an ECR of only 0.375. It squanders the majority of its contextual bandwidth on ignored evidence.

In stark contrast, intelligent orchestration drastically alters this consumption pattern. By utilizing causal feedback to penalize redundant information, our proposed submodular scheduler (Ascp) explicitly forces the model to ingest fresh evidence, driving the ECR up to a dominant 0.626. This proves our first empirical law: to maximize the generative utility of a retrieved pool, the system must actively steer the context policy using feedback, rather than passively rotating documents.

## 5<sub>.</sub>3 E<sub>mp</sub>i<sub>r</sub>i<sub>ca</sub>l L<sub>aw</sub> II<sub>:</sub> Th<sub>e</sub> Dil<sub>u</sub>ti<sub>on</sub> L<sub>aw o</sub>f C<sub>on</sub>t<sub>ex</sub>t Width

If intelligent scheduling is required at a fixed width, what happens if we simply bypass the problem by expanding the context window to fit all documents at once?

To answer this, we return to the width elasticity of 0.68 0.02 established via our causal probe (Section 4.4). The massive gap between a flat elasticity (which would imply perfect attention capacity) and our rigorously calibrated negative slopes (centering around 0.68 under protocol isolation) exposes a severe cognitive limitation.

While 0.68 represents our canonical estimate under strictly isolated diagnostic protocols, it is crucial to recognize that the exact coeficient is modulated by operational factors such as task complexity and intrinsic pool redundancy. For instance, depending on the severity of answer-attestation redundancy within the retrieved pool, the empirical slope varies between 0.43 and 0.67 (detailed extensively in Appendix B.2). However, the overarching generative physics remain absolute: across all evaluated strata, models, and semantic granularities, the elasticity remains profoundly negative. The dilution law is defined not by a singular universal constant, but by the inescapable sub-linear decay of evidence utilization as context expands.

This leads to our second empirical law: expanding the context width aggressively dilutes the magnitude of individual document contributions. Because the prompts in our experimental grid peak at 5, 485 tokens—safely below the LLM’s absolute hardware limits—this 0.68 dilution is not a hardware truncation artifact. It is a fundamental generative constraint. As the set of provided evidence grows, the LLM’s attention is inherently fractured.

Taken together, these two laws unequivocally dictate the design of our architecture: since monolithic wide contexts inevitably dilute attention (Law II), the optimal strategy is to break the budget into narrower sequential windows, actively steered by causal feedback to maximize extraction eficiency (Law I).

## 6 S<sub>y</sub>stem Architecture: The Closed-Loo<sub>p</sub> Orchestration Framework

Guided by the empirical laws established in Section 5—specifically that wide contexts dilute attention and optimal extraction requires active, multi-round scheduling—we now design a multi-tiered architecture to operationalize these findings. Rather than treating the RAG pipeline as a static, open-loop black box, we introduce a closed-loop system that dynamically orchestrates what evidence the LLM sees (the Scheduler) and how forcefully it extracts it (the Decoder).

Manuscript submitted to ACM

## 6.1 The Baseline: O<sub>p</sub>en-Loo<sub>p</sub> Context Rotation (Rotate)

Before introducing our intelligent system, we define the structural baseline: Rotate. This is a pure throughpu mechanism that blindly cycles disjoint windows down the retrieved ranking. For instance, round 1 exposes ranks 1 . . . �, round 2 exposes ranks $k + 1 \ldots 2 k .$ , and so on.

While Rotate guarantees that �� distinct documents are physically exposed to the LLM, it operates completely “open-loop.” It has no idea if the LLM actually utilized the top documents, nor does it penalize redundant information. Because relevance density sharply decays down a ranked list, this blind rotation aggressively squanders its budget on lower-ranked, noisy tail documents.

## 6.2 The Brain: Feedback-Driven Submodular Schedulin<sub>g</sub> (Ascp)

To optimize document allocation under concentrated relevance, we introduce Ascp, a dynamic scheduling engine that acts as the system’s brain.

Instead of blindly feeding documents, Ascp groups the retrieved documents into semantic clusters (knowledge facets). Crucially, after each generation round, it reads the causal utilization feedback from our LOO probe to see exactly which documents—and consequently, which knowledge facets—the LLM has already successfully consumed.

We formulate the next round’s context selection as a greedy maximization of a monotone submodular objective (formalized in Appendix A.1). In simple terms, submodularity mathematically enforces a “diminishing returns” penalty: if the causal probe reports that a specific knowledge facet has already been deeply utilized in round 1, the scheduler aggressively discounts any remaining documents belonging to that same facet.

This mechanism actively shifts the ofered context toward unexplored, fresh evidence. Our rigorous ablation studies (detailed later in Section 8.5) prove that this dynamic discounting is strictly dependent on our causal probe. Swapping our causal feedback for a standard embedding-similarity proxy causes the scheduling gains to completely collapse, proving that true causal sensitivity is the indispensable engine driving this scheduler.

## 6<sub>.</sub>3 Th<sub>e</sub> E<sub>n</sub>f<sub>orcer:</sub> At<sub>r</sub>ib<sub>u</sub>ti<sub>on-</sub>St<sub>eere</sub>d C<sub>on</sub>t<sub>ras</sub>ti<sub>ve</sub> D<sub>eco</sub>di<sub>ng</sub>

While the submodular scheduler dictates what enters the context window, we face a secondary generative hurdle: the LLM’s microscopic attention inertia. Even when the scheduler brilliantly provides a prompt full of fresh evidence, LLMs frequently fixate on familiar concepts they already generated, resisting the integration of novel facts.

To break this generative stagnation, we intervene directly inside the LLM’s generation process. After the initial round, we define an “over-used” set of documents � (those the probe flagged as heavily utilized). During subsequent rounds, we dynamically decode the text using a targeted contrastive distribution:

$$
\ell ^ { \prime } = \ell ( \cdot \mid q , C ) + \alpha \left[ \ell ( \cdot \mid q , C \setminus O ) - \ell ( \cdot \mid q , O ) \right] .\tag{5}
$$

In plain English, this subtractive formula acts as a cognitive override. It systematically subtracts probability mass from words associated with the over-used documents, and shifts that mass toward words supported by the fresh, under-used evidence.

To prevent the LLM from hallucinating when pushed away from its default distribution, this shift is strictly bounded by an adaptive-plausibility constraint (APC) [68]. Ultimately, this micro-level intervention guarantees that the fresh evidence curated by the scheduler is forcefully and safely extracted into the final portfolio.

Manuscript submitted to ACM

T<sub>a</sub>bl<sub>e</sub> 3<sub>.</sub> R<sub>e</sub>l<sub>evance</sub> d<sub>ens</sub>it<sub>y</sub> $\rho ( i ) \colon$ <sub>pro</sub>b<sub>a</sub>bilit<sub>y re</sub>t<sub>r</sub>i<sub>eve</sub>d <sub>ran</sub>k � <sub>a</sub>t<sub>es</sub>t<sub>s a go</sub>ld <sub>answer,</sub> f<sub>rom go</sub>ld <sub>anno</sub>t<sub>a</sub>ti<sub>ons over</sub> 200 <sub>quer</sub>i<sub>es</sub>/t<sub>as</sub>k<sub>.</sub> N<sub>ew</sub> h<sub>ea</sub>d<sub>-</sub>5 <sub>s</sub>h<sub>are coun</sub>t<sub>s on</sub>l <sub>answer mass no</sub>t <sub>a</sub>t<sub>es</sub>t<sub>e</sub>d <sub>ear</sub>li<sub>er.</sub> ELI5 <sub>om</sub>it<sub>e</sub>d<sub>:</sub> f<sub>ree-</sub>t<sub>ex</sub>t <sub>c</sub>l<sub>a</sub>i<sub>ms</sub> l<sub>ac</sub>k <sub>ran</sub>k<sub>-</sub>l<sub>eve</sub>l l<sub>ex</sub>i<sub>ca</sub>l <sub>a</sub>t<sub>es</sub>t<sub>a</sub>ti<sub>on.</sub>
<table><tr><td>Task</td><td>gold units per query</td><td> $\rho ( 1 . . 5 )$ </td><td> $\rho ( 1 1 . . 3 0 )$ </td><td>head-5 share head-5 share dlog (any)</td><td>(new)</td><td> $\rho / \mathrm { d } \log i$ </td></tr><tr><td>ASQA</td><td>3.4</td><td>0.469</td><td>0.211</td><td>0.290</td><td>0.713</td><td>-0.40</td></tr><tr><td>QAMPARI</td><td>13.6</td><td>0.295</td><td>0.170</td><td>0.247</td><td>0.398</td><td>-0.27</td></tr><tr><td>RECIPES</td><td>20.7</td><td>0.890</td><td>0.870</td><td>0.170</td><td>0.391</td><td>-0.01</td></tr></table>

## 6<sub>.</sub>4 O<sub>pera</sub>ti<sub>ona</sub>l Vi<sub>a</sub>bilit<sub>y an</sub>d C<sub>ompu</sub>t<sub>a</sub>ti<sub>ona</sub>l O<sub>ver</sub>h<sub>ea</sub>d

A persistent concern with multi-round RAG and causal probing is deployment latency. However, our architecture is meticulously designed for operational eficiency.

The greedy submodular selection operates in $O ( | \mathcal { P } | k Z )$ time, constituting a negligible microsecond overhead. More importantly, while our causal attribution probe requires �  1 forward passes per generation, these are executed as teacher-forced passes on an already-generated text. By completely bypassing the prohibitive autoregressive decoding bottleneck, these passes can be heavily parallelized. The system achieves deep causal introspection and intelligent multi-round orchestration while maintaining strict temporal viability for complex search tasks.

## 7 Ex<sub>p</sub>erimental Methodolo<sub>gy</sub> and Setu<sub>p</sub>

To isolate the precise causal efects of budget allocation and context scheduling, we enforce a highly controlled, zero-leakage evaluation framework. Across all comparative experiments, the foundational RAG pipeline components, specifically the retriever architecture and the retrieved document pools, are strictly frozen. Consequently, all evaluated systems are exposed to the exact same raw evidence, guaranteeing that performance deltas are directly and exclusively attributable to scheduling logic, instruction prompting, and decoding interventions

## 7<sub>.</sub>1 T<sub>as</sub>k<sub>s an</sub>d E<sub>v</sub>id<sub>ence</sub> P<sub>oo</sub>l<sub>s</sub>

We evaluate our portfolio-generation framework across three rigorous benchmarks explicitly designed to necessitate diverse, multi-faceted answer coverage, supplemented by an open-domain stress test:

ASQA [110]: A challenging dataset of ambiguous factoid questions requiring the synthesis of multiple disambiguated sub-answers (exhibiting a concentrated head-5 relevance share of 0.713, detailed in Table 3).

QAMPARI [2]: A broad answer-set generation task averaging 21 annotated entities per query, of which 13.6 are empirically attested in the retrieval pools. It presents a highly dispersed answer distribution (Table 3), testing extreme generative recall.

• <sup>ELI5</sup> <sup>[25]:</sup> <sup>An</sup> <sup>explanatory</sup> <sup>QA</sup> <sup>benchmark</sup> <sup>evaluated</sup> <sup>against</sup> <sup>claim</sup> <sup>sentences.</sup> <sup>To</sup> <sup>ensure</sup> <sup>rigorous</sup> <sup>semantic</sup> evaluation, we strictly match ELI5 claims via NLI cross-encoder entailment rather than brittle string containment.

• <sup>Cross-Cultural</sup> <sup>Recipe</sup> <sup>Adaptation:</sup> <sup>Detailed</sup> <sup>extensively</sup> <sup>in</sup> <sup>Appendix</sup> <sup>D,</sup> <sup>this</sup> <sup>serves</sup> <sup>as</sup> <sup>a</sup> <sup>non-English,</sup> <sup>open-</sup> <sup>domain</sup> <sup>stress</sup> <sup>test</sup> <sup>featuring</sup> <sup>an</sup> <sup>almost</sup> <sup>perfectly</sup> <sup>flat</sup> <sup>relevance</sup> <sup>density</sup> <sup>(log-log</sup> <sup>slope</sup> −<sup>0.01,</sup> <sup>Table</sup> <sup>3)</sup> <sup>for</sup> <sup>optimizing</sup> broad structural coverage beyond traditional QA paradigms.

For ASQA, QAMPARI, and ELI5, we adopt the standardized ALCE retrieval pools [30] (GTR for ASQA/QAMPARI, BM25 for ELI5), strictly preserving the exact top �=30 passages to ensure universal evidence parity. The HotpotQA extension used in the decoupling analysis (Table 8) follows the identical retrieval and truncation pipeline over its own corpus, likewise preserving the top �=30 passages so that every task enters the factorial under matched evidence parity

The deep 400-passage extensions utilized for extreme throughput limits preserve the original top-30 scores and ranks, maintaining absolute structural integrity for all � = 5 control conditions.

## 7<sub>.</sub>2 S<sub>y</sub>stems and Baselines Com<sub>p</sub>ared

We comprehensively benchmark our proposed architecture against a wide spectrum of classical IR diversification algorithms and state-of-the-art RAG baselines. To ensure absolute hardware and computational equivalency, all held-out comparisons rigorously equalize generation capacity: enforcing �=5, �=5, temperature 0.7, top-� = 1, disabled top-� filtering, and a strict repetition penalty of 1.

Static Schedulers (Single-Context Paradigms). These classical paradigms reuse one static context, mathematically upper-bounding distinct exposure to � documents regardless of generation rounds �:

Vanilla RAG: Naive top-� relevance sampling, serving as the standard industry baseline.

• <sup>MMR-RAG</sup> <sup>[10]:</sup> <sup>Re-ranks</sup> <sup>documents</sup> <sup>by</sup> <sup>aggressively</sup> <sup>balancing</sup> <sup>query</sup> <sup>relevance</sup> <sup>against</sup> <sup>content</sup> <sup>redundancy,</sup> mimicking deployed diverse RAG systems.

xQuAD-RAG [102] & PM-2-RAG [20]: Explicit intent-aware diversification algorithms operating over the exact same induced semantic facets utilized by our own scheduler, isolating the exact gain of our causal feedback.

Sequential Schedulers (Dynamic-Context Paradigms). These state-of-the-art frameworks intelligently mutate the context across generation rounds:

DPP-RAG: Samples diverse contexts via greedy MAP inference over a quality-weighted determinantal kernel.

• <sup>Carriage</sup> <sup>[41]:</sup> <sup>A</sup> <sup>cutting-edge</sup> <sup>diverse</sup> <sup>RAG</sup> <sup>pipeline</sup> <sup>integrating</sup> <sup>output-aware</sup> <sup>MMR,</sup> <sup>sequential</sup> <sup>prompt</sup> <sup>listings,</sup> and a sliding window. We evaluate both its main configuration and a restricted Carriage-narrow variant.

Our Orchestration Interventions. We deploy Rotate (cycling disjoint windows to isolate pure throughput efects without feedback) and Ascp (our causal-feedback-driven submodular scheduler optimizing Eq. (6)). For pristine causal isolation, the �  � factorial grid strictly enforces ordinary, unsteered generation on both sides of every measured contrast.

## 7<sub>.</sub>3 G<sub>enera</sub>t<sub>ors an</sub>d E<sub>m</sub>b<sub>e</sub>ddi<sub>ng</sub> C<sub>on</sub>fi<sub>gura</sub>ti<sub>ons</sub>

We power the generative backends utilizing three leading open-weight LLMs in half-precision: Qwen2.5-7B [93], Llama-3.1-8B [34], and Mistral-7B-v0.3 [48]. To validate scale generalization and real-world deployment robustness (Section 8.7), we seamlessly upscale to the 14B and 32B variants of Qwen2.5. Semantic facets are robustly induced via all-mpnet-base-v2 for English tasks and paraphrase-multilingual-mpnet-base-v2 for the recipe task [97].

## 7<sub>.</sub>4 Evaluation Protocol and Zero-Leaka<sub>g</sub>e Ri<sub>g</sub>or

To fundamentally prevent algorithmic overfitting and guarantee out-of-distribution generalizability, we enforce a strict separation of query pools. All architectural hyperparameters $( \beta = 0 . 3 , \beta _ { \mathrm { d o c } } = 0 . 3 , \lambda = 0 . 2 5 , \kappa = 0 , \gamma = 0 . 6 , m _ { \mathrm { m a x } } = 3 ,$ � = 0.1) were stabilized exactly once on a disjoint 40-query ASQA development split.

Crucially, the primary evaluation tables execute on a completely fresh, zero-leakage evaluation frame of 100 queries per task across four tasks, three generators, and two stochastic decoding seeds. All inference runs are strictly paired structurally (within task, model, seed, and query). We establish statistical significance through crossed bootstrap Manuscript submitted to ACM

![](images/2520ad17f32f63632b62b8822a8edaec5443042a28e788629a14357327b56bc0.jpg)

![](images/cf29473c2b734762bcc64c36bc6d7263f8b38c95448a5be111efa39e14ad9121.jpg)

![](images/ebb987413549c2ae992cd5a0c260d80e04d7bf3a38590059e4d5ed2816ed1265.jpg)  
Fi<sub>g.</sub> 5<sub>.</sub> D<sub>econ</sub>f<sub>oun</sub>d<sub>e</sub>d <sub>w</sub>idth<sub>–coun</sub>t <sub>gr</sub>id<sub>.</sub> S<sub>o</sub>lid li<sub>nes represen</sub>t <sub>ro</sub>t<sub>a</sub>ti<sub>ng</sub> th<sub>roug</sub>h f<sub>res</sub>h <sub>ev</sub>id<sub>ence;</sub> d<sub>as</sub>h<sub>e</sub>d li<sub>nes repea</sub>t<sub>e</sub>dl<sub>y samp</sub>l<sub>e</sub> th<sub>e</sub> <sub>same</sub> fi<sub>xe</sub>d <sub>con</sub>t<sub>ex</sub>t<sub>.</sub>

Table 4. Deconfounded width–count grid: rotation cycles disjoint rank windows; fixed repeats top-� for � samples. $\Delta \mathsf { P R } ( \varpi T$ is <sub>p</sub>aired within <sub>q</sub>uer<sub>y</sub> and bootstra<sub>pp</sub>ed b<sub>y q</sub>uer<sub>y,</sub> isolatin<sub>g</sub> fresh-evidence <sub>g</sub>ain. Pool $N = 4 0 0$ <sup>k</sup>ee<sub>p</sub>s $k T \leq 2 8 8 < N$ with no re<sub>p</sub>eats<sub>;</sub> earlier $N = 3 0$ <sub>ma</sub>d<sub>e</sub> �=12 <sub>o</sub>f<sub>er</sub> th<sub>e same</sub> 24<sub>–</sub>30 d<sub>ocs</sub> $^ { * } / ^ { \dagger } \colon p < 0 . 0 5 / p < 0 . 0 1$
<table><tr><td rowspan="2">k</td><td rowspan="2">T</td><td rowspan="2">kT</td><td colspan="2">offered</td><td colspan="2">PR@T</td><td rowspan="2">∆PR@T</td><td rowspan="2">95% CI</td></tr><tr><td>rotation</td><td>fixed</td><td>rotation</td><td>fixed</td></tr><tr><td>2</td><td>1</td><td>2</td><td>2.0</td><td>2.0</td><td>0.204</td><td>0.204</td><td>+0.000</td><td>[+0.000, +0.000]</td></tr><tr><td>2</td><td>5</td><td>10</td><td>10.0</td><td>2.0</td><td>0.328</td><td>0.238</td><td>+0.090†</td><td>[+0.068, +0.113]</td></tr><tr><td>2</td><td>12</td><td>24</td><td>24.0</td><td>2.0</td><td>0.397</td><td>0.257</td><td> $+ 0 . 1 4 0 ^ { \dagger }$ </td><td>[+0.113, +0.169]</td></tr><tr><td>5</td><td>1</td><td>5</td><td>5.0</td><td>5.0</td><td>0.218</td><td>0.218</td><td>+0.000</td><td>[+0.000, +0.000]</td></tr><tr><td>5</td><td>5</td><td>25</td><td>25.0</td><td>5.0</td><td>0.339</td><td>0.266</td><td> $+ 0 . 0 7 3 ^ { \dagger }$ </td><td>[+0.052, +0.094]</td></tr><tr><td>5</td><td>12</td><td>60</td><td>60.0</td><td>5.0</td><td>0.423</td><td>0.290</td><td>+0.132†</td><td>[+0.105, +0.159]</td></tr><tr><td>12</td><td>1</td><td>12</td><td>12.0</td><td>12.0</td><td>0.243</td><td>0.244</td><td>-0.001</td><td>[-0.002, +0.000]</td></tr><tr><td>12</td><td>5</td><td>60</td><td>60.0</td><td>12.0</td><td>0.365</td><td>0.296</td><td> $+ 0 . 0 6 9 ^ { \dagger }$ </td><td>[+0.048, +0.091]</td></tr><tr><td>12</td><td>12</td><td>144</td><td>144.0</td><td>12.0</td><td>0.428</td><td>0.331</td><td> $+ 0 . 0 9 7 ^ { \dagger }$ </td><td> $\left[ + 0 . 0 7 2 , + 0 . 1 2 2 \right]$ </td></tr><tr><td>24</td><td>1</td><td>24</td><td>24.0</td><td>24.0</td><td>0.253</td><td>0.254</td><td>-0.000</td><td> $[ - 0 . 0 0 1 , + 0 . 0 0 0 ]$ </td></tr><tr><td>24</td><td>5</td><td>120</td><td>120.0</td><td>24.0</td><td>0.371</td><td>0.316</td><td>+0.056†</td><td> $\left[ + 0 . 0 3 3 , + 0 . 0 7 9 \right]$ </td></tr><tr><td>24</td><td>12</td><td>288</td><td>288.0</td><td>24.0</td><td>0.421</td><td>0.336</td><td>+0.085†</td><td> $\left[ + 0 . 0 6 1 , + 0 . 1 1 0 \right]$ </td></tr></table>

resampling, weighting tasks equally, and conservatively applying the Benjamini-Hochberg False Discovery Rate (BH-FDR) within each designated family to mathematically account for testing multiplicity. Reporting multiple decoding seeds and treating them as replicates follows the standing recommendation that IR efect sizes be estimated with the run-to-run variance component made explicit [120].

## 8 Ex<sub>pe</sub>rim<sub>e</sub>nt<sub>a</sub>l R<sub>esu</sub>lt<sub>s</sub>

## 8.1 The � � Factorial Grid: Formulatin<sub>g</sub> the Laws of Allocation

The fundamental architectural question for diverse RAG is how to optimally allocate a constrained inference budget. To rigorously deconfound the generative efects of context width (�) from sequential generation count (� ), we executed a massive factorial evaluation crossing $k \in \{ 2 , 5 , 1 2 , 2 4 \}$ with $T \in \{ 1 , 5 , 1 2 \}$ (summarized in Table 4 and Figure 5). Evaluated symmetrically over four tasks and two generators with strict fixed-context controls, this grid explicitly isolates the marginal return of expanding the prompt versus querying the model iteratively.

Manuscript submitted to ACM

T<sub>a</sub>bl<sub>e</sub> 5<sub>.</sub> D<sub>ec</sub>i<sub>s</sub>i<sub>ve</sub> d<sub>econ</sub>f<sub>oun</sub>d<sub>e</sub>d<sub>-gr</sub>id <sub>con</sub>t<sub>ras</sub>t<sub>s</sub> b<sub>y</sub> t<sub>as</sub>k <sub>an</sub>d <sub>genera</sub>t<sub>or, as w</sub>ithi<sub>n-query pa</sub>i<sub>re</sub>d <sub>por</sub>tf<sub>o</sub>li<sub>o-reca</sub>ll dif<sub>erences.</sub> L<sub>as</sub>t <sub>co</sub>l<sub>umn:</sub> <sub>ro</sub>t<sub>a</sub>ti<sub>on over ma</sub>t<sub>c</sub>h<sub>e</sub>d fi<sub>xe</sub>d<sub>-con</sub>t<sub>ex</sub>t <sub>con</sub>t<sub>ro</sub>l<sub>.</sub> $^ { * } / ^ { \dagger } \colon p < 0 . 0 5 / p < 0 . 0 1$
<table><tr><td>condition</td><td></td><td></td><td></td><td></td><td>count T12 vs T1 at k24 count T12 vs T1 at k2 width k24 vs k2 at T12 width k24 vs k2 at T1 rot over fix at k24T12</td></tr><tr><td>asqa/llama</td><td> $+ 0 . 2 0 4 ^ { \dagger }$ </td><td>+0.289†</td><td> $- 0 . 0 6 1 ^ { * }$ </td><td>+0.023</td><td> $+ 0 . 0 6 6 ^ { \dagger }$ </td></tr><tr><td>asqa/qwen</td><td> $+ 0 . 1 4 4 ^ { \dagger }$ </td><td> $+ 0 . 2 2 9 ^ { \dagger }$ </td><td>-0.042</td><td>+0.043</td><td> $+ 0 . 0 7 4 ^ { \dagger }$ </td></tr><tr><td>qampari/llama</td><td> $+ 0 . 1 8 7 ^ { \dagger }$ </td><td> $+ 0 . 1 4 1 ^ { \dagger }$ </td><td> $+ 0 . 1 3 9 ^ { \dagger }$ </td><td> $+ 0 . 0 9 3 ^ { \dagger }$ </td><td> $+ 0 . 1 1 7 ^ { \dagger }$ </td></tr><tr><td>qampari/qwen</td><td> $+ 0 . 1 3 4 ^ { \dagger }$ </td><td> $+ 0 . 1 1 5 ^ { \dagger }$ </td><td> $+ 0 . 0 5 8 ^ { \dagger }$ </td><td> $+ 0 . 0 3 9 ^ { \dagger }$ </td><td> $+ 0 . 0 8 2 ^ { \dagger }$ </td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 6<sub>.</sub> Hi<sub>erarc</sub>hi<sub>ca</sub>l <sub>ro</sub>t<sub>a</sub>ti<sub>on w</sub>idth <sub>con</sub>t<sub>ras</sub>t $k = 2 4$ min<sub>us</sub> $k = 2 ,$ <sub>p</sub>aired within <sub>q</sub>uer<sub>y</sub>. Condition rows are <sub>p</sub>artiall<sub>y p</sub>ooled task– <sub>g</sub>enerator efects (Paule–Mandel); last rows <sub>p</sub>redict a new condition. With $K = 4 ,$ interval uses Hi<sub>gg</sub>ins–Thom<sub>p</sub>son–S<sub>p</sub>ie<sub>g</sub>elhalter $t _ { K - 2 } .$ <sub>, no</sub>t <sub>norma</sub>l<sub>.</sub> At � = 12 th<sub>e mean</sub> i<sub>s sma</sub>ll <sub>an</sub>d <sub>pre</sub>di<sub>c</sub>ti<sub>on</sub> i<sub>n</sub>t<sub>erva</sub>l <sub>w</sub>id<sub>e;</sub> � = 4 li<sub>m</sub>it<sub>s prec</sub>i<sub>s</sub>i<sub>on.</sub>
<table><tr><td>estimate</td><td> $T = 1$ </td><td> $T = 5$ </td><td> $T = 1 2$ </td></tr><tr><td>asqa/llama</td><td>+0.036</td><td>-0.033</td><td>-0.053</td></tr><tr><td>asqa/qwen</td><td>+0.047</td><td>-0.000</td><td>-0.038</td></tr><tr><td>qampari/llama</td><td>+0.081</td><td>+0.148</td><td>+0.134</td></tr><tr><td>qampari/qwen</td><td>+0.041</td><td>+0.066</td><td>+0.057</td></tr><tr><td>grand mean</td><td>+0.051</td><td>+0.045</td><td>+0.025</td></tr><tr><td>between-condition SD</td><td>+0.025</td><td>+0.083</td><td>+0.090</td></tr><tr><td>prediction interval  $\left( t _ { K - 2 } \right)$ </td><td> $\left[ - 0 . 0 7 4 , + 0 . 1 7 6 \right]$ </td><td> $\left[ - 0 . 3 5 6 , + 0 . 4 4 7 \right]$ </td><td> $\left[ - 0 . 4 1 2 , + 0 . 4 6 2 \right]$ </td></tr><tr><td>normal quantile, for comparison</td><td> $\left[ - 0 . 0 0 6 , + 0 . 1 0 8 \right]$ </td><td> $\left[ - 0 . 1 3 8 , + 0 . 2 2 8 \right]$ </td><td> $\left[ - 0 . 1 7 4 , + 0 . 2 2 4 \right]$ </td></tr></table>

The empirical data yield a paradigm-defining conclusion: scaling sequential generation count delivers universal, transformative gains, whereas expanding context width represents an architectural trap fundamentally constrained by relevance decay.

As Table 4 illustrates, holding context width constant and iteratively raising � from one to twelve drives massive portfolio recall improvements across the entire matrix. We observe robust absolute gains hovering around 0.20 at narrow widths $( k = 2 , 5 )$ and 0.17 at extreme widths $( k = 2 4 )$ . Crucially, these iterative gains remain globally positive and highly significant across all evaluated task-generator combinations (Table 5), establishing sequential generation as a universally transferable scaling lever.

Conversely, the marginal return on expanding context width � is highly brittle and rapidly collapses under iteration. <sup>While</sup> <sup>widening</sup> <sup>the</sup> <sup>context</sup> <sup>from</sup> <sup>2</sup> <sup>to</sup> <sup>24</sup> <sup>documents</sup> <sup>nominally</sup> <sup>improves</sup> <sup>a</sup> <sup>single-pass</sup> <sup>generation</sup> <sup>(</sup>+<sup>0.050</sup> <sup>at</sup> $T = 1 ) ;$ , this <sup>perceived</sup> <sup>benefit</sup> <sup>aggressively</sup> <sup>attenuates</sup> <sup>as</sup> <sup>rounds</sup> <sup>increase,</sup> <sup>collapsing</sup> <sup>to</sup> <sup>a</sup> <sup>negligible</sup> +<sup>0.024</sup> <sup>at</sup> $T = 1 2$ . Pushing the context beyond twelve slots yields no detectable statistical benefit under any configuration.

This dynamic unequivocally resolves the orchestration dilemma regarding informational extraction limits. When comparing the mathematical equivalent of 24 allocated document slots, the multi-round 2, 12 configuration systematically extracts a massively higher portfolio recall ( 0.144, 95% CI 0.119, 0.170 ) from the exact same evidence footprint than the single-pass 24, 1 baseline. While the monolithic 24, 1 approach minimizes computational latency, it hits a rigid cognitive ceiling, leaving severe informational blind spots. Even a maximally expensive 24, 12 configuration only marginally exceeds the narrow-iterative baseline in recall, proving that aggressively scaling test-time compute over tight, sequential windows is the only mechanism capable of maximizing the generative yield of retrieved evidence.

The catastrophic degradation of width utility under sequential rotation is not a generative artifact, but a fundamental information retrieval penalty: relevance density. Expanding the context window inherently forces the retriever to ingest deeper, lower-quality ranks. A random-efects hierarchical analysis of the width contrast (Table 6) exposes a highly dispersed prediction interval of $\left[ - 0 . 4 1 2 , + 0 . 4 6 2 \right]$ . This extreme variance masks a harsh structural split: width expansion Manuscript submitted to ACM

T<sub>a</sub>bl<sub>e</sub> 7<sub>.</sub> L<sub>ong-con</sub>t<sub>ex</sub>t Q<sub>wen</sub>2<sub>.</sub>5<sub>-</sub>7B<sub>-</sub>I<sub>ns</sub>t<sub>ruc</sub>t<sub>-</sub>1M <sub>w</sub>idth<sub>–coun</sub>t <sub>con</sub>t<sub>ras</sub>t<sub>s a</sub>t <sub>w</sub>idth<sub>s</sub> b<sub>eyon</sub>d th<sub>e</sub> 7<sub>–</sub>8B <sub>gr</sub>id<sub>;</sub> $k = 9 6$ i<sub>s a</sub>b<sub>ou</sub>t 11K <sub>promp</sub>t tokens. Entries are within-<sub>q</sub>uer<sub>y p</sub>aired PR@� diferences; $^ { * } / ^ { \dagger } \colon p < 0 . 0 5 / p < 0 . 0 1$ <sub>.</sub> G<sub>enera</sub>ti<sub>on-coun</sub>t <sub>ga</sub>i<sub>ns rema</sub>i<sub>n pos</sub>iti<sub>ve</sub> ( 0.085 t<sub>o</sub> 0.157)<sub>; no w</sub>idth <sub>ga</sub>i<sub>n</sub> b<sub>eyon</sub>d $k = 2 4$ is <sub>p</sub>ositive/si<sub>g</sub>nificant, and ASQA worsens.
<table><tr><td>contrast</td><td colspan="2"> $\mathsf { A S Q A }$ </td><td colspan="2">QAMPARI</td></tr><tr><td></td><td>Δ</td><td>n</td><td>Δ</td><td>n</td></tr><tr><td> $k { = } 4 8 \ \mathrm { v s } k { = } 2 4 , T { = } 1$ </td><td>-0.033*</td><td>120</td><td>+0.004</td><td>120</td></tr><tr><td> $k { = } 9 6 \mathrm { ~ v s } k { = } 2 4 , T { = } 1$ </td><td>-0.031</td><td>30</td><td>+0.040</td><td>29</td></tr><tr><td> $k { = } 4 8 \ \mathrm { v s } k { = } 2 4 , T { = } 5$ </td><td> $- 0 . 0 3 0$ </td><td>58</td><td> $+ 0 . 0 0 2$ </td><td>59</td></tr><tr><td> $T = 5 \mathrm { ~ v s ~ } T = 1 , k = 2 4$ </td><td> $+ 0 . 0 8 9 ^ { \dagger }$ </td><td>60</td><td> $+ 0 . 1 1 2 ^ { \dagger }$ </td><td>60</td></tr><tr><td> $T = 1 2 \mathrm { ~ v s ~ } T = 1 , k = 2 4$ </td><td> $+ 0 . 1 0 1 ^ { \dagger }$ </td><td>60</td><td> $+ 0 . 1 5 7 ^ { \dagger }$ </td><td>60</td></tr><tr><td> $T = 5 \mathrm { \ v s \ } T = 1 , k = 4 8$ </td><td> $+ 0 . 1 0 9 ^ { \dagger }$ </td><td>107</td><td> $+ 0 . 0 8 5 ^ { \dagger }$ </td><td>108</td></tr></table>

is only viable for tasks with highly dispersed answer sets $( \mathrm { e . g . , Q A M P A R I } )$ , but turns actively toxic for tasks where evidence is densely concentrated at the top ranks $( \mathrm { e . g . , A S Q A ) }$ . Generation count therefore stands alone as the globally optimal allocation strategy, structurally immune to the rank-degradation penalty that plagues wide-context packing.

## 8<sub>.</sub>2 I<sub>so</sub>l<sub>a</sub>ti<sub>ng</sub> th<sub>e</sub> M<sub>ec</sub>h<sub>an</sub>i<sub>cs:</sub> Th<sub>e</sub> S<sub>upremacy</sub> <sub>o</sub>f F<sub>res</sub>h E<sub>v</sub>id<sub>ence</sub> O<sub>ver</sub> St<sub>oc</sub>h<sub>as</sub>ti<sub>c</sub> R<sub>esamp</sub>li<sub>ng</sub>

Raising the generation count � unequivocally improves portfolio recall, but this architectural lever inherently confounds two fundamentally distinct generative mechanisms: the algorithmic benefit of drawing multiple stochastic samples from the decoder (resampling), and the epistemological benefit of exposing the model to new retrieved documents (fresh evidence).

To cleanly decouple these forces, we deploy a matched fixed-context control alongside the sequential rotation arm. By locking the exact same top-� documents in the prompt across all � rounds and drawing independent decoder samples, this control arm isolates the pure utility of stochastic re-reading. The paired diference (ΔPR@�) directly unmasks the true marginal utility of fresh informational exposure (Table 4 and Figure 5).

The resulting decomposition shatters the prevailing NLP assumption that decoding stochasticity alone is suficient for diverse generation. While repeated sampling over a fixed context does provide a reliable baseline bump (improving recall by 0.054 to 0.087 as the model re-evaluates the same text), this isolated mechanism rapidly hits a strict cognitive asymptote. At twelve generations, actively injecting fresh evidence via rotation overwhelmingly obliterates the fixed-context resampling baseline, driving decisive absolute margins ranging from 0.087 at � = 24 up to 0.140 at $k = 2 ( \mathrm { a l l } \rho < 0 . 0 0 1 )$

Deconstructing these gains reveals a profound structural insight: fresh evidence explicitly dictates the performance ceiling. Exposure to unseen documents strictly accounts for 72% of the total generation-count utility at narrow widths, and maintains a dominant 52% share even within massive 24-document windows where one might incorrectly assume all necessary information was already present.

The physical grounding footprints of the generated portfolios provide the definitive proof of this mechanism. At the 24, 12 extreme, the rotational policy structurally exposes 288 unique documents and successfully grounds its portfolio on a massive 50.9 of them. In stark contrast, the fixed-context arm, despite having twelve attempts to squeeze information out of its static 24-document window, mathematically stagnates, grounding on merely 10.3 documents.

Ultimately, this isolates a critical law of generative orchestration: an LLM cannot hallucinate genuine diversity from a stagnant prompt. Extra generative rounds extract their transformative value not through stochastic paraphrasing, but by serving as deliberate, sequential vehicles for unseen physical evidence.

Table 8. Relevance densit<sub>y</sub> versus <sub>g</sub>old-set size over four task <sub>p</sub>rofiles; the multi-ho<sub>p</sub> Hot<sub>p</sub>otQA re<sub>p</sub>laces the reci<sub>p</sub>e stress test here so that <sub>g</sub>old-set size and relevance densit<sub>y</sub> var<sub>y</sub> inde<sub>p</sub>endentl<sub>y</sub>. � mean <sub>g</sub>old answers, $\rho$ <sub>ran</sub>k<sub>-</sub>b<sub>an</sub>d <sub>answer pro</sub>b<sub>a</sub>bilit<sub>y,</sub> h<sub>ea</sub>d<sub>-</sub>5 concentration. $\mathsf { A t } \ T { = } 1 2 ,$ to<sub>p</sub>-5 share tracks width: onl<sub>y</sub> dis<sub>p</sub>ersed QAMPARI (0.398) <sub>g</sub>ains, not ASQA (0.713) or Hot<sub>p</sub>otQA (0.630) d<sub>esp</sub>it<sub>e go</sub>ld <sub>se</sub>t<sub>s</sub> 3.4/1.0<sub>; w</sub>ith <sub>one genera</sub>ti<sub>on, mu</sub>lti<sub>-</sub>h<sub>op ga</sub>i<sub>ns</sub> 0.33<sub>.</sub> ELI5 <sub>s</sub>t<sub>r</sub>i<sub>ng</sub> $\rho \mathrm { ~ i ~ } s$ <sub>no</sub>t <sub>es</sub>ti<sub>ma</sub>bl<sub>e.</sub> $^ { * } / ^ { \dagger } \colon p < 0 . 0 5 / p < 0 . 0 1$
<table><tr><td>task</td><td> $| A |$ </td><td>head-5 share</td><td> $\rho \ \mathrm { t o p } { - } 5 $ </td><td> $\mathrm { a t } T { = } 1$ </td><td>width ∆ width ∆ count ∆  $\operatorname { a t } T { = } 1 2$ </td><td> $\operatorname { a t } k { = } 2$ </td></tr><tr><td>ASQA</td><td>3.4</td><td>0.713</td><td>0.469</td><td>+0.033</td><td> $- 0 . 0 5 2 ^ { * }$ </td><td> $+ 0 . 2 5 9 ^ { \dagger }$ </td></tr><tr><td>QAMPARI</td><td>13.6</td><td>0.398</td><td>0.295</td><td>+0.066†</td><td> $+ 0 . 0 9 9 ^ { \dagger }$ </td><td> $+ 0 . 1 2 8 ^ { \dagger }$ </td></tr><tr><td>ELI5</td><td>一</td><td>一</td><td></td><td>+0.011</td><td> $+ 0 . 0 1 0 ^ { \dagger }$ </td><td> $+ 0 . 2 4 2 ^ { \dagger }$ </td></tr><tr><td>HotpotQA</td><td>1.0</td><td>0.630</td><td>0.169</td><td>+0.333†</td><td> $- 0 . 0 4 6 ^ { * }$ </td><td> $+ 0 . 4 5 0 ^ { \dagger }$ </td></tr></table>

Table 9. ELI5 factorial: small claim <sub>g</sub>old set but dis<sub>p</sub>ersed BM25 <sub>p</sub>rofile, se<sub>p</sub>aratin<sub>g</sub> relevance densit<sub>y</sub> from <sub>g</sub>old-set size; Hot<sub>p</sub>otQA extension in Table 8. Entries are within-<sub>q</sub>uer<sub>y p</sub>aired PR@� diferences; ∗/† mark $p < 0 . 0 5 / p < 0 . 0 1$ b<sub>y p</sub>aired bootstra<sub>p</sub>.
<table><tr><td></td><td></td><td></td><td></td><td></td><td>task generator count T at k2 count T at k24 width k at T12 budget (2,12) vs (24,1) rot over fix at k24T12</td><td></td></tr><tr><td>eli5 llama</td><td></td><td> $+ 0 . 2 2 8 ^ { \dagger }$ </td><td> $+ 0 . 1 5 6 ^ { \dagger }$ </td><td> $- 0 . 0 8 7 ^ { \dagger }$ </td><td> $+ 0 . 2 4 2 ^ { \dagger }$ </td><td></td></tr><tr><td></td><td>eli5 qwen</td><td> $+ 0 . 2 5 6 ^ { \dagger }$ </td><td> $+ 0 . 2 9 2 ^ { \dagger }$ </td><td> $+ 0 . 0 7 2 ^ { * }$ </td><td> $+ 0 . 2 1 9 ^ { \dagger }$ </td><td></td></tr></table>

## 8<sub>.</sub>3 The Tra<sub>p</sub> of Context Saturation and Relevance Densit<sub>y</sub>

A fundamental question arising from the $k \times T$ laws is whether the observed saturation of context width is a temporary artifact of the 7–8B models’ attention capacities, or a permanent structural bottleneck. To isolate the physical limits of context scaling, we stress-test our findings using Qwen2.5-7B-Instruct-1M, pushing the context windows to massive extremes of �  24, 48, 96 (consuming up to approximately 11, 000 prompt tokens).

The results (Table 7) unequivocally demonstrate that our established generation-count scaling laws transcend architectural context limits. Even with a 1-million token capacity, raising generation rounds from 1 to 12 at $k = 2 4$ strictly drives massive absolute gains of 0.101 and 0.157 across tasks. In stark contrast, aggressively expanding the context window beyond � = 24 yields zero statistically significant portfolio coverage benefits. In fact, on the complex ASQA benchmark, pushing to $k = 4 8$ actively degrades performance. A massively expanded trained window alone does not, and cannot, convert extreme context width into a superior informational budget.

This rigid saturation is not a generative failure, but rather a fundamental Information Retrieval (IR) limitation governed by relevance density. Expanding a context window mathematically forces the system to ingest deeper, lower-quality retrieval ranks, inevitably exhausting the high-density relevance head of the retrieved pool.

To cleanly decouple this relevance density from the sheer size of the gold answer set, we inject ELI5 (which exhibits dispersed BM25 retrieval but few claim-level units) and HotpotQA [129] (requiring a singular multi-hop answer strictly dependent on two specific co-occurring paragraphs) into the factorial analysis (Tables 9 and 8).

The HotpotQA dynamics perfectly isolate the necessity of sequential exploration. At a single generation pass, widening the prompt from two to twenty-four documents drives a massive 0.333 gain, purely because a narrow � = 2 window rarely captures both required multi-hop paragraphs simultaneously. However, when the budget permits twelve sequential generations, rotational policies successfully assemble the multi-hop pair across rounds, completely neutralizing the wide-context advantage and collapsing the width efect to a detrimental 0.046. Under matched hardware budgets, the iterative 2, 12 allocation continues to strictly dominate the monolithic 24, 1 allocation.

Across all evaluated benchmarks, the utility of context expansion strictly tracks evidence concentration, quantified as the head-5 answer-mass share, rather than gold-set size. Context expansion remains slightly viable exclusively on highly Manuscript submitted to ACM

T<sub>a</sub>bl<sub>e</sub> 10<sub>.</sub> F<sub>rozen</sub> h<sub>e</sub>ld<sub>-ou</sub>t <sub>compar</sub>i<sub>son: s</sub>t<sub>eer</sub>i<sub>ng c</sub>h<sub>osen once</sub> b<sub>y one-s</sub>t<sub>an</sub>d<sub>ar</sub>d<sub>-error on</sub> d<sub>eve</sub>l<sub>opmen</sub>t<sub>,</sub> th<sub>en</sub> t<sub>es</sub>t<sub>e</sub>d <sub>on</sub> di<sub>sjo</sub>i<sub>n</sub>t <sub>quer</sub>i<sub>es</sub> (100 er task), four tasks, three enerators, two decodin seeds. Contrasts are aired on uer / enerator/seed/ rom t/width/count/code S<sub>amp</sub>li<sub>ng:</sub> t<sub>empera</sub>t<sub>ure</sub> 0.7<sub>,</sub> t<sub>op-</sub>� = 1<sub>,</sub> t<sub>op-</sub>� di<sub>sa</sub>bl<sub>e</sub>d<sub>, repe</sub>titi<sub>on pena</sub>lt<sub>y</sub> 1<sub>.</sub> C<sub>rosse</sub>d b<sub>oo</sub>t<sub>s</sub>t<sub>rap, equa</sub>l t<sub>as</sub>k <sub>we</sub>i<sub>g</sub>ht<sub>s,</sub> BH<sub>-</sub>FDR f<sub>or</sub> PR@�<sub>;</sub> ECR fi<sub>xe</sub>d<sub>-response.</sub> R<sub>oun</sub>d<sub>e</sub>d dif<sub>erences may s</sub>hift 0.001<sub>; au</sub>dit <sub>requ</sub>i<sub>res</sub> 2400 <sub>comp</sub>l<sub>e</sub>t<sub>e ce</sub>ll<sub>s.</sub>
<table><tr><td>baseline</td><td>baseline PR</td><td>AscP PR</td><td>∆PR</td><td>BHq</td><td>ΔECR</td><td>∆ ground</td></tr><tr><td>AsCP without steering</td><td>0.300</td><td>0.309</td><td>+0.009</td><td>0.102</td><td>+0.056</td><td>+0.029</td></tr><tr><td>deep rotation</td><td>0.303</td><td>0.309</td><td>+0.006</td><td>0.343</td><td>+0.307</td><td>+0.050</td></tr><tr><td>vanilla RAG</td><td>0.237</td><td>0.309</td><td>+0.072</td><td>&lt; .001</td><td>+0.104</td><td>+0.003</td></tr><tr><td>MMR</td><td>0.239</td><td>0.309</td><td>+0.070</td><td>&lt; .001</td><td>+0.095</td><td>+0.011</td></tr><tr><td>DPP-RAG</td><td>0.274</td><td>0.309</td><td>+0.035</td><td>&lt; .001</td><td>+0.156</td><td>+0.011</td></tr><tr><td>xQuAD</td><td>0.238</td><td>0.309</td><td>+0.071</td><td>&lt; .001</td><td>+0.111</td><td>+0.002</td></tr><tr><td>PM-2-RAG</td><td>0.228</td><td>0.309</td><td>+0.081</td><td>&lt; .001</td><td>+0.111</td><td>+0.004</td></tr><tr><td>CARRIAGE</td><td>0.276</td><td>0.309</td><td>+0.033</td><td>&lt; .001</td><td>+0.215</td><td>+0.052</td></tr><tr><td>CARRIAGE-narrow</td><td>0.261</td><td>0.309</td><td>+0.048</td><td>&lt; .001</td><td>-0.074</td><td>+0.089</td></tr></table>

T<sub>a</sub>bl<sub>e</sub> 11<sub>.</sub> S<sub>c</sub>h<sub>e</sub>d<sub>u</sub>l<sub>er-on</sub>l<sub>y</sub> h<sub>e</sub>ld<sub>-ou</sub>t <sub>compar</sub>i<sub>son:</sub> A<sub>scp w</sub>ith<sub>ou</sub>t <sub>s</sub>t<sub>eere</sub>d d<sub>eco</sub>di<sub>ng, so</sub> b<sub>o</sub>th <sub>s</sub>id<sub>es use or</sub>di<sub>nary genera</sub>ti<sub>on an</sub>d i<sub>so</sub>l<sub>a</sub>t<sub>e</sub> <sub>ev</sub>id<sub>ence sc</sub>h<sub>e</sub>d<sub>u</sub>li<sub>ng.</sub> C<sub>rosse</sub>d b<sub>oo</sub>t<sub>s</sub>t<sub>rap over</sub> d<sub>eco</sub>di<sub>ng see</sub>d<sub>s an</sub>d <sub>w</sub>ithi<sub>n-</sub>t<sub>as</sub>k <sub>quer</sub>i<sub>es; equa</sub>l t<sub>as</sub>k <sub>we</sub>i<sub>g</sub>ht<sub>s,</sub> fi<sub>xe</sub>d th<sub>ree-mo</sub>d<sub>e</sub>l <sub>se</sub>t<sub>;</sub> BH-FDR covers the ei<sub>g</sub>ht PR@� contrasts.
<table><tr><td>baseline</td><td>baseline PR</td><td>scheduler PR</td><td>∆PR</td><td>BHq</td></tr><tr><td>deep rotation</td><td>0.303</td><td>0.300</td><td>-0.003</td><td>0.717</td></tr><tr><td>vanilla RAG</td><td>0.237</td><td>0.300</td><td>+0.063</td><td>&lt; .001</td></tr><tr><td>MMR</td><td>0.239</td><td>0.300</td><td>+0.061</td><td>&lt; .001</td></tr><tr><td>DPP-RAG</td><td>0.274</td><td>0.300</td><td>+0.026</td><td>0.002</td></tr><tr><td>xQuAD</td><td>0.238</td><td>0.300</td><td>+0.062</td><td>&lt; .001</td></tr><tr><td>PM-2-RAG</td><td>0.228</td><td>0.300</td><td>+0.072</td><td>&lt; .001</td></tr><tr><td>CARRIAGE</td><td>0.276</td><td>0.300</td><td>+0.024</td><td>&lt; .001</td></tr><tr><td>CARRIAGE-narrow</td><td>0.261</td><td>0.300</td><td>+0.039</td><td>&lt; .001</td></tr></table>

dispersed tasks like QAMPARI (head-5 share 0.398), but turns actively toxic on concentrated tasks like ASQA (0.713) and HotpotQA (0.630). Ultimately, pushing deeper into a ranked list to populate a massive generative prompt dilutes the LLM’s attention with low-yield noise. This physical IR constraint renders sequential, narrow-window generations the globally optimal strategy for portfolio coverage, regardless of underlying hardware capacity

## 8<sub>.</sub>4 E<sub>n</sub>d<sub>-</sub>t<sub>o-</sub>E<sub>n</sub>d E<sub>va</sub>l<sub>ua</sub>ti<sub>on:</sub> Th<sub>e</sub> T<sub>r</sub>i<sub>ump</sub>h <sub>o</sub>f Cl<sub>ose</sub>d<sub>-</sub>L<sub>oop</sub> O<sub>rc</sub>h<sub>es</sub>t<sub>ra</sub>ti<sub>on</sub>

Having established that sequential generation over narrow windows dictates optimal portfolio coverage, we now evaluate how efectively concrete scheduling policies capture this theoretical potential under a strictly fixed hardware budget. We rigorously benchmark our submodular evidence scheduler (Ascp) against a spectrum of selection and rotation baselines over 2, 400 strictly paired evaluation frames (Table 10).

The empirical results reveal a fundamental limitation in classical IR adaptations. When classical algorithms like MMR, xQuAD, or PM-2 are naively ported into RAG pipelines, they operate as open-loop systems. They attempt to diversify the prompt text based purely on document similarity, remaining entirely blind to what the generative model actually consumes. Consequently, they stagnate at portfolio recalls of 0.228 to 0.239. Similarly, cutting-edge diverse RAG frameworks like Carriage manage to push recall to 0.276, but ultimately hit an architectural ceiling.

In stark contrast, our Ascp scheduler systematically shatters this ceiling, dominating every evaluated selection baseline. Operating over an equal-task estimand, Ascp achieves a terminal portfolio recall of 0.309, securing a robust absolute 0.033 gain over Carriage (95% CI 0.022, 0.044 , BH � < 0.001) and soaring up to 0.081 absolute points over PM-2-RAG.

Crucially, this scheduling superiority is structurally driven by front-end context orchestration rather than generative post-processing tricks. To definitively prove this, we strip Ascp of its custom steered decoder (Table 11), forcing it to operate via ordinary generation perfectly matching baseline conditions. Even stripped of decoding interventions, the <sup>pure</sup> <sup>scheduler</sup> <sup>retains</sup> <sup>significant</sup> <sup>dominance,</sup> <sup>maintaining</sup> <sup>absolute</sup> <sup>gains</sup> <sup>between</sup> +<sup>0.024</sup> <sup>and</sup> +<sup>0.072</sup> <sup>over</sup> <sup>all</sup> <sup>selection</sup> baselines.

The true architectural elegance of Ascp emerges when we analyze how it achieves this coverage. While exhaustive deep rotation nominally matches the full method in pure portfolio recall (a statistically insignificant contrast of 0.006, BH $q = 0 . 3 4 3 )$ , it achieves this equivalence through brute-force ineficiency. To match Ascp’s recall, deep rotation blindly forces 25.00 documents into the contexts, hoping the model will randomly extract value. This squanders massive token bandwidth, yielding an Evidence Coverage Rate (ECR) of merely 0.375

Ascp, however, operates as a true closed-loop system. By actively reading causal attribution feedback, it dynamically penalizes redundant information and forcefully steers the context toward unexplored semantic facets. Consequently, it achieves the exact same terminal recall by ofering only 10.55 highly curated documents, driving ECR up to a dominant 0.626. This proves that submodular scheduling driven by causal feedback does not merely inflate coverage; it achieves peak generative utility while maintaining substantially tighter, highly grounded, and token-eficient context bounds

## 8<sub>.</sub>5 Th<sub>e</sub> F<sub>ee</sub>db<sub>ac</sub>k E<sub>ng</sub>i<sub>ne:</sub> C<sub>ausa</sub>l P<sub>ro</sub>bi<sub>ng vs.</sub> Si<sub>m</sub>il<sub>ar</sub>it<sub>y</sub> Ill<sub>us</sub>i<sub>ons</sub>

The architectural supremacy of the Ascp scheduler hinges fundamentally on its ability to dynamically discount evidence that the generator has already consumed. Consequently, the performance ceiling of the entire orchestration loop is strictly bound by the accuracy of its underlying feedback signal. While Section 4 proved that our causal LOO probe isolates necessary documents far better than similarity heuristics on constructed diagnostic pools, we must validate whether this discriminative advantage actually translates into end-to-end generative portfolio utility.

To definitively test this, we execute a structural ablation within the scheduling loop (detailed in Table 15), isolating the feedback mechanism while holding all other orchestration logic constant.

If the scheduler operates open-loop, blindly assuming that the generator perfectly utilizes every single documen ofered in the context (the uniform-use assumption), portfolio utility severely stagnates. By actively reading the causal LOO probe’s utilization signal, Ascp yields a statistically significant 0.014 absolute portfolio recall gain (95% CI 0.003, 0.025 , � = 0.013) over this naive uniform assumption. This proves that dynamically tracking actual consumption is essential for budget optimization.

Crucially, attempting to recover this operational gain by swapping our causal probe for a standard embeddingsimilarity proxy results in a catastrophic systemic failure. When operated with similarity-based feedback, the scheduler produces a marginal utility completely indistinguishable from the naive open-loop assumption (a meaningless 0.002 contrast, $ { p } = 0 . 7 3 )$ .

This end-to-end collapse aligns perfectly with our initial diagnostic findings in Section 4: because all documents retrieved for a given query naturally reside in the exact same semantic space as the generated response, embedding similarities severely saturate. They are mathematically incapable of distinguishing between the documents that truly drove the generation and dense, same-query near-misses.

The strategic conclusion is absolute: attempting to steer a RAG context scheduler using superficial text similarity provides zero operational leverage. The orchestration advantage over brute-force rotation is uniquely and exclusively unlocked by measuring exact counterfactual necessity, cementing our causal probe as the indispensable engine of diverse generative search.

Manuscript submitted to ACM

T<sub>a</sub>bl<sub>e</sub> 12<sub>.</sub> H<sub>e</sub>ld<sub>-ou</sub>t d<sub>eco</sub>d<sub>er</sub> d<sub>ecompos</sub>iti<sub>on w</sub>ith id<sub>en</sub>ti<sub>ca</sub>l <sub>samp</sub>li<sub>ng:</sub> t<sub>empera</sub>t<sub>ure</sub> 0.7<sub>,</sub> t<sub>op-</sub>� = 1<sub>,</sub> t<sub>op-</sub>� di<sub>sa</sub>bl<sub>e</sub>d<sub>, repe</sub>titi<sub>on pena</sub>lt<sub>y</sub> 1<sub>.</sub> APC is ada<sub>p</sub>tive-<sub>p</sub>lausibilit<sub>y</sub> constraint; contrasts isolate contrast direction, APC filterin<sub>g</sub>, and com<sub>p</sub>lement. Intervals/�: two-sided � <sub>over</sub> fi<sub>ve see</sub>d <sub>means</sub> $( d f = 4 ) ,$ , e ual task wei hts, fixed three-model set; audit needs all 6000 frames; BH-FDR covers PR@�.
<table><tr><td>contrast</td><td>∆PR@T</td><td>seed  $- t _ { 4 } \mathrm { C I }$ </td><td>p</td><td>ΔECR</td><td>∆ ground</td><td>Δ words</td></tr><tr><td>full decoder – off</td><td>+0.0107</td><td>[+0.0072, +0.0142]</td><td>0.001</td><td>+0.0555</td><td>+0.0275</td><td>-1.0</td></tr><tr><td>contrast, APC held on</td><td>+0.0116</td><td>[+0.0045, +0.0187]</td><td>0.011</td><td>+0.0583</td><td>+0.0232</td><td>-0.6</td></tr><tr><td>APC only</td><td>-0.0009</td><td>[-0.0086, +0.0069]</td><td>0.775</td><td>-0.0028</td><td>+0.0042</td><td>-0.4</td></tr><tr><td>contrast, APC held off</td><td>+0.0187</td><td>[+0.0145, +0.0229]</td><td>&lt; .001</td><td>+0.1012</td><td>+0.0366</td><td>-0.2</td></tr><tr><td>APC, contrast held on</td><td>-0.0080</td><td>[-0.0107, -0.0052]</td><td>0.001</td><td>-0.0457</td><td>-0.0092</td><td>-0.8</td></tr><tr><td>contrast × APC</td><td>-0.0071</td><td>[-0.0163, +0.0021]</td><td>0.098</td><td>-0.0429</td><td>-0.0134</td><td>-0.5</td></tr></table>

## 8<sub>.</sub>6 D<sub>eco</sub>d<sub>er</sub> I<sub>n</sub>t<sub>erven</sub>ti<sub>ons:</sub> O<sub>verr</sub>idi<sub>ng</sub> At<sub>en</sub>ti<sub>on</sub> I<sub>ner</sub>ti<sub>a</sub>

While our submodular scheduler successfully optimizes the macroscopic diet of the LLM (what enters the context), generative models inherently sufer from microscopic attention inertia. Even when provided with fresh evidence, LLMs frequently fixate on familiar, already-extracted concepts, resisting the integration of novel facts. To break this generative stagnation, our attribution-steered decoding intervenes directly at the logit level, executing a cognitive override that systematically shifts probability mass away from over-used evidence.

To definitively isolate the pure causal impact of this intra-generation intervention, we execute a rigorous five-seed decomposition on a completely disjoint evaluation frame (Table 12). By strictly enforcing identical baseline sampler configurations (temperature 0.7, top-� = 1) across all paths, we ensure the observed gains represent true architectural enhancements rather than stochastic anomalies.

The results (Table 12) unequivocally prove that this micro-level intervention successfully reprograms the LLM’s evidence consumption. Enabling the full attribution-steered decoder yields a highly robust, orthogonal 0.0107 absolute portfolio recall gain. Crucially, the mathematical stability of this override is absolute: across all five independent stochastic decoding seeds, the intervention reliably forces the extraction of new knowledge (BH $q = 0 . 0 0 1 \mathrm { ) }$ ). Beyond terminal recall, the decoder physically alters the generation footprint: it elevates the Evidence Coverage Rate (ECR) by 0.0555 and boosts strict lexical grounding by 0.0275. Most remarkably, it achieves this superior informational density while mathematically generating 1.0 fewer words per portfolio, proving that the intervention eliminates repetitive rambling in favor of dense, factual extraction

Decomposing the internal mechanics reveals a brilliant synergy between exploration and safety. The subtractive logit contrast acts as the exploratory engine: when deployed alone (APC held of), it aggressively forces the model into unexplored semantic territory, unleashing a massive 0.0187 (� < 0.001) recall gain alongside soaring ECR ( 0.1012). However, unconstrained contrastive generation inherently risks hallucination by forcing the model too far from its probability manifold.

This is where the adaptive-plausibility constraint (APC) proves vital. While APC alone provides zero coverage utility (a statistically null 0.0009), when fused with the logit contrast, it acts as a strict hallucination-resistant leash. It sacrifices a marginal 0.0080 of the raw contrastive recall to mathematically clamp the candidate tokens within a safe plausibility boundary. Ultimately, this fusion guarantees that the LLM’s attention is forcefully yet safely redistributed The decoder intervention systematically ensures that the fresh evidence ofered by the scheduler physically translates into diverse, deeply grounded portfolio construction.

Table 13. Bud<sub>g</sub>et-matched scalin<sub>g</sub> validation on 14B and 32B models. Entries re<sub>p</sub>resent <sub>p</sub>aired <sub>p</sub>ortfolio recall<sub>,</sub> strictl<sub>y</sub> contrastin<sub>g</sub> the <sub>mu</sub>lti<sub>-roun</sub>d it<sub>era</sub>ti<sub>ve genera</sub>ti<sub>on</sub> 2, 12 <sub>aga</sub>i<sub>ns</sub>t <sub>mono</sub>lithi<sub>c w</sub>id<sub>e-con</sub>t<sub>ex</sub>t <sub>genera</sub>ti<sub>on</sub> 24, 1 <sub>.</sub> Th<sub>e a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>super</sub>i<sub>or</sub>it<sub>y o</sub>f <sub>mu</sub>lti<sub>-roun</sub>d <sub>ev</sub>id<sub>ence a</sub>ll<sub>oca</sub>ti<sub>on ro</sub>b<sub>us</sub>tl h<sub>o</sub>ld<sub>s a</sub>t <sub>sca</sub>l<sub>e, ma</sub>i<sub>n</sub>t<sub>a</sub>i<sub>n</sub>i<sub>n mass</sub>i<sub>ve os</sub>iti<sub>ve</sub> b<sub>oun</sub>d<sub>s across</sub> t<sub>as</sub>k<sub>s.</sub>
<table><tr><td>model</td><td>task</td><td>(2,12)</td><td>(24,1)</td><td>Δ</td><td>95% CI</td></tr><tr><td>qwen14</td><td>asqa</td><td>0.569</td><td>0.401</td><td>+0.168</td><td>[+0.107,+0.233]</td></tr><tr><td>qwen14</td><td>pooled</td><td>0.400</td><td>0.266</td><td>+0.134</td><td>[+0.093,+0.175]</td></tr><tr><td>qwen14</td><td>qampari</td><td>0.231</td><td>0.131</td><td>+0.100</td><td>[+0.050,+0.157]</td></tr><tr><td>qwen32</td><td>asqa</td><td>0.560</td><td>0.380</td><td>+0.180</td><td>[+0.116,+0.250]</td></tr><tr><td>qwen32</td><td>pooled</td><td>0.429</td><td>0.291</td><td>+0.138</td><td>[+0.098,+0.181]</td></tr><tr><td>qwen32</td><td>qampari</td><td>0.298</td><td>0.202</td><td>+0.097</td><td>[+0.055, +0.140]</td></tr></table>

(a) token cost  
![](images/3e7a723d0cf5deed04ae1958b632a9ff259f3576d38c242b9325535157ecda10.jpg)

(b) wall-clock cost  
![](images/9616bf533b8610ad0a3fe547ff7858622dcc3e5a8aa518f9080726636b7d0c93.jpg)  
Fi<sub>g</sub>. 6. Cost versus <sub>q</sub>ualit<sub>y</sub> across ever<sub>y</sub> <sub>g</sub>rid cell (Qwen/ASQA, A100, <sub>p</sub>robe removed). (a) Total tokens <sub>p</sub>er <sub>q</sub>uer<sub>y</sub>; (b) Se<sub>q</sub>uential latenc<sub>y</sub>. S<sub>o</sub>lid li<sub>nes represen</sub>t <sub>ro</sub>t<sub>a</sub>ti<sub>on;</sub> d<sub>as</sub>h<sub>e</sub>d li<sub>nes represen</sub>t th<sub>e ma</sub>t<sub>c</sub>h<sub>e</sub>d fi<sub>xe</sub>d<sub>-con</sub>t<sub>ex</sub>t <sub>arm.</sub> Th<sub>e anno</sub>t<sub>a</sub>t<sub>e</sub>d <sub>pa</sub>i<sub>r</sub> i<sub>s</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>-ma</sub>t<sub>c</sub>h<sub>e</sub>d 24<sub>-s</sub>l<sub>o</sub>t 1<sub>comparison. Crucially, the two cost axes disagree on the preferred packaging, illustrating why no single cell serves as a universal</sub> <sub>recommen</sub>d<sub>a</sub>ti<sub>on.</sub>

## 8.7 Inference-Time Scaling: The Compute-Quality Frontier

To rigorously confirm that the superiority of iterative portfolio generation is a fundamental physical law of LLMs rather than an artifact of small-capacity models, we executed a budget-matched scaling replication at the extreme 14B and 32B scales (detailed in Table 13). Measured purely by informational yield, the multi-round 2, 12 configuration <sup>decisively</sup> <sup>obliterates</sup> <sup>the</sup> <sup>monolithic</sup> <sup>single-pass</sup> (<sup>24,</sup> <sup>1</sup>) <sup>baseline</sup> <sup>by</sup> <sup>an</sup> <sup>absolute</sup> +<sup>0.134</sup> <sup>(95%</sup> <sup>CI</sup> [+<sup>0.093,</sup> +<sup>0.175</sup>]<sup>)</sup> <sup>utilizing</sup> <sup>Qwen2.5-14B,</sup> <sup>and</sup> <sup>by</sup> +<sup>0.138</sup> <sup>utilizing</sup> <sup>Qwen2.5-32B.</sup>

However, translating these slot-matched gains into physical deployment exposes the fundamental mechanics of inference-time scaling. Achieving the massive recall of 2, 12 inherently demands deliberately investing greater test-time compute: sequential auto-regressive generation naturally increases temporal latency (e.g., 17.5 seconds versus a singular 1.7 seconds for 24, 1 ), and our causal LOO probe mandates strictly parallelizable, yet non-zero, teacher-forced forward passes.

Rather than viewing this computational overhead as a defect, Figure 6 maps this dynamic as a strict Compute-Quality scaling frontier. Monolithic single-pass architectures like 24, 1 are computationally cheap, but they hit a rigid cognitive ceiling. If a production system targets a modest 0.35 portfolio recall floor, generating a single wide response sufices. However, as the quality requirement scales to 0.40 or 0.45, the standard for complex exploratory search, no single-generation configuration remains mathematically capable of reaching the threshold, regardless of parameter scale.

This forcefully dictates a paradigm shift in generative search deployment. To breach the single-response quality ceiling, systems cannot simply widen the context; they must aggressively scale test-time compute. The multi-round Manuscript submitted to ACM

structures 2, 5 and 2, 12 prove that explicitly investing inference budget into sequential exploration and causal feedback is the only architectural mechanism capable of unlocking transformative gains in comprehensive evidence coverage.

## 9 C<sub>onc</sub>l<sub>us</sub>i<sub>on</sub>

Transitioning retrieval-augmented generation (RAG) from single-answer extraction to diverse portfolio generation is fundamentally stymied by flawed measurement heuristics and arbitrary resource allocation. In this work, we deconstructed the pervasive diagnostic illusion in RAG evaluation: we proved that the apparent perfection of traditional IR proxies is a structural mirage reliant on trivial of-query padding. Evaluated on rigorous same-query hard negatives, standard proxies collapse to random chance, whereas our intervention-based causal probe maintains highly robust discrimination. By formally calibrating out protocol-shift artifacts, we quantified the inherent dilution of generative attention—yielding a canonical width elasticity of 0.68 under controlled isolation—securing a rigorous causal instrument capable of genuinely auditing LLM evidence consumption.

Armed with this validated measurement capability, our deconfounded � � factorial grid definitively resolved the context budget dilemma. We established a fundamental generative law: scaling sequential generation count uniformly drives massive portfolio recall gains of 16.8 to 20.5 absolute percentage points. In stark contrast, simply expanding context width is an architectural trap actively penalized by rank degradation, yielding highly unstable returns that frequently harm tasks with concentrated relevance.

Exploiting this structural superiority of fresh evidence, we operationalized an attribution-steered submodular orchestration framework. Driven by causal feedback, our submodular scheduler systematically dominated all evaluated open-loop selection baselines. Augmented by an orthogonal contrastive decoder that acts as a cognitive override against attention inertia, our end-to-end architecture proves that intelligent, multi-round context orchestration fundamentally maximizes the generative yield of a retrieved pool.

Ultimately, this work formally introduces the paradigm of inference-time scaling to generative search. While classical RAG pipelines attempt to minimize latency by cramming evidence into a single, monolithic context pass, we prove this approach encounters a rigid cognitive ceiling. Instead, we demonstrate that deliberately investing computational budget during inference, via sequential autoregressive passes and causal teacher-forced probing, unlocks transformative gains in portfolio coverage. For future systems required to comprehensively cover a diverse evidence space, the paradigm is clear: aggressively scaling structured, feedback-driven test-time compute fundamentally supersedes static context maximization.

## R<sub>e</sub>f<sub>erences</sub>

[1] Rakesh Agrawal, Sreenivas Gollapudi, Alan Halverson, and Samuel Ieong. 2009. Diversifying search results. In Proceedings of the second ACM international conference on web search and data mining. 5–14

[2] Samuel Joseph Amouyal, Tomer Wolfson, Ohad Rubin, Ori Yoran, Jonathan Herzig, and Jonathan Berant. 2022. Qampari: An open-domain question answering benchmark for questions with many answers from multiple paragraphs. arXiv preprint arXiv:2205.12665 (2022).

[3] Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2024. L-eval: Instituting standardized evaluation for long context language models. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 14388–14411.

[4] Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In International conference on learning representations, Vol. 2024. 9112–9141.

[5] Ashwinkumar Badanidiyuru and Jan Vondrák. 2014. Fast algorithms for maximizing submodular functions. In Proceedings ofthe twenty-fifth annual ACM-SIAM symposium on Discrete algorithms. SIAM, 1497–1514.

Manuscript submitted to ACM

[6] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. 2024. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings ofthe 62nd annual meeting ofthe association for computational linguistics (volume 1: Long papers). 3119–3137.

[7] Bernd Bohnet, Vinh Q Tran, Pat Verga, Roee Aharoni, Daniel Andor, Livio Baldini Soares, Massimiliano Ciaramita, Jacob Eisenstein, Kuzman Ganchev, Jonathan Herzig, et al. 2022. Attributed question answering: Evaluation and modeling for attributed large language models. arXiv preprint arXiv:2212.08037 (2022).

[8] Sebastian Borgeaud, Arthur Mensch, Jordan Hofmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022. Improving language models by retrieving from trillions of tokens. In International conference on machine learning. PMLR, 2206–2240.

[9] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. 2024. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787 (2024).

[10] Jaime G Carbonell and Jade Goldstein. 1998. The use of MMR, diversity-based reranking for reordering documents and producing summaries.. In SIGIR, Vol. 98. 290941–291025.

[11] Olivier Chapelle, Thorsten Joachims, Filip Radlinski, and Yisong Yue. 2012. Large-scale validation and analysis of interleaved search evaluation ACM Transactions on Information Systems (TOIS) 30, 1 (2012), 1–41.

[12] Harr Chen and David R Karger. 2006. Less is more: probabilistic models for retrieving fewer relevant documents. In Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval. 429–436.

[13] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374 (2021).

[14] Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu, Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei Cao, Jie Ma, et al. 2025. A survey on knowledge-oriented retrieval-augmented generation. ACM Transactions on Information Systems (2025).

[15] Cheng-Han Chiang and Hung-yi Lee. 2023. Can large language models be an alternative to human evaluations?. In Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 15607–15631.

[16] Charles LA Clarke, Maheedhar Kolla, Gordon V Cormack, Olga Vechtomova, Azin Ashkan, Stefan Büttcher, and Ian MacKinnon. 2008. Novelty and diversity in information retrieval evaluation. In Proceedings ofthe 31st annual international ACM SIGIR conference on Research and development in information retrieval. 659–666.

[17] Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev, and Aleksander Mądry. 2024. Contextcite: Attributing model generation to context Advances in Neural Information Processing Systems 37, 95764–95807.

[18] Ian Covert, Scott Lundberg, and Su-In Lee. 2021. Explaining by removing: A unified framework for model explanation. Journal ofMachine Learning Research 22, 209 (2021), 1–90.

[19] Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola Tonellotto, and Fabrizio Silvestri. 2024. The power of noise: Redefining retrieval for rag systems. In Proceedings ofthe 47th international ACM SIGIR conference on research and development in information retrieval. 719–729.

[20] Van Dang and W Bruce Croft. 2012. Diversity by proportionality: an election-based approach to search result diversification. In Proceedings of the 35th international ACM SIGIR conference on Research and development in information retrieval. 65–74

[21] Zhirui Deng, Zhicheng Dou, Zhan Su, and Ji-Rong Wen. 2024. Multi-grained document modeling for search result diversification. ACM Transactions on Information Systems 42, 5 (2024), 1–22.

[22] Zhirui Deng, Zhicheng Dou, Yutao Zhu, and Ji-Rong Wen. 2025. A model-agnostic pre-training framework for search result diversification. ACM Transactions on Information Systems 44, 1 (2025), 1–23.

[23] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130 (2024).

[24] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. Ragas: Automated evaluation of retrieval augmented generation. In Proceedings of the 18th conference of the european chapter of the association for computational linguistics: system demonstrations. 150–158.

[25] Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019. ELI5: Long form question answering. In Proceedings ofthe 57th annual meeting ofthe association for computational linguistics. 3558–3567.

[26] Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings ofthe 56th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 889–898

[27] Hui Fang, Tao Tao, and Chengxiang Zhai. 2011. Diagnostic evaluation of information retrieval models. ACM Transactions on Information Systems (TOIS) 29, 2 (2011), 1–42.

[28] Marshall L Fisher, George L Nemhauser, and Laurence A Wolsey. 2009. An analysis of approximations for maximizing submodular set functions—II. In Polyhedral Combinatorics: Dedicated to the memory ofDR Fulkerson. Springer, 73–87.

[29] Dan Friedman and Adji Bousso Dieng. 2022. The vendi score: A diversity evaluation metric for machine learning. arXiv preprint arXiv:2210.02410 (2022).

[30] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling large language models to generate text with citations. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing. 6465–6488

[31] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2023. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997 (2023).

[32] Yunfan Gao, Yun Xiong, Wenlong Wu, Bohan Li, Yijie Zhong, and Haofen Wang. 2026. U-niah: Unified rag and llm evaluation for long context needle-in-a-haystack. ACM Transactions on Information Systems 44, 3 (2026), 1–30.

[33] Sreenivas Gollapudi and Aneesh Sharma. 2009. An axiomatic approach for result diversification. In Proceedings ofthe 18th international conference on World wide web. 381–390

[34] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).

[35] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. (2020), 3929–3938.

[36] Jordan Hofmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks Johannes Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556 (2022).

[37] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2019. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751.

[38] Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, and Been Kim. 2019. A benchmark for interpretability methods in deep neural networks. Advances in neural information processing systems 32.

[39] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. RULER: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654 (2024)

[40] Tianyi Hu, Maria Maistro, and Daniel Hershcovich. 2024. Bridging cultures in the kitchen: A framework and benchmark for cross-cultural recipe retrieval. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 1068–1080.

[41] Tianyi Hu, Andrea Morales-Garzón, Jingyi Zheng, Maria Maistro, and Daniel Hershcovich. 2026. Culinary crossroads: A rag framework for enhancing diversity in cross-cultural recipe adaptation. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2408–2423.

[42] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM transactions on information systems 43, 2 (2025), 1–55.

[43] Gautier Izacard and Edouard Grave. 2021. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings ofthe 16th conference ofthe european chapter ofthe association for computational linguistics: main volume. 874–880.

[44] Amir H Jadidinejad, Craig Macdonald, and Iadh Ounis. 2021. The simpson’s paradox in the ofline evaluation of recommendation systems. ACM Transactions on Information Systems (TOIS) 40, 1 (2021), 1–22

[45] Sarthak Jain and Byron C Wallace. 2019. Attention is not explanation. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 3543–3556.

[46] Kalervo Jarvelin and Eero Sormunen. 2024. A blueprint of IR evaluation integrating task and user characteristics. ACM Transactions on Information Systems 42, 6 (2024), 1–38.

[47] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM computing surveys 55, 12 (2023), 1–38.

[48] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7B. (2023). arXiv:2310.06825 [cs.CL] https://arxiv.org/abs/2310.06825

[49] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. Llmlingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 conference on empirical methods in natural language processing. 13358–13376

[50] Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024. Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 1658–1677.

[51] Zhengbao Jiang, Ji-Rong Wen, Zhicheng Dou, Wayne Xin Zhao, Jian-Yun Nie, and Ming Yue. 2017. Learning to diversify search results via subtopic attention. In Proceedings ofthe 40th international ACM SIGIR Conference on Research and Development in Information Retrieval. 545–554

[52] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. (2023), 7969–7992.

[53] Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan Arik. 2025. Long-context llms meet rag: Overcoming challenges for long inputs in rag. In International Conference on Learning Representations, Vol. 2025. 37784–37822.

[54] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jefrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361 (2020).

[55] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings ofthe 2020 conference on empirical methods in natural language processing (EMNLP). 6769–6781.

[56] Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. 2022. Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive nlp. arXiv preprint arXiv:2212.14024 (2022).

Manuscript submitted to ACM

[57] Alex Kulesza and Ben Taskar. 2012. Determinantal point processes for machine learning. Foundations and Trends® in Machine Learning 5, 2-3 (2012), 123–286.

[58] Youngwon Lee, Seung-won Hwang, Daniel F Campos, Filip Gralinski, Zhewei Yao, and Yuxiong He. 2025. Inference scaling for bridging retrieval and augmented generation. In Findings ofthe Association for Computational Linguistics: NAACL 2025. 7339–7354.

[59] Jurek Leonhardt, Koustav Rudra, and Avishek Anand. 2023. Extractive explanations for interpretable text ranking. ACM Transactions on Information Systems 41, 4 (2023), 1–31.

[60] Mosh Levy, Alon Jacoby, and Yoav Goldberg. 2024. Same task, more tokens: the impact of input length on the reasoning performance of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 15339–15353.

[61] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems 33, 9459–9474.

[62] Hang Li, Ahmed Mourad, Shengyao Zhuang, Bevan Koopman, and Guido Zuccon. 2023. Pseudo relevance feedback with deep language models and dense retrievers: Successes and pitfalls. ACM Transactions on Information Systems 41, 3 (2023), 1–40

[63] Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and William B Dolan. 2016. A diversity-promoting objective function for neural conversation models. In Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. 110–119.

[64] Jiwei Li, Will Monroe, and Dan Jurafsky. 2016. Understanding neural networks through representation erasure. arXiv preprint arXiv:1612.08220 (2016).

[65] Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. 2024. Long-context llms struggle with long in-context learning. arXiv preprint arXiv:2404.02060 (2024).

[66] Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yuyao Zhang, Peitian Zhang, Yutao Zhu, and Zhicheng Dou. 2025. From matching to generation: A survey on generative information retrieval. ACM Transactions on Information Systems 43, 3 (2025), 1–62.

[67] Xinze Li, Hanbin Wang, Zhenghao Liu, Shi Yu, Shuo Wang, Yukun Yan, Yukai Fu, Yu Gu, and Ge Yu. 2025. Building a coding assistant via the retrieval-augmented language model. ACM Transactions on Information Systems 43, 2 (2025), 1–25.

[68] Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori B Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2023. Contrastive decoding: Open-ended text generation as optimization. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: Long papers). 12286–12312.

[69] Shangsong Liang, Emine Yilmaz, Hong Shen, Maarten De Rijke, and W Bruce Croft. 2017. Search result diversification in short text streams. ACM Transactions on Information Systems (TOIS) 36, 1 (2017), 1–35.

[70] Hui Lin and Jef Bilmes. 2011. A class of submodular functions for document summarization. In Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies. 510–520.

[71] Jimmy Lin. 2007. An exploration of the principles underlying redundancy-based factoid question answering. ACM Transactions on Information Systems (TOIS) 25, 2 (2007), 6–es.

[72] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe association for computational linguistics 12 (2024), 157–173

[73] Nelson F Liu, Tianyi Zhang, and Percy Liang. 2023. Evaluating verifiability in generative search engines. In Findings ofthe Association for Computational Linguistics: EMNLP 2023. 7001–7025

[74] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using gpt-4 with better human alignment. In Proceedings ofthe 2023 conference on empirical methods in natural language processing. 2511–2522.

[75] Zeyang Liu, Ke Zhou, and Max L Wilson. 2021. Meta-evaluation of conversational search evaluation metrics. ACM Transactions on Information Systems (TOIS) 39, 4 (2021), 1–42

[76] Scott M Lundberg and Su-In Lee. 2017. A unified approach to interpreting model predictions. Advances in neural information processing systems 30.

[77] Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong, Bo Tang, Wenjin Wang, Hao Wu, Huanyong Liu, Tong Xu, and Enhong Chen. 2025. Crud-rag: A comprehensive chinese benchmark for retrieval-augmented generation of large language models. ACM Transactions on Information Systems 43, 2 (2025), 1–32.

[78] Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth Sieber, Mark Yatskar, and Dan Roth. 2024. ExpertQA: Expert-curated questions and attributed answers. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 3025–3045.

[79] Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. On faithfulness and factuality in abstractive summarization. In Proceedings ofthe 58th annual meeting ofthe association for computational linguistics. 1906–1919.

[80] Chuan Meng, Negar Arabzadeh, Arian Askari, Mohammad Aliannejadi, and Maarten de Rijke. 2025. Query performance prediction using relevance judgments generated by large language models. ACM Transactions on Information Systems 43, 4 (2025), 1–35.

[81] Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song, Martin Chadwick, Mia Glaese, Susannah Young, Lucy Campbell-Gillingham, Geofrey Irving, et al. 2022. Teaching language models to support answers with verified quotes. arXiv preprint arXiv:2203.11147 (2022).

[85] Fengran Mo, Kelong Mao, Ziliang Zhao, Hongjin Qian, Haonan Chen, Yiruo Cheng, Xiaoxi Li, Yutao Zhu, Zhicheng Dou, and Jian-Yun Nie. 2025. A survey of conversational search. ACM Transactions on Information Systems 43, 6 (2025), 1–50.

[82] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings ofthe 2023 conference on empirical methods in natural language processing. 12076–12100.

[83] Michel Minoux. 2005. Accelerated greedy algorithms for maximizing submodular set functions. In Optimization Techniques: Proceedings ofthe 8th IFIP Conference on Optimization Techniques Würzburg, September 5–9, 1977. Springer, 234–243.

[84] Baharan Mirzasoleiman, Ashwinkumar Badanidiyuru, Amin Karbasi, Jan Vondrák, and Andreas Krause. 2015. Lazier than lazy greedy. 29, 1 (2015).

[86] Alistair Mofat, Peter Bailey, Falk Scholer, and Paul Thomas. 2017. Incorporating user expectations and behavior into the measurement of search efectiveness. ACM Transactions on Information Systems (TOIS) 35, 3 (2017), 1–38

[87] Alistair Mofat and Justin Zobel. 2008. Rank-biased precision for measurement of retrieval efectiveness. ACM Transactions on Information Systems (TOIS) 27, 1 (2008), 1–27

[88] Andrea Morales-Garzón, Oscar A Rocha, Sara Benel Ramirez, Gabriel Tuco Casquino, and Alberto Medina. 2024. Healthy cooking with large language models, supervised fine-tuning, and retrieval augmented generation. In Proceedings ofthe LatinX in AI Workshop at NAACL.

[89] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jef Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. 2021. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332 (2021).

[90] George L Nemhauser, Laurence A Wolsey, and Marshall L Fisher. 1978. An analysis of approximations for maximizing submodular set functions—I. Mathematical programming 14, 1 (1978), 265–294.

[91] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. 2025. Graph retrieval-augmented generation: A survey. ACM Transactions on Information Systems 44, 2 (2025), 1–52.

[92] Xubo Qin, Zhicheng Dou, Yutao Zhu, and Ji-Rong Wen. 2023. GDESA: Greedy diversity encoder with self-attention for search results diversification ACM Transactions on Information Systems 41, 2 (2023), 1–36.

[93] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical Report. (2025). arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115

[94] Filip Radlinski and Susan Dumais. 2006. Improving personalized web search using result diversification. In Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval. 691–692

[95] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023. In-context retrievalaugmented language models. Transactions of the Association for Computational Linguistics 11 (2023), 1316–1331.

[96] Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and David Reitter. 2023. Measuring attribution in natural language generation models. Computational Linguistics 49, 4 (2023), 777–840.

[97] Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP). 3982–3992.

[98] Mohammad Reza Rezaei and Adji Bousso Dieng. 2025. Vendi-rag: Adaptively trading-of diversity and quality significantly improves retrieval augmented generation with llms. arXiv preprint arXiv:2502.11228 (2025)

[99] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. 1135–1144.

[100] Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. 2024. Ares: An automated evaluation framework for retrieval-augmented generation systems. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 338–354

[101] Tetsuya Sakai and Ruihua Song. 2011. Evaluating diversified search results using per-intent graded relevance. In Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval. 1043–1052.

[102] Rodrygo LT Santos, Craig Macdonald, and Iadh Ounis. 2010. Exploiting query reformulations for web search result diversification. In Proceedings ofthe 19th international conference on World wide web. 881–890.

[103] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher Manning. 2024. Raptor: Recursive abstractive processing for tree-organized retrieval. In International Conference on Learning Representations, Vol. 2024. 32628–32649

[104] Sofia Serrano and Noah A Smith. 2019. Is attention interpretable? (2019), 2931–2951.

[105] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language models can be easily distracted by irrelevant context. In International conference on machine learning. PMLR, 31210–31227.

[106] Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, and Wen-tau Yih. 2024. Trusting your evidence: Hallucinate less with context-aware decoding. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers). 783–791.

[107] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2024. Replug: Retrievalaugmented black-box language models. In Proceedings ofthe 2024 conference ofthe north american chapter ofthe association for computational linguistics: Human language technologies (volume 1: Long papers). 8371–8384.

[108] Zhengliang Shi, Lingyong Yan, Weiwei Sun, Yue Feng, Pengjie Ren, Xinyu Ma, Shuaiqiang Wang, Dawei Yin, Maarten de Rijke, and Zhaochun Ren. 2026. Direct retrieval-augmented optimization: Synergizing knowledge selection and language models. ACM Transactions on Information Systems 44, 4 (2026), 1–30.

[109] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling llm test-time compute optimally can be more efective than scaling model parameters. arXiv preprint arXiv:2408.03314 (2024).

[110] Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. 2022. ASQA: Factoid questions meet long-form answers. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing. 8273–8288.

[111] Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier. 2022. A contrastive framework for neural text generation Advances in neural information processing systems 35, 21548–21561.

[112] Zhan Su, Zhicheng Dou, Yutao Zhu, and Ji-Rong Wen. 2024. Passage-aware search result diversification. ACM Transactions on Information Systems 42, 5 (2024), 1–29.

[113] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic attribution for deep networks. In International conference on machine learning PMLR, 3319–3328.

[114] Yubao Tang, Ruqing Zhang, Jiafeng Guo, Maarten De Rijke, Wei Chen, and Xueqi Cheng. 2024. Listwise generative retrieval models via a sequential learning process. ACM Transactions on Information Systems 42, 5 (2024), 1–31

[115] Leila Tavakoli, Johanne R Trippas, Hamed Zamani, Falk Scholer, and Mark Sanderson. 2024. Online and ofline evaluation in search clarification ACM Transactions on Information Systems 43, 1 (2024), 1–30

[116] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663

[117] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In Proceedings ofthe 61st annual meeting ofthe association for computational linguistics (volume 1: long papers). 10014–10037.

[118] Mehmet Deniz Türkmen, Mucahid Kutlu, Bahadir Altun, and Gokalp Cosgun. 2025. Gentrec: The first test collection generated by large language models for evaluating information retrieval systems. ACM Transactions on Information Systems (2025).

[119] Ashwin Vijayakumar, Michael Cogswell, Ramprasaath Selvaraju, Qing Sun, Stefan Lee, David Crandall, and Dhruv Batra. 2018. Diverse beam search for improved description of complex scenes. 32, 1 (2018).

[120] Ellen M Voorhees, Daniel Samarov, and Ian Soborof. 2017. Using replicates in information retrieval evaluation. ACM Transactions on Information Systems (TOIS) 36, 2 (2017), 1–21.

[121] Lei Wang, Jingsen Zhang, Hao Yang, Zhi-Yuan Chen, Jiakai Tang, Zeyu Zhang, Xu Chen, Yankai Lin, Hao Sun, Ruihua Song, et al. 2025. User behavior simulation with large language model-based agents. ACM Transactions on Information Systems 43, 2 (2025), 1–37.

[122] Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi, Zhengyuan Wang, Shizheng Li, Qi Qian, et al. 2024 Searching for best practices in retrieval-augmented generation. In Proceedings of the 2024 conference on empirical methods in natural language processing. 17716–17736.

[123] Zhichao Wang, Bin Bi, Yanqi Luo, Sitaram Asur, and Claire Na Cheng. 2025. Diversity Enhances an LLM’s Performance in RAG and Long-context Task. arXiv preprint arXiv:2502.09017 (2025).

[124] Zilong Ryan Wang, Zifeng Wang, Long Le, Huaixiu Steven Zheng, Swaroop Mishra, Vincent Perot, Yuwei Zhang, Anush Mattapalli, Ankur Taly, Jingbo Shang, et al. 2025. Speculative rag: Enhancing retrieval augmented generation through drafting. In International Conference on Learning Representations, Vol. 2025. 18483–18505.

[125] Sarah Wiegrefe and Yuval Pinter. 2019. Attention is not not explanation. In Proceedings ofthe 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP). 11–20.

[126] Long Xia, Jun Xu, Yanyan Lan, Jiafeng Guo, and Xueqi Cheng. 2015. Learning maximal marginal relevance model via directly optimizing diversity evaluation measures. In Proceedings ofthe 38th international ACM SIGIR conference on research and development in information retrieval. 113–122.

[127] Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. RECOMP: Improving retrieval-augmented LMs with context compression and selective augmentation. In International Conference on Learning Representations, Vol. 2024. 43478–43502.

[128] Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. 2024. Retrieval meets long context large language models. In International Conference on Learning Representations, Vol. 2024. 49569–49584.

[129] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing. 2369–2380.

[130] Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. 2024. Making retrieval-augmented language models robust to irrelevant context. In International Conference on Learning Representations, Vol. 2024. 29862–29883

[131] Tan Yu, Anbang Xu, and Rama Akkiraju. 2024. In defense of rag in the era of long-context language models. arXiv preprint arXiv:2409.01666 (2024). Manuscript submitted to ACM

[132] Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang. 2022. Generate rather than retrieve: Large language models are strong context generators. arXiv preprint arXiv:2209.10063

[133] Xiang Yue, Boshi Wang, Ziru Chen, Kai Zhang, Yu Su, and Huan Sun. 2023. Automatic evaluation of attribution by large language models. In Findings of the Association for Computational Linguistics: EMNLP 2023. 4615–4635.

[134] Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, RolfJagerman, Hansi Zeng, Zhen Qin, Dong Wang, Xuanhui Wang, and Michael Bendersky. 2025 Inference scaling for long-context retrieval augmented generation. In International Conference on Learning Representations, Vol. 2025. 72914–72938.

[135] ChengXiang Zhai, William W Cohen, and John Laferty. 2015. Beyond independent relevance: methods and evaluation metrics for subtopic retrieval. In Acm sigirforum, Vol. 49. ACM New York, NY, USA, 2–9.

[136] Chao Zhang, Yuhao Wang, Derong Xu, Haoxin Zhang, Yuanjie Lyu, Yuhao Chen, Shuochen Liu, Tong Xu, Xiangyu Zhao, Yan Gao, et al. 2026 Tearag: A token-eficient agentic retrieval-augmented generation framework. ACM Transactions on Information Systems 44, 6 (2026), 1–35.

[137] Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and Joseph E Gonzalez. 2024. Raft: Adapting language model to domain specific rag. arXiv preprint arXiv:2403.10131 (2024)

[138] Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. 2025. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems 43, 6 (2025), 1–47.

[139] Wayne Xin Zhao, Jing Liu, Ruiyang Ren, and Ji-Rong Wen. 2024. Dense text retrieval based on pretrained language models: A survey. ACM Transactions on Information Systems 42, 4 (2024), 1–60.

[140] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems 36, 46595–46623.

[141] Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. 2018. Texygen: A benchmarking platform for text generation models. In The 41st international ACM SIGIR conference on research & development in information retrieval. 1097–1100.

[142] Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Haonan Chen, Zheng Liu, Zhicheng Dou, and Ji-Rong Wen 2025. Large language models for information retrieval: A survey. ACM Transactions on Information Systems 44, 1 (2025), 1–54.

## A F<sub>orma</sub>l M<sub>a</sub>th<sub>ema</sub>ti<sub>ca</sub>l F<sub>ramewor</sub>k <sub>an</sub>d P<sub>roo</sub>f<sub>s</sub>

To establish the theoretical rigor underpinning our portfolio generation architecture, this appendix details the submodular optimization guarantees of our scheduling lever and formalizes the idealized theoretical bounds against which our empirical violations are measured.

## A<sub>.</sub>1 S<sub>u</sub>b<sub>mo</sub>d<sub>u</sub>l<sub>ar</sub> O<sub>p</sub>ti<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> f<sub>or</sub> E<sub>v</sub>id<sub>ence</sub> O<sub>rc</sub>h<sub>es</sub>t<sub>ra</sub>ti<sub>on</sub>

At round �, the Ascp scheduler greedily constructs a size-� context � by maximizing a discrete set function that inherently penalizes redundancy through causal feedback. The objective is defined as:

$$
F _ { t } ( C ) = \sum _ { z \in \mathcal { Z } } \pi ( z ) \beta ^ { n _ { t } ( z ) } \Big [ 1 - \prod _ { d \in C } \left( 1 - v _ { t } ( d ) W [ d , z ] \right) \Big ] + \lambda \sum _ { d \in C } \frac { r ( d ) } { k \left( 1 + u _ { t } ( d ) \right) } ,\tag{6}
$$

where dynamic document scaling $v _ { t } ( d ) = \beta _ { \mathrm { d o c } } ^ { u _ { t } ( d ) }$ operates alongside measured document attribution $u _ { t }$ and facet attribution $n _ { t }$ . Static parameters include document–facet coverage $W [ d , z ]$ and query-conditional facet importance � � . This construction ensures that historical evidence utilization directly decays the marginal utility of future redundant exposures.

Proposition 1. For any fixed causal feedback state $( u _ { t } , n _ { t } )$ at round $t ,$ the context orchestration set function $F _ { t }$ defined in Eq. (6) satisfies $F _ { t } ( \boldsymbol { 0 } ) = 0$ and is strictly monotone non-decreasing and submodular.

Proof. We decompose the objective into coverage and relevance components: $F _ { t } = G _ { t } + \lambda M _ { t }$ . The relevance term $\begin{array} { r } { M _ { t } ( C ) = \sum _ { d \in C } r ( d ) / ( k ( 1 + u _ { t } ( d ) ) ) } \end{array}$ ) <sup>operates</sup> <sup>as</sup> <sup>a</sup> <sup>direct</sup> <sup>sum</sup> <sup>of</sup> <sup>non-negative</sup> <sup>per-element</sup> <sup>weights,</sup> <sup>rendering</sup> <sup>it</sup> <sup>inherently</sup> modular, monotone, and zero on the empty set. For the coverage component, we isolate $c _ { z } = \pi ( z ) \beta ^ { n _ { t } ( z ) } \geq 0$ and $w _ { z } ( d ) = v _ { t } ( d ) W [ d , z ] \in [ 0 , 1 ]$ , expressing $\begin{array} { r } { G _ { t } ( C ) = \sum _ { z } c _ { z } g _ { z } ( C ) } \end{array}$ where $\begin{array} { r } { g _ { z } ( C ) = 1 - \prod _ { d \in C } ( 1 - w _ { z } ( d ) ) } \end{array}$ . Because $c _ { z }$ and $w _ { z }$ are strictly state-dependent and invariant to the current context candidate $C ,$ we evaluate a fixed facet �. Trivially,

Manuscript submitted to ACM

$g _ { z } ( 0 ) = 0$ . For any document � $\notin C ,$ the marginal gain is mathematically bounded:

$$
g _ { z } ( C \cup \{ d \} ) - g _ { z } ( C ) = w _ { z } ( d ) \prod _ { e \in C } \left( 1 - w _ { z } ( e ) \right) ~ \ge ~ 0 ,\tag{7}
$$

proving $g _ { z }$ is monotone. To establish submodularity, consider subsets $A \subseteq B$ and a document � ∉ �. Because every factor $1 - w _ { z } ( e )$ is constrained within 0, 1 , the product inequality $\begin{array} { r } { \prod _ { e \in A } ( 1 - w _ { z } ( e ) ) \geq \prod _ { e \in B } ( 1 - w _ { z } ( e ) ) } \end{array}$ holds universally. Substituting this into $\operatorname { E q . } \left( 7 \right)$ satisfies the strict diminishing-returns property. As non-negative linear combinations preserve monotone submodularity, $F _ { t }$ is submodular. □

Consequently, standard greedy selection securely attains the robust $F _ { t } ( C ^ { g } ) \geq ( 1 - e ^ { - 1 } ) \operatorname* { m a x } _ { | C | \leq k } F _ { t } ( C )$ approximation guarantee. If core evidence � dictates mandatory pinning, optimizing the residual $F _ { t } ^ { \prime } ( S ) = F _ { t } ( K \cup S ) - F _ { t } ( K )$ maintains $F _ { t } ( K \cup S ^ { g } ) \geq ( 1 - e ^ { - 1 } ) \operatorname* { m a x } _ { K \subseteq C , | C | \leq k } F _ { t } ( C ) + e ^ { - 1 } F _ { t } ( K )$ . Because attribution feedback dynamically updates across rounds, the absolute objective evolves sequentially, yet intra-round submodularity is mathematically preserved.

## A<sub>.</sub>2 Brid<sub>g</sub>in<sub>g</sub> the Ga<sub>p</sub>: Theoretical Inde<sub>p</sub>endence vs<sub>.</sub> Generative Com<sub>p</sub>lexit<sub>y</sub>

While mathematical bounds provide an elegant structural ceiling, real-world generative evidence consumption systematically diverges from idealized theoretical assumptions, necessitating our rigorous empirical design. Consider the baseline assumption of strictly independent utilization, which posits that a generator grounds on a random subset of a provided context with an expected size $\mathbb { E } | S | = q \left| C \right|$ , where $q \in ( 0 , 1 )$ is entirely independent of the context’s specific contents or historical rounds.

Evaluating this baseline assumption via a mixed-efects model against our actual generative data decisively rejects policy invariance $( \chi ^ { 2 } = 1 7 1 . 4 , p = 6 \times 1 0 ^ { - 3 1 } )$ ; empirical utilization is over-dispersed by a factor of 1.39 and exhibits heavy cross-round correlation dictated by structural redundancy.

This profound generative complexity directly impacts optimal budget allocation. If we formalize relevance density $\rho ( i )$ as the non-increasing probability that a document at rank � carries a gold answer unit, a rotation policy ofering sequential ranks $1 , \ldots , k T$ yields an expected answer-bearing coverage of $\begin{array} { r } { q \sum _ { i \leq k T } \rho ( i ) } \end{array}$ . Conversely, a selection policy operating over a restricted top- $. N ^ { \prime }$ pool yields $\begin{array} { r } { \sum _ { i \leq m } \rho ( i ) \left[ 1 - ( 1 - q ) ^ { n _ { i } } \right] \leq \sum _ { i \leq m } \rho ( i ) } \end{array}$

Proposition 2 (Selection versus Rotation Regime). Under idealized consumption, sequential rotation mathematically dominates bounded selection whenever � $\begin{array} { r } { \sum _ { i = m + 1 } ^ { k T } \rho ( i ) \ > \ \sum _ { i \leq m } \rho ( i ) \Big [ ( 1 - q ) - ( 1 - q ) ^ { n _ { i } } \Big ] } \end{array}$ , where the right-hand side represents the marginal yield extracted from re-ofering previously chosen documents.

This inequality formally dictates that selection is theoretically optimal when relevance decays sharply, whereas rotation dominates when tail evidence retains value. However, pushing this logic to a global budget allocation boundary exposes the limits of pure mathematical bounding. Our full factorial grid actively confirms that empirical portfolio recall is governed by highly correlated sequential generative rounds where � carries the true causal efect, proving that allocation logic must be strictly governed by the deconfounded empirical laws of LLM behavior rather than isolated mathematical abstraction.

## B Com<sub>p</sub>rehensive S<sub>y</sub>stem Dia<sub>g</sub>nostics and Robustness

To ensure the statistical validity of our core systemic findings, we conduct exhaustive diagnostic testing across stochastic seed variation, decoder architecture parity, multiple hypothesis testing, and intrinsic pool redundancy. Manuscript submitted to ACM

T<sub>a</sub>bl<sub>e</sub> 14<sub>.</sub> S<sub>ee</sub>d <sub>rep</sub>li<sub>ca</sub>ti<sub>on o</sub>f th<sub>e</sub> f<sub>ac</sub>t<sub>or</sub>i<sub>a</sub>l <sub>un</sub>d<sub>er a severe</sub>l<sub>y cons</sub>t<sub>ra</sub>i<sub>ne</sub>d � = 30 <sub>re</sub>t<sub>r</sub>i<sub>eva</sub>l <sub>poo</sub>l<sub>.</sub> U<sub>n</sub>lik<sub>e</sub> th<sub>e</sub> d<sub>eep</sub> � = 400 <sub>poo</sub>l i<sub>n</sub> th<sub>e</sub> main text (which <sub>p</sub>revents evidence exhaustion), this artificiall<sub>y</sub> restricted setin<sub>g</sub> starves the se<sub>q</sub>uential iterations of fresh evidence, <sub>na</sub>t<sub>ura</sub>ll<sub>y compress</sub>i<sub>ng</sub> th<sub>e a</sub>b<sub>so</sub>l<sub>u</sub>t<sub>e s</sub>t<sub>ruc</sub>t<sub>ura</sub>l <sub>gaps</sub> (<sub>e.g.,</sub> th<sub>e</sub> b<sub>u</sub>d<sub>ge</sub>t<sub>-ma</sub>t<sub>c</sub>h<sub>e</sub>d <sub>ga</sub>i<sub>n s</sub>h<sub>r</sub>i<sub>n</sub>k<sub>s</sub> t<sub>o</sub> 0.065)<sub>.</sub> C<sub>ruc</sub>i<sub>a</sub>ll<sub>y,</sub> d<sub>esp</sub>it<sub>e</sub> thi<sub>s</sub> <sub>compresse</sub>d <sub>e</sub>f<sub>ec</sub>t <sub>s</sub>i<sub>ze,</sub> th<sub>e cross-see</sub>d <sub>s</sub>t<sub>an</sub>d<sub>ar</sub>d d<sub>ev</sub>i<sub>a</sub>ti<sub>on rema</sub>i<sub>ns neg</sub>li<sub>g</sub>ibl<sub>e</sub> $( \leq 0 . 0 1 1 )$ . This <sub>p</sub>roves that al<sub>g</sub>orithmic stochasticit<sub>y</sub> <sub>canno</sub>t <sub>exp</sub>l<sub>a</sub>i<sub>n our a</sub>ll<sub>oca</sub>ti<sub>on</sub> l<sub>aws, even un</sub>d<sub>er ex</sub>t<sub>reme resource s</sub>t<sub>arva</sub>ti<sub>on.</sub>
<table><tr><td>contrast</td><td>seed 0</td><td>seed 1</td><td>seed 2</td><td>mean</td><td>s.d.</td></tr><tr><td>count T12 vs T1 @k2</td><td>0.141</td><td>0.137</td><td>0.154</td><td>0.144</td><td>0.0090</td></tr><tr><td>count T12 vs T1 @k24</td><td>0.159</td><td>0.160</td><td>0.141</td><td>0.153</td><td>0.0109</td></tr><tr><td>width k24 vs k2 @T12</td><td>0.097</td><td>0.099</td><td>0.090</td><td>0.095</td><td>0.0049</td></tr><tr><td>width k24 vs k2 @T1</td><td>0.092</td><td>0.094</td><td>0.097</td><td>0.094</td><td>0.0024</td></tr><tr><td>rot over fix @k2T12</td><td>一</td><td>1</td><td>一</td><td>一</td><td>一</td></tr><tr><td>rot over fix @k24T12</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>budget (2,12) vs (24,1)</td><td>0.062</td><td>0.061</td><td>0.072</td><td>0.065</td><td>0.0058</td></tr></table>

## B<sub>.</sub>1 Stochastic Consistenc<sub>y</sub> and Decoder Parit<sub>y</sub>

A critical vulnerability in generative evaluation is the confounding variance introduced by stochastic decoding algorithms. To definitively prove that our context allocation laws are structurally driven rather than transient artifacts of a specific generation trace, we fully replicated the factorial grid across multiple independent decoding seeds.

To subject our findings to an extreme stress test, we executed this seed replication under a deliberately constrained retrieval pool $( N = 3 0$ , as opposed to the unconstrained � = 400 pool utilized in the primary Table 4). This severe restriction forces early evidence exhaustion, naturally compressing the absolute magnitude of the structural gains (e.g., attenuating the 2, 12 vs 24, 1 budget-matched contrast to an average of 0.065). However, holding the query sample strictly fixed, we observed cross-seed standard deviations tightly constrained between 0.002 and 0.011 (Table 14). Because these stochastic deviations remain an order of magnitude below even these artificially compressed structural contrasts, we definitively conclude that the structural benefit of iterative portfolio generation entirely eclipses baseline generative stochasticity, even under severe resource starvation.

Furthermore, any performance deltas attributed to our custom attribution-steered decoder (our cognitive override mechanism) must be absolutely isolated from arbitrary sampling discrepancies. We rigorously established baseline parity: by setting our custom steering strength (�) and adaptive-plausibility cutof (�) to zero, our specialized loop mathematically reproduces ordinary generation exactly across 240 held-out query conditions. This yields structurally identical texts, selected document traces, and resulting attribution vectors. Consequently, the performance deltas reported in our five-seed decomposition (Table 12) strictly isolate the true, unconfounded causal impact of the logit contrast mechanism.

## B<sub>.</sub>2 M<sub>u</sub>lti<sub>p</sub>li<sub>c</sub>it<sub>y</sub> C<sub>on</sub>t<sub>ro</sub>l <sub>an</sub>d R<sub>e</sub>d<sub>un</sub>d<sub>ancy</sub> St<sub>ra</sub>tifi<sub>ca</sub>ti<sub>on</sub>

Given the scale of our factorial interactions, spurious significance via multiple comparisons is a severe risk. We predeclared eight distinct testing families and enforced strict Benjamini-Hochberg False Discovery Rate (BH-FDR) control at $\alpha = 0 . 0 5$ . This correction proved our findings exceptionally robust: 108 of the 123 executed statistical tests survive correction entirely unchanged. Crucially, even our most marginal preliminary observations (e.g., the ASQA/Llama width contrast at $T = 1 2$ and the integrative-instruction width contrast) maintained absolute significance post-correction at a stable $q = 0 . 0 3 4$ , while all primary generation-count contrasts securely rested at the absolute bootstrap floor.

Beyond statistical correction, we isolated the physical confound of natural-pool redundancy. By stratifying 2, 880 query-level observations across six context widths using pairwise entailment, answer attestation, and maximum embedding similarity (exhibiting loose internal correlation $r = 0 . 1 2 \ – 0 . 3 5 )$ , we tracked continuous log � redundancy interactions. The structural dilution of evidence utilization firmly survives in even the lowest-redundancy environments.

![](images/01145ac305aef267ec2847612d102a1b5e1a8c0dab90f8559c61d75dcd01fb65.jpg)

![](images/5ce3a6665cda0c5d65761c565df5cb3a70ccffdaef7f155834f887bdac1a8c38.jpg)

![](images/a9b36b74d1e668d4f402668462a1e04876c7e3aac7c4b51643d98e832f7105c5.jpg)  
Fi<sub>g.</sub> 7<sub>.</sub> J<sub>u</sub>d<sub>ge</sub> <sub>me</sub>t<sub>a-eva</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>over</sub> 179 <sub>por</sub>tf<sub>o</sub>li<sub>os</sub> <sub>an</sub>d 858 d<sub>ocumen</sub>t<sub>-</sub>l<sub>eve</sub>l <sub>ju</sub>d<sub>gemen</sub>t<sub>s.</sub> (<sub>a</sub>) Id<sub>en</sub>tif<sub>y</sub>i<sub>ng</sub> th<sub>e</sub> <sub>suppor</sub>ti<sub>ng</sub> d<sub>ocumen</sub>t<sub>:</sub> b<sub>o</sub>th t<sup>h</sup>e probe and t<sup>h</sup>e embedding proxy trac<sup>k</sup> t<sup>h</sup>e judge<sup>’</sup>s pro<sup>f</sup>i<sup>l</sup>e equa<sup>ll</sup>y we<sup>ll</sup> overa<sup>ll</sup>, but t<sup>h</sup>e probe is signi<sup>f</sup>icant<sup>l</sup>y beter at identi<sup>f</sup>ying t<sup>h</sup>e <sub>p</sub>rimar<sub>y</sub> source. (b) A com<sub>p</sub>arison of <sub>p</sub>robe-derived and jud<sub>g</sub>e-derived covera<sub>g</sub>e measures. (c) Correlation with the jud<sub>g</sub>e’s assessment <sub>o</sub>f textual <sub>nove</sub>lt<sub>y.</sub> A <sub>response-on</sub>l<sub>y ju</sub>d<sub>ge eva</sub>l<sub>ua</sub>t<sub>es</sub> t<sub>ex</sub>t<sub>ua</sub>l <sub>nove</sub>lt<sub>y</sub> b<sub>y</sub> h<sub>eav</sub>il<sub>y</sub> t<sub>rac</sub>ki<sub>ng</sub> th<sub>e</sub> di<sub>s</sub>ti<sub>nc</sub>t<sub>-</sub>2 <sub>me</sub>t<sub>r</sub>i<sub>c ra</sub>th<sub>er</sub> th<sub>an ac</sub>t<sub>ua</sub>l <sub>ev</sub>id<sub>ence</sub> <sup>covera</sup>g<sup>e.</sup>

Table 15. Com<sub>p</sub>onent ablations and <sub>p</sub>robe substitutions (<sub>q</sub>wen). Atribution-feedback removal leaves o<sub>p</sub>en-loo<sub>p</sub> schedulin<sub>g</sub>; <sub>p</sub>robe-<sub>su</sub>b<sub>s</sub>tit<sub>u</sub>ti<sub>on rows</sub> k<sub>eep</sub> th<sub>e sc</sub>h<sub>e</sub>d<sub>u</sub>l<sub>er</sub> fi<sub>xe</sub>d <sub>an</sub>d <sub>swap on</sub>l<sub>y</sub> th<sub>e ev</sub>id<sub>ence-use s</sub>i<sub>gna</sub>l<sub>.</sub>
<table><tr><td rowspan="2">Variant</td><td colspan="2">ASQA</td><td colspan="2">QAMPARI</td><td colspan="2">ELI5</td><td colspan="2">Recipes</td></tr><tr><td>PR@T</td><td>ECR</td><td>PR@T</td><td>ECR</td><td>PR@T</td><td>ECR</td><td>PR@T</td><td>ECR</td></tr><tr><td>ASCP (full)</td><td>0.493</td><td>0.331</td><td>0.162</td><td>0.324</td><td>0.273</td><td>0.459</td><td>0.238</td><td>0.371</td></tr><tr><td>– attribution feedback</td><td>0.465</td><td>0.320</td><td>0.163</td><td>0.309</td><td>0.248</td><td>0.459</td><td>0.242</td><td>0.398</td></tr><tr><td>– submodular selection</td><td>0.470</td><td>0.264</td><td>0.158</td><td>0.265</td><td>0.257</td><td>0.408</td><td>0.220</td><td>0.319</td></tr><tr><td>– steered decoding</td><td>0.488</td><td>0.325</td><td>0.161</td><td>0.321</td><td>0.278</td><td>0.465</td><td>0.231</td><td>0.378</td></tr><tr><td>steered decoding only</td><td>0.398</td><td>0.150</td><td>0.107</td><td>0.122</td><td>0.232</td><td>0.177</td><td>0.221</td><td>0.197</td></tr><tr><td>+ guardrail (κ=1)</td><td>0.434</td><td>0.229</td><td>0.136</td><td>0.221</td><td>0.237</td><td>0.271</td><td>0.225</td><td>0.299</td></tr><tr><td>+ guardrail (κ=2)</td><td>0.417</td><td>0.210</td><td>0.133</td><td>0.188</td><td>0.237</td><td>0.233</td><td>0.215</td><td>0.282</td></tr><tr><td>probe → hierarchical LOO</td><td>0.495</td><td>0.341</td><td>0.165</td><td>0.327</td><td>0.258</td><td>0.496</td><td>0.246</td><td>0.377</td></tr><tr><td>probe → similarity</td><td>0.458</td><td>0.513</td><td>0.161</td><td>0.469</td><td>0.248</td><td>0.665</td><td>0.239</td><td>0.462</td></tr><tr><td>probe → uniform</td><td>0.465</td><td>0.516</td><td>0.163</td><td>0.510</td><td>0.248</td><td>0.665</td><td>0.241</td><td>0.464</td></tr></table>

When enforcing strict pairwise entailment filtering, the utilized fraction � still aggressively decays from 0.575 at $k = 2$ down to 0.164 at � = 24, yielding an elasticity of 0.528 0.021 . Similarly, enforcing zero duplicated answer-bearing passages forces decay from 0.561 to 0.204, securing an elasticity of 0.434 0.024 . While intense answer attestation redundancy mathematically steepens the decay slope to 0.667 (exhibiting a highly significant log � interaction at $ { p } = 2 \times 1 0 ^ { - 7 } )$ , redundancy strictly modulates, but critically cannot unilaterally manufacture, the fundamental dilution penalty inherent to context window expansion. Note that these strata are estimated on natural retrieval pools under free generation, so their slopes are expected to be shallower than the protocol-clean 0.68 of Section 5.3, which isolates the structural decay under a fixed target and a width-invariant threshold. The relevant evidence here is therefore no the absolute magnitude, but that a steep decay persists in every low-redundancy stratum

## C R<sub>epro</sub>d<sub>uc</sub>ibilit<sub>y an</sub>d E<sub>xper</sub>i<sub>men</sub>t<sub>a</sub>l A<sub>r</sub>tif<sub>ac</sub>t<sub>s</sub>

To satisfy rigorous ACM artifact review standards and ensure complete systematic reproducibility, this section details the underlying computational protocols, dataset construction methodologies, and hyperparameter stabilization processes that govern our end-to-end portfolio architecture.

Manuscript submitted to ACM

![](images/a1b45fb1e24b53de2da2514799ddb89639efeabeba47fa8cfbe9f336eef471bb.jpg)

![](images/69a4705b9ab9cb9ed44d4a2e2f96eb736cb0c9d9c16a63df0c57c34476165682.jpg)

![](images/db96b7da02d3d9c3fe829f15550e8ca43c5d8dba0eb2847572f5d1e1d85745a4.jpg)

![](images/27b21cd34dffeace592e85d4824b923a6fef1efec33b87d95a2e659903d3cda2.jpg)

![](images/b696a25236927856879d6019d577ec3f87325df0c43353b48194096627b32fdd.jpg)

![](images/195544280dc334acbbccf247a8e7b56244b08a24c5008a6c08413dd669543c42.jpg)

![](images/98b52b87e5518ca1e5c6f14689dda568730f5af6475633cbf9fe5a4788e62927.jpg)

![](images/d8f484a2007d3201d599447b12b9311705c09724121a486f131d47de0f694e50.jpg)  
deployed seting of the frozen configuration; sweeps are one-at-a-time on the disjoint ASQA development split  
Fi<sub>g</sub>. 8. One-at-a-time sensitivit<sub>y</sub> swee<sub>p</sub>s conducted on the disjoint ASQA develo<sub>p</sub>ment s<sub>p</sub>lit to ensure re<sub>p</sub>roducible h<sub>yp</sub>er<sub>p</sub>arameter stabilization. The ex<sub>p</sub>licitl<sub>y</sub> circled <sub>p</sub>oint indicates the selected o<sub>p</sub>timal value for each distinct architectural <sub>p</sub>arameter. The strictl<sub>y</sub> 1<sub>frozen</sub> <sub>evaluation</sub> <sub>configuration</sub> <sub>deployed</sub> <sub>uniformly</sub> <sub>across</sub> <sub>primary</sub> <sub>experiments</sub> <sub>is</sub> $\beta = 0 . 3 , \beta _ { \mathrm { d o c } } = 0 . 3 , \lambda = 0 . 2 5 , \kappa = 0 , \alpha = 0 . 5 , k = 5 ,$ <sub>an</sub>d $N = 3 0$

The foundation of our experimental fairness relies on strict parity across evaluated frameworks. All benchmarked systems, regardless of their internal scheduling logic or instructional prompting, are architecturally forced to share the identical underlying retriever, retrieved evidence pool, context size, generated portfolio size, and generation temperature. To eliminate overfitting, all architectural selection and semantic facet hyperparameters were independently stabilized exactly once on a disjoint 40-query ASQA development split. Figure 8 comprehensively visualizes the one-at-a-time sensitivity sweeps executed on this split, validating that no single hyperparameter artificially dominates the system’s robustness. The resulting frozen configuration $( \beta = 0 . 3 , \beta _ { \mathrm { d o c } } = 0 . 3 , \lambda = 0 . 2 5 , \kappa = 0 , \alpha = 0 . 5 , k = 5 , \mathrm { a n d } N = 3 0 )$ was locked prior to executing any of the primary held-out algorithmic evaluations.

Crucially, validating our causal attribution probe (Section 4) mandated the construction of highly controlled groundtruth pools utilizing the identical ALCE candidate passages. To maintain strict isolation, a document is mathematically classified as answer-bearing if and only if any normalized alias exists as a contiguous token subsequence. The essential necessary set systematically comprises the first � = 2 answer-bearing documents, ordered by answer coverage and strictly constrained to attest entirely disjoint answer sets. Extraneous padding is aggressively regulated: true duplicates strictly attest to already resolved answers, whereas hard distractors are aggressively sampled from up to sixty unrelated queries, completely incapable of resolving the target query. These stringent requirements ensure that our queries provide at least � 2 verified answer-bearing documents alongside twenty-four verified distractors, preserving approximately one-fifth of the raw ASQA corpus and one-third of QAMPARI. Within these validated pools, documents are cryptographically shufled via a per-case deterministic seed, ensuring that semantic document labels remain immutable across all � 1 counterfactual ablation loops.

Our accompanying submission artifact encompasses the complete functional probe architecture, the operating point calibration scripts, the exact validation-pool construction pipelines, and the comprehensive job specifications necessary to precisely regenerate all reported tables and figures. We emphasize that GPU-free validation checks structurally cover the critical � � calibration, crossed bootstrap uncertainty bounds, ContextCite ablation routines, and all similarity/lexical baseline trajectories. The total computational footprint of this analysis spans 188 unified execution runs (148 utilizing the primary commit instruction and 40 under the integrate baseline). A standard single evaluation run operates eficiently at 126 seconds per query. The primary budget and context width sweeps utilize 150 discrete queries at narrow widths $( k \leq 8 )$ and 60 at extreme limits $( k \in \{ 1 2 , 2 4 \} )$ , while the expansive $k \times T$ factorial mandates exactly 120 strictly paired observations across every distinct cell, driving 11,579 individually scored generation rows. This exhaustive scale directly necessitates the hierarchical predictive modeling applied throughout our results, ensuring tha our architectural conclusions represent fundamental generative properties rather than isolated statistical anomalies.

Table 16. Utilit<sub>y</sub>-side factorial under commit versus inte<sub>g</sub>rate instructions. Inte<sub>g</sub>rate asks for ever<sub>y</sub> su<sub>pp</sub>orted inter<sub>p</sub>retation and <sub>a</sub>ll<sub>ows enumera</sub>ti<sub>on.</sub> Dif<sub>erences are</sub> i<sub>n</sub>t<sub>egra</sub>t<sub>e m</sub>i<sub>nus comm</sub>it<sub>, pa</sub>i<sub>re</sub>d <sub>w</sub>ithi<sub>n query</sub> (� = 314<sub>–</sub>480 <sub>per ce</sub>ll)<sub>,</sub> t<sub>es</sub>ti<sub>ng a</sub>ll<sub>oca</sub>ti<sub>on un</sub>d<sub>er</sub> th<sub>e</sub> i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on mos</sub>t f<sub>avoura</sub>bl<sub>e</sub> t<sub>o one w</sub>id<sub>e res onse.</sub>
<table><tr><td>k</td><td> $T$ </td><td>commit PR@T</td><td>integrate PR@T</td><td> $\Delta$ </td><td>95% CI</td></tr><tr><td>2</td><td>1</td><td>0.203</td><td>0.253</td><td>+0.050</td><td> $\left[ + 0 . 0 3 1 , + 0 . 0 7 0 \right]$ </td></tr><tr><td>2</td><td>5</td><td>0.326</td><td>0.394</td><td>+0.068</td><td> $\left[ + 0 . 0 4 9 , + 0 . 0 8 8 \right]$ </td></tr><tr><td>2</td><td>12</td><td>0.393</td><td>0.453</td><td>+0.060</td><td> $\dot { \cdot } + 0 . 0 4 1 , + 0 . 0 8 0 \dot { ] }$ </td></tr><tr><td>5</td><td>1</td><td>0.219</td><td>0.292</td><td>+0.073</td><td> $\left[ + 0 . 0 5 3 , + 0 . 0 9 5 \right]$ </td></tr><tr><td>5</td><td>5</td><td>0.336</td><td>0.406</td><td>+0.070</td><td> $+ 0 . 0 5 2 , + 0 . 0 9 1 \bar { ] }$ </td></tr><tr><td>5</td><td>12</td><td>0.411</td><td>0.475</td><td>+0.065</td><td> $\dot { } + 0 . 0 4 4 , + 0 . 0 8 7 \dot { }$ </td></tr><tr><td>12</td><td>1</td><td>0.245</td><td>0.322</td><td>+0.077</td><td> $= + 0 . 0 5 5 , + 0 . 1 0 1 ]$ </td></tr><tr><td>12</td><td>5</td><td>0.365</td><td>0.439</td><td>+0.074</td><td> $\left[ + 0 . 0 5 3 , + 0 . 0 9 8 \right]$ </td></tr><tr><td>12</td><td>12</td><td>0.423</td><td>0.505</td><td>+0.082</td><td> $\left[ + 0 . 0 5 9 , + 0 . 1 0 7 \right]$ </td></tr><tr><td>24</td><td>1</td><td>0.259</td><td>0.355</td><td>+0.096</td><td> $\left[ + 0 . 0 7 3 , + 0 . 1 2 1 \right] ^ { 3 }$ </td></tr><tr><td>24</td><td>5</td><td>0.397</td><td>0.495</td><td>+0.098</td><td> $\left[ + 0 . 0 7 0 , + 0 . 1 2 9 \right]$ </td></tr><tr><td>24</td><td>12</td><td>0.411</td><td>0.484</td><td>+0.073</td><td> $\left[ + 0 . 0 5 2 , + 0 . 0 9 4 \right]$ </td></tr></table>

## D Su<sub>pp</sub>lementar<sub>y</sub> Setin<sub>g</sub>s and Full Em<sub>p</sub>irical Grids

To exhaustively validate that our architectural allocation laws and scheduling advantages extend beyond standard English short-answer question answering, we provide the complete empirical grids encompassing alternative instructional bounds, structured generative baselines, full algorithmic ablations, and open-domain cross-cultural stress tests.

## D<sub>.</sub>1 O<sub>p</sub>en-Domain Cross-Cultural Generalization

We evaluate the cross-cultural recipe adaptation setting as a rigorous non-English open-domain check [41]. Operating over a corpus of $^ { 9 , }$ 486 Spanish-origin recipes [88], the system is tasked with rewriting Latin-American query recipes to incorporate authentic Spanish culinary practices. Gold answer units are strictly defined as valid Spanish ingredients empirically attested by the retrieved pool but explicitly absent from the source recipe, directly rewarding generative substitution breadth. Retrieved via a multilingual sentence encoder, this specific domain exhibits an almost perfectly flat relevance density (log-log slope 0.01, Table 3). Unlike the concentrated ASQA benchmark, this flat relevancy physically insulates the wide-context configurations from deep-rank decay, structurally dictating that unconstrained rotationa scheduling theoretically should, and empirically does, achieve peak portfolio extraction in this isolated regime.

## D<sub>.</sub>2 Instructional Bounds and Structured Out<sub>p</sub>ut Controls

A prevailing assumption is that the limitations of single-pass context utilization can be bypassed simply through aggressive prompt engineering. To test this, we override our primary commit instruction with an expansive integrate instruction, explicitly commanding the generator to enumerate every supported interpretation and cite all relevant sources. As detailed in Table 16, while this integrative prompt successfully elevates absolute marginal recall across the board (gains ranging from $+ 0 . 0 5 0 \mathrm { ~ t o ~ } { + 0 . 0 9 8 } )$ , it completely fails to disrupt the fundamental factorial scaling laws. Raising the generation count from $T = 1 \mathrm { t o } T = 1 2$ continues to yield massive gains of 0.142 to 0.200. Budget-matched Manuscript submitted to ACM

T<sub>a</sub>bl<sub>e</sub> 17<sub>.</sub> E<sub>numera</sub>t<sub>e</sub>d <sub>s</sub>i<sub>ng</sub>l<sub>e-response</sub> b<sub>ase</sub>li<sub>ne versus ma</sub>t<sub>c</sub>h<sub>e</sub>d �=12 <sub>por</sub>tf<sub>o</sub>li<sub>o, pa</sub>i<sub>re</sub>d <sub>w</sub>ithi<sub>n query.</sub> Th<sub>e s</sub>i<sub>ng</sub>l<sub>e response ge</sub>t<sub>s</sub> th<sub>e</sub> <sub>p</sub>ortfolio decode-token bud<sub>g</sub>et and must list distinct su<sub>pp</sub>orted inter<sub>p</sub>retations with <sub>g</sub>roundin<sub>g,</sub> sto<sub>pp</sub>in<sub>g</sub> rather than <sub>p</sub>addin<sub>g</sub>. It <sub>rema</sub>i<sub>ns</sub> b<sub>e</sub>l<sub>ow por</sub>tf<sub>o</sub>li<sub>os a</sub>t <sub>every w</sub>idth/i<sub>ns</sub>t<sub>ruc</sub>ti<sub>on</sub> ( 0.109 t<sub>o</sub> 0.029)<sub>, w</sub>ith <sub>narrow</sub>i<sub>ng gaps as con</sub>t<sub>ex</sub>t <sub>w</sub>id<sub>ens.</sub> $^ { * } / ^ { \dagger } \colon p < 0 . 0 5 / p < 0 . 0 1$
<table><tr><td>instruction</td><td>k</td><td>n</td><td> $\mathrm { p o r t f o l i o }$ </td><td>structured</td><td>gap</td><td>95% CI</td></tr><tr><td>commit</td><td>2</td><td>480</td><td>0.398</td><td>0.289</td><td>+0.109†</td><td>[+0.084, +0.135]</td></tr><tr><td></td><td>5</td><td>480</td><td>0.423</td><td>0.335</td><td>+0.088†</td><td>[+0.063, +0.114]</td></tr><tr><td></td><td>12</td><td>480</td><td>0.426</td><td>0.371</td><td>+0.055†</td><td>[+0.028, +0.082]</td></tr><tr><td></td><td>24</td><td>480</td><td>0.416</td><td>0.381</td><td>+0.035†</td><td>[+0.009, +0.061]</td></tr><tr><td>integrate</td><td>2</td><td>480</td><td>0.398</td><td>0.295</td><td>+0.103†</td><td>[+0.077, +0.130]</td></tr><tr><td></td><td>5</td><td>480</td><td>0.423</td><td>0.342</td><td>+0.081†</td><td>[+0.056, +0.108]</td></tr><tr><td></td><td>12</td><td>480</td><td>0.426</td><td>0.382</td><td>+0.044†</td><td>[+0.017, +0.071]</td></tr><tr><td></td><td>24</td><td>480</td><td>0.416</td><td>0.388</td><td>+0.029*</td><td>[+0.002, +0.054]</td></tr></table>

<sup>superiority</sup> <sup>remains</sup> <sup>absolute:</sup> <sup>the</sup> <sup>multi-round</sup> (<sup>2,</sup> <sup>12</sup>) <sup>configuration</sup> <sup>systematically</sup> <sup>beats</sup> <sup>the</sup> <sup>single-pass</sup> (<sup>24,</sup> <sup>1</sup>) <sup>allocation</sup> by 0.111 (95% CI 0.087, 0.136 ).

Furthermore, we explicitly eliminate the confound of output-token volume limits. We construct a heavily optimized structured single-response baseline: allocating this single pass the exact equivalent decode-token budget of a full $T = 1 2$ portfolio, and coercing the model to output a structured list of distinct, grounded interpretations without premature stopping. Despite these extreme guardrails, the structured single response systematically trails the sequential portfolio across all widths and instructions (Table 17). While the performance gap expectedly narrows as the single context window expands ( 0.109 at � = 2 collapsing to 0.035 at � = 24 under the commit instruction), the deficit remains strictly positive and statistically significant across all 480 paired queries, proving that iterative context isolation mechanically extracts evidence that a single wide-context pass fundamentally ignores.

## D<sub>.</sub>3 C<sub>ompre</sub>h<sub>ens</sub>i<sub>ve</sub> S<sub>ys</sub>t<sub>em</sub> Abl<sub>a</sub>ti<sub>ons:</sub> Th<sub>e</sub> N<sub>ecess</sub>it<sub>y o</sub>f C<sub>ausa</sub>l F<sub>ee</sub>db<sub>ac</sub>k

Finally, Table 15 provides the exhaustive component-level ablation grid for the Ascp architecture across all four evaluated tasks. These structural substitutions cleanly isolate the precise marginal utility of attribution feedback, submodular selection, and steered decoding.

Notably, the final diagnostic rows strictly control the scheduling mechanism while substituting only the underlying evidence-use signal. Replacing our causal probe with a naive embedding similarity proxy or a uniform-utilization assumption triggers a systemic collapse in scheduling eficacy. This cleanly mirrors the diagnostic illusion exposed in Section 4, conclusively establishing that counterfactual causal sensitivity is the absolute, irreplaceable driver of the scheduler’s ability to maximize evidence coverage within a bounded generative budget.

Received