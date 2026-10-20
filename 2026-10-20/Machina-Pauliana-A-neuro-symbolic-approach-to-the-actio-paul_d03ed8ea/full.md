# Machina Pauliana: A neuro-symbolic approach to the actio pauliana in Dutch insolvency law

Daan BOVEN, <sup>a,1</sup> Gregor BEHNKE <sup>a</sup> and David GRAUS <sup>a</sup>

<sup>a</sup> University of Amsterdam

ORCiD ID: Daan Boven https://orcid.org/0009-0003-5031-9634, Gregor Behnke https://orcid.org/0000-0002-1445-9934, David Graus https://orcid.org/0000-0002-6245-0870

Abstract. Applying a legal provision frequently requires both structured rule-based reasoning and sensitivity to open-textured terms developed in case law. The actio pauliana of Article 42 of the Dutch Bankruptcy Act (Faillissementswet, Fw), which allows an insolvency practitioner to avoid transactions that prejudice creditors, requires both. We present a neuro-symbolic model of Article 42 Fw. Its symbolic core is written in Catala in a literate style that keeps statute and formalisation aligned and was developed through pair programming, with open-textured terms represented as boolean inputs where neural components couple to the symbolic solver. A neural routine structures a case’s natural-language facts into boolean and date inputs. Neural subroutines resolve the open-textured terms through case-based reasoning over Dutch civil judgments, retrieved through a pipeline that filters on the requirement being resolved and expands along the procedural chain of a case. Segmenting and labelling those judgments attains high precision and recall against a manually annotated gold standard of 100 judgments. On ten hand-annotated cases the neurosymbolic model handles most criteria reasonably but struggles with the knowledge requirement. Retrieval depth shows no clear quantitative effect, although retrieved judgments provide explicit grounding in prior case law.

Keywords. neuro-symbolic AI, case-based reasoning, retrieval-augmented generation, large language models, rules as code

## 1. Introduction

Research in artificial intelligence and law dates back to the 1970s, and has since often focused on symbolic, rule-based methods, which normalise legal texts into logic-based models that are also intelligible to computers [1,2]. Today, increasing attention is devoted to data-driven methods, in which AI is used for the automated interpretation of legal texts, document analysis, and the prediction of legal outcomes.

We apply a hybrid neuro-symbolic model to the avoidance action in Dutch insolvency law (the actio pauliana) as codified in Article 42 of the Dutch Bankruptcy Act (Faillissementswet, Fw). This provision allows an insolvency practitioner to avoid legal acts (transactions) that cause prejudice to creditors and decrease the estate. If creditors are disadvantaged in their possibilities of recovery as a result of legal transactions by the debtor, such transactions may, under certain conditions, be avoided by the insolvency practitioner for the benefit of the estate [3]. Consider, e.g., a director who sells the company car to himself, months before the company is declared bankrupt, by offsetting the car’s price against a loan he claims to have granted the company. An insolvency practitioner may avoid such a transaction: the sale was voluntary, it allowed the director to recover his own claim in full at the expense of the remaining creditors, and, as it took place within one year of the bankruptcy, knowledge of this prejudice is presumed.

While the legal structure of Article 42 Fw lends itself to formalisation in terms of logical rules, interpreting several of its open-textured terms requires reference to case law [4]. For that reason we use neural subroutines that resolve open-textured terms through case-based reasoning (CBR) over a corpus of publicly available judgments sourced from rechtspraak.nl (the Rechtspraak corpus).

This paper’s contribution is a full neuro-symbolic workflow for Article 42 Fw: a Catala formalisation [5] coupled to neural extraction and case-based reasoning over a segmented, labelled, and procedurally linked corpus of Dutch judgments.<sup>2</sup>

## 2. Background: Article 42 Fw

Legal definition and description. Upon bankruptcy, control of the debtor’s estate shifts to the insolvency practitioner, whose powers include avoiding legal acts that prejudice creditors through the actio pauliana [3,6]. The insolvency practitioner can invoke the actio pauliana by an extrajudicial declaration, and in practice only contested cases reach the courts [6,7]. We focus on the avoidance of voluntary legal acts under Article 42 Fw, together with Articles 43 and 45 Fw, which provide evidentiary presumptions that assist the insolvency practitioner. Article 42 Fw can be decomposed into five principal elements. The transaction: (a) is a legal act, (b) was voluntary, (c) resulted in prejudice to the general body of creditors, (d) the debtor knew or ought to have known that the legal act would result in prejudice to the creditors, and (e) where the legal act was for consideration, the counterparty also knew or ought to have known that the legal act would result in prejudice to the creditors [7]. We visualise these elements as part of the broader decision scheme of the actio pauliana in Figure 1.

Open-textured terms. Some requirements of Article 42 Fw are open-textured and factsensitive, and are not suitable for purely symbolic representation. For those elements (green, purple, and orange boxes in Figure 1), case law is not just illustrative but necessary to understand how the rule operates in practice. We address the following terms:

Voluntary performance: A legal act is voluntary where it is performed without a legal obligation arising from statute or contract, even where the factual circumstances left the debtor practically no alternative but to perform it. Case law further illustrates the boundary between legal obligation and mere factual or economic necessity [6,9].

Prejudice to creditors: Prejudice is assessed when the actio pauliana is invoked, or, if contested in court, when the court decides on it [9]. At that time, creditors’ actual recovery position is compared with the hypothetical situation in which the contested transaction had not occurred. This hypothetical does not concern all possible alternatives, but what would most likely have happened had the transaction been omitted [7]. Case law illustrates how this comparison is applied across different transactions.

![](images/79b9133c390cd9cf5f40112660788e1108d6294bf5a590945ddff1d356d5367c.jpg)  
Figure 1. Actio pauliana model based on Pannevis para. 7.2.1 [6], translation based on Warendorf [8].

Knowledge of prejudice: The knowledge requirements of the debtor and counterparty have been objectified by equating actual knowledge with an ‘ought to have known’ standard, so that the insolvency practitioner does not need to prove the debtor’s or counterparty’s actual state of mind [7,9]. The Supreme Court further specified that knowledge is present when, at the time of the transaction, bankruptcy and a deficit in the estate were foreseeable with a reasonable degree of probability [6,10]. Articles 43 and 45 Fw further assist the insolvency practitioner through rebuttable presumptions of knowledge. For acts within one year before bankruptcy, Article 43 presumes knowledge for specified transactions for consideration, while Article 45 does so for acts for no consideration [6].

## 3. Background: computational approaches

Executable law and Catala. The “rules-as-code” paradigm builds on the idea that normalised, machine-executable versions of law can improve transparency, consistency, and traceability between legal text and implementation [2]. Tax law is particularly suited to this approach because its rules are comparatively calculative and contain fewer opentextured terms; Dutch initiatives such as RegelRecht and Regelspraak illustrate this in practice [11,12], whereas Dutch private and insolvency law remain largely unexplored. Catala is a language for executable law that has been applied primarily to tax law. It brings a systematic approach in which literate programming (annotating the law line by line) and pair programming (a computer scientist and a legal scholar working side by side) are the norm ex ante [5,13]. Symbolic approaches, however, may struggle with vagueness and semantic or syntactic ambiguities [2,14].

Generative AI and Retrieval-Augmented Generation. These limitations motivate neural components that can reason over open-textured legal concepts. Large Language Models (LLMs), however, introduce their own limitations, including hallucination, poor calibration, and static knowledge [15]. Retrieval-augmented generation (RAG) addresses part of this problem by combining a model’s parametric memory with the non-parametric memory of a retrieved document collection [16]. This early approach, now commonly termed Naive RAG, has limitations such as over-reliance on or sensitivity to retrieved text.

Advanced RAG addresses these by adding pre- and post-retrieval stages, such as metadata filtering, re-ranking, and context compression [17]. Its effectiveness also depends on how documents are segmented: semantic chunking preserves meaningful sections, which aligns well with the structure of judicial decisions. RAG has accordingly been applied to legal question answering [18] and legal advice generation [19], while LLMs have also been used for legal information extraction and document segmentation [20].

Neuro-symbolic Legal AI. To address LLMs’ shortcomings, we draw on Kautz’s type 2 Symbolic[Neuro] architecture [21], increasingly applied in legal AI [22,23]: a symbolic solver is the primary system, and neural components supply structured inputs and resolve open-textured terms through retrieval-augmented case-based reasoning.

The first neuro-symbolic legal system, Split-Up (1995), addresses the Australian Family Law Act 1975 by combining rule-based and neural reasoning across separate subtasks [24], much like our routines. Its networks, however, are trained on past judgments and cannot explain their inferences, so Toulmin argument structures provide explanations separately. Our LLM routines can produce textual explanations grounded in retrieved case law. More recently, Kant et al. [25] prompt LLMs to encode insurance contracts and claims in Prolog, finding that structured guidance substantially improves the generated encodings. There the LLM writes the symbolic program, whereas our Catala model is written by hand and the LLM only resolves its inputs. Closest to our method, Sadowski and Chudziak [26] propose a three-step process of term identification, predicate extraction, and rule application through symbolic verification, applied to hearsay formalised in first-order logic, which outperforms few-shot prompting. Their rule is likewise defined by experts but their LLM determines predicates from the facts alone, while our CBR subroutines additionally rely on retrieved judgments to resolve open-textured terms.

## 4. Method

We propose an approach in which a symbolic problem solver is the primary system. Its structured inputs are derived from the target case’s natural-language facts by neural components: one factual-extraction routine and three RAG-powered CBR subroutines that resolve the open-textured terms identified in Section 2.

## 4.1. Symbolic model

Our symbolic core is a machine-readable model of Articles 42, 43, and 45 Fw that, given input variables describing a contested transaction, decides whether the transaction is avoidable. The model is written in Catala, with the decisive advantage here being its support for pair programming between programmers and lawyers. Our model was developed with the help of an insolvency practitioner and a legal scholar. Building the model follows the normalisation Allen and Engholm [14] described for statutes: atomic propositions are identified, replaced by labels, made explicit with propositional logic, and the original text restored to validate the result, in predicate form where possible [2]. Normalisation also exposed several syntactic ambiguities; for instance, we model the evidentiary presumption of Article 43 Fw as rebuttable per side. Semantic ambiguities include counting one year back from 29 February, where we round down to 28 February.

The symbolic model follows the standard decision scheme of the actio pauliana, as outlined in Figure 1. Some elements, namely whether an act is voluntary, prejudice to creditors, and the knowledge requirement, require more complex case-law-dependent reasoning. Rather than forcing them into further symbolic rules, the model represents them as boolean inputs whose truth values are resolved by neural subroutines.

## 4.2. Corpus construction

The CBR subroutines rely on an advanced RAG approach, which retrieves relevant segments of judgments from a corpus of judicial decisions concerning the actio pauliana under Article 42 Fw. The judgments are sourced from rechtspraak.nl, the official open data publication channel of the Dutch judiciary [27].

The initial document selection for corpus construction is made in two steps. First, we extract a metadata dump from all published judicial decisions through WooZM, a publicly accessible Dutch open government search engine. We select from this dump all decisions whose area oflaw metadata is classified as civil law. This yields over 270,000 civil-law decisions as XML documents. Next, we apply regular expressions to filter for Articles 42, 43, and/or 45 Fw, the latter two containing evidentiary presumptions; we deliberately do not use ‘(actio) pauliana’ as a search term, as it also denotes the civil-law actio pauliana of Article 3:45 BW, which is outside our scope. This yields our final corpus of 846 XML documents representing judicial decisions that involve the actio pauliana.

## 4.2.1. Semantic chunking

Rechtspraak XML is structurally inconsistent [28], making its markup insufficient for reliable section segmentation. Existing efforts, however, are either no longer accessible [29] or do not identify the substantive judgment sections required here [30].

We therefore build a domain-specific deterministic segmenter. Its lexicon is derived from an LLM-based corpus survey. A separate few-shot prompt is used per court instance. Dutch civil judgments generally follow a recurring structure grounded in Article 230 Rv and reflected in judicial drafting practice [31,32]. Each prompt receives one judgment in XML and asks the model to identify the opening marker of five recurring sections: course ofproceedings,facts, dispute, grounds ofdecision, and decision. We use mistral-small-3.2-24b-instruct-2506 with temperature 0.1, top-p 0.2, and seed 42. The identified opening markers are manually consolidated and iteratively refined into a lexicon of heading words and regex-to-label rules. The corpus survey thus informs a fixed rule set rather than making downstream segmentation itself LLM-dependent.

The resulting segmenter is a cascade of finite-state transducers in the spirit of FAS-TUS [33]: (i) recognition identifies candidate section boundaries using this lexicon; (ii) refinement recovers missed boundaries and removes incorrectly detected ones; and (iii) closing, the final stage, pairs each section start with the next and assigns the text between them to the corresponding section. The resulting labelled sections are validated against a manually segmented sample and subsequently serve as the semantic chunks used for retrieval. We evaluate this segmentation approach in Section 5.1.

## 4.3. Neural components

The neural components comprise one factual-extraction routine and three CBR subroutines, all implemented using LLMs. Each routine uses a system prompt containing the relevant statutory provisions and legal doctrine, while the user prompt provides the facts and dispute of the case to be resolved in natural language. The routines run in parallel and do not communicate. Their outputs are combined by the Catala symbolic solver.

Structuredfactual-input extraction. This routine transforms unstructured facts of a contested transaction in natural language into structured inputs of the Catala model. The factual-extraction routine uses an LLM to extract eight structured case inputs from the target case: three boolean inputs (is legal act, performed by the debtor, and for no consideration); two dates (the date of the legal act and bankruptcy date); and three textual explanations supporting the extracted boolean values.

Open-textured term resolution. To resolve the open-textured terms discussed in Section 2 (voluntary performance, prejudice to creditors, and the knowledge requirements), three RAG-powered CBR subroutines retrieve relevant judgments and return booleanvalued inputs to Catala, alongside a textual explanation of the reasoning. The voluntaryperformance and prejudice subroutines each resolve to a single boolean input. The knowledge subroutine resolves to eight boolean inputs corresponding to the knowledge and evidentiary-presumption branches in Figure 1: directly established debtor and counterparty knowledge; transaction at an undervalue; payment of, or security for, a debt not yet due and payable; affiliated party; pre-existing obligation before the one-year period; and rebuttal of the debtor- and counterparty-knowledge presumptions.

The RAG embeds judgment sections with qwen3-embedding-8b, a multilingual model previously used for legal example retrieval [34], with a 32k-token context window accommodating an entire section as a single chunk [35]. We score each judgment against the natural-language input of the target case, using the judgment’s facts and grounds of decision sections. Where multiple sections of the same type occur, as in judgments addressing several issues, maxP aggregation [36] retains the best-matching chunk for each type before the scores are combined using fixed weights, inspired by the CBR-RAG approach of Wiratunga et al. [18].

Pre-retrieval. We go beyond Naive RAG by adding the following pre- and post-retrieval steps. Each judgment in our corpus carries boolean labels indicating whether it rules on each of the three open-textured terms. For each subroutine, we restrict retrieval to judgments labelled as addressing the corresponding term, reducing the search space and the probability of retrieving irrelevant chunks.

To select a model for labelling our corpus, we manually annotated a gold standard of 100 judgments for the presence of each of the three elements. We compared gpt-5.4 to mistral-medium-3.5-128b:fp8, and picked the latter as it achieved slightly higher recall (Table 2), which we prefer over precision as missing a relevant case is costlier than retrieving a less directly relevant one, in addition to the open-weights nature of the model, which is an advantage for reproducibility.

Post-retrieval. Higher instance courts frequently rely on facts established in lower courts without restating them. At the same time, a lower-instance decision may have been overruled, with the higher-instance decision establishing the final reasoning and decision. In post-retrieval we therefore expand the result set along the procedural chain: when a retrieved decision belongs to a case litigated at another instance, those decisions are added to the retrieved context supplied to the CBR subroutine.

Parameters used in neural components. For reproducibility, the inference parameters were held constant where supported. For mistral-medium-3.5-128b:fp8, used in both the pre-retrieval annotation and the neural routines, we set the temperature to 0.1, top-p to 0.2, and the seed to 42; top-k was not set. For gpt-5.4, used in the pre-retrieval annotation experiment, and $\mathtt { g p t - 5 . 5 } _ { \mathtt { i } }$ , used in the neuro-symbolic model, we set the seed to 42 and reasoning effort to medium. For gemini-3.5-flash, used in the neurosymbolic model, we set the temperature to 0.1, top-p to 0.2, and reasoning effort to medium. Parameters not mentioned were left at their default values.

## 5. Results

Each component is evaluated by precision, recall, and $F _ { 1 }$ against a gold standard annotated manually by the first author. Due to practical constraints that we discuss in Section 6, the end-to-end model is evaluated on a small set of hand-annotated cases. For that reason, we also report qualitative findings alongside the quantitative metrics.

## 5.1. Prerequisites

Document segmentation. First, we evaluate the document segmenter against a manually annotated random sample of 100 judgments, consisting of 582 segments. As Table 1 shows, the segmenter attains high precision, recall, and $F _ { 1 }$ across all five target sections, with a micro average of 0.98 precision and 0.97 recall; the lowest per-section score is on the grounds of decision section $( F _ { 1 } = 0 . 9 4 )$ . The segmenter was built through a corpus survey specifically for judgments concerning the Dutch transaction avoidance under Art. 42 Fw, so these results speak only to the actio pauliana of Art. 42 Fw, and do not necessarily extend to judgments in other areas of civil law.

Table 1. Section segmentation performance on the Rechtspraak corpus, reporting precision, recall, and F1 per section against 100 manually segmented judgments.
<table><tr><td></td><td colspan="3">Score</td><td></td><td></td><td colspan="3">Score</td><td></td></tr><tr><td>Section</td><td>P</td><td>R</td><td>F1</td><td>n</td><td>Section</td><td>P</td><td>R</td><td>F1</td><td>n</td></tr><tr><td>Course of proceedings</td><td>1.00</td><td>0.99</td><td>0.99</td><td>100</td><td>Established facts</td><td>1.00</td><td>0.97</td><td>0.99</td><td>76</td></tr><tr><td>Dispute</td><td>0.98</td><td>0.94</td><td>0.96</td><td>63</td><td>Grounds of decision</td><td>0.95</td><td>0.92</td><td>0.94</td><td>114</td></tr><tr><td>Decision</td><td>0.99</td><td>0.99</td><td>0.99</td><td>101</td><td>Other</td><td>0.98</td><td>1.00</td><td>0.99</td><td>128</td></tr><tr><td>Micro average (all)</td><td>0.98</td><td>0.97</td><td>0.98</td><td>582</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Document labelling. Accurate segmentation of judicial decision documents enables the grounds of the decision and the core reasoning by the court to be annotated for whether the court ruled on one of the three open-textured legal criteria presented in Section 2. We apply an LLM to label document segments with whether the court rules on these (voluntary performance, prejudice, and the knowledge requirements) in the grounds of its decision. We evaluate them against the same sample of 100 documents used for evaluating the segmenter, which we also manually annotated for these labels; the results are presented in Table 2. We report precision, recall, and F<sub>1</sub>-score, but an important focus here is on recall, since for jurists the cost of missing information in a legal context is substantially higher than the cost of reviewing extra, irrelevant results.

The two models seem to trade precision against recall: Mistral reaches higher recall (micro-average 0.94) at lower precision (0.80), whereas GPT-5.4 achieves higher precision (0.93) at the cost of slightly lower recall (0.88), with micro-averaged $F _ { 1 }$ scores of 0.86 and 0.91 respectively. Given the importance of recall and the greater reproducibility and portability of an open-weight model, we select Mistral, which misses the fewest relevant rulings, whereas GPT-5.4 returns fewer false positives.

Table 2. Pre-retrieval annotation performance per element against 100 manually annotated judgments.
<table><tr><td></td><td colspan="4">mistral-medium-3.5-128b:fp8</td><td colspan="4">gpt-5.4</td></tr><tr><td>Element</td><td>P</td><td>R</td><td>Fl</td><td>n</td><td>P</td><td>R</td><td>F1</td><td>n</td></tr><tr><td>Voluntary performance</td><td>0.69</td><td>0.97</td><td>0.81</td><td>39</td><td>0.85</td><td>0.90</td><td>0.88</td><td>39</td></tr><tr><td>Prejudice to creditors</td><td>0.89</td><td>0.91</td><td>0.90</td><td>34</td><td>1.00</td><td>0.82</td><td>0.90</td><td>34</td></tr><tr><td>Knowledge</td><td>0.86</td><td>0.92</td><td>0.89</td><td>39</td><td>0.97</td><td>0.92</td><td>0.95</td><td>39</td></tr><tr><td>Micro average</td><td>0.80</td><td>0.94</td><td>0.86</td><td>112</td><td>0.93</td><td>0.88</td><td>0.91</td><td>112</td></tr></table>

## 5.2. Neuro-symbolic model

Finally, we evaluate the neuro-symbolic model against ten hand-annotated cases from outside our Rechtspraak corpus and not used in developing the model, drawn mainly from Jurisprudentie Onderneming & Recht (JOR), a Dutch jurisprudence journal that publishes legally noteworthy cases. Their facts and the dispute at hand served as the natural-language input for the neuro-symbolic model. Our symbolic Catala model computes a single output from a given set of inputs (truth values of legal criteria, and dates). To provide a more meaningful comparison, we compare both the model’s output and its inputs against the gold standard. We additionally vary the number k of documents retrieved for case-based reasoning. We report micro-averaged precision, recall, and $F _ { 1 }$ for different values of k. Because the validation set is small and consists largely of cases selected for publication in a jurisprudence journal, the results should be interpreted as indicative rather than conclusive. In Table 3, we compare three LLMs for the end-to-end neuro-symbolic approach. Micro-averaged scores for all three models fall in a comparable range (precision between 0.63 and 0.68, recall between 0.82 and 0.90, and $F _ { 1 }$ between 0.72 and 0.77). The retrieval depth does not meaningfully affect these aggregate figures: for each model the scores at k = 0, k = 5, and k = 10 are similar, even though the retrieved decisions do visibly change the reasoning the model puts forward for its answers. Neither a clear ranking between the models nor a clear effect of retrieval depth can be drawn from these figures.

Table 3. Neuro-symbolic model performance, reporting micro-averaged precision, recall, and $F _ { 1 }$ of the boolean criteria and the final verdict over ten judgments against the gold standard, at retrieval depths $k = 0 ,$ k = 5, and k = 10.
<table><tr><td></td><td colspan="3">k= 0</td><td colspan="3">k=5</td><td colspan="3">k = 10</td></tr><tr><td>Model</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td> $F _ { 1 }$ </td></tr><tr><td>gemini-3.5-flash</td><td>0.64</td><td>0.88</td><td>0.74</td><td>0.67</td><td>0.90</td><td>0.77</td><td>0.63</td><td>0.88</td><td>0.74</td></tr><tr><td>gpt-5.5</td><td>0.68</td><td>0.90</td><td>0.77</td><td>0.67</td><td>0.90</td><td>0.77</td><td>0.67</td><td>0.90</td><td>0.77</td></tr><tr><td>mistral-medium-3.5-128b:fp8</td><td>0.66</td><td>0.82</td><td>0.73</td><td>0.65</td><td>0.82</td><td>0.72</td><td>0.66</td><td>0.82</td><td>0.73</td></tr></table>

Table 4. Per-subroutine micro-averaged precision, recall, and $F _ { 1 }$ at retrieval depths k = 0, k = 5, and $k = 1 0 .$
<table><tr><td></td><td></td><td colspan="3"> $k = 0$ </td><td colspan="3"> $k = 5$ </td><td colspan="3"> $k = 1 0$ </td></tr><tr><td>Model</td><td>Subroutine</td><td>P</td><td>R</td><td> $F _ { 1 }$ </td><td>P</td><td>R</td><td> $F _ { 1 }$ </td><td>P</td><td>R</td><td> $F _ { 1 }$ </td></tr><tr><td rowspan="3">gemini-3.5-flash</td><td>Voluntary</td><td>0.78</td><td>0.88</td><td>0.82</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.78</td><td>0.88</td><td>0.82</td></tr><tr><td>Prejudice</td><td>0.67</td><td>1.00</td><td>0.80</td><td>0.60</td><td>1.00</td><td>0.75</td><td>0.67</td><td>1.00</td><td>0.80</td></tr><tr><td>Knowledge</td><td>0.10</td><td>0.40</td><td>0.16</td><td>0.16</td><td>0.60</td><td>0.25</td><td>0.10</td><td>0.40</td><td>0.16</td></tr><tr><td rowspan="3">gpt-5.5</td><td>Voluntary</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.88</td></tr><tr><td>Prejudice</td><td>0.67</td><td>1.00</td><td>0.80</td><td>0.75</td><td>1.00</td><td>0.86</td><td>0.75</td><td>1.00</td><td>0.86</td></tr><tr><td>Knowledge</td><td>0.16</td><td>0.60</td><td>0.25</td><td>0.14</td><td>0.60</td><td>0.23</td><td>0.14</td><td>0.60</td><td>0.23</td></tr><tr><td rowspan="3">mistral-medium-  $3 . 5 \mathrm { - } 1 2 8 \mathrm { b } : \mathrm { f } \mathsf { p } 8$ </td><td>Voluntary</td><td>0.80</td><td>1.00</td><td>0.89</td><td>0.80</td><td>1.00</td><td>0.89</td><td>0.80</td><td>1.00</td><td>0.89</td></tr><tr><td>Prejudice</td><td>0.60</td><td>1.00</td><td>0.75</td><td>0.60</td><td>1.00</td><td>0.75</td><td>0.60</td><td>1.00</td><td>0.75</td></tr><tr><td>Knowledge</td><td>0.07</td><td>0.20</td><td>0.11</td><td>0.07</td><td>0.20</td><td>0.10</td><td>0.07</td><td>0.20</td><td>0.11</td></tr></table>

Retrieval can provide explicit textual grounding in prior judgments even where aggregate scores remain similar. In the validation case ECLI:NL:RBOVE:2026:4561, GPT-5.5 at $k = 5$ supports its finding that the payment was voluntary by citing ECLI:NL:GHSHE:2006:BC2959, “where performance of a natural obligation does not remove the voluntary character of the legal act.” This reference supports treating a gift as a voluntary legal act even if it fulfilled a natural obligation, as the court explains in paragraph 7.5 of the retrieved judgment. Since GPT-5.5 already reaches the same conclusion at $k = 0 ,$ , the visible addition is primarily explicit, verifiable grounding in prior case law.

In another case, JOR 2026/137, GPT-5.5 at $k = 5$ cites ECLI:NL:GHLEE:2007: BB3627 to support its finding that the onward payment to a third party fulfilled an existing obligation. The court itself refers to the same judgment as a “somewhat comparable case” in footnote 1 to paragraph 3.4. The retrieval step of the neuro-symbolic model thus surfaces a legally relevant case that the court itself also cites in the gold standard.

To better understand the model’s performance, we break down the results per CBR subroutine in Table 4. The voluntary-performance and prejudice subroutines reach high recall and moderate-to-high $F _ { 1 }$ across all three models, whereas the knowledge subroutine attains markedly lower precision, recall, and F throughout. For the knowledge subroutine, only Gemini scores slightly higher at k = 5 than at $k = 0$ . The other two models remain largely unchanged, so the retrieved prior decisions do not systematically raise performance on this element.

Qualitative inspection shows differences in how the models reach their conclusions. In JOR 2026/137, GPT-5.5 and Gemini at $k = 5$ cite ECLI:NL:GHSHE:2014:4568 to reach opposite conclusions on prejudice to creditors. Gemini follows the insolvency practitioner’s argument: the onward payment to a third party removes funds available to the general body of creditors, and it applies ECLI:NL:GHSHE:2014:4568 by analogy. GPT-5.5 instead distinguishes the present case from that precedent, treating the onward payment as putting right a mistake made by the bank, and finds no prejudice, in line with the court in the gold standard. The same case-law reference can thus, particularly in hard cases, support different but defensible positions grounded in prior judgments.

In JOR 2014/78, all three models at $k = 5$ find the debtor’s knowledge of prejudice unproven, leading the symbolic model to an incorrect verdict that the transaction is not avoidable. The court instead treats knowledge as established because it was insufficiently disputed, a finding that is hard to infer from the model input. This is consistent with Nouwens et al.’s [37] observation that imperfect accuracy may reflect differences between the knowledge available to the system and the human decision-maker. Mistral’s explanation adds a second error: it invokes the evidentiary presumption by incorrectly placing the 2005 legal act within one year of the 2009 bankruptcy order. The symbolic model correctly rejects the presumption because the dates are over one year apart, but the explanation is not updated accordingly. Such drift in textual explanations follows from the loose symbolic-neural coupling and could be addressed in future work through chain-of-thought prompting. The same incorrect verdict thus masks different reasoning failures that boolean scores alone cannot distinguish.

Finally, the accuracy of the final verdict, whether the transaction is avoidable under Art. 42 Fw, is reported in Table 5 as the number of correct verdicts out of the ten cases. The accuracy is modest for this sample. Manual inspection of the neuro-symbolic model’s output reveals no single recurring cause for a diverging final verdict. For none of the models does $k = 1 0$ yield more correct verdicts than $k = 5$

Table 5. Correct final verdicts (avoidable under Article 42 Fw) out of ten gold-standard cases, by model and retrieval depth.
<table><tr><td>Model</td><td>k = 0</td><td>k = 5 k = 10</td></tr><tr><td>gemini-3.5-flash</td><td>7</td><td>7</td><td>6</td></tr><tr><td>gpt-5.5</td><td>7</td><td>7</td><td>7</td></tr><tr><td>mistral-medium-3.5-128b:fp8</td><td>6</td><td>6</td><td>6</td></tr></table>

## 6. Conclusion

We show how Article 42 Fw can be modelled neuro-symbolically by combining a symbolic Catala model that follows the decision scheme of the actio pauliana with neural routines for factual extraction and open-textured legal reasoning. A pipeline for collecting, segmenting, and LLM-labelling judgments provides the corpus used for retrieval, with both segmentation and labelling performing strongly against our gold standard.

On ten hard, out-of-corpus validation cases, performance of the neuro-symbolic model is strongest for voluntary performance and prejudice to creditors, while the results indicate particular difficulty in resolving the knowledge requirement. Increasing retrieval depth does not yield a clear quantitative improvement, but upon inspection, retrieved judgments do seem to strengthen the model’s textual reasoning through explicit and legally relevant grounding in prior case law.

The present court-centric evaluation captures only one way in which the model may be assessed. The validation set consists largely of cases published in a jurisprudence journal precisely because they are legally interesting, and is therefore skewed towards harder cases. Uncontested, routine cases are settled by an extrajudicial declaration and therefore leave no public record, so they could not be evaluated. In the harder cases that do reach the courts, a model conclusion that differs from the court may still reflect a defensible argument for the opposing side, as the qualitative inspection of the model’s reasoning illustrates. Future work should therefore evaluate the model with insolvency practitioners, who can steer its inputs and argumentation and thereby enable further validation of its legal reasoning.

## Acknowledgements

We thank Henk Jan Boven, insolvency practitioner, and Gert-Jan Boon, assistant professor of company and insolvency law at Leiden University, for their valuable legal expertise and guidance throughout this research.

David Graus is partly funded by ICAI (AI for Open Government Lab). Views expressed in this paper are not necessarily shared or endorsed by those funding the research.

## References

[1] Rissland EL, Ashley KD, Loui RP. AI and Law: A fruitful synergy. Artificial Intelligence. 2003;150(1):1-15. Available from: https://www.sciencedirect.com/science/article/pii/ S000437020300122X.

[2] Ashley KD. Modeling Statutory Reasoning. In: Artificial Intelligence and Legal Analytics: New Tools for Law Practice in the Digital Age. Cambridge: Cambridge University Press; 2017. p. 38-72. Available from: https://doi.org/10.1017/9781316761380.

[3] De Weijs RJ. Faillissementspauliana, Insolvenzanfechtung & Transaction Avoidance in Insolvencies. No. 1 in Recht en praktijk. Insolventierecht. Deventer: Wolters Kluwer; 2010.

[4] Araszkiewicz M. Precedent as Interpretive Resource: A Case Frame Analysis of Statutory Reasoning. In: Markovich R, Di Caro L, Rapp A, Schifanella C, editors. Legal Knowledge and Information Systems: JURIX 2025: The Thirty-eighth Annual Conference. vol. 416 of Frontiers in Artificial Intelligence and Applications. Amsterdam: IOS Press; 2025. p. 2-13. Available from: https://doi.org/10.3233/ FAIA251572.

[5] Merigoux D, Chataing N, Protzenko J. Catala: A Programming Language for the Law. Proceedings of the ACM on Programming Languages. 2021;5(ICFP):1-29. Available from: https://doi.org/10. 1145/3473582.

[6] Pannevis NB. Schuldeisersbenadeling. In: Insolventierecht. 16th ed. Deventer: Wolters Kluwer; 2026. Chapter 7.

[7] Vriesendorp RD. Insolventierecht. 3rd ed. No. 8 in Studiereeks Burgerlijk Recht. Deventer: Wolters Kluwer; 2024.

[8] Warendorf HCS, Thomas RL. The Dutch Bankruptcy Act: De faillissementswet in het Engels vertaald. Deventer: Wolters Kluwer; 2015.

[9] Wessels B. Gevolgen van faillietverklaring. vol. III of Insolventierecht. 5th ed. Deventer: Wolters Kluwer; 2019.

[10] Supreme Court of the Netherlands (Hoge Raad). ECLI:NL:HR:2009:BI8493 Van Dooren q.q./ABN AMRO III. 2009.

[11] Ministerie van Binnenlandse Zaken en Koninkrijksrelaties. RegelRecht: van wet naar digitale werking; 2025. Accessed: 2026-03-17. Available from: https://minbzk.github.io/regelrecht/.

[12] Ministerie van Binnenlandse Zaken en Koninkrijksrelaties. Regelspraak; 2025. Accessed: 2026-03-17. Available from: https://regels.overheid.nl/docs/methods/REGELSPRAAK.

[13] Huttner L, Merigoux D. Catala: Moving towards the future of legal expert systems. Artificial Intelligence and Law. 2022 Aug. Available from: https://doi.org/10.1007/s10506-022-09328-5.

[14] Allen LE, Engholm CR. Normalized Legal Drafting and the Query Method. Journal of Legal Education. 1978;29:380-412. Available from: https://repository.law.umich.edu/facarticles/29.

[15] Jurafsky D, Martin JH. Large Language Models. In: Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition, with Language Models. 3rd ed.; 2026. Online manuscript, draft of January 6, 2026. Available from: https://web. stanford.edu/ jurafsky/slp3/.

[16] Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V, Goyal N, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In: Larochelle H, Ranzato M, Hadsell R, Balcan MF, Lin H, editors. Advances in Neural Information Processing Systems 33 (NeurIPS 2020). Red Hook, NY: Curran Associates; 2020. p. 9459-74. Available from: https://proceedings.neurips.cc/paper/2020/ file/6b493230205f780e1bc26945df7481e5-Paper.pdf.

[17] Gao Y, Xiong Y, Gao X, Jia K, Pan J, Bi Y, et al.. Retrieval-Augmented Generation for Large Language Models: A Survey; 2024. Available from: https://arxiv.org/abs/2312.10997.

[18] Wiratunga N, Abeyratne R, Jayawardena L, Martin K, Massie S, Nkisi-Orji I, et al. CBR-RAG: Case-Based Reasoning for Retrieval Augmented Generation in LLMs for Legal Question Answering. In: Recio-Garcia JA, Orozco-del Castillo MG, Bridge D, editors. Case-Based Reasoning Research and Development. Cham: Springer Nature Switzerland; 2024. p. 445-60.

[19] Van der Meer V, Rossi J. LegalCheck: Retrieval- and Context-Augmented Generation for Drafting Municipal Legal Advice Letters. In: Proceedings of the Twenty-First International Conference on Artificial Intelligence and Law. ICAIL ’26; 2026. Accepted for publication. Available from: https://arxiv.org/abs/2605.12012.

[20] Li H, Liu W, Qu Z, Xia Y, Jiang H, Ding H. LEE: An Efficient Scheme of Legal Element Extraction based on Large Language Model. In: 2025 IEEE International Symposium on Parallel and Distributed Processing with Applications (ISPA). Los Alamitos, CA, USA: IEEE Computer Society; 2025. p. 1372-7. Available from: https://doi.ieeecomputersociety.org/10.1109/ISPA67752.2025. 00189.

[21] Kautz H. The Third AI Summer: AAAI Robert S. Engelmore Memorial Lecture. AI Magazine. 2022;43(1):105-25. Available from: https://doi.org/10.1002/aaai.12036.

[22] Hude Z. Where Has Legal Knowledge Gone: Constraining LLMs with Knowledge Graphs for Interpretable Reasoning; 2025. Paper presented at the JURIX 2025 Doctoral Consortium, Turin, Italy. Available from: https://doi.org/10.2139/ssrn.6149326.

[23] von Cossel O. Beyond Imperfect Alternatives with Rulemapping: A Neuro-Symbolic Case Study on Online Hate Speech. In: Proceedings of the Twenty-First International Conference on Artificial Intelligence and Law. ICAIL ’26; 2026. Accepted for publication. Available from: https://arxiv.org/ abs/2605.16280.

[24] Zeleznikow J, Stranieri A. The split-up system: integrating neural networks and rule-based reasoning in the legal domain. In: Proceedings of the 5th International Conference on Artificial Intelligence and Law. ICAIL ’95. New York, NY, USA: Association for Computing Machinery; 1995. p. 185-94. Available from: https://doi.org/10.1145/222092.222235.

[25] Kant M, Nabi S, Kant M, Scharrer R, Ma M, Nabi M. Towards Robust Legal Reasoning: Harnessing Logical LLMs in Law; 2025. Available from: https://arxiv.org/abs/2502.17638.

[26] Sadowski A, Chudziak JA. On Verifiable Legal Reasoning: A Multi-Agent Framework with Formalized Knowledge Representations. In: Proceedings of the 34th ACM International Conference on Information and Knowledge Management. CIKM ’25. ACM; 2025. p. 2535-45. Available from: http://dx.doi. org/10.1145/3746252.3761057.

[27] IVO Rechtspraak. Open Data van de Rechtspraak. Raad voor de Rechtspraak; 2019. Version 1.15. Available from: https://www.rechtspraak.nl/binaries/\_rts\_1768910542320/content/ assets/ivo/wi/ivo-wi-technische-documentatie-open-data-van-de-rechtspraak. pdf.

[28] Tjong Tjin Tai E. Methodologische aspecten van een data pipeline voor rechtspraakanalyse. In: Verbruggen P, editor. Methoden van systematische rechtspraakanalyse: Tussen juridische dogmatiek en data science. Den Haag: Boom Juridische Uitgevers; 2021. p. 163-84. Available from: https://research. tilburguniversity.edu/en/publications/075561be-d320-44d3-a342-dc5f102ca516.

[29] Trompper M, Winkels R. Automatic Assignment of Section Structure to Texts of Dutch Court Judgments. In: Bex F, Villata S, editors. Legal Knowledge and Information Systems: JURIX 2016: The Twenty-Ninth Annual Conference. vol. 294 of Frontiers in Artificial Intelligence and Applications. Amsterdam: IOS Press; 2016. p. 167-72.

[30] Crombach C, Chakravarthy S, Wicker DHF. rechtspraak-segmentation. GitHub; 2025. Last code update: September 2024; README updated June 2025. Accessed: 2026-05-13. Available from: https:// github.com/maastrichtlawtech/rechtspraak-segmentation.

[31] Ahsmann MJAM. De weg naar het civiele vonnis. 2nd ed. Den Haag: Boom Juridisch; 2020.

[32] Bartelds J, Burg E, Lewin G, Schreuder D, Veldhuijzen van Zanten E, Ter Veer C, et al.. Professionele standaard Schrijven: Best practices en aanbevelingen van de civiele afdelingen van de hoven; 2020. Available from: https://www.rechtspraak.nl/binaries/\_rts\_1768396290578/ content/assets/lov/ps/lov-ps-professionele-standaard-schrijven.pdf.

[33] Hobbs JR, Appelt D, Bear J, Israel D, Kameyama M, Stickel M, et al. FASTUS: A Cascaded Finite-State Transducer for Extracting Information from Natural-Language Text. In: Roche E, Schabes Y,

editors. Finite-State Language Processing. Cambridge, MA: MIT Press; 1997. p. 383-406. Available from: https://arxiv.org/abs/cmp-lg/9705013.

[34] Xue J, Nguyen PM, Nguyen HT, Zin MM, Satoh K. Legal2LogicICL: Improving Generalization in Transforming Legal Cases to Logical Formulas via Diverse Few-Shot Learning. In: Proceedings of the Twenty-First International Conference on Artificial Intelligence and Law. ICAIL ’26; 2026. Accepted for publication. Available from: https://arxiv.org/abs/2604.11699.

[35] Zhang Y, Li M, Long D, Zhang X, Lin H, Yang B, et al. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv preprint arXiv:250605176. 2025.

[36] Dai Z, Callan J. Deeper Text Understanding for IR with Contextual Neural Language Modeling. In: Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. SIGIR ’19. ACM; 2019. p. 985-8. Available from: http://dx.doi.org/10.1145/ 3331184.3331303.

[37] Nouwens J, Borg A, Prakken H. An Application of Case-Based Reasoning to Decision-Making in Dutch Administrative Law. In: Savelka J, Harasta J, Novotna T, Misek J, editors. Legal Knowledge and Information Systems: JURIX 2024: The Thirty-seventh Annual Conference. vol. 395 of Frontiers in Artificial Intelligence and Applications. Amsterdam: IOS Press; 2024. p. 131-41. Available from: https://doi.org/10.3233/FAIA241240.