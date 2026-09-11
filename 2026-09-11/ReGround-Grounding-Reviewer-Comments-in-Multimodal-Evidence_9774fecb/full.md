# ReGround: Grounding Reviewer Comments in Multimodal Evidence

Serwar Basch<sup>1</sup>, Lizhen Qu<sup>2</sup>, Iryna Gurevych<sup>1</sup>

<sup>1</sup>Ubiquitous Knowledge Processing Lab (UKP Lab), Department of Computer Science and Hessian Center for AI (hessian.AI), TU Darmstadt <sup>2</sup>Department of Data Science & AI, Monash University, Australia www.ukp.tu-darmstadt.de

## Abstract

Reviewer comments naturally relate to specific parts of the reviewed paper, yet grounding these comments to the underlying evidence is difficult due to long multimodal documents. Existing benchmarks do not capture this setting and largely focus on explicit, information-seeking queries. We introduce ReGround, a large-scale dataset for reviewer comment grounding that links 10,267 reviewer comments to 16,274 evidence in the original anonymous submissions of 3,656 papers. We build on a simple observation: author rebuttals often include explicit references to content of the submission used to address reviewer comments, providing a highprecision annotation source. We cast grounding as a retrieval task and evaluate a wide range of retrieval methods. Results show that retrieval over the entire paper content performs poorly, evidence-type inference is a major bottleneck, and multimodal evidence provides complementary signals that text alone misses. Our dataset exposes grounding reviewer comments as a difficult and practically important problem for scientific document understanding.<sup>1</sup>

## 1 Introduction

Peer review produces a large volume of expert comments about scientific papers, and the load is growing fast enough that top AI venues have begun deploying reviewer assistance systems into their official workflows (Thakkar et al., 2025). A core challenge for any such system is grounding a reviewer comment in the paper content it depends on: a methodological detail, a result table, a sectionlevel claim, or a visual pattern in a figure. Beyond peer review, this instantiates a broader scientificdocument problem: grounding natural, underspecified queries in long multimodal documents. The challenge is twofold: evidence is heterogeneous in modality, appearing as text, tables, or figures, and in granularity, from a single line to a full section.

![](images/41e70d69c90821c672b69b09f54d024acd72dda2802575e4e0da435b2a1cffd3.jpg)  
Figure 1: Overview of dataset construction pipeline and a representative example. We use explicit paper references in author rebuttals to derive high-precision links between reviewer comments and content in the original anonymous submission. (1) Detect rebuttal sentences that reference specific paper content. (2) Align each reference-bearing rebuttal sentence to the reviewer span it addresses. (3) Resolve the references in the submitted PDF, and extract the corresponding content, yielding a reviewer comment–paper content pair.

Existing resources do not directly support this setting (Table 1). Scientific QA datasets such as QASPER (Dasigi et al., 2021) and PEERQA (Baumgärtner et al., 2025) focus on informationseeking questions with relatively well-defined answer spans. Reviewer comments, by contrast, are often evaluative, underspecified, and implicitly connected to the paper content, as illustrated in Figure 1. Revision-based datasets such as ARIES (D’Arcy et al., 2024) link comments to edits in later versions of a paper rather than to the evidence available in the original submission. Finally, resources that target fine-grained links to the original paper, such as F1000RD (Kuznetsov et al., 2022) and CLAIMCHECK (Ou et al., 2025), are manually annotated, small, and text-only.

<table><tr><td>Dataset</td><td>Annotation Method</td><td>Original Sub.</td><td>Multimodal</td><td>Evidence Types</td><td>Size</td></tr><tr><td>ARIES</td><td>manual</td><td>x</td><td>x</td><td>text spans</td><td>42 papers</td></tr><tr><td>PEERQA</td><td>manual</td><td>X</td><td>x</td><td>sentences/paragraphs</td><td>208 papers</td></tr><tr><td>QASPER²</td><td>manual</td><td>X</td><td>√</td><td>paragraphs</td><td>1585 papers</td></tr><tr><td>F1000RD</td><td>manual</td><td>V</td><td>X</td><td>lines/paragraphs/sections</td><td>172 papers</td></tr><tr><td>CLAIMCHECK</td><td>manual</td><td>V</td><td>X</td><td>text spans</td><td>41 papers</td></tr><tr><td>Ours</td><td>automatic</td><td></td><td></td><td>lines/sections/figures/tables</td><td>3656 papers</td></tr></table>

Table 1: Comparison of datasets that link queries and peer-review comments to full manuscript content.

We identify author rebuttals as a high-precision source of grounding annotations. When responding to reviews, authors often explicitly point to specific parts of the submitted paper as evidence for addressing a concern, clarifying a point, or highlighting content that may have been overlooked. We refer to these mentions as paper references, and the referenced content as evidence. Because rebuttals discuss the paper as it was reviewed, these references expose links between reviewer comments and the original anonymous submission, rather than links to a revised or camera-ready version.

On this basis, we introduce ReGround, a largescale dataset for grounding reviewer comments in multimodal evidence from scientific papers. ReGround contains 3,656 papers and 16,274 reviewer comment–evidence pairs covering paragraphs, sections, tables, and figures. We formalize grounding as a retrieval task: given a reviewer comment, the goal is to retrieve the evidence authors used to address it. Beyond peer review, the dataset targets a capability shared with scientific QA, fact-checking, and document-grounded assistance: retrieving multimodal evidence for naturallanguage queries over long papers. We benchmark sparse, dense, cross-encoder, LLM-based, and multimodal retrievers across text and visual modalities.

Our findings expose grounding as a problem with several distinct difficulties: (1) retrieval over a heterogeneous evidence pool is hard across all model families, with even the best LLM-based ranker reaching only 21% Recall@10; (2) inferring the type of evidence (paragraph, section, figure, table) is a major bottleneck; (3) models also struggle with comments grounded to multiple evidence; (4) for figures and tables, visual and textual signals are complementary, with substantial modalityspecific failure modes that motivate multimodal fusion rather than replacement.

## 2 Related Work

A growing line of work has introduced peer-review corpora to study the reviewing process and reviewer–author interactions (Kang et al., 2018; Dycke et al., 2023; Zhang et al., 2026). Several datasets further annotate review–rebuttal exchanges with discourse or argument structure, such as APE (Cheng et al., 2020) and DISAPERE (Kennard et al., 2022). Although valuable for modeling the review process, these corpora work on document level, while our work directly targets the fine-grained links between reviewer comments and paper content.

Other related datasets attempt to link reviews with paper content, but differ substantially from our setting. ARIES (D’Arcy et al., 2024) and RE3 (Ruan et al., 2024) align reviewer comments with paper revisions, capturing how feedback manifests in revised papers rather than how comments relate to the original submission. PEERQA (Baumgärtner et al., 2025) reframes reviewer questions as document-level QA with manually annotated answer spans, but operates on camera-ready papers and is limited to text-only evidence. More closely related, Kuznetsov et al. (2022) introduce the F1000RD dataset, and study implicit linking between review sentences and paper content, highlighting the difficulty of grounding underspecified review comments in long documents. However, their work is text-only and relies on manual annotation. Other efforts, such as CLAIMCHECK (Ou et al., 2025) and ABCD-LINK (Basch et al., 2026), focus on claim-centric or sentence-level linking and remain small in scale. Concurrent to our work, PRISMM-Bench (Selch et al., 2026) compiles 384 reviewer-flagged multimodal inconsistencies from ICLR papers as a multiple-choice benchmark. In contrast, our dataset is derived automatically from author rebuttals, targets the original submission, scales to thousands of papers, and covers finegrained multimodal evidence.

Our task is also related to scientific evidence retrieval and document-level QA, where systems must identify supporting evidence in long documents. Prior benchmarks include fact-checking datasets such as SCIFACT (Wadden et al., 2020, 2022), citation-span identification (Li et al., 2020), and QA over full papers such as QASPER (Dasigi et al., 2021). Recent work has extended these settings to multimodal and PDF-based retrieval too (Dong et al., 2025). Our setting differs in several key respects: reviewer comments are naturally occurring not crowdsourced, are often evaluative rather than information-seeking; and figures and tables play a central role in addressing them. By grounding reviewer comments to fine-grained, multimodal evidence in original submissions, our dataset exposes a challenging and practically important retrieval problem that is not captured by existing benchmarks.

## 3 Dataset Construction

## 3.1 Task Definition

We cast reviewer comment grounding as a retrieval task. Given a reviewer comment c, the goal is to retrieve author-cited evidence: paper content that addresses, clarifies, refutes, or contextualizes c. We build a candidate pool $D = \{ d _ { 1 } , \ldots , d _ { n } \}$ of evidence units extracted from the same original submission using a structured document representation (§3.3). A reviewer comment may be grounded in one or more units, so we evaluate both singleand multi-evidence retrieval (§4).

Retrieval is a natural framing because it is the shared prerequisite for multiple downstream applications. We further target the implicit setting, in which the comment does not itself contain an explicit reference (e.g., “Figure 3”, “Section 4.2”, “line 120”) to the gold evidence, since such references are trivially resolved by string matching.

Evidence units. The retrieval pool contains textual evidence units (paragraphs, sections, and figure/table captions) and visual evidence units (figure and table images). Line references are mapped to their containing paragraphs, because ACL-style line numbers are formatting artifacts rather than semantic units.

## 3.2 Data Source

Our task requires the original submission to preserve the evidence that authors and reviewers actually discuss, in contrast to revised or camera-ready versions that may have been edited based on the reviews. To that end, we use version 2 of NLPeer (Dycke et al., 2023) with original submissions, reviews, and rebuttals from EMNLP 24/25, COL-ING 25, NAACL 25 and ACL 25.

## 3.3 Preprocessing

Review and rebuttal segmentation. We segment reviews and rebuttals into sentences using a custom rule-based splitter tailored to scientific writing. The splitter handles common reference patterns (e.g., “Fig. 4”, “Sec. 6.6”) and list-style formatting that breaks off-the-shelf tools.<sup>3</sup>

Paper content extraction. We parse the original submission PDFs to create a structured document representation for each paper. Textual content is extracted at multiple levels: (i) individual numbered lines; (ii) section-level spans, identified via section headers and numbering; and (iii) figure and table captions. Since ACL-style papers follow standardized templates, we use PDF parsing via pypdf<sup>4</sup> rather than OCR, because it ensures deterministic and consistent results. However, because paragraph boundaries cannot be reliably reconstructed from parsing PDFs, we use GROBID (Lopez, 2009) to parse them. Finally, we extract figures and tables as images along with their captions using PDFFigures 2.0 (Clark and Divvala, 2016).

## 3.4 Constructing Review-Evidence Links

Detecting paper references. We identify rebuttal sentences that explicitly reference paper content using regular expressions developed iteratively over a random sampling of 250 rebuttals, and refined after a final manual inspection of the resulting dataset. The regex patterns cover references to lines, sections, tables, figures, appendices, equations, pages and footnotes.<sup>5</sup>

Aligning rebuttals to review comments. Authors often quote or index the reviewer comment they are addressing (e.g., “W1”, “Q5”). Using these cues, we employ an LLM (gpt-oss-120b<sup>6</sup>) to align each reference-bearing rebuttal sentence to its corresponding reviewer comment.<sup>7</sup> When a rebuttal addresses multiple reviewer segments, we concatenate them. Manual evaluation of alignment quality is detailed in §3.6.

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Papers</td><td>3656</td></tr><tr><td>Reviewer comments (queries) Paper references (evidence)</td><td>10267</td></tr><tr><td>Unique Paper references</td><td>16274 13391</td></tr><tr><td></td><td></td></tr><tr><td>Avg. queries per paper Avg. evidence per query</td><td>2.81</td></tr><tr><td>% Queries with  $\geq 2$  evidence</td><td>1.58 25.77</td></tr></table>

Table 2: Summary statistics of the final dataset.

Resolving paper references. We resolve each detected paper reference against the structured document representation. Line references are mapped to the paragraphs that contain them, section references are matched by number and title, and figure/table references are matched by index and linked to both their captions and extracted images.

## 3.5 Filtering

The raw links extracted from rebuttals contain three sources of noise, which we remove with successive filters (see §D for details). First, authors frequently quote reviewer text verbatim, which can trigger false-positive reference detection; we discard rebuttal sentences that overlap with the review using fuzzy string matching (\~20% of sentences). Second, to focus on the implicit setting, if a reviewer comment already references the same paper content referenced in the rebuttal, grounding becomes trivially solvable with string matching, so we drop such pairs (\~18% of pairs). Third, we prompt gpt-oss-120b to remove pairs where the author-cited evidence is not actually present in the original submission<sup>8</sup>, namely (i) promised cameraready changes, (ii) new experiments introduced in the rebuttal, or (iii) content from external papers (\~23% of pairs); we validate this step manually in §3.6.

## 3.6 Quality Assurance

We manually evaluate the two LLM-based stages of the pipeline and inspect the final dataset.

Review–rebuttal alignment. Two annotators<sup>9</sup> verify alignments for 100 randomly sampled papers, achieving 96% alignment acceptance (with Cohen’s κ=0.83). Most prevelant errors (13%) are under-selection of the reviewer comment span rather than incorrect alignment.

Filtering validation. We annotate 300 reviewer comment–rebuttal sentence pairs using the same criteria (§3.5) given to the model (κ = 0.84). The LLM filter achieves 97% precision and 93% recall against this gold subset, with a small bias toward removing ambiguous cases. The high precision is expected since the filtering criteria rely on surfacelevel cues such as future-tense references ("we will update..."), mentions of new experiments, and standard citation patterns.<sup>10</sup>

Manual inspection. We inspect 200 reviewer comments and their 316 referenced evidence to assess reference resolution and evidence validity. We find that 1.58% of evidence spans are coarsegrained (e.g., a whole section cited when a subsection would suffice); we retain these since they reflect realistic author behavior and the appropriate boundary is often subjective. On a 100-comment subset, we find that all referenced evidence is relevant to the reviewer comment, and in 84% of cases the reviewer explicitly acknowledged the rebuttal, indirectly validating their relevancy. We note that this acknowledgment represents the reviewer being satisfied with the rebuttal, and not a signal that the review was refuted.<sup>11</sup>

## 3.7 Dataset Statistics

Dataset scale. Our final dataset contains 3,656 papers, comprising 10,267 reviewer comments and 16,274 paper evidence (Table 2). Notably, 25.77% of comments have multiple evidence, underscoring the need for methods that can effectively handle multi-evidence scenarios.

Evidence types. Figure 2 shows that references to lines, sections, figures, and tables account for nearly 90% of all evidence, motivating our focus on them, while appendix references comprise around 8%. The distribution of evidence varies substantially by type, and notably, textual evidence varies widely in length, from short captions to long section-level spans, resulting in uneven retrieval pools across evidence types. This complicates retrieval, as models must retrieve from candidate units that differ in length and information density.

Limits of semantic similarity. To quantify the relation between reviewer comments and their gold evidence, we compute their cosine similarity and obtain an average of 0.377, indicating limited semantic overlap. For comparison, the similarity between the corresponding rebuttal sentence and gold evidence is 0.431. This is expected: reviewer comments are often evaluative and underspecified, and the referenced evidence frequently corresponds to content that reviewers may have missed or overlooked. This weak similarity suggests that retrieval based on semantic matching alone is insufficient, motivating the need for better methods. 12

![](images/ffa6d841785f10cce5f5082b20e13588722ea5a4597cb9dacba694d59d6a295c.jpg)

![](images/ae4f9d54bfb4e04d7dbd355470771592713aac3dca257e4df7a8a46ccb507247.jpg)

![](images/e421683573f39270eaebdf416ee396a677d90da63cac4c2ba7c9d625c3461ceb.jpg)

![](images/bde3f2e74f6215cc25aa4120d3fa8c89bcbeaa488ddc6b06b32ef9180c6384ad.jpg)

![](images/bb080153ede8e3bad13b7f784fbae0405fa247f51389898b5cf86f5684cc97cc.jpg)

![](images/e1b72dcf2807eebfea35ca652daa96588a2367d668408d11aa05031d70df1aee.jpg)

![](images/820cc1bf61f2a9eb5e7799c3df3733bf24369e7374f2c657d872936f83a56c1a.jpg)  
Figure 2: Dataset composition and evidence types. Left: Distribution of number of candidate units of each type (paragraphs, sections, figures, tables) per paper, showing substantial variation in retrieval pool size across types. Center: Distribution of reference types in the dataset. Rest includes references to pages, footnotes, and equations. Right: Token-length distributions of textual evidence units by type, illustrating large disparities in semantic scope.

Types of reviewer comments. Finally, we manually annotate 200 randomly sampled comments with a taxonomy of addressed paper issues. The most common category is Empirical Rigor (31%), encompassing concerns about experimental design, evaluation, and analysis. Two other categories– Technical Soundness and Deployment & Impact– each account for \~16%. This distribution confirms that the dataset captures a broad range of reviewer concerns grounded in diverse parts of the paper.<sup>13</sup>

## 3.8 Use Cases

ReGround supports reviewer-facing systems that connect review comments to the specific parts of a paper they concern. First, in a human reviewerassistance workflow, a system can retrieve relevant paper locations for a drafted comment. This helps reviewers verify a claim against the submission, identify potentially overlooked material, and attach precise citations to the review. The reviewer remains responsible for the final judgment, while retrieval reduces the effort required to check and ground comments.

Second, ReGround provides supervision and evaluation data for LLM-based reviewer systems. Such systems should not only generate criticisms or questions, but also link them to concrete passages, sections, figures, and tables in the paper. This grounding can make generated reviews more transparent, easier for humans to verify, and less likely to rely on unsupported claims.

## 4 Experimental Setup

ReGround supports tasks that require grounding natural-language inputs in scientific paper evidence (review assistance, scientific QA, fact-checking), where retrieval of relevant evidence is the shared prerequisite. We therefore evaluate grounding as retrieval, and study research questions that isolate the different aspects making it hard:

• RQ1: How well do models retrieve over a heterogeneous textual evidence pool, when the evidence type is unknown?

• RQ2: How well can models retrieve when the evidence type is known?

• RQ3: How well do models retrieve multiple nonredundant evidence units?

• RQ4: How much does visual information add to grounding figures and tables, beyond captions?

## 4.1 Evaluation Settings

Unified text retrieval (RQ1). We retrieve from a single heterogeneous textual candidate pool containing paragraphs, sections, and figure/table captions. In this setting, models must both identify what content is relevant and infer which evidence type it appears in. Evaluation is performed for each referenced evidence individually then averaged over all comment-evidence pairs.

Type-aware (oracle) text retrieval (RQ2). To separate retrieval difficulty from type inference, we define an oracle setting restricted to the gold evidence type. For example, comments grounded to sections are evaluated against section candidates only, and comments grounded to tables against table captions. This setting quantifies the performance loss due to heterogeneous candidate pools.

Type-aware joint evidence retrieval (RQ3). To directly test multi-evidence grounding, we additionally evaluate a joint setting on comments with $\geq 2$ grounded targets. Here, models are evaluated on their ability to retrieve the full set of relevant evidence. We assume oracle type information as above to focus on the multi-evidence challenge rather than type inference.

Visual evidence retrieval (RQ4). In this setup, we use figures and tables represented as images. We evaluate two image representations: (i) the raw figure/table image alone, and (ii) the image augmented with its caption, where the caption text is concatenated visually to the image input.

## 4.2 Baselines

Text retrieval models. We use a range of retrieval models: Sparse baselines (BM25 (Robertson and Zaragoza, 2009) and SPLADEv3 (Lassance et al., 2024)). Dense encoders (all-mpnet-base-v2<sup>1415</sup>, BGE-M3 (Chen et al., 2024), Qwen3-Embedding-4B (Zhang et al., 2025), and EmbeddingGemma (Vera et al., 2025)) which independently encode comments and evidence and rank by cosine similarity. To assess joint query–document modeling, we also include cross-encoders: ms-marco-MiniLM-L12-v2<sup>1617</sup> and bge-reranker-v2-m3 (Chen et al., 2024).

LLM-based ranking. We evaluate LLMs as pointwise relevance scorers. Given a reviewer comment and a candidate evidence, the model outputs a binary relevance judgment (Yes/No). We convert the binary decision to a continuous relevance score using a two-class softmax over the decision tokens, $s ~ = ~ \exp ( \ell _ { \mathsf { Y e s } } ) / ( \exp ( \ell _ { \mathsf { Y e s } } ) + \exp ( \ell _ { \mathsf { N o } } ) )$

<table><tr><td>Model</td><td>MRR</td><td>R@1</td><td>R@2</td><td>R@10</td></tr><tr><td>Sparse encoders</td><td></td><td></td><td></td><td></td></tr><tr><td>BM25</td><td>7.62</td><td>3.57</td><td>5.65</td><td>14.90</td></tr><tr><td>SPLADEv3</td><td>7.39</td><td>3.04</td><td>5.38</td><td>15.17</td></tr><tr><td>Dense encoders</td><td></td><td></td><td></td><td></td></tr><tr><td>all-mpnet</td><td>9.08</td><td>4.08</td><td>6.78</td><td>19.12</td></tr><tr><td>BGE-M3</td><td>9.41</td><td>4.45</td><td>7.14</td><td>19.05</td></tr><tr><td>Qwen-3 Embedding 4B</td><td>6.92</td><td>3.26</td><td>5.03</td><td>14.59</td></tr><tr><td>EmbeddingGemma</td><td>8.78</td><td>4.17</td><td>6.64</td><td>17.68</td></tr><tr><td>Cross encoders</td><td></td><td></td><td></td><td></td></tr><tr><td>BGE-M3-Reranker</td><td>7.85</td><td>3.42</td><td>5.67</td><td>16.20</td></tr><tr><td>MiniLM-L12</td><td>7.92</td><td>3.62</td><td>5.82</td><td>15.81</td></tr><tr><td>LLMs</td><td></td><td></td><td></td><td></td></tr><tr><td>Gemma-3 27B</td><td>10.12</td><td>4.68</td><td>7.74</td><td>20.34</td></tr><tr><td>Qwen-3 30B Instruct</td><td>10.87</td><td>5.12</td><td>8.29</td><td>21.15</td></tr></table>

Table 3: Unified text retrieval performance. Retrieval is performed over a heterogeneous pool of textual evidence (paragraphs, sections, and captions) without evidence type hints.

where $\ell _ { \mathsf { Y e s } }$ and $\ell _ { \mathsf { N o } }$ are the model-assigned token log-probabilities. We evaluate non-reasoning LLMs: Gemma 3 (12B, 27B) (Team et al., 2025), Qwen 3 (4B, 30B Instruct) (Yang et al., 2025), and reasoning LLMs: Qwen 3 30B Thinking (Yang et al., 2025) and gpt-oss 20b (OpenAI, 2025), and GPT-5.1<sup>18</sup> as a representative commercial model. Due to cost, GPT-5.1 is evaluated on a random 50% subset.<sup>19</sup>

Visual retrieval models. For image-based retrieval, we evaluate vision–text encoders including SigLIP2 (Tschannen et al., 2025), OpenCLIP (Radford et al., 2021), and Jina Embeddings v4 (Günther et al., 2025), which rank candidates by cosine similarity. We also evaluate a lateinteraction model, ColQwen2.5, and evaluate multimodal LLMs (Qwen 3 VL 8B, 32B (Yang et al., 2025)) using the same scoring protocol.

## 4.3 Evaluation Metrics

We report Recall@k to measure the fraction of the relevant set in the top-k ranked candidates averaged across queries, and Mean Reciprocal Rank (MRR) to measure how highly the first relevant unit is ranked, emphasizing early precision.

We evaluate at the level of evidence units with exact unit-ID matching: a prediction is counted as correct only when it matches the gold unit exactly, and hierarchically related units (e.g., a section containing the gold paragraph) receive no partial credit. Each reference resolves to a single evidence unit to avoid inflating scores via ancestor units.

<table><tr><td></td><td colspan="4">Paragraph</td><td colspan="3">Section</td><td colspan="3">Table Caption</td><td colspan="3">Figure Caption</td></tr><tr><td>Model</td><td>MRR</td><td>R@1</td><td>R@2</td><td>R@10</td><td>MRR</td><td>R@1</td><td>R@2</td><td>MRR</td><td>R@1</td><td>R@2</td><td>MRR</td><td>R@1</td><td>R@2</td></tr><tr><td>Sparse encoders</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BM25</td><td>18.65</td><td>8.11</td><td>13.37</td><td>35.58</td><td>25.55</td><td>11.38</td><td>19.58</td><td>47.28</td><td>24.38</td><td>41.10</td><td>44.25</td><td>22.57</td><td>38.65</td></tr><tr><td>SPLADEv3</td><td>27.92</td><td>15.13</td><td>23.44</td><td>48.89</td><td>29.81</td><td>15.24</td><td>24.54</td><td>53.51</td><td>30.77</td><td>49.14</td><td>48.59</td><td>26.51</td><td>44.69</td></tr><tr><td>Dense encoders</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>all-mpnet</td><td>19.38</td><td>8.39</td><td>13.29</td><td>36.81</td><td>28.98</td><td>14.02</td><td>23.70</td><td>49.50</td><td>26.75</td><td>44.12</td><td>48.17</td><td>27.07</td><td>43.26</td></tr><tr><td>BGE-M3</td><td>24.38</td><td>12.63</td><td>18.91</td><td>42.59</td><td>27.55</td><td>13.25</td><td>21.62</td><td>51.61</td><td>28.99</td><td>45.80</td><td>49.05</td><td>26.43</td><td>46.00</td></tr><tr><td>Qwen-3 Embedding 4B</td><td>14.54</td><td>7.13</td><td>11.02</td><td>25.93</td><td>22.87</td><td>12.10</td><td>19.58</td><td>39.62</td><td>22.78</td><td>36.22</td><td>37.10</td><td>21.32</td><td>35.20</td></tr><tr><td>EmbeddingGemma</td><td>27.11</td><td>14.08</td><td>22.35</td><td>47.55</td><td>31.76</td><td>17.31</td><td>26.69</td><td>53.25</td><td>30.15</td><td>48.99</td><td>50.91</td><td>28.49</td><td>47.98</td></tr><tr><td>Cross encoders</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BGE-M3-Reranker</td><td>27.05</td><td>14.58</td><td>22.05</td><td>46.57</td><td>29.32</td><td>14.69</td><td>24.02</td><td>53.86</td><td>30.92</td><td>49.83</td><td>50.65</td><td>28.19</td><td>48.26</td></tr><tr><td>MiniLM-L12</td><td>24.38</td><td>13.25</td><td>19.85</td><td>41.94</td><td>28.48</td><td>14.39</td><td>22.60</td><td>48.51</td><td>26.26</td><td>42.67</td><td>47.19</td><td>25.54</td><td>42.85</td></tr><tr><td>LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemma-3 12B</td><td>32.55</td><td>18.73</td><td>27.72</td><td>53.05</td><td>30.76</td><td>14.87</td><td>26.15</td><td>53.41</td><td>30.89</td><td>49.89</td><td>53.20</td><td>32.59</td><td>50.68</td></tr><tr><td>Gemma-3 27B</td><td>34.87</td><td>20.32</td><td>30.01</td><td>57.62</td><td>31.36</td><td>14.48</td><td>26.37</td><td>53.72</td><td>31.65</td><td>49.87</td><td>52.25</td><td>32.50</td><td>49.00</td></tr><tr><td>Qwen-3 4B</td><td>31.08</td><td>18.02</td><td>27.17</td><td>54.16</td><td>30.49</td><td>14.47</td><td>25.12</td><td>54.31</td><td>31.59</td><td>49.95</td><td>52.01</td><td>31.30</td><td>48.80</td></tr><tr><td>Qwen-3 30B Instruct</td><td>32.97</td><td>19.19</td><td>27.69</td><td>53.64</td><td>33.88</td><td>18.16</td><td>30.46</td><td>56.17</td><td>33.31</td><td>53.29</td><td>55.40</td><td>34.37</td><td>55.16</td></tr><tr><td>Qwen-3 30B Thinking</td><td>25.44</td><td>13.76</td><td>20.76</td><td>42.39</td><td>30.26</td><td>15.20</td><td>25.05</td><td>50.91</td><td>28.92</td><td>44.87</td><td>46.46</td><td>25.11</td><td>42.31</td></tr><tr><td>gpt-oss-20b</td><td>28.36</td><td>15.71</td><td>24.44</td><td>44.14</td><td>29.39</td><td>14.89</td><td>24.32</td><td>47.30</td><td>24.36</td><td>40.83</td><td>45.29</td><td>24.39</td><td>41.01</td></tr><tr><td>GPT-5.1</td><td>36.26</td><td>21.62</td><td>31.34</td><td>59.07</td><td>32.98</td><td>16.51</td><td>28.73</td><td>54.26</td><td>31.69</td><td>50.63</td><td>52.64</td><td>32.86</td><td>49.86</td></tr></table>

Table 4: Type-aware (oracle) text retrieval performance. Retrieval is restricted to the gold evidence type, isolating retrieval difficulty from type inference. Note that GPT-5.1 is evaluated on a 50% subset due to cost. Cutoff for figures and tables is at 2 because of the smaller pool size, and for sections at 2 because higher cutoffs correspond to more than one page which is impractical.

## 5 Results

## 5.1 Unified Text Retrieval (RQ1)

Table 3 shows that unified retrieval over a heterogeneous textual pool is difficult across all model families: MRR remains below 11 and R@10 peaks at 21.15. LLM-based rankers achieve the best results, but the gains over the dense models are modest (10.87 vs. 9.41 MRR), indicating substantial remaining headroom. More notably, cross-encoders fail to outperform bi-encoders despite their stronger joint query–document modeling, a counterintuitive result that motivates a closer look.

We analyze the evidence length effect in detail in §M, and find that cross-encoders (BGE-M3-Reranker) collapse on the longest section quantile, with MRR dropping from 44 at Q5 to near zero at Q10, while LLM-based rankers continue to improve up to Q8 and degrade only gracefully thereafter. Since the unified pool mixes short captions with long section-level spans (cf. Figure 2), cross-encoders are penalized on these instances. These results establish the difficulty of unified text retrieval, and motivate model choices that are robust to long, heterogeneous candidates.

## 5.2 Type-aware (oracle) Text Retrieval (RQ2)

Table 4 shows that restricting retrieval to the gold evidence type substantially improves performance across all models and evidence types. These performance gains indicate that evidence-type uncertainty is a major source of error in the unified setting.

Nonetheless, performance still varies by type. Paragraph- and section-level retrieval remain challenging: even the strongest models achieve only around 31% Recall@2, consistent with the observation of semantic gap between reviewer comments and gold evidence (§3.7). At higher cutoffs, recall improves, but retrieving 10 paragraphs can correspond to more than a page of text in some papers, which reduces its practical usefulness.

Caption retrieval achieves the highest scores overall. However, caption candidate pools are also smaller than paragraph/section pools, which likely contributes to the higher recall (cf. Figure 2).

Across model families, larger LLMs perform best under oracle conditions, but the gap between the models narrows compared to unified retrieval. This pattern suggests that improving routing to the correct evidence type is an essential bottleneck.

We also test whether this oracle gain can be recovered by predicting the evidence type before retrieval. Concretely, hard filtering by predicted type drops Qwen-3 30B Instruct’s R@10 from 21.15 to 6.11 (§O), because errors on figure/table prediction remove the correct evidence from the pool. Type prediction is thus better viewed as a soft reranking signal than a hard filter, a finding that shapes how systems should integrate routing.

<table><tr><td>Model</td><td>MRR</td><td>R@1</td><td>R@2</td><td>R@10</td></tr><tr><td>EmbeddingGemma</td><td>37.63</td><td>10.25</td><td>18.79</td><td>55.94</td></tr><tr><td rowspan="2">Qwen-3 30B Instruct GPT-5.1</td><td>43.29</td><td>12.69</td><td>23.53</td><td>62.43</td></tr><tr><td>39.53</td><td>8.99</td><td>19.67</td><td>62.33</td></tr></table>

Table 5: Best models on type-aware joint retrieval (RQ3), evaluated on comments with ≥2 evidence units. Full results in §N.

<table><tr><td></td><td colspan="3">Image+Caption</td><td colspan="3">Image-only</td></tr><tr><td>Model</td><td>MRR</td><td>R@1 R@2</td><td></td><td>MRR</td><td>R@1</td><td>R@2</td></tr><tr><td colspan="7">Table Retrieval</td></tr><tr><td>SigLIP2 OpenCLIP L14</td><td></td><td>49.60 30.04 48.94</td><td></td><td>44.54</td><td>24.40</td><td>42.56</td></tr><tr><td></td><td>47.23</td><td>28.48</td><td>44.82</td><td>43.60</td><td>24.37</td><td>40.20</td></tr><tr><td>Jina Embeddings v4</td><td>55.67</td><td>37.19</td><td>55.92</td><td>52.80</td><td>34.46</td><td>51.70</td></tr><tr><td>ColQwen2.5 v0.2</td><td>57.95</td><td>40.48</td><td>58.33</td><td>53.73</td><td>35.65</td><td>52.16</td></tr><tr><td>Qwen-3 VL 8B</td><td>57.61</td><td>39.82</td><td>58.45</td><td>55.31</td><td>37.87</td><td>54.00</td></tr><tr><td>Qwen-3 VL 32B</td><td>59.53</td><td>42.39</td><td>60.34</td><td>56.79</td><td>39.22</td><td>56.97</td></tr><tr><td colspan="7">Figure Retrieval</td></tr><tr><td>SigLIP2</td><td colspan="2">47.84 27.93</td><td colspan="2">46.31</td><td>24.02</td><td>42.53</td></tr><tr><td>OpenCLIP L14</td><td colspan="2">46.99</td><td colspan="2">26.54</td><td>44.23 44.92</td><td></td></tr><tr><td>Jina Embeddings v4</td><td colspan="2">52.11</td><td colspan="2">46.26</td><td>24.80</td><td>43.96</td></tr><tr><td></td><td colspan="2">32.88 52.53</td><td colspan="2">52.87</td><td>46.65 26.80</td><td>45.57</td></tr><tr><td>ColQwen2.5 v0.2</td><td colspan="2">32.82</td><td colspan="2">53.79</td><td>46.54 26.10</td><td>46.14</td></tr><tr><td>Qwen-3 VL 8B Qwen-3 VL 32B</td><td colspan="2">53.08 56.39</td><td colspan="2">33.49 54.95 37.66 58.47</td><td>49.01 28.93 50.85 30.67</td><td>48.65 51.13</td></tr></table>

Table 6: Visual evidence retrieval for tables and figures.

## 5.3 Type-aware joint evidence retrieval (RQ3)

Table 5 reports the best models on the 25.77% of comments grounded to multiple evidence units, evaluated jointly under oracle type information. Performance drops sharply compared to the singleevidence oracle setting in Table 4: even the strongest model (Qwen-3 30B Instruct) retrieves only 12.69% of the full evidence set at R@1 and 23.53% at R@2, despite the favorable type-aware conditions. This indicates that when evidence is distributed across non-redundant units, models struggle to surface them together.<sup>20</sup>

## 5.4 Visual Evidence Retrieval (RQ4)

Table 6 shows that across all visual models, augmenting images with captions consistently improves image retrieval performance, indicating that captions provide a strong signal for grounding reviewer comments. Nonetheless, image-only retrieval remains competitive, hinting that visual information alone often encodes cues relevant to reviewer concerns (e.g., trends, comparisons, etc.). The remaining gap suggests that visual and textual signals are complementary, supporting RQ4 and motivating multimodal approaches for grounding.

## 6 Error Analysis

## 6.1 Oracle text retrieval failures

To understand how and why models are failing, we analyze oracle text retrieval failures: for each type, we take the best model (GPT-5.1 for paragraphs; Qwen-3 30B Instruct for sections/captions) and analyze 50 failed instances.

For paragraphs a dominant failure mode is due to line references being imprecise and pointing to section headers rather than the paragraph containing the relevant evidence. For sections, rebuttals often cite coarse sections while evidence is localized in a subsection. Moreover, line/section references frequently act as navigational pointers to another evidence unit (e.g. a table), so the text alone does not contain sufficient information to address the reviewer comment.

Caption failures are mainly due to some captions being too generic (with key details in the figure/table) or being similar across multiple figures/tables, causing plausible-but-wrong matches.

Across all types, under-specified reviewer comments that require additional local context (e.g., ”How did you introduce the error in Line 412?”) remain difficult to ground, as successful retrieval requires first identifying the referenced context before locating the explanatory evidence.

## 6.2 Text vs. image retrieval disagreement

Caption-based text retrieval and image-pluscaption-based retrieval often succeed on different instances. Looking at Recall@2, for figures, image retrieval is slightly higher than retrieving caption text only (58.47% vs. 55.16%; McNemar p = 0.25) with substantial disagreement between both (both fail 26.0%; disagree 24.1%). For tables, image retrieval is higher and statistically significant (60.34% vs. 53.29%; p = 0.0156) with high disagreement (both fail 24.8%; disagree 29.7%). Taken together, these findings show that reviewer comment grounding exhibits substantial modality-specific failure modes. Even when one modality is clearly stronger, neither text nor image retrieval alone is sufficient. A practical consequence is that systems grounding reviewer comments should consider late fusion strategies that combine caption-based and imagebased scores, rather than committing to a single modality at retrieval time.

## 7 Conclusion

We introduce ReGround, a large-scale dataset that grounds reviewer comments in fine-grained, multimodal evidence from the original anonymous submission. We leverage author rebuttals as an annotation source: when authors point reviewers to a specific section, table, or figure, they indirectly annotate the link between a comment and the evidence that addresses it. This allowed us to scale to 3,656 papers and 16,274 reviewer comment–evidence pairs units while keeping the precision of expert human judgments. One limitation is that the target of retrieval is author-cited evidence, so there might be plausible evidence not covered by that.

Our grounding task presents a hard problem: unified retrieval peaks at 21% Recall@10, type inference emerges as the dominant bottleneck, multievidence comments remain difficult even under oracle conditions, and text and image signals succeed on different instances making them complementary. These findings frame reviewer comment grounding as a stress test for scientific document understanding, demanding type-aware routing, multi-evidence aggregation, and multimodal fusion, capabilities that any system grounding claims in scientific documents will need.

## Ethical considerations

This work uses peer-review data from NLPeer, consisting of original anonymous submissions, reviews, and rebuttals that were voluntarily donated by authors and reviewers, and publicly released under CC-BY-NC 4.0. All data were collected and processed in accordance with those terms, and will be released under the same CC-BY-NC 4.0 license. Our dataset contains no personally sensitive information beyond what is publicly shared by the authors and reviewers. All annotators for the manual evaluation are volunteers. Our dataset is intended strictly for research on grounding reviewer comments in paper content, not for evaluating, scoring, or profiling individual reviewers or authors. While peer-review text may include subjective or critical language, our work does not aim to automate reviewer decisions or replace human judgment, but to support research on scientific document understanding and evidence retrieval.

## Limitations

Our dataset focuses on grounding reviewer comments that are implicitly related to paper content through author rebuttals, and deliberately excludes comments that explicitly reference paper content– an easier but complementary setting. While this design choice isolates a more challenging and realistic grounding scenario, it does not capture the full spectrum of reviewer behaviors.

Our evaluation does not consider task-specific training or end-to-end systems that jointly perform evidence type selection and retrieval. We leave this for future work, and focus on analyzing the challenges in this work. In addition, dataset construction relies on paper references in author rebuttals, which reflect author interpretations of reviewer comments and emphasize content authors chose to address, potentially underrepresenting unresolved or weakly grounded comments. Moreover, author-provided evidence links can be coarse (e.g., section-level when a more specific subsection is appropriate), introducing unavoidable noise in the supervision signal, so retrieval performance should be interpreted with this limitation in mind.

Although we include multimodal evidence, text and image retrieval are evaluated using separate candidate pools; fully unified multimodal retrieval remains a non-trivial direction for future work. Finally, our dataset is limited to NLP papers due to inaccessibility of peer-review data from other domains. Other domains like biomedicine, geology or chemistry could have different evidence types and granularities not covered by this dataset. Investigating cross-domain reviewing dynamics is an important future direction, and we support this by releasing our dataset construction code publicly.

## Acknowledgments

This work has been co-funded by the German Federal Ministry of Research, Technology and Space (BMFTR) under the promotional reference 01ZZ2314H (GeMTeX), and by the European Union (ERC, InterText, 101054961). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them. We thank Qian Ruan, Nils Dycke, and Dennis Zyska for their feedback on an initial draft of this paper.

## References

Serwar Basch, Ilia Kuznetsov, Tom Hope, and Iryna Gurevych. 2026. ABCD-LINK: Annotation boot-

strapping for cross-document fine-grained links. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3399–3423, Rabat, Morocco. Association for Computational Linguistics.

Tim Baumgärtner, Ted Briscoe, and Iryna Gurevych. 2025. PeerQA: A scientific question answering dataset from peer reviews. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 508–544, Albuquerque, New Mexico. Association for Computational Linguistics.

Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. M3- embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through selfknowledge distillation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2318–2335, Bangkok, Thailand. Association for Computational Linguistics.

Liying Cheng, Lidong Bing, Qian Yu, Wei Lu, and Luo Si. 2020. APE: Argument pair extraction from peer review and rebuttal via multi-task learning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7000–7011, Online. Association for Computational Linguistics.

Christopher Clark and Santosh Divvala. 2016. Pdffigures 2.0: Mining figures from research papers. In Proceedings ofthe 16th ACM/IEEE-CS on Joint Conference on Digital Libraries, JCDL ’16, page 143–152, New York, NY, USA. Association for Computing Machinery.

Mike D’Arcy, Alexis Ross, Erin Bransom, Bailey Kuehl, Jonathan Bragg, Tom Hope, and Doug Downey. 2024. ARIES: A corpus of scientific paper edits made in response to peer reviews. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6985– 7001, Bangkok, Thailand. Association for Computational Linguistics.

Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. 2021. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of the 2021 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4599–4610, Online. Association for Computational Linguistics.

Kuicai Dong, Yujing Chang, Derrick Goh Xin Deik, Dexun Li, Ruiming Tang, and Yong Liu. 2025. MM-DocIR: Benchmarking multimodal retrieval for long documents. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 30959–30993, Suzhou, China. Association for Computational Linguistics.

Nils Dycke, Ilia Kuznetsov, and Iryna Gurevych. 2023. NLPeer: A unified resource for the computational study of peer review. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5049– 5073, Toronto, Canada. Association for Computational Linguistics.

Michael Günther, Saba Sturua, Mohammad Kalim Akram, Isabelle Mohr, Andrei Ungureanu, Bo Wang, Sedigheh Eslami, Scott Martens, Maximilian Werk, Nan Wang, and Han Xiao. 2025. jina-embeddingsv4: Universal embeddings for multimodal multilingual retrieval. In Proceedings of the 5th Workshop on Multilingual Representation Learning (MRL 2025), pages 531–550, Suzhuo, China. Association for Computational Linguistics.

Dongyeop Kang, Waleed Ammar, Bhavana Dalvi, Madeleine van Zuylen, Sebastian Kohlmeier, Eduard Hovy, and Roy Schwartz. 2018. A dataset of peer reviews (PeerRead): Collection, insights and NLP applications. In Proceedings ofthe 2018 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1647–1661, New Orleans, Louisiana. Association for Computational Linguistics.

Neha Nayak Kennard, Tim O’Gorman, Rajarshi Das, Akshay Sharma, Chhandak Bagchi, Matthew Clinton, Pranay Kumar Yelugam, Hamed Zamani, and Andrew McCallum. 2022. DISAPERE: A dataset for discourse structure in peer review discussions. In Proceedings ofthe 2022 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1234–1249, Seattle, United States. Association for Computational Linguistics.

Ilia Kuznetsov, Jan Buchmann, Max Eichler, and Iryna Gurevych. 2022. Revise and resubmit: An intertextual model of text-based collaboration in peer review. Computational Linguistics, 48(4):949–986.

Carlos Lassance, Hervé Déjean, Thibault Formal, and Stéphane Clinchant. 2024. Splade-v3: New baselines for splade. Preprint, arXiv:2403.06789.

Lei Li, Yang Xie, Wei Liu, Yinan Liu, Yafei Jiang, Siya Qi, and Xingyuan Li. 2020. CIST@CL-SciSumm 2020, LongSumm 2020: Automatic scientific document summarization. In Proceedings of the First Workshop on Scholarly Document Processing, pages 225–234, Online. Association for Computational Linguistics.

Patrice Lopez. 2009. GROBID: combining automatic bibliographic data recognition and term extraction for scholarship publications. In Research and Advanced Technologyfor Digital Libraries, 13th European Conference, ECDL 2009, Corfu, Greece, September 27 - October 2, 2009. Proceedings, volume 5714 of Lecture Notes in Computer Science, pages 473–474. Springer.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card. CoRR, abs/2508.10925.

Jiefu Ou, William Gantt Walden, Kate Sanders, Zhengping Jiang, Kaiser Sun, Jeffrey Cheng, William Jurayj, Miriam Wanner, Shaobo Liang, Candice Morgan, Seunghoon Han, Weiqi Wang, Chandler May, Hannah Recknor, Daniel Khashabi, and Benjamin Van Durme. 2025. CLAIMCHECK: How grounded are LLM critiques of scientific papers? In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 21712–21735, Suzhou, China. Association for Computational Linguistics.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389.

Qian Ruan, Ilia Kuznetsov, and Iryna Gurevych. 2024. Re3: A holistic framework and dataset for modeling collaborative document revision. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4635–4655, Bangkok, Thailand. Association for Computational Linguistics.

Lukas Selch, Yufang Hou, Muhammad Jehanzeb Mirza, Sivan Doveh, James R. Glass, Rogerio Feris, and Wei Lin. 2026. PRISMM-bench: A benchmark of peer-review grounded multimodal inconsistencies. In The Fourteenth International Conference on Learning Representations.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, and 197 others. 2025. Gemma 3 technical report. Preprint, arXiv:2503.19786.

Nitya Thakkar, Mert Yuksekgonul, Jake Silberg, Animesh Garg, Nanyun Peng, Fei Sha, Rose Yu, Carl Vondrick, and James Zou. 2025. Can llm feedback enhance review quality? a randomized study of 20k reviews at iclr 2025. Preprint, arXiv:2504.09737.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. 2025.

Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. Preprint, arXiv:2502.14786.

Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, Daniel Cer, Alice Lisak, Min Choi, Lucas Gonzalez, Omar Sanseviero, Glenn Cameron, Ian Ballantyne, Kat Black, Kaifeng Chen, and 70 others. 2025. Embeddinggemma: Powerful and lightweight text representations. Preprint, arXiv:2509.20354.

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or fiction: Verifying scientific claims. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7534–7550, Online. Association for Computational Linguistics.

David Wadden, Kyle Lo, Bailey Kuehl, Arman Cohan, Iz Beltagy, Lucy Lu Wang, and Hannaneh Hajishirzi. 2022. SciFact-open: Towards open-domain scientific claim verification. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 4719–4734, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Daoze Zhang, Zhijian Bao, Sihang Du, Zhiyi Zhao, Kuangling Zhang, Dezheng Bao, and Yang Yang. 2026. Re2: A consistency-ensured dataset for fullstage peer review and multi-turn rebuttal discussions. Preprint, arXiv:2505.07920.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Qwen3 embedding: Advancing text embedding and reranking through foundation models. Preprint, arXiv:2506.05176.

## A Sentence Segmentation

We segment reviews and rebuttals into sentencelevel spans using a custom rule-based splitter. Offthe-shelf sentence segmenters frequently fail on peer-review text due to scientific abbreviations (e.g., ‘Fig.’, ‘Sec.’), explicit manuscript references, and list-style formatting. Our splitter is designed to be deterministic and to preserve exact character offsets into the original text.

To prevent erroneous splits, we identify protected spans in which punctuation should be ignored. These include explicit manuscript references detected using the patterns described in Appendix B, as well as common scientific abbreviations (e.g., ‘e.g.’, ‘i.e.’, ‘et al.’). Sentence-ending punctuation inside protected spans is excluded from boundary detection.

We additionally introduce forced sentence boundaries at structural markers commonly used in reviews and rebuttals, including blank-line paragraph breaks, list items, block quotes, and headings. For headings followed by content on the same line (e.g., ‘Strengths:’), we split after the colon.

Outside protected spans, we split on sentenceending punctuation characters (., !, ?) when followed by whitespace or the end of the text. We avoid splitting on numbered list markers (e.g., ‘1. at line start) and include trailing closing characters such as quotes or parentheses with the preceding sentence.

Final sentence boundaries are obtained by merging punctuation-based split points with all forced boundaries. Each sentence is returned as a span with start and end character offsets into the original document, with leading and trailing whitespace trimmed. The complete implementation is released as part of our codebase.

## B Detecting Explicit Rebuttal References

This appendix describes how we detect explicit references to manuscript content in author rebuttals. These references form the supervision signal used to ground reviewer comments to fine-grained evidence in the original anonymous submission. Our goal in this step is to maximize precision while covering the broad range of reference formats used in scientific writing.

## B.1 Reference Types

We detect explicit references to the following categories of manuscript content: (i) line numbers and line ranges, (ii) sections and subsections, (iii) figures, (iv) tables, (v) appendices, (vi) equations, (vii) pages, and (viii) footnotes. Table 7 summarizes the reference types along with representative examples observed in rebuttals.

## B.2 Regex-Based Reference Detection

We identify reference-bearing rebuttal sentences using a set of regular expressions. These patterns were developed by manually inspecting 250 randomly sampled rebuttals and iteratively expanding coverage to capture common scientific reference conventions.

<table><tr><td>Reference Type</td><td>Example Mentions</td></tr><tr><td>Lines</td><td>lines 120–135,1. 45–47, L120–L130</td></tr><tr><td>Sections</td><td>Section 3.2, Sec. 4, §5, Results section</td></tr><tr><td>Figures</td><td>Figure 2, Fig. 3(a), Figs. 4–6</td></tr><tr><td>Tables</td><td>Table 1, Tab. 5</td></tr><tr><td>Appendices</td><td>Appendix A, Appendix C.2</td></tr><tr><td>Equations</td><td>Eq. (3), Equation 7</td></tr><tr><td>Pages</td><td>page 5, pp. 3–4</td></tr><tr><td>Footnotes</td><td>footnote 2, fn. 7</td></tr></table>

Table 7: Types of explicit manuscript references detected in rebuttal sentences.

Patterns are grouped by reference type and account for common abbreviations (e.g., Fig. vs. Figure), optional punctuation, ranges (e.g., 2–4), and series (e.g., 2, 3, and 5). Matching is caseinsensitive. A single rebuttal sentence may contain multiple references, in which case all references are extracted independently.

For readability, we describe the pattern families here and release the full implementation with our codebase.

Line references. We detect references to individual lines or line ranges, including formats such as lines 120–135, l. 45–47, and L120–L130. These patterns allow optional range markers (e.g., –, to) and optional parentheses.

Section references. We detect both numeric section references (e.g., Section 3.2, §4) and references to named section headers. Named section references include commonly used scientific sections such as Introduction, Related Work, Methodology, Experiments, Results, Analysis, Discussion, Limitations, and Conclusion. These are matched using a curated list of canonical section names derived from ACL-style papers and empirical inspection of submissions. Matching is case-insensitive and allows optional determiners (e.g., the methodology section).

Figures and tables. We detect references to figures and tables using standard prefixes (e.g., Fig., Figure, Tab., Table), supporting single indices, ranges, and series (e.g., Figs. 2–4, Tables 1 and 3).

Appendices, equations, pages, and footnotes. We additionally detect references to appendices (e.g., Appendix A.2), equations (e.g., Eq. (5)), page ranges (e.g., pp. 3–4), and footnotes (e.g., footnote 7) using dedicated patterns for each category.

## B.3 Ambiguities and Resolution Strategy

When a rebuttal sentence contains multiple explicit references, we extract all references and generate multiple candidate links. During the subsequent content extraction step, each reference is resolved against the original anonymous submission PDF.

If a reference cannot be resolved (e.g., due to inconsistent line numbering, missing figures, or malformed indices), the corresponding instance is discarded. We do not attempt to infer or repair unresolved references in order to preserve the precision of the supervision signal.

References that primarily point to future revisions (e.g., “we will update Figure 3 in the cameraready version”) or to external papers are filtered out in a later stage, as described in §D.

All regular expressions and reference resolution utilities are implemented in Python and released as part of our public codebase.

## C LLM-Based Rebuttal–Review Alignment

We align rebuttal sentences that contain explicit manuscript references to the reviewer comment they address.

Goal. Given a segmented reviewer report, a segmented rebuttal, and a target rebuttal sentence, the goal is to select the reviewer span(s) that the rebuttal sentence responds to. A rebuttal sentence may align to multiple reviewer spans.

Method. We perform alignment using gpt-oss-120b due to its long context window, and because it is one of the SOTA open-source models. The model is provided with (i) the full reviewer report with indexed spans, (ii) the full rebuttal text, and (iii) the target rebuttal sentence. It is instructed to return the identifiers of the reviewer span(s) being addressed, or an empty set if no clear alignment exists.

The full prompt template and output specification are shown in Figure 4. The model outputs span identifiers only for easier parsing.

Multiple and ambiguous cases. If a rebuttal sentence addresses multiple reviewer comments, all alignments are retained. Sentences that are generic or cannot be confidently aligned are discarded.

## D Filtering

To construct a dataset for implicit reviewer comment grounding, we apply a set of precisionoriented filters to remove cases where the target

evidence is explicitly identifiable or does not correspond to the reviewed submission. All filtering steps are applied after rebuttal–review alignment.

## D.1 Removing Quoted Review Text

Authors frequently quote reviewer comments verbatim in rebuttals, which can introduce spurious references originating from the review rather than the author response. We remove rebuttal sentences that closely match reviewer text using fuzzy string matching.

## D.2 Removing Trivial Explicit-Reference Cases

We remove instances where the reviewer comment already explicitly names the same evidence as the rebuttal (e.g., both reference the same figure, table, or section). This prevents trivial grounding via reference matching and ensures that the reviewer comment does not identify the target evidence.

## D.3 Removing References Outside the Reviewed Submission

We remove rebuttal sentences that do not correspond to content available in the original anonymous submission. Specifically, we exclude cases where the rebuttal points to (i) promised cameraready changes, (ii) new experiments or results introduced in the rebuttal, or (iii) content from external papers. This final step ensures that all retained pairs ground reviewer comments to content available in the original submission. This filter is implemented using a binary LLM; prompt template is in Fig. 5.

## E Manual Validation Protocol

We manually validated two stages of the dataset construction: (i) review–rebuttal alignment and (ii) filtering of invalid evidence.

Review–rebuttal alignment. Annotators were presented with the reviewer comment, the corresponding rebuttal text, and the automatically proposed alignment, with aligned spans highlighted. For each instance, annotators made a binary accept or reject decision. In cases of rejection, annotators provided a short keyword-based explanation (e.g., wrong reviewer comment, partial coverage, etc.). Annotators were not asked to provide corrected alignments.

Filtering validation. Filtering was validated using spreadsheet-based annotation. Each row corresponded to a single rebuttal sentence, and each column represented one filtering criterion (references to new experiments, future work, or external content). Annotators marked each criterion with a binary yes/no decision. A rebuttal sentence was considered invalid evidence if it violated any filtering criterion.

## F Manual Inspection

Our manual inspection had two main goals.

First, we checked the most objective—and most failure-prone—component of the pipeline: reference resolution, i.e., whether the author-cited evidence correctly maps to the intended location in the paper. To assess this, one author manually inspected 200 reviewer comments containing a total of 316 references to paper evidence (restricted to text-only evidence). Our analysis shows that the vast majority of references are correctly resolved. Approximately 1.58% (5/316) of the references may be considered coarse-grained, meaning that the cited span could potentially be narrowed to a more precise evidence boundary. However, determining appropriate evidence boundaries is inherently subjective, and in most cases it is difficult to definitively conclude that the boundaries selected by the authors are too broad.

Second, we checked the validity of the referenced evidence. We note that manually verifying whether a cited piece of evidence logically addresses a reviewer comment is inherently subjective and requires deep understanding of paper-specific context. In practice, the authors of the paper are the most qualified annotators for this task. By relying on their rebuttals as a supervision signal, we effectively leverage the authors themselves as expert annotators of whether a given piece of evidence addresses a reviewer concern.

Nonetheless, for a subset of 100 reviewer comments, we checked each referenced evidence manually to determine their relevancy to the reviewer comment. We find that all referenced evidence is relevant to the reviewer comment. More importantly, in the majority of cases (84%) the reviewer themselves acknowledged the rebuttal, which we take as a signal that the provided evidence was at least acceptable by the reviewer. Finally, our setup does not guarantee coverage: there might be other relevant evidence not referenced by the authors. However, we argue that this is unlikely, as authors would usually try to cover as much evidence as needed to strengthen their rebuttal

![](images/87cbf51e5c27b068f7f80771b966a2de7cd7066a79b966a5cd9e87810e1d33ee.jpg)  
Figure 3: Distribution of cosine similarity between a reviewer comment and its target content unit(s).

## G Cosine Similarity Analysis

We compute a simple embedding-based similarity measure between each reviewer comment and its grounded evidence unit(s) to characterize their lexical/semantic proximity.

Embedding model and similarity. For each comment–evidence pair, we encode the reviewer comment and the evidence text using the Sentence-Transformers model all-mpnet-base-v2. We then compute cosine similarity between the two embeddings. For figure/table evidence, we use the associated caption text as the evidence text. Similarities are computed independently per grounded pair, and pairs with multiple evidence units contribute one similarity value per unit.

Distributions by evidence type. Figure 3 shows histograms of cosine similarity values stratified by reference type (e.g., lines, sections, figures, tables). Across the dataset, the mean cosine similarity over all reviewer comment–evidence pairs is 0.377. We observe substantial overlap across types, with no evidence category exhibiting consistently high similarity scores.

We report these statistics solely as a descriptive property of the dataset and do not use them as supervision or a filtering criterion.

## H Multi-Evidence Type Combinations

Comments grounded to multiple evidence units are particularly challenging: retrieval must identify and aggregate all relevant units rather than retrieve a single passage. To characterize this setting, Table 8 breaks down the most frequent and substantively informative evidence-type combinations at the query level. Percentages are calculated within the multievidence subset. The displayed combinations account for 55% of all multi-evidence combinations.

<table><tr><td>Combination</td><td>% of multi-evidence comments</td></tr><tr><td>Paragraph + section</td><td>14.06 8.84</td></tr><tr><td>Multiple paragraphs Multiple sections</td><td>7.79</td></tr><tr><td>Section + table</td><td>6.31</td></tr><tr><td>Figure + section</td><td>5.86</td></tr><tr><td>Figure + table</td><td>5.14</td></tr><tr><td>Multiple tables</td><td>4.54</td></tr><tr><td>Paragraph + table</td><td>3.17</td></tr></table>

Table 8: Most frequent evidence-type combinations for comments grounded to multiple evidence units. Percentages are within the multi-evidence subset.

The challenge is therefore not confined to a single aggregation setting. The dataset includes sametype aggregation (e.g., multiple paragraphs, sections, or tables), cross-granularity aggregation (e.g., paragraph + section), and multimodal combinations involving figures or tables. This diversity makes multi-evidence retrieval a test of both finding individual units and combining evidence across content types.

## I Relation Directionality

ReGround formalizes retrieval of the paper content to which a reviewer comment and an author response refer. The direction of the relation is a separate question from retrieval: relevant evidence can be retrieved before determining whether it supports, qualifies, or contradicts the reviewer comment. Thus, our use of evidence does not imply a verdict or relation type.

As a descriptive check for data analysis, one annotator manually labeled a sample of 200 reviewercomment–rebuttal-sentence pairs using a narrower three-way scheme. A relation was labeled corrective when the authors used cited paper content to contradict a reviewer claim or show that allegedly missing content was already present. It was labeled explanatory when the cited content answered, clarified, justified, or otherwise addressed a concern without establishing that the reviewer was mistaken; the remaining cases were labeled other/unclear. In this sample, 126 pairs (63.0%) were explanatory, 64 (32.0%) corrective, and 10 (5.0%) other/unclear.

These single-annotator descriptive results should be interpreted cautiously, but they suggest that corrective uses of evidence are not the dominant pattern in the sample. More commonly, authors point to paper content to explain or clarify a concern. We therefore retain the retrieval framing and leave relation typing as a distinct downstream classification layer: evidence must be retrieved before its relation type to the reviewer comment can be assessed.

<table><tr><td>Primary Dimension</td><td>Percentage</td></tr><tr><td>Empirical Rigor</td><td>30.8%</td></tr><tr><td>Technical Soundness</td><td>15.7%</td></tr><tr><td>Deployment &amp; Impact</td><td>15.6%</td></tr><tr><td>Presentation &amp; Writing</td><td>12.8%</td></tr><tr><td>Data &amp; Reproducibility</td><td>11.5%</td></tr><tr><td>Conceptual &amp; Novelty</td><td>11.1%</td></tr><tr><td>Other</td><td>2.5%</td></tr></table>

Table 9: Distribution of primary reviewer comment dimensions across the annotated examples, based on taxonomy classification.

## J Reviewer Comment Taxonomy

Annotation setup. Each comment is assigned a single primary category corresponding to the dominant aspect of the paper discussed. Categories are defined to be mutually exclusive and collectively exhaustive. The resulting taxonomy is intended to be descriptive rather than normative. The annotation is done by one of the authors. Table 9 reports the resulting distribution over primary categories.

Taxonomy categories. The taxonomy consists of the following categories: (i) Conceptual & Novelty: originality, significance, and relation to prior work; (ii) Technical Soundness: correctness of theoretical claims and technical reasoning; (iii) Empirical Rigor: quality of experiments, evaluations, and analyses; (iv) Data & Reproducibility: data quality, annotation procedures, and reproducibility concerns; (v) Presentation & Writing: clarity, organization, and visual presentation; (vi) Deployment & Impact: efficiency, ethics, and real-world applicability; (vii) Other: administrative, metalevel, or vague comments not fitting elsewhere.

## K Coverage and Representativeness Analysis

Our rebuttal-based construction necessarily selects reviewer comments that authors address by referring to content in the paper. To quantify this selection, we measure coverage over complete review threads using the same sentence-level segmentation as the construction pipeline. Because rebuttals primarily address criticisms and suggestions, rather than paper summaries or strengths, we compute coverage only over the summary\_of\_weaknesses and comments\_suggestions\_and\_typos fields.

<table><tr><td>Dimension</td><td>Included</td><td>Excluded</td><td>Difference</td></tr><tr><td>Empirical Rigor</td><td>30.8%</td><td>29.5%</td><td>+1.3</td></tr><tr><td>Technical Soundness</td><td>15.7%</td><td>9.0%</td><td>+6.7</td></tr><tr><td>Deployment &amp; Impact</td><td>15.6%</td><td>3.0%</td><td>+12.6</td></tr><tr><td>Presentation &amp; Writing</td><td>12.8%</td><td>14.0%</td><td>-1.2</td></tr><tr><td>Data &amp; Reproducibility</td><td>11.5%</td><td>10.0%</td><td>+1.5</td></tr><tr><td>Conceptual &amp; Novelty</td><td>11.1%</td><td>16.5%</td><td>-5.4</td></tr><tr><td>Other</td><td>2.5%</td><td>18.0%</td><td>-15.5</td></tr></table>

Table 10: Comment-type distribution in ReGround and 200 excluded comments.

In this subset, ReGround covers 17,682 of 115,989 review spans (15.24%).

We also assess whether the included comments differ systematically from those excluded by this procedure. We randomly sampled 200 excluded spans and classified them using the same sevencategory taxonomy and definitions used elsewhere in our analysis. Table 10 compares their distribution with that of included comments.

The largest difference is in Other comments, which comprise administrative, meta-level, or vague remarks. Such comments generally do not identify paper content to retrieve; their exclusion is therefore a property of the grounding task rather than evidence of a content-related sampling bias. The same partly applies to Conceptual & Novelty comments, which often concern positioning relative to the field rather than a localized part of the paper. Conversely, ReGround over-represents concerns that authors can answer by pointing to paper content, particularly Technical Soundness and Deployment & Impact. These are precisely the comments for which a grounding system is intended to be used. Importantly, Empirical Rigor, the largest category, has nearly identical prevalence among included and excluded comments (30.8% versus 29.5%), indicating no material skew in the dominant reviewer concern.

## L LLM-based Pointwise Ranking

Setup. We use LLMs as pointwise relevance judges. Given a reviewer comment c and a candidate evidence unit d (paragraph/section/caption), the model predicts whether d helps address c. Each candidate is scored independently; candidates are then ranked by this score.

Scoring from token log-probabilities. We request token-level log-probabilities (logprobs) and a small set of top\_logprobs for each generated token. Let $\ell _ { A }$ and $\ell _ { B }$ denote the log-probabilities of emitting A and B at the final decision position.

We compute a calibrated relevance score as the normalized probability of A:

$$
p ( \mathsf { A } \mid c , d ) = \frac { \exp ( \ell _ { A } ) } { \exp ( \ell _ { A } ) + \exp ( \ell _ { B } ) } .\tag{1}
$$

This score is used for ranking (higher is more relevant). If only one of $\ell _ { A } , \ell _ { B }$ is present in top\_logprobs, we fall back to the emitted token’s own log-probability when it matches A or B.

Decision token identification. Because tokenizers may emit variants such as $\ " { \mathrm { ~  ~ { ~ \sf ~ A ~ } ~ } } ^ { \prime \prime } { \mathrm { ~  ~ { ~ \cal ~ V } ~ } } { \mathrm {  ~ { ~ \cal ~ S } ~ } } . \mathrm { ~  ~ { ~ \cal ~ \prime ~ } ~ } \mathsf { A } ^ { \prime \prime }$ , we normalize tokens by stripping whitespace. We select the decision index by (i) taking the last non-whitespace character in the generated text and matching it to A/B, then (ii) falling back to the last generated token whose normalized form is in {A,B}. If no decision token is found (formatting violation), we retry the call up to a fixed number of attempts and otherwise mark the instance as failed.

Inference settings and model-specific controls. All calls use the same system message and user prompt template. We set decoding parameters per model family to reduce formatting failures and support explicit “thinking” modes when available. Concretely, we use temperature 0 for non-thinking settings with a small max\_tokens budget (enough to emit only the label), and enable model-provided thinking modes where supported (e.g., via template flags) with a higher temperature. We request top\_logprobs to increase the chance that both labels appear in the returned distribution.

Failure handling. We treat three cases as failures: missing token log-probabilities, inability to locate the decision token, or neither label appearing in the returned top\_logprobs. We retry formatting-related failures up to a fixed maximum.

## M Effect of Evidence Length

To analyze the effect of section length on retrieval performance, we evaluate the Mean Reciprocal

<table><tr><td>Model</td><td>Q1</td><td>Q2</td><td>Q3</td><td>Q4</td><td>Q5</td><td>Q6</td><td>Q7</td><td>Q8</td><td>Q9</td><td>Q10</td></tr><tr><td>BGE-M3-Reranker</td><td>21.0</td><td>29.9</td><td>34.2</td><td>41.5</td><td>44.2</td><td>39.2</td><td>38.8</td><td>31.6</td><td>12.7</td><td>0.2</td></tr><tr><td>Qwen3-30B-Instruct</td><td>18.2</td><td>32.3</td><td>33.8</td><td>39.0</td><td>41.8</td><td>45.0</td><td>44.9</td><td>46.1</td><td>30.4</td><td>7.2</td></tr></table>

Table 11: MRR across section length quantiles for the best cross-encoder (BGE-M3-Reranker) and the best LLMbased retriever (Qwen3-30B-Instruct).

<table><tr><td>Model</td><td>MRR</td><td>R@1</td><td>R@2</td><td>R@10</td></tr><tr><td>Sparse encoders</td><td></td><td></td><td></td><td></td></tr><tr><td>BM25</td><td>29.91</td><td>6.88</td><td>13.31</td><td>46.78</td></tr><tr><td>SPLADEv3</td><td>37.17</td><td>10.28</td><td>18.70</td><td>54.51</td></tr><tr><td>Dense encoders</td><td></td><td></td><td></td><td></td></tr><tr><td>all-mpnet-base-v2</td><td>32.54</td><td>7.82</td><td>15.08</td><td>49.47</td></tr><tr><td>BGE-M3</td><td>34.85</td><td>9.18</td><td>16.62</td><td>51.75</td></tr><tr><td>Qwen-3 Embedding 4B</td><td>26.29</td><td>7.46</td><td>13.06</td><td>38.11</td></tr><tr><td>EmbeddingGemma</td><td>37.63</td><td>10.25</td><td>18.79</td><td>55.94</td></tr><tr><td>Cross encoders</td><td></td><td></td><td></td><td></td></tr><tr><td>BGE-M3-Reranker</td><td>36.91</td><td>9.93</td><td>17.99</td><td>54.80</td></tr><tr><td>MiniLM-L12-v2</td><td>33.88</td><td>8.98</td><td>16.08</td><td>50.30</td></tr><tr><td>LLMs</td><td></td><td></td><td></td><td></td></tr><tr><td>Gemma-3 12B</td><td>39.82</td><td>10.54</td><td>21.46</td><td>59.32</td></tr><tr><td>Gemma-3 27B</td><td>43.20</td><td>12.59</td><td>23.91</td><td>62.21</td></tr><tr><td>Qwen-3 30B Instruct</td><td>43.29</td><td>12.69</td><td>23.53</td><td>62.43</td></tr><tr><td>Qwen-3 30B Thinking</td><td>38.34</td><td>11.14</td><td>18.44</td><td>54.53</td></tr><tr><td>gpt-oss-20b</td><td>39.44</td><td>11.69</td><td>19.01</td><td>52.95</td></tr><tr><td>GPT-5.1</td><td>39.53</td><td>8.99</td><td>19.67</td><td>62.33</td></tr></table>

Table 12: Joint retrieval over all gold labels for a query, restricted to the gold evidence types.

Rank (MRR) across ten quantiles of section text length. Quantile Q1 corresponds to the shortest sections and Q10 to the longest.

The results in Table 11 show different behaviors for the two model types. The cross-encoder (BGEM3) improves as section length increases up to mid-length sections (Q4–Q5), after which performance declines sharply, with a substantial drop for the longest sections. This supports our hypothesis that cross-encoders struggle when ranking very long text segments.

In contrast, the LLM-based retriever (Qwen3- 30B-Instruct) continues to improve with increasing section length up to Q8, only decreasing in the final two quantiles. This behavior is expected, as longer sections provide more contextual information that the model can use to determine whether a section is relevant to a reviewer comment.

## N Joint Retrieval Remains Challenging for Multi-Evidence Comments

Table 12 reports joint retrieval results for reviewer comments grounded in multiple evidence units, assuming oracle evidence types. Even in this favorable setting, performance remains limited, showing that multi-evidence grounding poses challenges beyond type inference.

Across all models, recall at small cutoffs (R@1 and R@2) is substantially lower than in the singletarget oracle setting, showing that models struggle to surface multiple relevant evidence units among the top-ranked results equally. While larger LLMs achieve higher overall recall, the gains are modest, and no model reliably retrieves the full set of relevant evidence early in the ranking.

These results support RQ3 and suggest that reviewer comments often require aggregating complementary evidence that is distributed across the paper. Even when the correct evidence types are known, retrieving all relevant units within a single ranked list remains difficult, highlighting multievidence retrieval as a distinct challenge in reviewer comment grounding.

## O Predicted Evidence-Type Routing

The oracle type-aware setting in Section 5 assumes access to the gold evidence type. To test whether this information can be inferred automatically, we run a two-stage experiment: first predict the relevant evidence type(s) from the reviewer comment, then retrieve only from candidates of the predicted type(s). We evaluate a four-way setting (figure, table, paragraph, section) and a coarser three-way setting (figure, table, text), where text includes both paragraph- and section-level evidence. Table 13 reports the type-prediction quality, and Table 14 reports retrieval performance on the shared subset.

Table 13 shows that predictors identify the dominant text class reliably, but figure and table prediction remains substantially weaker. This explains the retrieval behavior in Table 14: the broad text class leaves many candidates in the pool, while mistakes on figure/table examples remove the correct evidence entirely. Thus, predicted evidence type is better interpreted as a soft reranking signal than as a hard retrieval constraint.

## P AI Assistants Usage

We used GitHub Copilot for some coding related tasks, as well as ChatGPT for light editing (phrasing, grammar proof-checking) to help writing the paper.

<table><tr><td>Setting</td><td>Predictor</td><td>Set Acc.</td><td>Micro-F1</td><td>Macro-F1</td><td>Avg. Gold</td><td>Avg. Pred.</td><td>Fig. F1</td><td>Tab. F1</td><td>Text F1</td><td>Par. F1</td><td>Sec. F1</td></tr><tr><td>3-way</td><td>EmbeddingGemma</td><td>30.28</td><td>60.20</td><td>49.36</td><td>1.07</td><td>1.62</td><td>33.82</td><td>37.15</td><td>77.12</td><td>一</td><td></td></tr><tr><td>3-way</td><td>Qwen-3 30B Instruct</td><td>55.98</td><td>61.43</td><td>34.31</td><td>1.07</td><td>1.04</td><td>12.82</td><td>12.72</td><td>77.41</td><td>一</td><td></td></tr><tr><td>3-way</td><td>GPT-5.1</td><td>50.12</td><td>61.51</td><td>42.33</td><td>1.07</td><td>1.19</td><td>20.58</td><td>29.48</td><td>76.95</td><td>一</td><td></td></tr><tr><td>4-way</td><td>EmbeddingGemma</td><td>3.25</td><td>46.36</td><td>45.33</td><td>1.14</td><td>2.46</td><td>37.50</td><td>40.53</td><td>一</td><td>51.62</td><td>51.69</td></tr><tr><td>4-way</td><td>Qwen-3 30B Instruct</td><td>29.15</td><td>39.39</td><td>27.29</td><td>1.13</td><td>1.11</td><td>17.99</td><td>16.27</td><td>一</td><td>18.84</td><td>56.07</td></tr><tr><td>4-way</td><td>GPT-5.1</td><td>19.14</td><td>42.88</td><td>38.58</td><td>1.14</td><td>1.54</td><td>24.26</td><td>38.37</td><td></td><td>39.04</td><td>52.64</td></tr></table>

Table 13: Standalone multi-label evidence-type prediction performance. Scores are percentages except for the average number of gold/predicted labels. In the 3-way setting, text merges paragraph- and section-level evidence.

<table><tr><td>Setting</td><td>Retriever</td><td>Base R@10</td><td>Filtered R@10</td><td>Base MRR</td><td>Filtered MRR</td></tr><tr><td>3-way</td><td>GPT-5.1</td><td>21.67</td><td> $1 5 . 9 2 _ { - 5 . 7 5 }$ </td><td>10.92</td><td> $7 . 8 4 _ { - 3 . 0 8 }$ </td></tr><tr><td>3-way</td><td>Qwen-3 30B Instruct</td><td>21.15</td><td> $1 4 . 5 7 _ { - 6 . 5 8 }$ </td><td>10.87</td><td> $6 . 6 8 _ { - 4 . 1 9 }$ </td></tr><tr><td>3-way</td><td>EmbeddingGemma</td><td>17.68</td><td> $1 6 . 4 1 _ { - 1 . 2 7 }$ </td><td>8.78</td><td> $9 . 0 9 _ { + 0 . 3 1 }$ </td></tr><tr><td>4-way</td><td>GPT-5.1</td><td>21.67</td><td> $1 5 . 0 2 _ { - 6 . 6 5 }$ </td><td>10.92</td><td> $7 . 1 8 _ { - 3 . 7 4 }$ </td></tr><tr><td>4-way</td><td>Qwen-3 30B Instruct</td><td>21.15</td><td> $6 . 1 1 _ { - 1 5 . 0 4 }$ </td><td>10.87</td><td> $3 . 0 5 _ { - 7 . 8 2 }$ </td></tr><tr><td>4-way</td><td>EmbeddingGemma</td><td>17.68</td><td> $1 8 . 1 6 _ { + 0 . 4 8 }$ </td><td>8.78</td><td> $9 . 7 1 _ { + 0 . 9 3 }$ </td></tr></table>

Table 14: Retrieval after hard filtering by predicted evidence type on the shared subset. Deltas in subscripts indicate the change from the base retriever. Filtering usually lowers R@10 and MRR because type prediction errors remove relevant evidence before ranking.

![](images/9a11c0aeda707633ed973abfcbe3a5c291a0942810d891ad5b5bfd003ac673dd.jpg)  
Figure 4: Prompt template for review–rebuttal alignment.

![](images/3b3338f42e6aaa77b284efdcd8761da3bac61617169a901686487e7b1d3e3433.jpg)  
Figure 5: Prompt template for LLM-based pairwise sentence classification.