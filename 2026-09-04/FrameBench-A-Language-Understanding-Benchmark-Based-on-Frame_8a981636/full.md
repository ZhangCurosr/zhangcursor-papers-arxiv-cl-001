# FrameBench: A Language Understanding Benchmark Based on Frame Semantics

Chihiro Yano

Ryohei Sasano

Graduate School of Informatics, Nagoya University yano.chihiro.j3@s.mail.nagoya-u.ac.jp sasano@i.nagoya-u.ac.jp

## Abstract

In frame semantics, sentence comprehension is assumed to proceed by relating lexical meaning to background knowledge called semantic frames, thereby enabling readers to implicitly enrich the text with unstated information. Recent large language models (LLMs) have achieved strong performance across a wide range of downstream tasks. However, it remains unclear whether they can reproduce the kinds of implicit enrichment that humans naturally make during comprehension. To address this question, we introduce FrameBench, a benchmark grounded in frame semantics. FrameBench consists of multiple-choice questions that test whether models distinguish the frames evoked by the same verb across contexts. We construct the benchmark for English and Japanese using FrameNet-style resources and a generationand-verification pipeline with native-speaker judgments. Our experiments on a diverse set of models reveal challenges for small models, while several large models surpass the human reference scores. We release the constructed FrameBench dataset and the code for dataset construction and evaluation at https: //github.com/SasanoLab/FrameBench

## 1 Introduction

The same verb can evoke different situations depending on context. Consider these examples:

1. He left the bank after talking with a friend.

2. He left the bank at the age of sixty.

Although both sentences contain the same verb, left, they evoke different situations. In Sentence 1, left describes a physical departure from the bank as a location. In Sentence 2, by contrast, it evokes a quitting interpretation rather than physical departure. Under this interpretation, readers infer an employment relation that is not explicitly stated.

![](images/abaa43eb28fee5cfa09d4e60354b9af0552280e1e62c4394e1c2f92cadb89318.jpg)  
Figure 1: An example from FrameBench.

This kind of context-dependent enrichment is central to frame semantics (Fillmore, 1982).

Despite the strong performance of recent large language models (LLMs) across a wide range of downstream tasks, it remains unclear whether they can reliably perform this kind of implicit, contextdependent enrichment. Most existing evaluations still rely on broad benchmark suites built from diverse downstream tasks, which do not directly test this capability (Wang et al., 2018, 2024). To address this gap, we introduce FrameBench, a multiple-choice benchmark that evaluates whether LLMs can distinguish context-dependent framesemantic interpretations evoked by the same verb. Rather than requiring explicit frame-label prediction, FrameBench probes such distinctions indirectly through natural-language questions about the situations implied by each sentence.

In this work, we construct FrameBench for English and Japanese, two typologically distant languages, using their respective FrameNet resources (Ruppenhofer et al., 2016; Ohara et al., 2004). FrameBench is built through LLM-based generation and native-speaker validation. Importantly, the generation process is grounded in humanauthored resources rather than solely in the implicit knowledge of LLMs. We also evaluate a diverse set of LLMs on the resulting benchmarks to analyze whether they can distinguish the background knowledge evoked by the same verb across contexts.

Figure 1 shows an example item from FrameBench. The verb left appears in both sentences, and the prompt asks the model to select all sentences that describe Noah quitting his job. The correct answer is “1: $\mathbf { S _ { A } } ^ { \prime \prime }$ because only $\mathrm { \bf S _ { A } }$ expresses quitting, whereas $\mathbf { S _ { B } }$ describes physical departure. The item cannot be solved by matching the predicate alone. It requires context-sensitive frame-semantic interpretation.

We make the following contributions:

• We introduce FrameBench, a FrameNetgrounded multiple-choice benchmark that evaluates whether LLMs can distinguish context-dependent frame-semantic interpretations evoked by the same verb.

• We construct English and Japanese versions of FrameBench through LLM-based generation and native-speaker validation. We release the English and Japanese FrameBench datasets, along with the construction and evaluation code.

• We benchmark a wide range of LLMs, showing that FrameBench performance is strongly affected by model scale, reasoning mode, and language. Through behavioral analyses and case studies, we identify error patterns and finegrained frame-semantic distinctions that can still challenge high-performing models.

## 2 Related Work

## 2.1 Evaluation of Language Models

Evaluating LLMs has attracted substantial attention in recent years. Because LLM capabilities are multifaceted, evaluations often rely on benchmarks that bundle diverse downstream tasks rather than a single task (Wang et al., 2018; Hendrycks et al., 2021; Wang et al., 2024; Srivastava et al., 2023).

Some benchmarks more directly test whether models can distinguish meanings based on context. For instance, Word Sense Disambiguation (WSD) tasks and the WiC dataset (Pilehvar and Camacho-Collados, 2019) ask whether the same word has the same meaning in two different contexts. Other approaches focus on sentence-level meaning representations, such as AMR (Knight et al., 2017), by evaluating how well models can parse sentences into graph-structured semantics. However, our goal is to test whether a model can discriminate context-dependent event interpretations evoked by the same verb. Unlike standard WSD, which focuses on dictionary senses, our evaluation is grounded in frame semantics, capturing broader conceptual situations, which differs from the objectives of these existing resources.

## 2.2 Frame-Semantic Resources and Datasets

FrameNet is a lexical knowledge base grounded in frame semantics (Fillmore, 1982), built through manual annotation of corpus sentences with evoked frames and participant roles (Baker et al., 1998). It has been widely used for frame-semantic parsing and related tasks such as semantic role labeling, and provides structured semantic information, including relations between frames.

Beyond English, FrameNet-style resources have been developed for many languages (Ohara et al., 2004; Hahm et al., 2020; You and Liu, 2005; Djemaa et al., 2016; Lyngfelt et al., 2018). More recently, datasets have been proposed that annotate multimodal data with frames and semantic roles (Belcavello et al., 2024; Viridiano et al., 2024).

## 2.3 Frame Semantics and LLMs

A growing body of work explores the use of LLMs for frame-semantic analysis (Chundru et al., 2025; Devasier et al., 2025; Garat et al., 2025; Yano et al., 2025; Rai et al., 2025). Other studies attempt to extend or assist FrameNet annotation with LLMs (Belcavello et al., 2026; Han et al., 2024). Overall, these results indicate that LLMs may be able to leverage frame-related information from context to some extent.

In addition, frameworks have been proposed to directly evaluate how well LLMs acquire conceptual structures grounded in frame semantics. Guo et al. (2024) propose NutFrame, which evaluates whether LLMs can induce explicit framesemantic structures from FrameNet, and report that such induction remains challenging. Moreover, evaluating generative or parsing tasks often suffers from formatting inconsistencies in LLM outputs. In contrast, FrameBench evaluates the discrimination of context-dependent event interpretations evoked by the same verb as a multiplechoice task. By avoiding tasks that require explicit prediction of frame labels or role inventories, FrameBench more directly evaluates whether models distinguish context-dependent event interpretations evoked by the same verb.

![](images/01d0fc3baa9a45eac5a1a43a5b800038ecdfafd5fa0168b38d432f5f1539db7b.jpg)

Figure 2: Overview of the benchmark construction. The numbered steps correspond to those described in Section 3.2.  
![](images/ca29fc1ba45a1fa925a85197da043d1f17c777cba1ff8fc509dc1ec13346ef85.jpg)  
(b) Evaluation Format  
Figure 3: Overview of the FrameBench task.

## 3 FrameBench: Task Design and Dataset Construction

FrameBench is a four-choice benchmark for evaluating whether a model can correctly distinguish between different semantic frames evoked by the same verb. This section first defines the FrameBench task and its evaluation protocol, then describes the construction pipeline and its application to English and Japanese.

## 3.1 Task Definition

As shown in Figure 3a, each FrameBench entry consists of a question Q and four candidate sentences $( S _ { \mathrm { { A } } } , \ S _ { \mathrm { { B } } } , \ S _ { \mathrm { { A } ^ { \prime } } } , \ S _ { \mathrm { { B } ^ { \prime } } } )$ , constructed from a polysemous verb and a pair of semantic frames (Frame $\mathbf { A } \mathbf { \cdot }$ , Frame<sub>B</sub>). The question $Q$ targets Frame<sub>A</sub>: $S _ { \mathrm { { A } } }$ and $S _ { \mathrm { { A } ^ { \prime } } }$ are designed to evoke Frame , while $S _ { \mathrm { B } }$ and $S _ { \mathrm { B } } ,$ are designed to evoke the contrasting Frame<sub>B</sub>. During dataset construction, we use human-authored frame-semantic re-<sup>Step</sup> sources that specify each frame’s name, definition, core frame elements, and example sentences. Thus, the semantic distinctions in FrameBench are grounded in external frame-semantic resources rather than in an LLM’s internal knowledge alone. Since these resources are not provided at evaluation time, solving the task using only the model’s internal knowledge is non-trivial. Each entry also includes human evaluation scores for acceptability and correctness.

For evaluation, the four sentences are split into two predefined pairs, each containing exactly one sentence that evokes the target frame: the base pair $( S _ { \mathrm { A } } , S _ { \mathrm { B } } )$ , which is constructed to be surfacesimilar and therefore more challenging, and the extended pair $( S _ { \mathrm { { A } } } , \ S _ { \mathrm { { B } } } , )$ , as illustrated in Figure 3b. Given the question Q and an evaluation pair, the model selects one of four labels: Sentence A, Sentence B, Both Sentences, or Neither Sentence. Although exactly one sentence is correct in each pair, we include the two dummy options to reduce noise from forced binary choices.

## 3.2 Construction Pipeline

Figure 2 provides an overview of the benchmark construction pipeline, which consists of the following three steps:

Step 1: Construction of Questions and Base Sentence Pairs We extract polysemous verbs and their evoked frame pairs from the target language’s frame-semantic resource. For each verb V and frame pair (Frame , Frame ), we use an LLM to generate the base sentence pair $( S _ { \mathrm { A } } , S _ { \mathrm { B } } )$ and a question Q for which only $S _ { \mathrm { { A } } }$ is the correct answer. The generation is conditioned on the frame names, definitions, core elements, and example usages of both frames. To increase task difficulty, we instruct the LLM to make $S _ { \mathrm { { A } } }$ and $S _ { \mathrm { B } }$ as lexically similar as possible while preserving their distinct frame assignments.

Step 2: Expansion of Target Sentence Pairs To diversify the evaluation set, we generate additional sentences to form the extended pair $( S _ { \mathrm { { A ^ { \prime } } } } ,$ $S _ { \mathbf { B } ^ { \prime } } )$ . Unlike the base pair, the extended pair is not constrained to be surface-similar. For each Frame $( x \in \{ \mathbf { A } , \mathbf { B } \} )$ , we generate an additional sentence $S _ { \mathrm { x } } ,$ using the $\mathrm { F r a m e _ { x } }$ information, the question $Q ,$ and the original sentence $S _ { \mathrm { x } }$ as input. $S _ { \mathrm { x } } ,$ is constrained to evoke the same frame as $S _ { \mathrm { x } }$ . Thus, $S _ { \mathrm { x } } ,$ retains the same ground-truth label as $S _ { \mathrm { x } }$ with respect to $Q .$ . For example, $S _ { \mathrm { { A } } } ,$ is a correct answer to $Q ,$ , matching the label of $S _ { \mathrm { { A } } }$

Step 3: Human Validation and Filtering To ensure benchmark quality, native speakers of the target language manually evaluated the constructed items. The evaluation consisted of two components: a correctness judgment, implemented as a four-choice task, and an acceptability judgment of the descriptions. Model evaluation uses only single-correct pairs, whereas human evaluation additionally includes auxiliary bothcorrect pairs $( S _ { \mathrm { A } } , S _ { \mathrm { A ^ { \prime } } } )$ and neither-correct pairs $( S _ { \mathrm { B } } , S _ { \mathrm { B } ^ { \prime } } )$ to reduce annotator bias toward selecting a single sentence. These auxiliary pairs are used only for validation and are excluded from entrylevel scoring. Correctness and acceptability scores are computed over the two single-correct pairs by taking the minimum across pairs for each dimension.

## 3.3 English and Japanese Versions

We constructed English and Japanese versions of FrameBench.

Resources and Generation Setup As sources of frame knowledge, we used FrameNet (Ruppenhofer et al., 2016) for English and Japanese FrameNet (Ohara et al., 2004) for Japanese. In both languages, $\mathrm { G P T } { \cdot } 5 ^ { 1 }$ (OpenAI, 2025) was used for generation in Step 1 and Step 2. Table 1 summarizes the resource and dataset statistics. Due to differences in resource scale, we randomly sampled 800 frame pairs for English, while using all 335 eligible frame pairs for Japanese. For Japanese, we generated two entries per frame pair $\left( k { = } 2 \right)$ to ensure sufficient dataset size. During construction, we removed invalid generations, yielding 731 final items for English and 549 for

<table><tr><td>Metric</td><td>English</td><td>Japanese</td></tr><tr><td>Source FrameNet # Frame Pairs (P)</td><td>800</td><td>335</td></tr><tr><td># Unique LUs</td><td></td><td></td></tr><tr><td></td><td>430</td><td>139</td></tr><tr><td># Entries per Pair (k)</td><td>1</td><td>2</td></tr><tr><td>FrameBench</td><td></td><td></td></tr><tr><td># Candidate Items  $( P \times k )$ </td><td>800</td><td>670</td></tr><tr><td># Final Items</td><td>731</td><td>549</td></tr><tr><td># Unique LUs</td><td>407</td><td>128</td></tr></table>

Table 1: Statistics of the FrameBench dataset.

(a) English
<table><tr><td rowspan="2" colspan="2"></td><td colspan="4">#Accpt.</td><td rowspan="2">Total</td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td></tr><tr><td rowspan="4">#Corr.</td><td>0</td><td>0</td><td>3</td><td>8</td><td>1</td><td>12</td></tr><tr><td>1</td><td>2</td><td>4</td><td>15</td><td>10</td><td>31</td></tr><tr><td>2</td><td>5</td><td>18</td><td>48</td><td>63</td><td>134</td></tr><tr><td>3</td><td>11</td><td>55</td><td>201</td><td>287</td><td>554</td></tr><tr><td colspan="2">Total</td><td colspan="2">18 80</td><td>272</td><td>361</td><td>731</td></tr><tr><td colspan="8">(b) Japanese</td></tr><tr><td rowspan="2" colspan="2"></td><td colspan="4">#Accpt.</td><td rowspan="2">Total</td></tr><tr><td colspan="2">0 1</td><td>3</td><td></td></tr><tr><td rowspan="4">#Corr.</td><td>0</td><td>0</td><td>0</td><td>1</td><td>9</td><td>10</td></tr><tr><td>1</td><td>0</td><td>1</td><td>7</td><td>26</td><td>34</td></tr><tr><td>2</td><td>0</td><td>1</td><td>12</td><td>117</td><td>130</td></tr><tr><td>3</td><td>0</td><td>4</td><td>34</td><td>337</td><td>375</td></tr><tr><td colspan="2">Total</td><td>0</td><td>6</td><td>54</td><td>489</td><td>549</td></tr></table>

Table 2: Distribution of entries by the numbers of annotators who judged each entry correct and acceptable. Highlighted cells indicate the high-quality subset used for evaluation.

Japanese.

Manual Revision Compared with the English outputs, the Japanese outputs tended to contain less natural phrasing. Therefore, one of the authors, a native Japanese speaker, manually revised the generated Japanese descriptions after Steps 1 and 2. Low-acceptability items were further revised after the initial human validation in Step 3 and re-evaluated by annotators.

Dataset Statistics and Human Validation Results Table 2 summarizes the distribution of human validation scores for FrameBench entries. Each entry consists of a question and two sentence pairs: the base pair and the extended pair. Following the filtering criterion described in Step 3, we retain only entries that at least two annotators judged correct and at least two judged acceptable. This subset defines the main evaluation set used throughout our experiments. As shown by the blue-highlighted cells in Tables 2a and 2b, the resulting set contains 599 English entries and 500 Japanese entries. Detailed inter-annotator agreement statistics are provided in Appendix A.3.

<table><tr><td rowspan="2">Model</td><td rowspan="2">R</td><td colspan="6">English</td><td colspan="5">Japanese</td></tr><tr><td>FrameBench</td><td></td><td>MMLU- Pro</td><td>GPQA- Diamond</td><td>HLE</td><td>Frame Ident.</td><td>FrameBench</td><td></td><td>JamC- QA</td><td>MMLU- , Frame</td><td>ProX Ident.</td></tr><tr><td>Human</td><td></td><td> $9 6 . 3 \pm . 1 . 9$ </td><td>I</td><td></td><td></td><td>I</td><td></td><td> $9 4 . 9 \pm \ : 3 . 7 \ : \ : \ : ^ { \ : }$ </td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Closed-Source Models</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5 nano</td><td>√</td><td> $9 3 . 5 \pm \ : \ : 0 . 8$ </td><td></td><td></td><td>67.0c</td><td>7.6c</td><td>77.8</td><td> $8 3 . 9 \pm \ : \ : 4 . 4$ </td><td></td><td></td><td></td><td>76.3</td></tr><tr><td>GPT-5</td><td>√</td><td> $9 9 . 1 \pm \ : \ : 0 . 2$ </td><td></td><td>86.5e</td><td>84.2c</td><td>23.5c  84.0</td><td></td><td> $9 6 . 3 \pm \ : \ : 2 . 0$ </td><td></td><td>85.8e</td><td>84.9e</td><td>78.3</td></tr><tr><td>Gemini 3.1 Flash-Lite</td><td>√</td><td> $9 7 . 6 \pm \ : \ : 0 . 4$ </td><td></td><td>86.2a</td><td>82.2c</td><td>16.2c  80.0</td><td></td><td> $9 7 . 8 \pm \ : \ : 0 . 2$ </td><td>I</td><td></td><td></td><td>72.3</td></tr><tr><td>Gemini 3.1 Pro</td><td>√</td><td> $9 9 . 3 \pm \ : \ : 0 . 2$ </td><td></td><td>91.2b</td><td>94.1c</td><td>44.7c 82.5</td><td></td><td> $9 7 . 5 \pm \ : \ : 0 . 7$ </td><td></td><td></td><td></td><td>71.5</td></tr><tr><td colspan="10">Open-Weight Models</td><td colspan="3"></td></tr><tr><td>Gemma 4 E2B</td><td>√</td><td> $8 6 . 3 \pm \ : \ : 1 . 9$ </td><td></td><td>60.0ª</td><td>43.3c</td><td>4.8c</td><td>74.8</td><td> $8 2 . 3 \pm \ : \ : 2 . 8$ </td><td></td><td>36.1e</td><td>58.0e</td><td>72.3</td></tr><tr><td>Gemma 4 E4B</td><td>√</td><td> $9 4 . 5 \pm \ : \ : 1 . 1$ </td><td></td><td>69.4a</td><td>57.6c</td><td>3.7º」 80.8</td><td></td><td> $9 2 . 2 \pm \ : \ : 1 . 0$ </td><td></td><td>42.4e</td><td>67.0e</td><td>76.8</td></tr><tr><td>Gemma 4 31B</td><td>√</td><td> $9 8 . 6 \pm \ : \ : 0 . 3$ </td><td></td><td>85.2a</td><td>85.7c</td><td>22.7c82.0</td><td></td><td> $9 9 . 1 \pm \ : \ : 0 . 1$ </td><td></td><td>68.6e</td><td>84.1e</td><td>73.8</td></tr><tr><td>Gemma 4 E2B</td><td></td><td> $6 8 . 6 \pm 2 1 . 5$ </td><td></td><td>57.9f</td><td>40.5c</td><td>4.5c</td><td>74.0</td><td> $4 1 . 5 \pm 1 1 . 0$ </td><td></td><td>33.7f</td><td>50.5f</td><td>67.8</td></tr><tr><td>Gemma 4 E4B</td><td></td><td> $8 2 . 5 \pm 2 0 . 1$ </td><td></td><td>67.0f</td><td>54.9c</td><td>4.7º71.5</td><td></td><td> $8 4 . 6 \pm \ : \ : 5 . 1$ </td><td></td><td>43.4f</td><td>63.6f</td><td>68.5</td></tr><tr><td>Gemma 4 31B</td><td>-</td><td> $9 7 . 5 \pm \ : \ : 0 . 5$ </td><td></td><td>83.8f</td><td>76.3c</td><td>11.5c 81.0</td><td></td><td> $9 7 . 6 \pm \ : \ : 0 . 5$ </td><td></td><td>67.6f</td><td>80.0f</td><td>71.5</td></tr><tr><td>Qwen3.5-0.8B</td><td>√</td><td> $3 3 . 3 \pm \ : \ : 7 . 2$ </td><td></td><td>42.3a</td><td>11.1º</td><td>1.2c</td><td>41.0</td><td> $2 4 . 2 \pm \ : 1 . 5$ </td><td></td><td>24.5e</td><td>23.6e</td><td>39.3</td></tr><tr><td>Qwen3.5-2B</td><td>√</td><td> $8 0 . 3 \pm \ : \ : 1 . 1$ </td><td></td><td>66.5ª</td><td>59.8c</td><td>5.1c</td><td>68.0</td><td>60.7 ± 3.3</td><td></td><td>28.3e</td><td>39.1e</td><td>54.3</td></tr><tr><td>Qwen3.5-4B</td><td>√</td><td> $9 5 . 5 \pm \ : \ : 0 . 5$ </td><td></td><td>79.1a</td><td>68.8c</td><td>6.7c | 81.0</td><td></td><td>89.9 ± 2.5</td><td></td><td>39.5e</td><td>75.0e</td><td>71.0</td></tr><tr><td>Qwen3.5-9B</td><td>√</td><td> $9 7 . 3 \pm \ : \ : 0 . 2$ </td><td></td><td>82.5ª</td><td>77.8c</td><td>7.5c</td><td>81.3</td><td> $9 1 . 8 \pm \ : \ : 2 . 4$ </td><td></td><td>48.9e</td><td>78.4e</td><td>72.0</td></tr><tr><td>Qwen3.5-27B</td><td>√</td><td> $9 8 . 6 \pm \ : \ : 0 . 2$ </td><td></td><td>86.1a</td><td>87.5c</td><td>16.6c</td><td>80.8</td><td> $9 7 . 3 \pm \ : \ : 0 . 9$ </td><td></td><td>59.1e</td><td>83.5e</td><td>70.8</td></tr><tr><td>Qwen3.5-0.8B</td><td></td><td> $4 7 . 6 \pm \ : \ : 8 . 6$ </td><td></td><td>29.7a</td><td>23.6c</td><td>4.9c</td><td>49.5</td><td> $2 5 . 4 \pm \ : 2 . 4$ </td><td></td><td> 24.0f</td><td>20.9f</td><td>32.5</td></tr><tr><td>Qwen3.5-2B</td><td></td><td> $4 8 . 5 \pm 1 1 . 0$ </td><td></td><td>55.3a</td><td>43.8c</td><td>4.9c</td><td>53.3</td><td> $3 0 . 0 \pm \ : \ : 3 . 3$ </td><td></td><td>28.5f</td><td>36.2f</td><td>40.0</td></tr><tr><td>Qwen3.5-4B</td><td></td><td> $8 1 . 1 \pm \ : \ : 1 . 9$ </td><td></td><td>69.8f</td><td>71.2c</td><td>7.5c</td><td>45.5</td><td> $5 6 . 7 \pm . 4 . 8$ </td><td></td><td>37.7f</td><td>60.1f</td><td>45.0</td></tr><tr><td>Qwen3.5-9B</td><td></td><td> $8 5 . 8 \pm \ : \ : 3 . 2$   $9 6 . 1 \pm 0 . 5$ </td><td></td><td>73.1f 78.3f</td><td>78.6c 84.2c</td><td>8.6c  64.0 13.2c  74.8</td><td></td><td> $7 2 . 9 \pm \ : \ : 3 . 9$ </td><td> $8 3 . 7 \pm \ : 3 . 1$ </td><td>46.9f 55.9f</td><td>65.5f</td><td>47.8</td></tr><tr><td colspan="10">Qwen3.5-27B</td><td></td><td>74.5f</td><td>56.5</td></tr><tr><td>Japanese-Oriented Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\overline { { \mathrm { L L M } \mathrm { - j p } 4 8 \mathrm { B } } }$ </td><td>√</td><td></td><td>1</td><td></td><td></td><td>I</td><td></td><td> $8 4 . 5 \pm \ : \ : 5 . 8$ </td><td></td><td>51.4e</td><td>62.3e</td><td>75.0</td></tr><tr><td> $\mathrm { L L M - j p 4 8 B }$ </td><td>-</td><td></td><td>I</td><td></td><td></td><td></td><td></td><td> $6 9 . 2 \pm \ : \ : 8 . 8$ </td><td></td><td>47.1f</td><td>46.1f</td><td>33.3</td></tr><tr><td>Qwen3 Swallow 8B</td><td>√ √</td><td></td><td>1 I</td><td></td><td></td><td>1 I</td><td></td><td> $8 4 . 0 \pm \ : \ : 4 . 4$   $9 3 . 2 \pm \ : \ : 1 . 1$ </td><td></td><td>46.9e 51.8e</td><td>70.8e</td><td>78.8 76.3</td></tr><tr><td colspan="10">Qwen3 Swallow 32B</td><td></td><td>76.1e</td><td></td></tr><tr><td>Models with Shared Base Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-8B Qwen3-32B</td><td>√ √</td><td></td><td></td><td></td><td></td><td></td><td></td><td> $8 5 . 0 \pm \ : \ : 6 . 4$   $9 3 . 9 \pm \ : \ : 0 . 8$ </td><td></td><td> $4 0 . 1 ^ { \mathrm { e } }$   $4 7 . 2 ^ { \mathrm { e } }$ </td><td>71.1e 75.9e</td><td>69.0 71.3</td></tr></table>

Table 3: FrameBench results with comparison benchmark scores. Model scores on FrameBench are reported as mean ± standard deviation over five prompt templates, while human scores are reported as mean ± standard deviation across three annotators. R denotes reasoning mode, and Frame Ident. denotes the frame identification task. Rows under “Models with Shared Base Models” share base models with the Qwen3 Swallow models. For non-FrameBench scores, superscripts indicate the source of each score: <sup>a</sup> model-provider reports, <sup>b</sup> MMLU-Pro leaderboard, <sup>c</sup> Artificial Analysis, <sup>e</sup> Swallow LLM Leaderboard, and <sup>f</sup> our own evaluations.

## 4 Evaluation on FrameBench

We evaluate LLMs on the English and Japanese versions of FrameBench and compare their performance with existing benchmarks.

## 4.1 Experimental Setup

We evaluate models on the English and Japanese main evaluation sets described in Section 3.3. For each entry, we evaluate both the base pair and the extended pair. To mitigate position bias, each pair is tested in both original and swapped orders. We report accuracy as the mean and standard deviation across five prompt templates per language. Answer options are randomly ordered for each question.

We test closed-source, open-weight, and Japanese-oriented models, using constrained or structured decoding for reliable answer extraction. As the human score, we report accuracy computed from the human validation results described in Section 3.3. Since the evaluation subset is filtered to entries answered correctly by at least two annotators, the human score may be upwardly biased. Note that an independently measured human score may be lower if the same task were administered in a separate evaluation round.

For comparison, we report results on widely used LLM evaluation benchmarks: MMLU-Pro (Wang et al., 2024), GPQA-Diamond (Rein et al., 2024), Humanity’s Last Exam (HLE) (Center for AI Safety et al., 2026), JamC-QA (Oka et al., 2026), and MMLU-ProX (Xuan et al., 2025). We additionally evaluate a frame identification task as a complementary measure of models’ ability to identify frames from sentences. Additional details of the experimental setup are provided in Appendix B.

## 4.2 Results

Table 3 shows the results of the English and Japanese evaluations. Across both languages, FrameBench performance increases with model scale and reasoning. Several high-performing models surpass the human reference scores, despite the potential upward bias in the human scores. The scale and reasoning trends are broadly consistent with the results on the comparison benchmarks. However, reasoning gains tend to be larger on FrameBench than on several of these benchmarks.

We first focus on the English results. Among closed-source models, Gemini 3.1 Pro achieves the highest score, reaching 99.3, followed by GPT-5 at 99.1. Among open-weight models, Qwen3.5- 27B and Gemma 4 31B perform best, both reaching 98.6 with reasoning. All of these scores exceed the human reference score of 96.3. Across the results, larger models consistently obtain higher scores, and reasoning improves performance for all matched open-weight models except Qwen3.5- 0.8B. The gains are particularly large for smaller and mid-sized models. For example, enabling reasoning improves Qwen3.5-2B from 48.5 to 80.3 and Qwen3.5-4B from 81.1 to 95.5.

Compared with existing LLM benchmarks, FrameBench shows broadly consistent model rankings. Frame identification scores also show a broadly similar trend to FrameBench, but with several reversals relative to the comparison benchmarks. Together, these patterns suggest that FrameBench reflects general languageunderstanding ability while remaining grounded in frame-semantic interpretation. This makes FrameBench a more suitable evaluation setting than direct frame identification for assessing frame-semantic language understanding in LLMs. At the same time, FrameBench shows larger performance gains than the comparison benchmarks as model capability improves through scaling or reasoning. For example, Qwen3.5 with reasoning improves from 33.3 at 0.8B to 80.3 at 2B in English, a much larger jump than the corresponding increase on MMLU-Pro. This pattern suggests that FrameBench is particularly sensitive to the ability to discriminate context-dependent framesemantic interpretations.

The LLM used in dataset construction could affect the resulting FrameBench entries and the evaluation results. To examine this possibility, we constructed an additional 100 English FrameBench entries using Gemini 3.1 Pro in place of GPT-5 and evaluated them. The overall performance pattern was similar to that observed on the original FrameBench. Detailed results are provided in Appendix C.

Turning to the Japanese results, we observe similar trends, with performance generally improving with model scale and reasoning. This is in line with the results on MMLU-ProX. Gemma 4 31B achieves the highest score of 99.1, followed by Gemini 3.1 Flash-Lite at 97.8, Gemini 3.1 Pro at 97.5, Qwen3.5-27B at 97.3, and GPT-5 at 96.3. All of these scores exceed the human reference score of 94.9. Reasoning also yields sharp gains, improving Qwen3.5-2B from 30.0 to 60.7, Qwen3.5-4B from 56.7 to 89.9, Gemma 4 E2B from 41.5 to 82.3, and Gemma 4 E4B from 84.6 to 92.2.

For Japanese-oriented models, we compare Qwen3 Swallow models with the corresponding Qwen3 models, since they are based on the same Qwen3-Base models. Although the Swallow models outperform their Qwen3 counterparts on JamC-QA, they perform comparably to or slightly below the corresponding Qwen3 models on FrameBench. Specifically, Qwen3 Swallow 8B scores 84.0 compared with 85.0 for Qwen3-8B, and Qwen3 Swallow 32B scores 93.2 compared with 93.9 for Qwen3-32B. This contrast suggests that target-language training does not necessarily improve frame-semantic interpretation.

Comparing the English and Japanese results, the human reference score decreases only slightly, from 96.3 to 94.9, whereas many LLMs show larger drops. Moreover, the Japanese results separate models more clearly than the English results. For example, strong reasoning models show small cross-lingual gaps: Gemma 4 31B changes only from 98.6 to 99.1, and Qwen3.5-27B from 98.6 to 97.3. In contrast, weaker models show larger drops, with GPT-5 nano falling from 93.5 to 83.9 and Qwen3.5-9B without reasoning from 85.8 to 72.9. These results suggest that Japanese FrameBench cannot be solved by the cross-lingual transfer ability of lower-capability models alone, and instead requires robust context-dependent semantic interpretation in Japanese.

<table><tr><td rowspan="2">Model</td><td rowspan="2">R</td><td rowspan="2">Acc.</td><td colspan="3">(i) Sent. Pair Type</td><td colspan="3">(ii) Correct Sent. Position</td><td colspan="3">(iii) Error Breakdown</td></tr><tr><td>Base</td><td>Ext.</td><td> $\underline { { \Delta \boldsymbol { B } - \boldsymbol { E } } }$ </td><td>1st</td><td>2nd</td><td> $\underline { { \Delta 1 - 2 } }$ </td><td>Both</td><td>Neither</td><td>Opposite</td></tr><tr><td>Gemma 4 E2B</td><td>=</td><td>68.6</td><td>66.9</td><td>70.3</td><td>-3.3</td><td>77.1</td><td>60.1</td><td>+17.0</td><td>11.7</td><td>7.7</td><td>12.0</td></tr><tr><td>Gemma 4 E4B</td><td></td><td>82.5</td><td>80.8</td><td>84.1</td><td>-3.3</td><td>84.2</td><td>80.8</td><td>+3.4</td><td>6.3</td><td>4.6</td><td>6.6</td></tr><tr><td>Gemma 4 31B</td><td>=</td><td>97.5</td><td>96.8</td><td>98.2</td><td>-1.4</td><td>97.5</td><td>97.5</td><td>0.0</td><td>2.2</td><td>0.2</td><td>0.1</td></tr><tr><td>Gemma 4 E2B</td><td>√</td><td>86.3</td><td>85.4</td><td>87.1</td><td>-1.7</td><td>89.2</td><td>83.4</td><td>+5.7</td><td>10.8</td><td>2.1</td><td>0.8</td></tr><tr><td>Gemma 4 E4B</td><td>√</td><td>94.5</td><td>93.8</td><td>95.1</td><td>-1.4</td><td>95.6</td><td>93.4</td><td>+2.2</td><td>4.6</td><td>0.6</td><td>0.3</td></tr><tr><td>Gemma 4 31B</td><td>√</td><td>98.6</td><td>98.2</td><td>99.1</td><td>-0.9</td><td>98.9</td><td>98.3</td><td>+0.6</td><td>1.1</td><td>0.1</td><td>0.2</td></tr><tr><td>Qwen3.5-0.8B</td><td></td><td>47.6</td><td>46.2</td><td>49.0</td><td>-2.7</td><td>52.9</td><td>42.4</td><td>+10.5</td><td>24.1</td><td>2.9</td><td>25.4</td></tr><tr><td>Qwen3.5-2B</td><td></td><td>48.5</td><td>46.4</td><td>50.5</td><td>-4.2</td><td>54.4</td><td>42.6</td><td>+11.9</td><td>15.3</td><td>22.5</td><td>13.7</td></tr><tr><td>Qwen3.5-4B</td><td></td><td>81.1</td><td>80.5</td><td>81.6</td><td>-1.1</td><td>90.8</td><td>71.4</td><td>+19.4</td><td>7.9</td><td>4.0</td><td>7.1</td></tr><tr><td>Qwen3.5-9B</td><td></td><td>85.8</td><td>85.4</td><td>86.2</td><td>-0.7</td><td>91.9</td><td>79.7</td><td>+12.3</td><td>7.7</td><td>1.6</td><td>4.9</td></tr><tr><td>Qwen3.5-27B</td><td></td><td>96.1</td><td>95.8</td><td>96.3</td><td>-0.6</td><td>96.8</td><td>95.4</td><td>+1.4</td><td>2.8</td><td>0.7</td><td>0.4</td></tr><tr><td>Qwen3.5-0.8B</td><td>√</td><td>33.3</td><td>32.3</td><td>34.2</td><td>-1.9</td><td>30.6</td><td>36.0</td><td>-5.4</td><td>25.8</td><td>20.1</td><td>20.9</td></tr><tr><td>Qwen3.5-2B</td><td>√</td><td>80.3</td><td>78.2</td><td>82.3</td><td>-4.1</td><td>81.3</td><td>79.3</td><td>+2.0</td><td>13.1</td><td>4.3</td><td>2.3</td></tr><tr><td>Qwen3.5-4B</td><td>√</td><td>95.5</td><td>94.7</td><td>96.3</td><td>-1.6</td><td>96.2</td><td>94.8</td><td>+1.4</td><td>3.8</td><td>0.4</td><td>0.2</td></tr><tr><td>Qwen3.5-9B</td><td>√</td><td>97.3</td><td>96.7</td><td>97.9</td><td>-1.2</td><td>98.2</td><td>96.5</td><td>+1.7</td><td>2.4</td><td>0.2</td><td>0.1</td></tr><tr><td>Qwen3.5-27B</td><td>√</td><td>98.6</td><td>98.1</td><td>99.1</td><td>-1.0</td><td>99.1</td><td>98.1</td><td>+1.0</td><td>1.0</td><td>0.3</td><td>0.1</td></tr></table>

Table 4: Breakdown of English FrameBench performance by sentence-pair type, position of the correct sentence, and error type. R denotes reasoning mode, and Acc. denotes overall accuracy. Base and Ext. report accuracy on base and extended sentence pairs, while 1st and 2nd report accuracy when the correct sentence appears first or second. $\Delta _ { B - E }$ and $\Delta _ { 1 - 2 }$ indicate performance differences. Both denotes selecting both sentences, Neither denotes selecting neither sentence, and Opposite denotes selecting the incorrect sentence. Acc. and the three errortype rates sum to 100% up to rounding.

<table><tr><td>Model</td><td>R</td><td>M</td><td>Frame Bench</td><td>MMLU- Pro</td></tr><tr><td>Qwen3-1.7B</td><td>√</td><td></td><td>77.8</td><td>35.6c</td></tr><tr><td>Qwen3-VL-2B</td><td>√</td><td>√</td><td>67.7</td><td>62.3ª</td></tr><tr><td>Qwen3-4B Qwen3-VL-4B</td><td>√ √</td><td>√</td><td>90.9</td><td>52.2c</td></tr><tr><td>Qwen3-8B</td><td>√</td><td></td><td>92.3 92.5</td><td>73.6ª</td></tr><tr><td>Qwen3-VL-8B</td><td>√</td><td>√</td><td>96.3</td><td>72.1c 77.3ª</td></tr><tr><td>Qwen3-32B</td><td>√</td><td></td><td>96.9</td><td>66.8c</td></tr><tr><td>Qwen3-VL-32B</td><td>√</td><td>√</td><td>95.7</td><td>82.1ª</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-1.7B</td><td></td><td></td><td>53.5</td><td>37.3f</td></tr><tr><td>Qwen3-VL-2B</td><td></td><td>√</td><td>71.4</td><td>49.0ª</td></tr><tr><td>Qwen3-4B</td><td></td><td></td><td>84.6</td><td>57.9f</td></tr><tr><td>Qwen3-VL-4B</td><td></td><td>√</td><td>89.8</td><td>67.1ª</td></tr><tr><td>Qwen3-8B</td><td></td><td></td><td>87.1</td><td> $\overline { { 5 9 . 9 ^ { f } } }$ </td></tr><tr><td>Qwen3-VL-8B</td><td></td><td>√</td><td>94.3</td><td>71.6ª</td></tr><tr><td>Qwen3-32B</td><td></td><td></td><td>92.5</td><td>72.7f</td></tr><tr><td>Qwen3-VL-32B</td><td></td><td>√</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>95.7</td><td>78.6a</td></tr><tr><td>Phi-3.5-mini</td><td></td><td></td><td>59.7</td><td>34.5f</td></tr><tr><td>Phi-3.5-vision</td><td></td><td>√</td><td>72.1</td><td>32.2f</td></tr><tr><td>Phi-4-mini</td><td></td><td></td><td>78.5</td><td>43.3f</td></tr><tr><td>Phi-4-multimodal</td><td></td><td>√</td><td>79.8</td><td>39.1f</td></tr></table>

Table 5: Performance comparison of matched text-only and multimodal models. R denotes reasoning mode, and M denotes multimodality. For MMLU-Pro scores, superscripts indicate the source of each score: <sup>a</sup> modelprovider reports, <sup>c</sup> Artificial Analysis, and <sup>f</sup> our own evaluations.

## 5 Analysis

We analyze model behavior on the English subset of FrameBench to better understand error patterns and remaining challenges.

## 5.1 Error and Bias Analysis

We further analyze model behavior on FrameBench along three dimensions: sentencepair type, sentence position, and error type. Japanese results for the same analyses are provided in Appendix D; they show similar tendencies except for a weaker sentence-pair type effect.

Difficulty by Sentence Pair Type Since base pairs are designed to be lexically similar, they are expected to be harder for models than extended pairs, which are not constrained to be surfacesimilar. Columns under (i) Sent. Pair Type in Table 4 show the accuracy for each pair type and the performance gap. All models achieved higher accuracy on extended pairs than on base pairs, as indicated by the negative values of $\Delta _ { B - E } .$ . The human score gap shows the same tendency, with extended pairs outperforming base pairs by 2.57 points. These results indicate that lexical similarity makes base pairs harder for LLMs. In the Japanese subset, however, this pair-type effect is weaker: the human score gap decreases to 0.93 points, and the LLM results show a less consistent advantage for extended pairs.

Analysis of Sentence Position Bias We also investigated order sensitivity by comparing performance when the correct sentence appears in the first versus second position. Columns under (ii) Correct Sent. Position in Table 4 report accuracy for these two positions and the corresponding performance gap. Since each sentence pair is evaluated in both original and swapped orders, the same items appear in both position conditions, controlling for item difficulty. Most models show a noticeable absolute difference between the two positions, indicating that sentence order can affect model predictions. This supports the main evaluation protocol described in Section 4, where each pair is evaluated in both possible orders to mitigate order sensitivity.

<table><tr><td rowspan="2">Question and Sentence Pair</td><td>Qwen3.5</td><td>Gemma 4</td></tr><tr><td>0.8B 2B 4B 9B 27B</td><td>E2B E4B 31B</td></tr><tr><td>Q1. Select all sentences where the speaker gets hurt. SA. Rounding the corner too fast, I grazed the wall with my knee. ([Impact]) SB. Rounding the corner too fast, I grazed my knee on the wall. ([Body_injury])</td><td></td><td></td></tr><tr><td>Q2. Select all sentences where someone carries out a ceremonial rite. SA. They christened their new boat the Sea Breeze at the pier. ([Name_conferral]) SB. They christened their new boat with a bottle of champagne at the pier. ([Rite])</td><td></td><td></td></tr><tr><td>Q3. Select all sentences where someone is trying to follow the person. SA. I lost the guards in the crowd. ([Losing_track_of_perceiver])</td><td></td><td></td></tr><tr><td>SB. I lost my wallet in the crowd. ([Losing]) Q4. Select all sentences where a person takes on a character in a play.  $S _ { \mathbf { A } } .$  He and Maya played Romeo and Juliet at the festival. ([Performers_and_roles])</td><td></td><td></td></tr></table>

Table 6: Examples from FrameBench. Bold text marks the correct answer, underlining marks the frame-evoking word, and parenthetical tags show the corresponding frame. Each model column reports whether the model answered the item correctly in the reasoning setting, where ‘✓’ denotes a correct prediction and ‘–’ denotes an incorrect prediction.

Error Tendencies in Incorrect Choices Columns under (iii) Error Breakdown in Table 4 show the distribution of model errors. For models scoring above 85%, “Opposite” errors become rare. For high-performing models, errors therefore tend to reflect over-selection or under-selection rather than outright selection of the opposite frame, suggesting that the remaining difficulty lies in borderline frame-semantic distinctions.

## 5.2 Impact of Multimodality

Frame semantics is a linguistic framework, but the situations it describes often involve visually grounded event knowledge. This motivates examining whether multimodal training helps models interpret frame-semantic distinctions. Table 5 reports matched comparisons between multimodal models and the language models on which they are based. Overall, multimodal models outperform the corresponding language models in most matched comparisons on FrameBench. However, these gains should be interpreted cautiously: many multimodal models also achieve higher MMLU-Pro scores, while the Phi models show FrameBench improvements despite lower MMLU-Pro scores. Thus, visually grounded training may help frame-semantic interpretation, but its effect cannot be isolated from other training differences.

## 5.3 Case Study

We examine examples selected to span different levels of model accuracy. Because we consider only items for which at least two of the three human annotators selected the correct answer, these examples are intended to be answerable by humans rather than inherently ambiguous. Table 6 reports per-model correctness for reasoning-enabled Qwen3.5 and Gemma 4 models using the first prompt template in Table 9. The scores are shown across model sizes, allowing us to examine whether each example is solved consistently, requires larger models, or remains difficult even for stronger models.

In Questions 1 and 2 of Table 6, even the largest Qwen3.5 and Gemma 4 models fail, incorrectly selecting both sentences. These examples require distinguishing subtle frame-semantic differences despite strong lexical or contextual overlap. In Question 1, a small change in argument structure shifts the interpretation from contact with an object to injury to a body part. In Question 2, christen carries a ceremonial nuance in both sentences, but only Sentence B explicitly evokes the [Rite] frame through the use of a bottle of champagne. These results suggest that even strong models can struggle when the relevant frame distinction depends on fine-grained semantic cues.

Question 3 is easier, but still challenging for smaller models. Both sentences use lost in closely related senses, but only Sentence A describes a situation in which someone trying to follow the speaker loses track of them. The score pattern suggests that distinguishing related senses within a close semantic domain requires a moderate level of semantic sensitivity.

By contrast, Question 4 is solved by all models in the table. Here, the two meanings of play belong to clearly different semantic domains, and the distinction can be identified from salient contextual cues such as Romeo and Juliet and tennis.

## 6 Conclusion

In this study, we introduced FrameBench, a FrameNet-grounded multiple-choice benchmark for evaluating whether LLMs can distinguish the frames evoked by the same verb across contexts. We constructed English and Japanese versions of FrameBench using FrameNet resources, a generation-and-verification pipeline, and nativespeaker judgments. Together, these datasets provide a new resource for evaluating frame-semantic interpretation in LLMs across two typologically distant languages.

Experiments across a wide range of LLMs showed that several frontier models reached or exceeded the human reference score, while performance varied substantially with model scale, reasoning mode, and language. Our analyses further characterized model behavior in terms of sentence-pair type, sentence position, and error type, and examined whether multimodal training may support frame-semantic interpretation.

## Limitations

First, FrameBench focuses on a specific aspect of semantic competence, namely frame-semantic discrimination for context-dependent verb interpretations in a four-choice setting. It does not directly evaluate broader language understanding, open-ended generation, or explicit frame prediction. Second, the human reference scores reported in our experiments are not based on an independent evaluation. They are calculated from the judgments of the same annotators whose responses were used to validate and filter FrameBench items. Because our experiments use only items that were answered correctly by at least two annotators during validation, the reported human scores may overestimate performance relative to an evaluation conducted with an independent group of annotators.

## Ethical considerations

Our dataset construction involved human annotation in both English and Japanese. English annotations were conducted by three expert native English annotators recruited through an annotation vendor and compensated according to the vendorʼs standard rates. Japanese evaluation was conducted by three native Japanese-speaking university students, who were compensated at the universitydefined hourly rate for research assistants. Manual revision of Japanese generations was performed by one of the authors as part of the research process. All annotators were informed in advance that their judgments would be used for research purposes and reported in a paper.

## Acknowledgments

We would like to express our gratitude to Dr. Kyoko Ohara of Keio University for providing the Japanese FrameNet data used in this study. This research was supported by JST FOREST Program JPMJFR216N and JST SPRING Program JPMJSP2125.

## References

Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Martin Cai, Qin Cai, Vishrav Chaudhary, Dong Chen, Dongdong Chen, and 110 others. 2024. Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. arXiv preprint arXiv:2404.14219.

Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dongdong Chen, Junkun Chen, Weizhu Chen, Yen-Chun Chen, Yi ling Chen, Qi Dai, Xiyang Dai, Ruchao Fan, and 55 others. 2025. Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs. arXiv preprint arXiv:2503.01743.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631.

Collin F Baker, Charles J Fillmore, and John B Lowe. 1998. The Berkeley FrameNet project. In Pro-

ceedings of the 36th Annual Meeting of the Association for Computational Linguistics and 17th International Conference on Computational Linguistics (ACL-COLING 1998), pages 86–90.

Frederico Belcavello, Ely E. Matos, Arthur Lorenzi, Lisandra Bonoto, Livia Pádua Ruiz, Luiz Fernando Pereira, Victor Herbst, Yulla Liquer Navarro, Helen de Andrade Abreu, Lívia Vicente Dutra, and Tiago Timponi Torrent. 2026. Evaluating the Impact of LLM-Assisted Annotation in a Perspectivized Setting: The Case of FrameNet Annotation. In Proceedings of the 22nd Joint ACL - ISO Workshop on Interoperable Semantic Annotation and Representation (ISA-22) @ LREC 2026, pages 77–87.

Frederico Belcavello, Tiago Timponi Torrent, Ely E. Matos, Adriana S. Pagano, Maucha Gamonal, Natalia Sigiliano, Lívia Vicente Dutra, Helen de Andrade Abreu, Mairon Samagaio, Mariane Carvalho, Franciany Campos, Gabrielly Azalim, Bruna Mazzei, Mateus Fonseca de Oliveira, Ana Carolina Loçasso Luz, Lívia Pádua Ruiz, Júlia Bellei, Amanda Pestana, Josiane Costa, and 5 others. 2024. Frame2: A FrameNet-based Multimodal Dataset for Tackling Text-image Interactions in Video. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 7429–7437.

Center for AI Safety, Scale AI, and HLE Contributors Consortium. 2026. A benchmark of expert-level academic questions to assess AI capabilities. Nature, 649(8099):1139<sup>‒</sup>1146.

Jayanth Krishna Chundru, Rudrashis Poddar, Jie Cao, and Tianyu Jiang. 2025. Do LLMs Encode Frame Semantics? Evidence from Frame Identification. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025), pages 29488–29500.

Jacob Devasier, Rishabh Mediratta, and Chengkai Li. 2025. Can LLMs Extract Frame-Semantic Arguments? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025), pages 30609–30622.

Marianne Djemaa, Marie Candito, Philippe Muller, and Laure Vieu. 2016. Corpus Annotation within the French FrameNet: a Domain-by-domain Methodology. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 3794–3801.

Clement Farabet and Olivier Lacombe. 2026. Gemma 4: Byte for byte, the most capable open models.

Charles J Fillmore. 1982. Frame Semantics. In Linguistics in the Morning Calm, pages 111–137.

Diego Garat, Guillermo Moncecchi, and Dina Wonsever. 2025. Exploring in-context learning for frame-semantic parsing. arXiv preprint arXiv:2507.23082.

Shaoru Guo, Yubo Chen, Kang Liu, Ru Li, and Jun Zhao. 2024. NutFrame: Frame-based Conceptual Structure Induction with LLMs. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 12330–12335.

Younggyun Hahm, Youngbin Noh, Ji Yoon Han, Tae Hwan Oh, Hyonsu Choe, Hansaem Kim, and Key-Sun Choi. 2020. Crowdsourcing in the Development of a Multilingual FrameNet: A Case Study of Korean FrameNet. In Proceedings of the Twelfth Language Resources and Evaluation Conference (LREC 2020), pages 236–244.

Yi Han, Ryohei Sasano, and Koichi Takeda. 2024. Definition Generation for Automatically Induced Semantic Frame. In Findings of the Association for Computational Linguistics: ACL 2024 (ACL2024 Findings), pages 11112–11118.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring Massive Multitask Language Understanding. In International Conference on Learning Representations (ICLR 2021).

Kevin Knight, Bianca Badarau, Laura Baranescu, Claire Bonial, Madalina Bardocz, Kira Griffitt, Ulf Hermjakob, Daniel Marcu, Martha Palmer, Tim O’Gorman, and Nathan Schneider. 2017. Abstract Meaning Representation (AMR) Annotation Release 2.0. LDC2017T10. Dataset.

LLM-jp. 2024. LLM-jp: A Cross-organizational Project for the Research and Development of Fully Open Japanese LLMs. arXiv preprint arXiv:2407.03963.

Benjamin Lyngfelt, Lars Borin, Kyoko Ohara, and Tiago Timponi Torrent, editors. 2018. Constructicography: Constructicon Development Across Languages. John Benjamins Publishing Company.

Kyoko Hirose Ohara, Seiko Fujii, Toshio Ohori, Ryoko Suzuki, Hiroaki Saito, and Shun Ishizaki. 2004. The Japanese FrameNet Project: An introduction. In Proceedings of the LREC 2004 Satellite Workshop “Building Lexical Resources from Semantically Annotated Corpora”, pages 9–11.

Teruaki Oka, Tomohide Shibata, and Nao Yoshida. 2026. JamC-QA: A Multiple-Choice Question Answering Benchmark for Japan-Specific Knowledge. In Proceedings of the Fifteenth Language Resources and Evaluation Conference(LREC 2026), pages 4536–4546.

OpenAI. 2025. Introducing GPT-5. August 7, 2025.

Mohammad Taher Pilehvar and Jose Camacho-Collados. 2019. WiC: the Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language

Technologies (Volume 1: Long and Short Papers), pages 1267–1273.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Shahid Iqbal Rai, Danilo Croce, and Roberto Basili. 2025. Injecting Frame Semantics into Large Language Models via Prompt-Based Fine-Tuning. In Proceedings ofthe 14th Joint Conference on Lexical and Computational Semantics (\*SEM 2025), pages 31–47.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. In First Conference on Language Modeling (CoLM 2024).

Josef Ruppenhofer, Michael Ellsworth, Myriam Schwarzer-Petruck, Christopher R Johnson, and Jan Scheffczyk. 2016. FrameNet II: Extended theory and practice.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, and 431 others. 2023. Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research.

Swallow LLM Team. 2026. Qwen3 Swallow.

Swallow LLM Team, Sakae Mizuki, Koshiro Saito, Masanari Oi, Tatsuya Ichinose, Naoya Matsushita, Sora Miyamoto, Tien Dung Nguyen, and Sangwhan Moon. 2025. swallow-evaluation-instruct: Evaluation Framework for Large Language Models (in Japanese).

Marcelo Viridiano, Arthur Lorenzi, Tiago Timponi Torrent, Ely E. Matos, Adriana S. Pagano, Natália Sathler Sigiliano, Maucha Gamonal, Helen de Andrade Abreu, Lívia Vicente Dutra, Mairon Samagaio, Mariane Carvalho, Franciany Campos, Gabrielly Azalim, Bruna Mazzei, Mateus Fonseca de Oliveira, Ana Carolina Luz, Livia Padua Ruiz, Júlia Bellei, Amanda Pestana, and 7 others. 2024. Framed Multi30K: A Frame-Based Multimodal-Multilingual Dataset. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 7438–7449.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 353–355.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. 2024. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. In Advances in Neural Information Processing Systems (NeurIPS 2024), volume 37, pages 95266–95290.

Weihao Xuan, Rui Yang, Heli Qi, Qingcheng Zeng, Yunze Xiao, Aosong Feng, Dairui Liu, Yun Xing, Junjue Wang, Fan Gao, Jinghui Lu, Yuang Jiang, Huitao Li, Xin Li, Kunyu Yu, Ruihai Dong, Shangding Gu, Yuekang Li, Xiaofei Xie, and 13 others. 2025. MMLU-ProX: A Multilingual Benchmark for Advanced Large Language Model Evaluation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025), pages 1513–1532.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.

Chihiro Yano, Kosuke Yamada, Hayato Tsukagoshi, Ryohei Sasano, and Koichi Takeda. 2025. FrameEOL: Semantic Frame Induction using Causal Language Models. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 11620–11632.

Liping You and Kaiying Liu. 2005. Building Chinese FrameNet database. In 2005 International Conference on Natural Language Processing and Knowledge Engineering (IEEE NLP-KE 2005), pages 301– 306.

## A Details of Human Annotation and Manual Revision

## A.1 Human Annotation Protocol

We manually evaluated correctness and acceptability for the finalized items in both English and Japanese FrameBench. For correctness, annotators answered the same four-choice question format used in model evaluation. For acceptability, Japanese items were judged with a binary naturalness label, whereas English items were judged with a three-way scale (Unacceptable, Acceptable, Natural) to reduce rater drift.

## A.2 Acceptability scale and binarization

To facilitate cross-lingual analysis, we convert acceptability judgments into a binary label. For English, we map Natural to the positive label and treat Acceptable and Unacceptable as negative. This mapping yields a more informative label distribution than treating only Unacceptable as negative, since Unacceptable is extremely rare in our data, and it aligns with our goal of using a stricter notion of linguistic naturalness. We report the three-way label distribution for English in Table 7.

<table><tr><td></td><td>Natural</td><td>Acceptable</td><td>Unacceptable</td></tr><tr><td>Ann1</td><td>83.8</td><td>14.9</td><td>1.3</td></tr><tr><td>Ann2</td><td>91.4</td><td>8.3</td><td>0.2</td></tr><tr><td>Ann3</td><td>82.6</td><td>16.8</td><td>0.6</td></tr></table>

Table 7: Distribution of 3-level acceptability labels for English. Values are percentages.

<table><tr><td colspan="3">Positive label (%) Ann1 Ann2 Ann3</td><td>Agreement Fleiss&#x27; (%)</td><td>κ</td></tr><tr><td colspan="5">English</td></tr><tr><td colspan="5">Correctness</td></tr><tr><td>Base</td><td>88.8</td><td>92.5 93.6</td><td>82.5</td><td>0.2406</td></tr><tr><td>Extended 93.8</td><td></td><td>96.0 96.0</td><td>89.3</td><td>0.2054</td></tr><tr><td colspan="5">Acceptability</td></tr><tr><td>Base</td><td>82.9 91.8</td><td>81.0</td><td>67.6</td><td>0.1417</td></tr><tr><td>Extended</td><td>84.5 91.9</td><td>83.2</td><td>68.8</td><td>0.1070</td></tr><tr><td colspan="5">Japanese</td></tr><tr><td colspan="5">Correctness</td></tr><tr><td>Base</td><td>86.2 87.3</td><td>96.2</td><td>77.2</td><td>0.1670</td></tr><tr><td>Extended</td><td>91.1 91.6</td><td>96.4</td><td>84.5</td><td>0.2054</td></tr><tr><td colspan="5">Acceptability</td></tr><tr><td>Base</td><td>96.0 99.1</td><td>97.1</td><td>92.9</td><td>0.0687</td></tr><tr><td>Extended</td><td>98.0 99.5</td><td>97.1</td><td>94.9</td><td>0.0494</td></tr></table>

Table 8: Inter-annotator agreement for binary judgments. Agreement is the percentage of pairs where all three annotators made the same judgment.

## A.3 Inter-annotator agreement

Table 8 reports positive-label rates, Fleiss’ κ, and the 3/3 agreement rate for the binary judgments. Because κ is sensitive to skewed label distributions, especially for acceptability, we report these additional statistics to aid interpretation. In English, acceptability contains more borderline cases than correctness, which can lower agreement. For the main experiments, we therefore use a highquality subset defined by a strict human validation criterion. Under this criterion, an entry is included only if at least two annotators judged it correct and at least two judged it acceptable.

## B Experimental Details

## B.1 Evaluation Prompts

Table 9 and Table 10 present the prompts used for English and Japanese evaluation. The variables {question}, {verb}, {sentence\_a}, {sentence\_b}, and {choices\_text} serve as placeholders for the question text, the target verb, the target sentence pair, and the options, respectively. To eliminate the influence of specific choice number output probabilities, the mapping between options and numbers was randomized for each question.

Prompt templates   
{question}   
Sentence A: {sentence\_a}   
Sentence B: {sentence\_b}   
Choices: {choices\_text}   
Instruction: Please make your decision by focusing on   
the verb “{verb}”.   
Respond using only the choice number: “1”, “2”, “3”, or   
“4”.   
Please make your decision by focusing on the verb   
“{verb}”.   
{question}   
Sentence A: {sentence\_a}   
Sentence B: {sentence\_b}   
Choices: {choices\_text}   
Respond using only the choice number: “1”, “2”, “3”, or   
“4”.   
{question}   
Sentence A: {sentence\_a}   
Sentence B: {sentence\_b}   
Choices: {choices\_text}   
Respond using only the choice number: “1”, “2”, “3”, or   
“4”.   
Your task is to compare the two sentences through the   
usage of “{verb}”.   
{question}   
Sentence A: {sentence\_a}   
Sentence B: {sentence\_b}   
Available choices:   
{choices\_text}   
Output strictly one number (1–4). No explanation.   
Problem: {question}   
[Sentence A]   
{sentence\_a}   
[Sentence B]   
{sentence\_b}   
Choices:   
{choices\_text}   
Final answer format: just the option number.  
Table 9: Prompt templates used for English evaluation.

## B.2 Evaluated Models List

## Closed-source Models

• GPT-5 Series (OpenAI, 2025): gpt-5-2025-08-07, gpt-5-nano-2025-08-07

• Gemini 3.1 Series: gemini-3.1-pro, gemini-3.1-flash-lite

Open-weight Models

• Gemma 4 Series (Farabet and Lacombe, 2026): Gemma-4-31B-it, E4B-it, E2B-it

Prompt templates  
{question}  
文 A: {sentence\_a}  
文 B: {sentence\_b}  
選択肢: {choices\_text}  
回答する際は、文の最後の動詞に注目して判断して  
ください。  
(When answering,focus on the sentence-final verb in  
each sentence.)  
回答は選択肢の番号「1」「2」「3」「4」のいずれかで  
答えてください。  
(Respond using only the option number 1, 2, 3, or 4.)  
{question}  
文 A: {sentence\_a}  
文 B: {sentence\_b}  
選択肢: {choices\_text}  
選択肢の番号「1」「2」「3」「4」のいずれかで答えて  
ください。  
(Answer with the option number 1, 2, 3, or 4.)  
それぞれの文の述語動詞に注意して、{question}  
(Pay attention to the predicate verb in each sentence:  
{question})  
文 A: {sentence\_a}  
文 B: {sentence\_b}  
選択肢: {choices\_text}  
選択肢の番号「1」「2」「3」「4」のいずれかで答えて  
ください。  
(Answer with the option number 1, 2, 3, or 4.)  
それぞれの文の述語動詞に注意して回答してくださ  
い。  
(Pay attention to the predicate verb in each sentence.)  
{question}  
<sup>文</sup> A: {sentence\_a}  
<sup>文</sup> B: {sentence\_b}  
<sup>選択肢</sup>: {choices\_text}  
選択肢の番号「1」「2」「3」「4」のいずれかで答えて  
ください。  
(Answer with the option number 1, 2, 3, or 4.)  
選択肢の番号で回答してください。  
(Please answer using the number ofthe correct  
option.)  
{question}  
<sup>文</sup> A: {sentence\_a}  
<sup>文</sup> B: {sentence\_b}  
<sup>選択肢</sup>: {choices\_text}  
Table 10: Prompt templates used for Japanese evaluation.

• Qwen 3.5 Series (Qwen Team, 2026): Qwen3.5-27B, 9B, 4B, 2B, 0.8B

• Qwen 3 Series (Yang et al., 2025): Qwen3- 1.7B, Qwen3-4B, Qwen3-8B, Qwen3-32B

• Phi Series: Phi-3.5-mini-instruct (Abdin et al., 2024), Phi-4-mini-instruct (Abouelenin et al., 2025)

## Japanese-oriented Models

• LLM-jp-4 Series (LLM-jp, 2024): llm-jp-4- 8b-thinking, 8b-instruct

• Swallow Series (Swallow LLM Team, 2026): Qwen3-Swallow-8B-RL, 32B-RL Qwen3-Swallow models are developed through continual pre-training, SFT, and

RL based on Qwen3-Base to enhance their Japanese language capabilities.

## Multimodal Models

• Qwen3-VL Series (Bai et al., 2025): Qwen3-VL-2B-Thinking, 4B-Thinking, 8B-Thinking, 32B-Thinking, 2B-Instruct, 4B-Instruct, 8B-Instruct, 32B-Instruct Qwen3-VL models are built on Qwen3- family language backbones and incorporate vision-specific components to support text and image inputs.

• Phi Series: Phi-3.5-vision-instruct (Abdin et al., 2024), Phi-4-multimodalinstruct (Abouelenin et al., 2025) Both Phi-3.5-vision-instruct and Phi-4- multimodal-instruct are multimodal Phiseries models that support text and image inputs, using language backbones derived from Phi-3.5-mini and Phi-4-Mini-Instruct, respectively.

## B.3 Decoding Procedure

For open-weight models evaluated in the nonreasoning setting, generation is restricted to the tokens corresponding to the answer choices. In the reasoning setting, models with reasoning capabilities generate tokens freely until they output a specific tag marking the end of the reasoning phase. After this tag is generated, decoding is restricted in the same way as in the non-reasoning setting, allowing the model to generate only the tokens corresponding to the valid answer choices. For models with a native reasoning process, such as the GPT-5 series, we utilize JSON-based structured output to facilitate reliable answer extraction.

## B.4 Details of the Comparison Benchmarks

We describe the LLM benchmarks used for comparison in each experiment. Unless otherwise noted, we adopt the evaluation protocol in swallow-evaluation-instruct<sup>2</sup> (Swallow LLM Team et al., 2025).

## English Benchmarks

• MMLU-Pro (Wang et al., 2024): A harder variant of MMLU that evaluates broad domain knowledge via multiple-choice questions. It spans 14 subject areas, including humanities, social sciences, and STEM.

• GPQA-Diamond (Rein et al., 2024): A four-choice, graduate-level science QA benchmark written by domain experts. It covers biology, physics, and chemistry and is designed to be difficult to solve via web search.

• Humanity’s Last Exam (Center for AI Safety et al., 2026): A cross-disciplinary benchmark comprising 2,500 questions across dozens of subjects, including mathematics, the humanities, and the natural sciences. It features a mix of multiple-choice and short-answer questions designed to evaluate expert-level reasoning across broad academic domains.

## Japanese Benchmarks

• JamC-QA (Oka et al., 2026): A highdifficulty four-choice QA benchmark specialized for knowledge of Japan-specific culture and customs. Questions cover eight categories including culture, customs, local environment, geography, administration, law, medicine, and related topics, requiring niche and diverse knowledge.

• MMLU-ProX (Xuan et al., 2025): A multilingual extension of MMLU-Pro covering 29 languages. We use its Japanese subset.

## B.5 Details of the Frame Identification Evaluation

This section describes the frame identification evaluation. Unlike FrameBench, this evaluation directly provides models with candidate frame names and their definitions from the corresponding frame-semantic resources: FrameNet for English and Japanese FrameNet for Japanese. It therefore tests whether models can identify the frame evoked by a target word when the relevant candidate frames are explicitly given.

Each test instance consists of a sentence containing a target frame-evoking verb and a set of answer options. The answer options include two candidate frames that the same verb can evoke, as well as a negative option, “Neither frame is evoked.” The order of the answer options is randomized.

We construct this evaluation from the same verb–frame pairs used in FrameBench. When a verb appears in multiple FrameBench pairs because it can evoke three or more frames, we randomly select one pair for that verb to avoid redundancy. For each selected verb–frame pair, we sample one sentence for each of the two candidate frames from the corresponding frame-semantic resource. Thus, each pair yields two independent frame identification instances, one for each candidate frame.

<table><tr><td>Model</td><td>R</td><td>FrameBench (Gemini-generated)</td></tr><tr><td>GPT-5 nano</td><td>√</td><td>92.7</td></tr><tr><td>GPT-5</td><td>√</td><td>97.9</td></tr><tr><td>Gemini 3.1 Flash-Lite</td><td>√</td><td>98.8</td></tr><tr><td>Gemini 3.1 Pro</td><td>V</td><td>100.0</td></tr><tr><td>Gemma 4 E2B</td><td>√</td><td>88.6</td></tr><tr><td>Gemma 4 E4B</td><td>√</td><td>95.5</td></tr><tr><td>Gemma 4 31B</td><td>√</td><td>98.9</td></tr><tr><td>Gemma 4 E2B</td><td>一</td><td>66.2</td></tr><tr><td>Gemma 4 E4B</td><td>一</td><td>85.5</td></tr><tr><td>Gemma 431B</td><td>一</td><td>97.9</td></tr><tr><td>Qwen3.5-0.8B</td><td>√</td><td>33.8</td></tr><tr><td>Qwen3.5-2B</td><td>√</td><td>80.8</td></tr><tr><td>Qwen3.5-4B</td><td>√</td><td>96.6</td></tr><tr><td></td><td>√</td><td></td></tr><tr><td>Qwen3.5-9B</td><td></td><td>96.9</td></tr><tr><td>Qwen3.5-27B</td><td>√</td><td>99.1</td></tr><tr><td>Qwen3.5-0.8B</td><td>一</td><td>47.6</td></tr><tr><td>Qwen3.5-2B</td><td>一</td><td>47.6</td></tr><tr><td>Qwen3.5-4B</td><td></td><td>82.4</td></tr><tr><td>Qwen3.5-9B</td><td></td><td>86.9</td></tr><tr><td>Qwen3.5-27B</td><td></td><td>97.1</td></tr></table>

Table 11: English FrameBench results on 100 entries generated using Gemini 3.1 Pro instead of GPT-5 in Steps 1 and 2. R denotes reasoning mode.

We evaluate 200 verb–frame pairs in a 3-shot setting. Because each pair contributes two independently evaluated target sentences, the resulting evaluation set contains 400 test instances in total.

## C Effect of the LLM Used for Dataset Construction

To assess the effect of the LLM used in dataset construction, we generated an additional 100 English candidate FrameBench entries using Gemini 3.1 Pro in place of GPT-5 for Steps 1 and 2. The verb–frame pairs used for these entries were selected from those used in the original GPT-5-based construction. To limit additional annotation costs, we did not conduct the human validation and filtering in Step 3 for these additional entries. We evaluated them using the same model evaluation protocol as in the main experiment.

Table 11 reports the results. We compared the results on entries constructed using Gemini 3.1 Pro with the original FrameBench results reported in Table 3. Performance tended to be slightly higher for models from the same family as the LLM used to construct the benchmark. However, the overall performance pattern was similar across the two sets, with a Spearman rank correlation of $\rho = 0 . 9 8 6 .$ . These results suggest that the overall performance trends are largely preserved when a different LLM is used in dataset construction.

<table><tr><td rowspan="2">Model R</td><td rowspan="2">Acc.</td><td rowspan="2"></td><td colspan="3">(i) Sent. Pair Type</td><td colspan="3">(ii) Correct Sent. Position</td><td colspan="3">(iii) Error Breakdown</td></tr><tr><td>Base</td><td>Ext.</td><td> $\underline { { \Delta _ { B - E } } }$ </td><td>1st</td><td>2nd</td><td>∆1-2</td><td>Both</td><td>Neither</td><td>Opposite</td></tr><tr><td>Gemma 4 E2B</td><td></td><td>41.5</td><td>38.8</td><td>44.2</td><td>-5.5</td><td>53.4</td><td>29.6</td><td>+23.8</td><td>20.1</td><td>22.1</td><td>16.4</td></tr><tr><td>Gemma 4 E4B</td><td></td><td>84.6</td><td>84.3</td><td>84.8</td><td>-0.4</td><td>84.0</td><td>85.0</td><td>-1.0</td><td>5.9</td><td>4.0</td><td>5.6</td></tr><tr><td>Gemma 4 31B</td><td>=</td><td>97.6</td><td>97.7</td><td>97.5</td><td>+0.2</td><td>98.5</td><td>96.8</td><td>+1.6</td><td>0.7</td><td>1.2</td><td>0.4</td></tr><tr><td>Gemma 4 E2B</td><td>√</td><td>82.3</td><td>78.6</td><td>86.0</td><td>-7.5</td><td>80.7</td><td>84.0</td><td>-3.3</td><td>8.5</td><td>8.3</td><td>0.8</td></tr><tr><td>Gemma 4 E4B</td><td>√</td><td>92.2</td><td>90.9</td><td>93.5</td><td>-2.6</td><td>92.0</td><td>92.5</td><td>-0.5</td><td>4.0</td><td>3.1</td><td>0.7</td></tr><tr><td>Gemma 4 31B</td><td>√</td><td>99.1</td><td>99.2</td><td>99.0</td><td>+0.2</td><td>99.5</td><td>98.6</td><td>+0.9</td><td>0.4</td><td>0.3</td><td>0.1</td></tr><tr><td>Qwen3.5-0.8B</td><td></td><td>25.4</td><td>25.5</td><td>25.4</td><td>+0.1</td><td>27.8</td><td>23.1</td><td>+4.8</td><td>28.8</td><td>20.4</td><td>25.4</td></tr><tr><td>Qwen3.5-2B</td><td></td><td>30.0</td><td>28.8</td><td>31.3</td><td>-2.5</td><td>34.5</td><td>25.5</td><td>+8.9</td><td>21.9</td><td>28.0</td><td>20.1</td></tr><tr><td>Qwen3.5-4B</td><td></td><td>56.7</td><td>57.6</td><td>55.7</td><td>+1.9</td><td>66.5</td><td>46.8</td><td>+19.6</td><td>12.1</td><td>19.0</td><td>12.3</td></tr><tr><td>Qwen3.5-9B</td><td></td><td>72.9</td><td>73.7</td><td>72.1</td><td>+1.6</td><td>74.2</td><td>71.6</td><td>+2.6</td><td>10.4</td><td>8.3</td><td>8.4</td></tr><tr><td>Qwen3.5-27B</td><td></td><td>83.7</td><td>85.4</td><td>81.9</td><td>+3.5</td><td>88.5</td><td>78.9</td><td>+9.6</td><td>10.2</td><td>2.8</td><td>3.3</td></tr><tr><td>Qwen3.5-0.8B</td><td>√</td><td>24.2</td><td>23.2</td><td>25.1</td><td>-1.9</td><td>24.1</td><td>24.3</td><td>-0.3</td><td>26.0</td><td>25.1</td><td>24.7</td></tr><tr><td>Qwen3.5-2B</td><td>√</td><td>60.7</td><td>58.2</td><td>63.0</td><td>-4.8</td><td>60.6</td><td>60.7</td><td>-0.1</td><td>19.0</td><td>12.6</td><td>7.7</td></tr><tr><td>Qwen3.5-4B</td><td>√</td><td>89.9</td><td>89.0</td><td>90.8</td><td>-1.8</td><td>89.9</td><td>90.0</td><td>-0.1</td><td>3.6</td><td>4.3</td><td>2.2</td></tr><tr><td>Qwen3.5-9B</td><td>√</td><td>91.8</td><td>90.8</td><td>92.8</td><td>-1.9</td><td>91.5</td><td>92.2</td><td>-0.7</td><td>2.8</td><td>4.0</td><td>1.4</td></tr><tr><td>Qwen3.5-27B</td><td>√</td><td>97.3</td><td>96.9</td><td>97.7</td><td>-0.8</td><td>97.5</td><td>97.1</td><td>+0.4</td><td>1.1</td><td>1.2</td><td>0.4</td></tr></table>

Table 12: Breakdown of Japanese FrameBench performance by sentence-pair type, position of the correct sentence, and error type. R denotes reasoning mode, and Acc. denotes overall accuracy. Base and Ext. report accuracy on base and extended sentence pairs, while 1st and 2nd report accuracy when the correct sentence appears first or second. $\Delta _ { B - E }$ and $\Delta _ { 1 - 2 }$ indicate performance differences. Both denotes selecting both sentences, Neither denotes selecting neither sentence, and Opposite denotes selecting the incorrect sentence. Acc. and the three errortype rates sum to 100% up to rounding.

## D Analysis on the Japanese Subset

Table 12 reports the behavioral analysis on the Japanese subset of FrameBench, covering sentence-pair type, correct sentence position, and error type. The Japanese results are generally consistent with the English analysis for correct sentence position and error type, but differ in sentence-pair type. In English, extended pairs show the expected advantage over base pairs, with a gap of 2.57 points in human scores. In Japanese, this effect is weaker: the corresponding gap decreases to 0.93 points, suggesting that the surfacelevel contrast between base and extended pairs is less pronounced. Consistent with this weaker contrast, the LLM results in Japanese show a less consistent advantage for extended pairs.