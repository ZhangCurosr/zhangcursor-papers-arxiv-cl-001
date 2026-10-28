# Japanese Discourse Structure Analysis Based on Segmented Discourse Representation Theory Using LLMs

Satomu Tomisaki<sup>1,2</sup>, Yosuke Mikami<sup>1,2</sup>, Amane Watahiki<sup>1</sup>, Hitomi Yanaka<sup>1,2,3</sup>

<sup>1</sup>The University of Tokyo, <sup>2</sup>RIKEN, <sup>3</sup>Tohoku University

{tomisaki-no-mail, ymikami, hyanaka}@is.u-tokyo.ac.jp, amanew@g.ecc.u-tokyo.ac.jp

## Abstract

Discourse parsing based on Segmented Discourse Representation Theory (SDRT) is crucial for capturing the structural dynamics of dialogues. However, most prior research focuses on English, and its effectiveness for Japanese remains unclear owing to the lack of Japanese SDRT corpora. To address this issue, we construct Japanese SDRT corpora by translating existing English SDRT corpora using an LLM. Through a comparative evaluation of promptbased and fine-tuned LLMs, we show that the models achieve comparable performance between English and Japanese, suggesting the cross-lingual applicability of SDRT-based discourse parsing. In order to investigate why the English and Japanese parsers achieve similar performance, we conduct ablation analyses on discourse markers and contextual inputs. Our results reveal that most of the Japanese discourse markers we examined have little impact, and the models’ predictions depend heavily on contextual discourse relations rather than on the content of the context clauses. This structural reliance might explain the comparable performance despite the substantial typological differences between English and Japanese.

## 1 Introduction

Understanding a dialogue requires one to comprehend not only the meaning of individual discourse units but also how they are logically and pragmatically related. Discourse parsing is the task of identifying such relations between individual discourse units. It benefits numerous downstream tasks, such as dialogue summarization (Gao et al., 2023), question answering (He et al., 2021), and AI-generated text detection (Kim et al., 2024).

To formalize these discourse relations, several theoretical frameworks have been developed. Segmented Discourse Representation Theory (SDRT; Asher and Lascarides, 2003) is a prominent example. Compared to other discourse analysis frameworks, SDRT is grounded in formal semantics and can handle multiparty conversations.

However, automating SDRT parsing remains computationally challenging. Recent studies have explored simple prompt-based approaches (Chan et al., 2024; Fan et al., 2024). Zero- or few-shot prompting with models such as ChatGPT has not achieved satisfactory performance, leaving considerable room for improvement. In contrast, models trained from scratch or fine-tuned on task-specific data have achieved high SDRT parsing performance in English (Bennis et al., 2023; Fan et al., 2025; Liu et al., 2025; Thompson et al., 2024a).

Although SDRT is intended to be languageindependent, existing SDRT resources and parsing studies predominantly focus on English. It therefore remains unclear how well existing SDRT parsing methods generalize to other languages. Languages differ in their lexical inventories and syntactic structures and may consequently use different cues to signal discourse relations. We hypothesize that parsers for different languages may rely on different information and achieve different levels of performance. We take Japanese, which differs substantially from English in both lexicon and syntax, as a test case. For example, Japanese sentence-final particles may provide important cues for predicting discourse relations because these particles express the speaker’s attitude toward information (Masuoka and Takubo, 1989). Differences in the use of connectives and sentence-final particles may therefore affect SDRT parsing performance.

However, to the best of our knowledge, no Japanese SDRT corpus is available. This makes it difficult to train Japanese discourse relation parsers. Without these parsers, we cannot study to what extent existing English SDRT parsing methods can be applied to Japanese, and what kind of information is crucial for Japanese discourse analysis.

In order to study the linguistic generalizability of SDRT, we first fill the resource gap. We construct Japanese SDRT corpora by translating existing English SDRT corpora automatically. The translated corpora will be made publicly available for research purposes<sup>1</sup>.

Using these corpora, we examine how well SDRT-based parsing performs across languages. We first evaluate LLM-based SDRT parsers on these translated corpora. We consider two kinds of parsers: fine-tuned ones and prompt-based ones. We then compare the best-performing Japanese and English models.

Finally, we study the information that is crucial for identifying discourse relations. We conduct three ablation analyses. The first one is on two types of Japanese discourse markers: connectives and sentence-final particles. The second one is on the previous discourse relations of the input. The third one is on the content of contextual clauses.

Our results show that parsers perform comparably between English and Japanese. The results of the first ablation show that, while most of the discourse markers we studied contribute little to the prediction of discourse relations, demo (でも; “but”) has a substantial impact on identifying Contrast relations. The results of the second and third ablations show that model predictions primarily depend on the previous discourse relations, instead of the content. The models’ dependence on contextual discourse relations may explain the small performance gap between English and Japanese. Because our Japanese corpora are translations of the English ones, these discourse relations are identical across the two languages, allowing the models to exploit the same information. As a result, they achieve comparable performance in both languages.

## 2 Background

## 2.1 Discourse Analysis Framework

Segmented Discourse Representation Theory (SDRT; Asher and Lascarides, 2003) is a theoretical framework for analyzing discourse structure. In SDRT, a discourse is considered to be a sequence of elementary units. There are two types of elementary units: Elementary Discourse Units (EDUs) and Elementary Event Units (EEUs). Additionally, SDRT defines a compound unit, called a Complex Discourse Unit (CDU), which consists of multiple elementary units. While an EDU refers to a clause in the text or conversation, an EEU refers to a salient nonlinguistic action or event in the surrounding context. The entire discourse can then be represented as a directed acyclic graph (DAG), whose nodes are EDUs, EEUs, or CDUs, and the edges represent discourse relations among those units. Figure 1 presents an example of such a discourse graph.

![](images/d65e0d15e116a3e95b6a23221eef71956967f71ec34d2f564f0f8b1743eb3d66.jpg)  
Figure 1: An extract from MSDC, represented as a discourse graph. EDUs and EEUs are shown in roman and italic fonts, respectively.

Compared with other frameworks, such as Rhetorical Structure Theory (RST; Mann and Thompson, 1988), SDRT has two main advantages. First, discourse relations in SDRT are defined more rigorously based on formal semantics. Second, SDRT can be used to analyze multiparty conversations.

## 2.2 SDRT parsing with LLMs

Like many other areas of computational linguistics, discourse relation parsing is increasingly being tackled with LLMs. In the context of SDRT parsing, approaches can be categorized into two groups: prompt-based methods and fine-tuning methods. Existing prompt-based methods have not reached satisfactory performance; both zeroand few-shot prompting leave substantial room for improvement (Chan et al., 2024; Fan et al., 2024). In contrast, fine-tuning methods achieved higher parsing performance. For example, BERTLine (Bennis et al., 2023) is a fine-tuned BERT. It employs a multi-task architecture that simultaneously learns the attachments between EDUs and EEUs, as well as the discourse relations holding between them. BERTLine outperformed previous neural models. More recently, Liu et al. (2025) and Fan et al. (2025) have proposed methods with complex multi-module architectures, in which each module is built around a fine-tuned LLM. Unlike the multi-module methods, Llamipa is a straightforward method that has a single module but shows competitive performance (Thompson et al., 2024a). In this study, we select Llamipa as our primary fine-tuned parser because of its simplicity and efficiency.

Llamipa takes three inputs: (i) a target clause, (ii) its context, i.e., the preceding clauses within a fixed window, and (iii) the discourse relations among the context clauses. A clause in the context is either an EDU or an EEU. Llamipa uses a window size of 15, since discourse relations in SDRT corpora in general rarely span more than 15 clauses (Thompson et al., 2024a). Llamipa then identifies which context clauses are related to the target clause and determines the discourse relation between each such clause and the target clause.

Llamipa incrementally predicts discourse relations. Initially, Llamipa receives the first clause and therefore an empty set of discourse relations. It then predicts the relation between the second clause and the first clause. At each subsequent step, the previously processed target clause is added to the context, and the next clause becomes the new target. Meanwhile, the discourse relations between the previous target clause and the clauses in the previous context are added to the discourse relation set.

There are two settings for using Llamipa, depending on which kind of relations we use to update the discourse relation set. In the gold structure setting, we use the gold relations. In the predicted structure setting, we use the relations predicted by Llamipa. In the predicted structure setting, Llamipa iterates over the entire discourse and eventually constructs the entire discourse structure.

## 2.3 SDRT Corpus

In this paper, we use two English SDRT corpora: Minecraft Structured Dialogue Corpus (MSDC; Thompson et al., 2024b) and the Strategic Conversation (STAC) corpus (Asher et al., 2016).

MSDC is the Minecraft Dialogue Corpus (MDC; Narayan-Chen et al., 2019) annotated with SDRT-based discourse relations<sup>2</sup>. MDC contains conversations between two players, a “Builder” and an “Architect”, as they collaborate on tasks in the game Minecraft. Specifically, the Architect provides instructions to the Builder. The Builder carries out these instructions and constructs specified structures. Chat messages exchanged between the two players are treated as EDUs, and the descriptions of the Builder’s actual behaviors (e.g., picking up or placing blocks) in the game are treated as EEUs. Figure 1 shows the discourse graph of an extract from MSDC.

STAC is a corpus of chat logs and in-game events from the online version of the board game The Settlers ofCatan, all of which are manually annotated according to SDRT. In The Settlers ofCatan, players aim to accumulate victory points by acquiring and trading resources to construct roads, settlements, and cities. In this context, players use the chat interface to negotiate resource exchanges. In STAC, chat messages are treated as EDUs and game information (e.g., dice rolls, card plays) are treated as EEUs.

## 3 Japanese SDRT Corpus

## 3.1 Corpus Construction

To prepare training data for the Japanese SDRT parser, we translated MSDC and STAC into Japanese using GPT-5.1<sup>3</sup>, a proprietary LLM accessed via the OpenAI API. We translated the corpora clause by clause in order to preserve the original order of the discourse units. Furthermore, in order to ensure context-aware and natural translations, we provided GPT-5.1 with the target clause and its corresponding dialogue session.

Because GPT-5.1 sometimes yielded inconsistent translations for EEUs in MSDC, we implemented a rule-based translation approach for all EEUs to ensure consistency. Specifically, we mapped predefined English keywords to their Japanese equivalents<sup>4</sup>.

In terms of STAC, we followed the methodology of Thompson et al. (2024a) and applied several simplifications. Details are provided in Appendix D.

## 3.2 Dataset Statistics

The training set of MSDC contains 288 dialogues, 12,594 EDUs, and 3,439 EEUs, and the test set contains 97 dialogues, 3,886 EDUs, and 1,048 EEUs. In MSDC, the relation Result is the most frequent, followed by Acknowledgement, Elaboration, and Narration. The training set of STAC contains 1,017 dialogues, 11,380 EDUs, and 2,488 EEUs, and the test set contains 113 dialogues, 1,208 EDUs, and 257 EEUs<sup>5</sup>. In STAC, Question-answer pair is the most frequent relation, followed by Comment, Acknowledgement, and Result.

## 3.3 Quality Analysis

We manually evaluated a sample of the Japanese MSDC corpus, our primary dataset, and automatically evaluated the entire MSDC corpus.

We performed two types of manual evaluation. First, one of the authors, a native Japanese speaker, verified the naturalness of the translations and the consistency of the relation labels after translation. Second, three of the authors, all native Japanese speakers, manually scored each speaker’s turn on a scale from 0 to 100 based on translation quality. We randomly sampled and evaluated 20 dialogue sessions out of all 97 test sessions in the Japanese MSDC. The mean score and standard error for each annotator are shown in Table 1. Inter-annotator agreement, measured by Krippendorff’s α with the interval metric, was 0.132. Although this value is low, limited agreement is a well-known challenge in human evaluation of machine translation (Graham et al., 2013). Examples of translation errors are presented in Appendix B.

<table><tr><td></td><td>Annotator1</td><td>Annotator2</td><td>Annotator3</td></tr><tr><td>Mean</td><td>97.2</td><td>88.3</td><td>88.7</td></tr><tr><td>Standard error</td><td>0.3</td><td>0.2</td><td>0.3</td></tr></table>

Table 1: Mean translation evaluation scores and standard errors for each annotator.

For the automatic evaluation, we used XCOMET-XL (Guerreiro et al., 2024). XCOMET-XL can take the source and target texts and output a quality score ranging from 0.0 to 1.0, where higher scores indicate better translation quality. We calculated scores for both the training and test sets of the Japanese MSDC. The resulting XCOMET-XL scores were 0.84 for the training set, 0.85 for the test set.

## 4 Japanese SDRT Parsers

## 4.1 Experimental Settings

Llamipa We fine-tune two LLMs, Llama-3-8B and Llama-3-Swallow-8B-v0.1 on our Japanese corpora, yielding two Japanese parsers. Regarding the English parsers, we fine-tune the Llama-3-8B model on the original English corpus. The Swallow model is based on Llama-3 and has been continually pre-trained on Japanese text<sup>6</sup>. For the evaluation, we employ a predicted structure setting (cf. Section 2.2).

Prompt-Based Parsers We compare Llamipa against several prompt-based parsers, evaluating each under zero-shot and few-shot (one-shot and four-shot) settings. Our prompt design follows the proposal by Fan et al. (2024). In the original dataset, each dialogue corresponds to the entire process of building one structure in the Minecraft game world. We count one such complete dialogue as a single “shot” and sample these exemplars from the training set. We use GPT-5.5<sup>7</sup> and Llama-3.1-8B-Instruct for English and Japanese experiments, along with Llama-3.1-Swallow-8B-Instruct-v0.3 for the Japanese experiments. Note that for prompt-based parsers, we adopt newer LLMs than those used for Llamipa because the context window of Llama-3 (8K tokens) is too short for these prompts.

Evaluation Metrics We calculate the F1-score for each discourse relation. In order to evaluate the overall performance, we use the weighted average F1-score, where the F1-score for each discourse relation is weighted by its support (i.e., its frequency in the dataset).

## 4.2 Results

We present the performance of Llamipa and prompt-based parsers on the Japanese MSDC in Table 2<sup>8</sup>. Overall, Llamipa achieves the numerically highest weighted average F1-score, which is around 0.79. The weighted average F1-scores of prompt-based parsers do not exceed 0.72.

For individual relations, Llamipa outperforms prompt-based parsers for the majority of the relations. However, prompt-based parsers outperform Llamipa for Conditional and Explanation relations.

<table><tr><td>Model</td><td>Llama-3</td><td>Swallow-3</td><td colspan="3">Llama-3.1</td><td colspan="3">Swallow-3.1</td><td colspan="3">GPT-5.5</td><td rowspan="2"></td></tr><tr><td>Method</td><td>Llamipa</td><td colspan="2"></td><td>Oshot</td><td>1shot 4shot</td><td>Oshot</td><td>1shot</td><td>4shot</td><td>Oshot</td><td>1shot</td><td>4shot</td><td>Supports</td></tr><tr><td>Ackn.</td><td>0.86</td><td>0.85</td><td>0.56</td><td>0.43</td><td>0.43</td><td>0.51</td><td>0.36</td><td>0.15</td><td>0.56</td><td>0.76</td><td>0.78</td><td>792</td></tr><tr><td>Alte.</td><td>0.81</td><td>0.90</td><td>0.00</td><td>0.00</td><td>0.42</td><td>0.00</td><td>0.00</td><td>0.13</td><td>0.76</td><td>0.65</td><td>0.88</td><td>24</td></tr><tr><td>Clari-Q.</td><td>0.67</td><td>0.68</td><td>0.08</td><td>0.24</td><td>0.27</td><td>0.01</td><td>0.19</td><td>0.08</td><td>0.40</td><td>0.58</td><td>0.60</td><td>134</td></tr><tr><td>Comm.</td><td>0.58</td><td>0.58</td><td>0.16</td><td>0.23</td><td>0.12</td><td>0.19</td><td>0.14</td><td>0.03</td><td>0.34</td><td>0.47</td><td>0.49</td><td>382</td></tr><tr><td>Cond.</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.22</td><td>0.22</td><td>0.00</td><td>13</td></tr><tr><td>Conf-Q.</td><td>0.91</td><td>0.90</td><td>0.07</td><td>0.56</td><td>0.56</td><td>0.01</td><td>0.55</td><td>0.14</td><td>0.78</td><td>0.89</td><td>0.90</td><td>159</td></tr><tr><td>Contin.</td><td>0.45</td><td>0.42</td><td>0.04</td><td>0.06</td><td>0.07</td><td>0.06</td><td>0.04</td><td>0.03</td><td>0.11</td><td>0.26</td><td>0.40</td><td>321</td></tr><tr><td>Contr.</td><td>0.76</td><td>0.81</td><td>0.03</td><td>0.10</td><td>0.07</td><td>0.04</td><td>0.10</td><td>0.00</td><td>0.55</td><td>0.77</td><td>0.73</td><td>68</td></tr><tr><td>Corr.</td><td>0.75</td><td>0.74</td><td>0.03</td><td>0.08</td><td>0.07</td><td>0.03</td><td>0.08</td><td>0.01</td><td>0.22</td><td>0.49</td><td>0.60</td><td>403</td></tr><tr><td>Elab.</td><td>0.76</td><td>0.75</td><td>0.32</td><td>0.37</td><td>0.41</td><td>0.30</td><td>0.38</td><td>0.12</td><td>0.55</td><td>0.72</td><td>0.73</td><td>720</td></tr><tr><td>Expl.</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.04</td><td>0.05</td><td>0.01</td><td>0.04</td><td>0.00</td><td>0.10</td><td>0.15</td><td>0.16</td><td>16</td></tr><tr><td>Narr.</td><td>0.80</td><td>0.81</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.17</td><td>0.68</td><td>693</td></tr><tr><td>QAP.</td><td>0.81</td><td>0.82</td><td>0.03</td><td>0.41</td><td>0.46</td><td>0.03</td><td>0.37</td><td>0.11</td><td>0.63</td><td>0.78</td><td>0.79</td><td>290</td></tr><tr><td>QElab.</td><td>0.39</td><td>0.42</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.10</td><td>0.12</td><td>0.17</td><td>38</td></tr><tr><td>Resu.</td><td>0.90</td><td>0.90</td><td>0.35</td><td>0.50</td><td>0.52</td><td>0.30</td><td>0.51</td><td>0.12</td><td>0.51</td><td>0.73</td><td>0.85</td><td>1765</td></tr><tr><td>Sequ.</td><td>0.29</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>6</td></tr><tr><td>Weighted Average</td><td>0.79</td><td>0.79</td><td>0.24</td><td>0.32</td><td>0.33</td><td>0.22</td><td>0.30</td><td>0.09</td><td>0.41</td><td>0.60</td><td>0.72</td><td></td></tr></table>

Table 2: F1-scores for Llamipa and prompt-based parsers on the Japanese MSDC. "Supports" represents the number of instances per relation in the test data, and discourse relations are abbreviated. Bold values indicate the highest F1-scores. The models were evaluated on a subset of gold relations from the MSDC corpus (cf. Appendix E).

These two relations have only a few instances in the training data. This sparsity might have hindered the fine-tuning quality, making it difficult for Llamipa to learn these two relations.

Among the prompt-based parsers, GPT-5.5 benefits from the increasing number of few-shot exemplars. Its performance improves as the number of few-shot exemplars increases. In contrast, Llama-3.1 and Swallow-3.1 do not show the same trend. For both models, the weighted average F1-score under the one-shot setting (0.32 for Llama-3.1 and 0.30 for Swallow-3.1) is higher than that under zero-shot setting (0.24 for Llama-3.1 and 0.22 for Swallow-3.1). Nevertheless, for the four-shot setting, the weighted average F1-scores increase only minimally for Llama-3.1 (0.33), or even decrease substantially for Swallow-3.1 (0.09).

This is likely because adding more exemplars also increases the input length. Compared with GPT-5.5, which has a context window of roughly one million tokens, Llama-3.1 and Swallow-3.1 are both based on architectures that support context windows of up to 128K tokens. Specifically, the publicly released Swallow-3.1 checkpoint is configured with a maximum sequence length of 8,192 tokens, and the prompt length for the four-shot setting exceeds this, although the inputs were not truncated. In contrast, GPT-5.5 might be better able to utilize information across long contexts and thus benefit more from additional exemplars.

We now turn to the best-performing parser, Llamipa, and compare its performance on the English and Japanese corpora. Table 3 shows the SDRT parsing performance of Llamipa on the MSDC and STAC corpora.<sup>9</sup> For both MSDC and STAC, performance on the original English corpus and the translated Japanese corpus is comparable. Across all parsers, the weighted average F1-score is around 0.79 for MSDC and 0.60 for STAC. Notably, the weighted average F1-score of the English Llama-3-based model on STAC is lower than the 0.78 reported by Thompson et al. (2024a). This might be due to differences in the experimental setup, including computational environments and simplification of training data.

## 5 Ablation Tests

We have hypothesized that parsers might achieve different levels of performance. However, as we have shown in the previous section, their performance is comparable. To investigate why, we study which information the models rely on to identify discourse relations. We conduct three ablation analyses. Our first study is about Japanese discourse markers. Japanese discourse markers, such as sentence-final particles, are linguistic expressions that help speakers and listeners understand how utterances are related in discourse. We evaluate the changes in performance before and after removing discourse markers from the input.

<table><tr><td>Dataset</td><td colspan="4">MSDC</td><td colspan="4">STAC</td></tr><tr><td>Language Base model</td><td>EN Llama-3</td><td>JA Llama-3</td><td>JA Swallow-3</td><td>Supports</td><td>EN Llama-3</td><td>JA Llama-3</td><td>JA Swallow-3</td><td>Supports</td></tr><tr><td>Ackn.</td><td>0.86</td><td>0.86</td><td>0.85</td><td>792</td><td>0.61</td><td>0.62</td><td>0.65</td><td>173</td></tr><tr><td>Alte.</td><td>0.90</td><td>0.81</td><td>0.90</td><td>24</td><td>0.67</td><td>0.59</td><td>0.62</td><td>10</td></tr><tr><td>Back.</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td><td>0.00</td><td>0.20</td><td>0.00</td><td>9</td></tr><tr><td>Clari-Q.</td><td>0.69</td><td>0.67</td><td>0.68</td><td>134</td><td>0.50</td><td>0.46</td><td>0.50</td><td>48</td></tr><tr><td>Comm.</td><td>0.58</td><td>0.58</td><td>0.58</td><td>382</td><td>0.60</td><td>0.60</td><td>0.60</td><td>222</td></tr><tr><td>Cond.</td><td>0.00</td><td>0.00</td><td>0.00</td><td>13</td><td>0.53</td><td>0.78</td><td>0.63</td><td>12</td></tr><tr><td>Conf-Q.</td><td>0.93</td><td>0.91</td><td>0.90</td><td>159</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td></tr><tr><td>Contin.</td><td>0.45</td><td>0.45</td><td>0.42</td><td>321</td><td>0.29</td><td>0.25</td><td>0.36</td><td>69</td></tr><tr><td>Contr.</td><td>0.76</td><td>0.76</td><td>0.81</td><td>68</td><td>0.50</td><td>0.43</td><td>0.49</td><td>60</td></tr><tr><td>Corr.</td><td>0.72</td><td>0.75</td><td>0.74</td><td>403</td><td>0.38</td><td>0.24</td><td>0.31</td><td>23</td></tr><tr><td>Elab.</td><td>0.77</td><td>0.76</td><td>0.75</td><td>720</td><td>0.51</td><td>0.46</td><td>0.44</td><td>100</td></tr><tr><td>Expl.</td><td>0.00</td><td>0.00</td><td>0.00</td><td>16</td><td>0.39</td><td>0.37</td><td>0.38</td><td>55</td></tr><tr><td>Narr.</td><td>0.82</td><td>0.80</td><td>0.81</td><td>693</td><td>0.00</td><td>0.00</td><td>0.00</td><td>8</td></tr><tr><td>Para.</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td><td>0.34</td><td>0.29</td><td>0.34</td><td>23</td></tr><tr><td>QAP.</td><td>0.81</td><td>0.81</td><td>0.82</td><td>290</td><td>0.81</td><td>0.81</td><td>0.78</td><td>273</td></tr><tr><td>QElab.</td><td>0.43</td><td>0.39</td><td>0.42</td><td>38</td><td>0.49</td><td>0.53</td><td>0.50</td><td>84</td></tr><tr><td>Resu.</td><td>0.90</td><td>0.90</td><td>0.90</td><td>1765</td><td>0.65</td><td>0.61</td><td>0.61</td><td>145</td></tr><tr><td>Sequ.</td><td>0.25</td><td>0.29</td><td>0.00</td><td>6</td><td>0.78</td><td>0.78</td><td>0.75</td><td>124</td></tr><tr><td>Weighted Average</td><td>0.79</td><td>0.79</td><td>0.79</td><td></td><td>0.60</td><td>0.59</td><td>0.59</td><td></td></tr></table>

Table 3: F1-scores of Llamipa on the MSDC and STAC datasets. "Supports" indicates the number of instances per relation in the test data. Discourse relations are abbreviated. "n/a" indicates that the relation is not included in the respective dataset. Bold values denote the best F1-scores.

Our second and third studies are about two types of information in the input. The input of Llamipa consists of three parts: the context clauses, the discourse relations in the context, and the target clause. The context clauses are largely different in English and Japanese. However, the discourse relations in the context are the same for both languages. A plausible hypothesis for the comparable performance in English and Japanese is that parsers rely more on the discourse relations than on the context clauses. We therefore ablate the discourse relations and the context clauses to verify this hypothesis.

## 5.1 Settings

Discourse Markers We select connectives that appear frequently (at least 40 times) at the beginning of EDUs in the MSDC test data. This yields five connectives: sorekara $( \frac { 7 } { 1 } ) \div n \cdots 5$ ; “and then”), sositara $( \frac { 7 } { 1 } L ) \neq 5$ ; “then”), jaa (じゃあ; “well/then”), dakara $( 1 \div x ) ^ { . } 5$ ; “so/therefore”), and demo $( \mathbb { C } \notin ; { \mathbb { \cdots } } { \mathbb { b u t } ^ { 3 } } )$

In the Japanese MSDC corpus, sorekara frequently co-occurs with the discourse relation Narration. We hypothesize that the parser uses this expression as a strong cue when predicting Narration. Similarly, sositara and jaa often co-occur with Result and Narration, dakara with Elaboration, and demo with Contrast; we hypothesize that these expressions are crucial for predicting the discourse relations with which they most frequently co-occur.

Apart from the five connectives mentioned above, we also include the sentence-final particles yo, ne, and ka according to the established theoretical discussion. The particle yo draws the listener’s attention to information they are presumed not to know (Masuoka and Takubo, 1989). The particle ne confirms information that is assumed to be shared between the speaker and the listener (Masuoka and Takubo, 1989). Based on these functions, we expect yo to be crucial for Correction relations and ne to be crucial for Acknowledgement. The particle ka expresses questions or doubts, so we expect it to be crucial for Clarification-Q, Confirmation-Q, and Question-answer pair. Table 4 presents the statistics of these expressions, relations, and their co-occurrences.

Contextual Discourse Relations For the discourse relations, we simply compare the parsers’ performance with and without discourse relations given. We conduct an ablation analysis by removing them all from the preceding context. Specifically, we provide the input context without any discourse relation information.

<table><tr><td>Expression</td><td>Relation</td><td>Occurrence</td><td>Position</td></tr><tr><td>それから(sorekara)</td><td>Narr.</td><td>36/54</td><td>y</td></tr><tr><td rowspan="2">そしたら(sositara)</td><td>Narr.</td><td>36/40</td><td>y</td></tr><tr><td>Resu.</td><td>35/40</td><td>y</td></tr><tr><td rowspan="2">じやあ(jaa)</td><td>Narr.</td><td>284/379</td><td>y</td></tr><tr><td>Resu.</td><td>316/379</td><td>y</td></tr><tr><td>だから(dakara)</td><td>Elab.</td><td>41/47</td><td>y</td></tr><tr><td>でも(demo)</td><td>Contr.</td><td>46/54</td><td>y</td></tr><tr><td>よ (yo)</td><td>Corr.</td><td>22/222</td><td>y</td></tr><tr><td>ね (ne)</td><td>Ackn.</td><td>145/369</td><td>y</td></tr><tr><td rowspan="3">か(ka)</td><td>Clari-Q.</td><td>24/60</td><td>y</td></tr><tr><td>Conf-Q.</td><td>8/60</td><td>y</td></tr><tr><td>QAP.</td><td>37/60</td><td>x</td></tr></table>

Table 4: The statistics of the ablation study on target expressions. The “Occurrence” column is presented in the format (a/b), where a is the number of times the linguistic expression appears at the position specified in the “Position” column, and b is the total number of clauses in which the expression occurs. In the “Position” column, x and y denote the first and second arguments of the discourse relation, respectively.

Context Clauses In order to investigate model performance when provided with only the discourse relations and the target clause, we mask the content of the context clauses. We replace each context clause with the string [MASKED] while leaving other information unchanged. An example of this process is shown in Figure 2.

For the aforementioned three studies, we use the Llama-3-based Llamipa parser for English and Japanese, and the Llama-3-Swallow-based Llamipa parser for Japanese only. These studies are conducted on the MSDC test set and under the gold structure setting (cf. Section 2.2) in order to keep the contextual discourse relations of the input fixed. Under the predicted structure setting, the input itself could differ before and after each ablation, making it difficult to isolate the effect of the ablated component.

![](images/e6c3b3708c3ad50e938fccf00c177eab69aee371238e1c26d9da10004011bfa8.jpg)  
Figure 2: Depiction of the context clause masking.

<table><tr><td colspan="2">Model</td><td colspan="2">Llama-3-base</td><td colspan="2">Swallow-base</td></tr><tr><td>Expression</td><td>Rel.</td><td>F1</td><td>Pred.</td><td>F1</td><td>Pred.</td></tr><tr><td>それから (sorekara)</td><td>Narr.</td><td>0.89/0.90</td><td>46/31</td><td>0.89/0.90</td><td>43/33</td></tr><tr><td>そしたら (sositara)</td><td>Narr. Resu.</td><td>0.89/0.89 0.91/0.91</td><td>35/33 35/34</td><td>0.89/0.89 0.91/0.91</td><td>36/34 36/35</td></tr><tr><td rowspan="2">じゃあ (jaa)</td><td>Narr.</td><td>0.89/0.89</td><td>282/272</td><td>0.89/0.88</td><td>290/274</td></tr><tr><td>Resu.</td><td>0.91/0.91</td><td>308/300</td><td>0.91/0.91</td><td>313/298</td></tr><tr><td>だから (dakara)</td><td>Elab.</td><td>0.77/0.77</td><td>42/41</td><td>0.76/0.76</td><td>39/39</td></tr><tr><td>でも(demo)</td><td>Contr.</td><td>0.73/0.11</td><td>46/2</td><td>0.77/0.14</td><td>48/2</td></tr><tr><td>よ (yo)</td><td>Corr.</td><td>0.79/0.79</td><td>30/28</td><td>0.81/0.81</td><td>22/20</td></tr><tr><td>ね (ne)</td><td>Ackn.</td><td>0.86/0.85</td><td>145/142</td><td>0.86/0.86</td><td>140/135</td></tr><tr><td rowspan="2">か(ka)</td><td>Clari-Q.</td><td>0.68/0.69</td><td>23/23</td><td>0.68/0.67</td><td>26/24</td></tr><tr><td>Conf-Q.</td><td>0.92/0.92</td><td>5/5</td><td>0.91/0.91</td><td>4/4</td></tr><tr><td rowspan="2"></td><td>QAP.</td><td>0.83/0.83</td><td>35/30</td><td>0.85/0.85</td><td>34/33</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 5: F1-scores and the number of predictions (Pred.) before and after ablation. In the $^ { 6 6 } \mathrm { F } 1 ^ { \prime \prime }$ column, the notation $b / a$ indicates the scores before (b) and after (a) ablation, respectively. Similarly, in the “Pred.” column, $b / a$ denotes that out of b original predictions, a remained unchanged after the target expressions were ablated.

## 5.2 Results

Table 5 presents the results of the ablation tests for the discourse markers (connectives and sentencefinal particles). We compare the F1-scores before and after ablation and measure the number of initial predictions that remained unchanged after the target expressions were removed.

When demo is removed, the F1-score for predicting Contrast decreases from 0.73 to 0.11 for the Llama-3-based model and from 0.77 to 0.14 for the Llama-3-Swallow-based model. Furthermore, of the 46 Contrast predictions in which demo appears in the second argument, only 2 remained unchanged after ablation for the Llama-3-based model (2 of 48 for the Llama-3-Swallow-based model). These results suggest that the constructed Japanese SDRT parsers rely heavily on the presence of demo to identify Contrast.

In contrast, removing sorekara changed a considerable proportion of the Narration predictions: 15 of the 46 predictions changed after ablation for the Llama-3-based model (10 of 43 for the Llama-3- Swallow-based model). Nevertheless, the F1-score for Narration remained almost unchanged (0.89 → 0.90 for both models), possibly because these instances account for only a small fraction of the Narration relations in the test set. This suggests that the models use sorekara as a cue for predicting Narration, although its impact on overall perfor-

<table><tr><td rowspan="4">Model Language</td><td colspan="2">Llama-3-base</td><td>Swallow-base</td></tr><tr><td>EN</td><td>JA</td><td>JA</td></tr><tr><td>Before/After</td><td>Before/After</td><td>Before/After</td></tr><tr><td>0.86/0.71</td><td>0.86/0.70</td><td>0.86/0.74</td></tr><tr><td>Ackn. Alte.</td><td>0.88/0.65</td><td>0.79/0.67</td><td>0.90/0.83</td></tr><tr><td>Clari-Q.</td><td>0.70/0.33</td><td>0.68/0.21</td><td>0.68/0.52</td></tr><tr><td>Comm.</td><td>0.59/0.40</td><td>0.58/0.46</td><td>0.59/0.48</td></tr><tr><td>Cond.</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Conf-Q.</td><td>0.93/0.82</td><td>0.92/0.90</td><td>0.91/0.90</td></tr><tr><td>Contin.</td><td>0.51/0.27</td><td>0.48/0.21</td><td>0.46/0.24</td></tr><tr><td>Contr.</td><td>0.77/0.70</td><td>0.73/0.66</td><td>0.77/0.70</td></tr><tr><td>Corr.</td><td>0.79/0.33</td><td>0.79/0.55</td><td>0.81/0.57</td></tr><tr><td>Elab.</td><td>0.77/0.67</td><td>0.77/0.59</td><td>0.76/0.71</td></tr><tr><td>Expl.</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Narr.</td><td>0.90/0.03</td><td>0.89/0.04</td><td>0.89/0.02</td></tr><tr><td>QAP.</td><td>0.82/0.16</td><td>0.83/0.29</td><td>0.85/0.30</td></tr><tr><td>QElab.</td><td>0.52/0.42</td><td>0.34/0.27</td><td>0.42/0.33</td></tr><tr><td>Resu.</td><td>0.91/0.26</td><td>0.91/0.53</td><td>0.91/0.59</td></tr><tr><td>Sequ.</td><td>0.25/0.22</td><td>0.29/0.10</td><td>0.00/0.16</td></tr><tr><td>Weighted</td><td>0.81/0.38</td><td>0.81/0.47</td><td>0.81/0.52</td></tr></table>

Table 6: F1-scores before and after ablation of input discourse relations. The “Weighted” row indicates the overall performance weighted by relation frequency.

mance is limited.

However, for the other connectives (sositara, jaa, and dakara), and the sentence-final particles (yo, ne, and ka), no substantial changes in performance are observed when predicting their respective discourse relations. This suggests that our hypothesis that these expressions contribute to predicting specific discourse relations was not empirically supported.

Table 6 presents the results of the ablation study on the gold discourse relations among the context clauses. The performance substantially decreases; the weighted average F1-scores drop by at least 0.29 for all parsers. The largest decrease is observed for Narration, which drops by at least 0.85. Notably, the decrease for Contrast in the Japanese models is relatively small. This is consistent with the result of the first ablation, which suggests that the models rely heavily on demo. Since demo appears in the target clause, it remains available to the models even after the contextual discourse relations are removed.

Table 7 presents the performance of the models before and after masking the content of the context clauses. Across all models, we only observe a slight decline in the weighted average. Combined with the results for the ablation on gold discourse relations in input, these results suggest that the parsers primarily rely on the discourse relations in the input rather than on the context clauses, supporting our hypothesis. Thus, the reason why Japanese and English SDRT parsers show comparable performance might be that parsers rely on the discourse structure, which is shared across Japanese and English corpora, rather than on the content of the context clauses.

<table><tr><td rowspan="2">Model Language</td><td colspan="2">Llama-3-base</td><td>Swallow-base</td></tr><tr><td>EN</td><td>JA</td><td>JA</td></tr><tr><td></td><td>Before/After</td><td>Before/After</td><td>Before/After</td></tr><tr><td>Ackn.</td><td>0.86/0.81</td><td>0.86/0.80</td><td>0.86/0.70</td></tr><tr><td>Alte.</td><td>0.88/0.88</td><td>0.79/0.82</td><td>0.90/0.91</td></tr><tr><td>Clari-Q.</td><td>0.70/0.62</td><td>0.68/0.65</td><td>0.68/0.66</td></tr><tr><td>Comm.</td><td>0.59/0.54</td><td>0.58/0.52</td><td>0.59/0.54</td></tr><tr><td>Cond.</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.00</td></tr><tr><td>Conf-Q.</td><td>0.93/0.88</td><td>0.92/0.89</td><td>0.91/0.89</td></tr><tr><td>Contin.</td><td>0.51/0.40</td><td>0.48/0.44</td><td>0.46/0.40</td></tr><tr><td>Contr.</td><td>0.77/0.77</td><td>0.73/0.59</td><td>0.77/0.73</td></tr><tr><td>Corr.</td><td>0.79/0.70</td><td>0.79/0.57</td><td>0.81/0.68</td></tr><tr><td>Elab.</td><td>0.77/0.72</td><td>0.77/0.75</td><td>0.76/0.71</td></tr><tr><td>Expl.</td><td>0.00/0.00</td><td>0.00/0.00</td><td>0.00/0.11</td></tr><tr><td>Narr.</td><td>0.90/0.73</td><td>0.89/0.78</td><td>0.89/0.73</td></tr><tr><td>QAP.</td><td>0.82/0.79</td><td>0.83/0.71</td><td>0.85/0.73</td></tr><tr><td>QElab.</td><td>0.52/0.36</td><td>0.34/0.32</td><td>0.42/0.36</td></tr><tr><td>Resu.</td><td>0.91/0.86</td><td>0.91/0.87</td><td>0.91/0.84</td></tr><tr><td>Sequ.</td><td>0.25/0.15</td><td>0.29/0.09</td><td>0.00/0.00</td></tr><tr><td>Weighted</td><td>0.81/0.75</td><td>0.81/0.75</td><td>0.81/0.74</td></tr></table>

Table 7: F1-scores before and after masking of the content of the context clauses. The “Weighted” indicates the overall performance weighted by relation frequency.

## 6 Conclusion

In this paper, we used Japanese as a test case to investigate the cross-linguistic generalizability of LLM-based Segmented Discourse Representation Theory (SDRT) parsing. We first constructed Japanese corpora by automatically translating the English MSDC and STAC corpora. We then compared the performance of fine-tuning-based and prompt-based parsing approaches on Japanese and English. Our results show that fine-tuned parsers achieve higher SDRT parsing performance than prompt-based parsers and perform comparably across English and Japanese.

In order to explain why English and Japanese models perform comparably, we conducted three ablation studies to analyze the types of information on which the models rely when identifying discourse relations. First, we studied the influence of Japanese connectives and sentence-final particles that frequently co-occur with particular discourse relations in our translated Japanese MSDC corpus. The results show that demo (でも; “but”) particularly influences the prediction of the Contrast relation. However, the impact of the other expressions was not as large as that of demo. Our second and third ablation studies examined the effects of removing the gold discourse relations among the context clauses and of masking the context clauses in the input, respectively. The results suggest that the models rely heavily on the discourse relations provided in the input. This reliance may explain the small performance difference between our English and Japanese SDRT parsers.

Overall, our findings suggest the cross-linguistic generalizability of LLM-based SDRT parsing to Japanese and show that language-specific cues, such as demo, can facilitate the prediction of particular discourse relations. We achieved strong Japanese discourse parsing performance using a simple approach based on clause-by-clause translation and model training, suggesting that generative LLMs offer considerable potential for cross-lingual transfer in discourse parsing.

Building on these results, one promising direction for future work is to investigate the generalizability of LLM-based discourse parsing to languages beyond English and Japanese. Such an investigation is crucial for extending discourse parsing to a wider range of languages. Another worthwhile direction is to conduct more fine-grained analyses that distinguish linguistic expressions providing critical cues for discourse-relation prediction from those that provide little or no useful information. Such analyses may clarify the discourse functions of specific lexical items in Japanese and ultimately improve Japanese discourse parsing performance. Finally, a promising direction for future work is to analyze how LLMs process discourse relations using interpretability methods such as that of Miao and Kan (2025), which would shed light on how LLMs represent and interpret discourse structure.

## Limitations

In this study, we used only MSDC and STAC, which are corpora of chat logs collected during gaming sessions. Moreover, due to time and staffing constraints, we did not manually evaluate the entire Japanese translation corpus. One direction for future research is to expand such resources to include other domains and genres. Another issue regarding corpora is that the Japanese corpora are constructed based on machine translation. Translated Japanese might be unnatural and different from what native Japanese speakers produce. This unnaturalness and difference might introduce confounding factors that affect parser performance. Therefore, constructing a native Japanese SDRT corpus is also necessary as future work.

## Ethical Statement

In this study, we construct and release a Japanese Segmented Discourse Representation Theory (SDRT) corpus by translating existing English corpora (the MSDC and STAC datasets) using a Large Language Model (LLM). This dataset is released strictly for academic and research purposes. In doing so, we carefully consider all licensing and copyright implications. Because our Japanese corpus is derived from the original English datasets, we strictly adhere to their terms of use and licenses. The translated dataset will be distributed under an appropriate license that aligns with the restrictions of the source materials, ensuring that the original authors’ intellectual property is fully respected.

## Acknowledgements

We thank the lab members for their insightful advice and the anonymous reviewers for their helpful comments and feedback. This work was supported by JSPS KAKENHI Grant Number JP24H00809, Japan.

## References

N. Asher and A. Lascarides. 2003. Logics of Conversation. Cambridge University Press.

Nicholas Asher, Julie Hunter, Mathieu Morey, Farah Benamara, and Stergos Afantenos. 2016. Discourse structure and dialogue acts in multiparty dialogue: the STAC corpus. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 2721–2727, Portorož, Slovenia. European Language Resources Association (ELRA).

Zineb Bennis, Julie Hunter, and Nicholas Asher. 2023. A simple but effective model for attachment in discourse parsing with multi-task learning for relation labeling. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 3412–3417, Dubrovnik, Croatia. Association for Computational Linguistics.

Chunkit Chan, Cheng Jiayang, Weiqi Wang, Yuxin Jiang, Tianqing Fang, Xin Liu, and Yangqiu Song. 2024. Exploring the potential of ChatGPT on sentence level relations: A focus on temporal, causal, and discourse relations. In Findings of the Association for Computational Linguistics: EACL 2024,

pages 684–721, St. Julian’s, Malta. Association for Computational Linguistics.

Yaxin Fan, Feng Jiang, Peifeng Li, and Haizhou Li. 2024. Uncovering the potential of ChatGPT for discourse analysis in dialogue: An empirical study. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 16998–17010, Torino, Italia. ELRA and ICCL.

Yaxin Fan, Peifeng Li, and Qiaoming Zhu. 2025. Improving dialogue discourse parsing through discourse-aware utterance clarification. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 18800–18816, Vienna, Austria. Association for Computational Linguistics.

Shen Gao, Xin Cheng, Mingzhe Li, Xiuying Chen, Jinpeng Li, Dongyan Zhao, and Rui Yan. 2023. Dialogue summarization with static-dynamic structure fusion graph. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13858– 13873, Toronto, Canada. Association for Computational Linguistics.

Yvette Graham, Timothy Baldwin, Alistair Moffat, and Justin Zobel. 2013. Continuous measurement scales in human evaluation of machine translation. In Proceedings of the 7th Linguistic Annotation Workshop and Interoperability with Discourse, pages 33–41, Sofia, Bulgaria. Association for Computational Linguistics.

Nuno M. Guerreiro, Ricardo Rei, Daan van Stigt, Luisa Coheur, Pierre Colombo, and André F. T. Martins. 2024. xCOMET: Transparent machine translation evaluation through fine-grained error detection. Transactions of the Association for Computational Linguistics, 12:979–995.

Yuchen He, Zhuosheng Zhang, and Hai Zhao. 2021. Multi-tasking dialogue comprehension with discourse parsing. In Proceedings of the 35th Pacific Asia Conference on Language, Information and Computation, pages 551–561, Shanghai, China. Association for Computational Linguistics.

Zae Myung Kim, Kwang Lee, Preston Zhu, Vipul Raheja, and Dongyeop Kang. 2024. Threads of subtlety: Detecting machine-generated texts through discourse motifs. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5449–5474, Bangkok, Thailand. Association for Computational Linguistics.

Shannan Liu, Peifeng Li, Yaxin Fan, and Qiaoming Zhu. 2025. Enhancing multi-party dialogue discourse parsing with explanation generation. In Proceedings of the 31st International Conference on Computational Linguistics, pages 1531–1544, Abu Dhabi, UAE. Association for Computational Linguistics.

William C. Mann and Sandra A. Thompson. 1988. Rhetorical structure theory: Toward a functional theory of text organization. Text & Talk, 8:243 – 281.

Takashi Masuoka and Yukinori Takubo. 1989. [Basic Japanese Grammar] kiso nihongo bunpou (in Japanese). Kurosio Syuppan (Kurosio Publishers).

Yisong Miao and Min-Yen Kan. 2025. Discursive circuits: How do language models understand discourse relations? In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 32570–32589, Suzhou, China. Association for Computational Linguistics.

Anjali Narayan-Chen, Prashant Jayannavar, and Julia Hockenmaier. 2019. Collaborative dialogue in Minecraft. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5405–5415, Florence, Italy. Association for Computational Linguistics.

Kate Thompson, Akshay Chaturvedi, Julie Hunter, and Nicholas Asher. 2024a. Llamipa: An incremental discourse parser. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 6418–6430, Miami, Florida, USA. Association for Computational Linguistics.

Kate Thompson, Julie Hunter, and Nicholas Asher. 2024b. Discourse structure for the Minecraft corpus. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 4957–4967, Torino, Italia. ELRA and ICCL.

## A Semantics of Discourse Relations used in MSDC and STAC

Below are the relations used in MSDC and STAC, along with illustrative examples of each. In the dataset, relation instances are represented as pairs of indices associated with a specific discourse relation label. The first and second arguments (x, y) represent the source and target of the directed edge in the discourse graph, respectively. In the following examples, the upper clause corresponds to argument x and the lower clause to argument y.

• Acknowledgement(x, y)...y entails that the speaker of y has accepted or achieved the intention of the speaker of x

(1) a. Builder: “place orange 2 1 2, place orange 0 1 3”

b. Architect: “Great!”

• Alternation(x, y)...Either x or y is true.

(2) a. Builder: “so parallel?”

b. Builder: “or perpendicular?”

• Background(x, y)...x provides some stage setting for the event that happens in y, or vice versa.

(3) a. Dave: While we’re talking about trading,

b. Dave: does anyone have sheep?

• Clarification-Q(x, y)...y provides a question about x. It occurs when one does not follow what happened or what is said.

(4) a. Architect: “anyway please place the first purple block on the rightmost red block”

b. Builder: “is it connected?”

• Comment(x, y)...y provides an opinion or evaluation of the content associated with x.

(5) a. Architect: “I am sorry for the incorrect instructions”

b. Builder: “you did a good job!”

• Conditional(x, y)...x is a hypothesis and y is a consequence of the hypothesis.

(6) a. Architect: “so if you take a directly top-down view,”

b. Architect: “it will look like a staircase going up and right”

• Confirmation-Q(x, y)...y is a question about whether x is correct.

(7) a. Builder: “place orange -1 4 1, place orange -1 4 0, place orange -1 4 -1”

b. Builder: “like that?”

• Continuation(x, y)...Both x and y are subject to the same topic.

(8) a. Architect: “you will be removing the middle block of the last row you added.”

b. Architect: “and the middle three blocks of the row youre about to add.”

• Contrast(x, y)...x and y have similar semantic structures, but contrasting themes.

(9) a. Architect: “I have bad news”

b. Architect: “but it’s an easy fix”

• Correction(x, y)...y corrects a part of x.

(10) a. Architect: “and 1 space to the right”

b. Architect: “I meant to say 1 space to the left”

• Elaboration(x, y)...y provides more information about the eventuality introduced in x.

(11) a. Architect: “we’ll now do orange”

b. Architect: “place on orange to the right of the red closest to you”

• Explanation(x, y)...y explains why, or gives the cause of, what happened in x.

(12) a. Builder: “I’ll stay in a fixed position”

b. Builder: “so it’s easier to give me directions with respect to what I’m looking at”

• Narration(x, y)...The main eventualities of x and y occur in the sequence in which they are described.

(13) a. Architect: “then 1 red on top of the right red”

b. Architect: “then another red to the bottom right of the right ground red block.”

• Parallel(x, y)...x and y share a common topic and have similar semantic structures.

(14) a. You can trade for more sheep with Rennoc,

b. You can also trade with me

• Question-answer pair(x, y)...x is a question, and y is an answer to it.

(15) a. Builder: “Whats our mission this time?”

b. Architect: “we have a difficult one ahead of us this time builder.”

• Q-Elaboration(x, y)...y is a question intended to get more information about x.

(16) a. Architect: “we have a difficult one ahead of us this time builder.”

b. Builder: “describe the structure?”

• Result(x, y)...The main eventuality of x is understood to cause the eventuality given by y.

(17) a. Architect: “now connect these red blocks with three more red blocks making a 3x3”

b. Builder: “place red 0 7 0, place red 0 6 0, place red 0 5 0”

• Sequence(x, y)...x and y happen consecutively.

(18) a. Builder: “Mission has started”

b. Builder: “place purple -1 1 3, pick purple -1 1 3”

## B Translation Examples and Error Analysis for MSDC

In this section, we provide examples of translations and several error analyses of MSDC. Table 8 shows examples of the translations. The sentence above is the original English, while the sentence below is the Japanese translation.

Examples of translation errors are presented in Table 9. In each example, the upper segment provides the original English excerpt, while the lower segment presents its corresponding Japanese translation.

The first example is shown in the upper section of the table. In this excerpt, the Builder intends to ask the Architect if any color of block can be placed, saying “any color.” However, the resulting translation fails to reflect the interrogative intent of the original utterance. Therefore, a more appropriate translation would be “どんな色でもいい？” (“Is any color fine?”) rather than “どんな色でも いいよ” (“Any color is fine.”).

The second excerpt is located in the middle section of the table. In this instance, the translation of the preposition “on” is incorrect. The speaker intends for the Builder to place a block to the left of the topmost block. Consequently, the translation $^ { \ast \bullet } \mathbin { \longrightarrow } \mathop { \exists } \mathscr { E } \mathscr { O } \mathbin { \uparrow } V \mathscr { V } \mathscr { V } \mathscr { O } \mathscr { O } \mathscr { T } \mathbin { \ i } \mathscr { D } \mathscr { V } \mathscr { D } \mathscr { O } \mathscr { L } \mathscr { V } ^ { \flat } ,$ (“on top of the top orange block”) is inappropriate. The correct translation should be “一番上のオレンジ $\textcircled { 1 } \because \square \forall \forall \cdots ( \cdots$ “on the top orange block”).

The final example is located in the bottom sec-<sub>tion</sub> <sub>of</sub> <sub>the</sub> <sub>table.</sub> <sub>In</sub> <sub>this</sub> <sub>case,</sub> <sub>the</sub> <sub>phrase</sub> <sub>“</sub>上と下 に” (“on top and bottom”) is redundantly repeated. <sub>T</sub>h<sub>ere</sub>f<sub>ore,</sub> <sub>t</sub>h<sub>e</sub> <sub>p</sub>h<sub>rase</sub> <sub>“</sub>上と下に<sub>”</sub> i<sub>n</sub> <sub>t</sub>h<sub>e</sub> <sub>secon</sub>d clause should be removed to ensure a more natural translation.

## C Detailed Experimental Results

Tables 11 and 12 present the detailed performance metrics (precision, recall, and F1-score) of the constructed SDRT parsers on the MSDC and STAC corpora, respectively. Table 13 details the performance of the prompt-based methods on the MSDC corpus in both English and Japanese.

## D Simplification of STAC

We applied several simplifications to the dataset following the methodology of Thompson et al. (2024a) because Llamipa cannot deal with CDUs directly. First, we flattened CDUs into their constituent EDUs or EEUs. Second, we merged consecutive EEUs into single units. Finally, we removed any EEUs that lacked relations to EDUs.

<table><tr><td>Index</td><td>Speaker</td><td>Text</td></tr><tr><td>2</td><td>Builder</td><td>“lets do this” “やろうか”</td></tr><tr><td>3</td><td>Architect</td><td>“place a red block in front of the blue block&quot; “青いブロックの前に赤いブロ</td></tr><tr><td>4</td><td>Architect</td><td>“along the line toward the purple block&quot; “紫のブロックに向かう線に沿</td></tr><tr><td>5</td><td>Builder</td><td>“place red 41 0”  $ \frac {  } { \mid \overrightarrow { \Xi } \mid } \frac { \scriptscriptstyle { \mathrm { D } } \scriptscriptstyle { \mathrm { D } } \scriptscriptstyle { \mathrm { D } } } { \mid \Xi \mid } \frac {  } { \mathcal { D } \ / \mathrm { D } } 4 \ 1 \ 0 ^ { \circ }$ </td></tr></table>

Table 8: Examples of translation of MSDC. The sentence above is the original English, while the sentence below is the Japanese translation.

<table><tr><td>Speaker</td><td>Text</td></tr><tr><td>Builder Builder</td><td>“any color&quot; “or still orange?&quot;</td></tr><tr><td>Builder Builder</td><td>“どんな色でもいいよ” “それともオレンジのままでいい?”</td></tr><tr><td>Architect Architect</td><td>“on the top orange block&quot; “put a yellow block to the left of it”</td></tr><tr><td>Architect Architect</td><td>“一番上のオレンジのブロックの上に” “その左側に黄色のブロックを置いてくだ  $\yen 1,3$ </td></tr><tr><td>Architect Architect</td><td>“On the top and bottom” “we are going to add 2 block extensions”</td></tr><tr><td>Architect Architect</td><td>“上と下に， ,, “上と下に，ブロックをニつ分突き出させ  $\ddagger { \ddagger } \ '$ </td></tr></table>

Table 9: Examples of translation errors. In each example, the upper segment provides the original English excerpt, and the lower segment presents its corresponding Japanese translation.

## E Evaluation on All Relations of MSDC

In Tables 2 and 3, some gold relations in the original MSDC dataset are excluded from the evaluation following Thompson et al. (2024a). This is because Llamipa does not predict relations whose distance exceeds its context window size. Therefore, such relations are excluded from the training and test data. The evaluation for all relations is presented in Table 14.

## F LLM-translation inconsistencies for EEUs in MSDC

Directly translating EEUs using GPT-5.1 frequently introduces inconsistencies. The following examples illustrate this issue:

(19) a. place green 0 3 -1, place green 0 4 -1

b. place green 0 3 -1, place green 0 4 -1

(20) a. pick red 2 7 5

b. 赤を275から取る

(21) a. pick purple -2 4 1, pick purple 1 4 1b<sub>.</sub> 紫を<sub>-241</sub>を回収、紫を<sub>141</sub>を回収

In each example, the first line (a) represents the original English EEU, and the second line (b) shows the LLM-generated translation. In Example (19), the EEU remains completely untranslated, whereas in Example (20), it is translated into Japanese. Furthermore, the translation in Example (21) is ungrammatical.

Because such inconsistencies could negatively impact the performance of SDRT parsers, we implemented a rule-based translation approach for all EEUs to ensure structural and linguistic consistency. Specifically, “place” was mapped to “設<sup>置</sup>” and “pick” to “<sup>採取</sup>”. The color words “orange,” “blue,” “green,” “yellow,” “red,” and “purple” were <sub>mapped to “</sub>オレンジ<sub>,” “</sub>青<sub>,” “</sub>緑<sub>,” “</sub>黄色<sub>,” “</sub>赤<sub>,”</sub> and “<sup>紫</sup>,” respectively.

## G The prompt used for the translation

This appendix provides the prompts utilized to translate the original MSDC and STAC into Japanese. The prompts below are reproduced as used in our experiments, including typographical errors in the original prompts (e.g., "Ouptput" for "Output" and "as possible as naturally" for "as naturally as possible"). We retain them so that our experiments can be replicated exactly. The following prompt is used for MSDC. Note that although the MSDC prompt mentions translations of the two previous clauses, these were not included in the actual input; the model received only the dialogue session and the target clause, as described in Section 3.1.

```jsonl
{"role": "system",
"content":
11 11 11
## Task
You are given a multi-turn Minecraft
chat log in JSON format,a target
clause,
and your translation of the two previous
clauses of the target clause.
The target clause is one of the clauses
in the chat log.
Translate the clause into Japanese as
possible as naturally.
## INPUT FORMAT
You are given the following JSON
structure as input:
# Context
[{"text": "<clause1>","speaker":
"<speaker1>"},{"text":
"<clause2>","speaker":
"<speaker2>"},...]
# Target clause
{"text": "<target_clause>","speaker":
"<target_speaker>"}
## Rules
- No comments
Translate only the target clause
Ouptput only the translated text
The transcript includes descriptions
of in-game actions. Keep them as
they are.
- For example, if the text is "pick
yellow 0 0 1, place red 1 0 0",
Do not translate the action
commands. Keep them in English
Translate as naturally as possible
while considering the context and
previous translated clauses.
11 11 11
}
{"role": "user", "content":
"#Context\n" + <dialogue session>}
{"role": "user", "content": "# Target
clause\n" + <target clause>}
```

The following is used for STAC.

{"role": "system",   
"content":   
n1 11 1   
## Task   
You are an expert translator   
specializing in board games.   
Your task is to translate logs from the   
online version of the game The   
Settlers of Catan (STAC corpus) into   
natural Japanese.   
You will be provided with a target   
clause and the surrounding dialogue   
context.   
## INPUT FORMAT

```jsonl
# Context
[{"text": "<clause1>","speaker":
"<speaker1>"},{"text":
"<clause2>","speaker":
"<speaker2>"},...]
# Target clause
{"text": "<target_clause>","speaker":
"<target_speaker>"}
## Rules
- Translate only the target clause.
Output only the translated text. Do
not include any explanations or
comments.
Tone & Style:
- If the speaker is a player: Use a
natural, conversational chat
style.
- If the speaker is "Server" or
"UI": These represent system
actions or game events.
Translate them as clear, concise
system messages or objective
descriptions.
Ensure the translation reflects the
strategic intent of the negotiation
if the context involves trading.
- Do not translate player names. Keep
them exactly as they appear in the
original text.
11 11 11
}
{"role": "user", "content":
"#Context\n" + <dialogue session>}
{"role": "user", "content": "# Target
clause\n" + <target clause>}
```

## H Prompts for Prompt-Based Discourse Parsing

This appendix presents the full text of the prompts utilized in our prompt-based experiments. The system prompt consists of general instructions and examples for in-context learning (ICL), while the user prompt contains the target dialogue for discourse parsing. The prompt is reproduced as used in our experiments. Note that it contains errors, including "Segmented Discourse Rhetorical Theory" (for "Segmented Discourse Representation Theory") and "Dialougue" (for "Dialogue"). We retain them because all reported results were obtained with this prompt.

{"role": "system",   
"content":   
According to the Segmented Discourse   
Rhetorical Theory, the rhetorical   
structure of a dialogue can be   
represented by a directed acyclic   
graph, where nodes are utterances or   
actions and edges are the following   
16 relations:   
["Elaboration", "Acknowledgement",   
"Continuation", "Contrast",   
"Correction", "Result", "Comment",   
"Question-answer\_pair",   
"Narration",   
"Clarification\_question",   
"Confirmation\_question", "Q-Elab",   
"Explanation", "Alternation",   
"Conditional", "Sequence"]   
please annotate the rhetorical   
structure of the following dialogue   
and represent it in the form of   
[index1, index2, 'relation'], where   
index1 and index2 are the index of   
two utterances, and the 'relation'   
is one of the above relations to   
connect the two utterances.

Dialougue:   
0: Builder said, Mission has started .   
1: Builder said, hi   
Annotation:   
[0, 1, "Acknowledgement"],   
{"role": "user",   
"content":   
"0: Builder said, Mission has started.   
1: Architect said, We'll be making a   
pair of horseshoes   
2: Architect said, First, start by   
making a 3x3 square of red bricks on   
the ground

## I Hyperparameters

Table 10 details the hyperparameters used for training the SDRT parsers.

<table><tr><td>GPUs</td></tr><tr><td>NVIDIA Hopper H100</td></tr><tr><td>Hyperparameters</td></tr><tr><td>training epochs 3</td></tr><tr><td>batch size 1</td></tr><tr><td>optimizer Adam</td></tr><tr><td>learning rate 2e-4</td></tr><tr><td>linear warm-up and learning rate scheduler cosine annealing</td></tr><tr><td>warm-up ratio 0.03</td></tr><tr><td>gradient clipping 0.3</td></tr><tr><td>LoRA r 64</td></tr><tr><td>LoRA (alpha) 16</td></tr><tr><td>LoRA dropout ratio 0.1</td></tr><tr><td>Only Attention Blocks</td></tr><tr><td>LoRA target modules (q_proj, v_proj)</td></tr><tr><td></td></tr><tr><td>quantization 4-bit NormalFloat</td></tr></table>

Table 10: Hyperparameters for the training.

<table><tr><td></td><td colspan="3">EN (Llama-3-base)</td><td colspan="3">JA (Llama-3-base)</td><td colspan="3">JA (Swallow-base)</td><td></td></tr><tr><td>Relation</td><td>Prec.</td><td>Recall</td><td>F1</td><td>Prec.</td><td>Recall</td><td>F1</td><td>Prec.</td><td>Recall</td><td>F1</td><td>Supports</td></tr><tr><td>Ackn.</td><td>0.86</td><td>0.85</td><td>0.86</td><td>0.86</td><td>0.87</td><td>0.86</td><td>0.85</td><td>0.85</td><td>0.85</td><td>792</td></tr><tr><td>Alte.</td><td>0.85</td><td>0.96</td><td>0.90</td><td>0.70</td><td>0.96</td><td>0.81</td><td>0.85</td><td>0.96</td><td>0.90</td><td>24</td></tr><tr><td>Clari-Q.</td><td>0.67</td><td>0.72</td><td>0.69</td><td>0.66</td><td>0.69</td><td>0.67</td><td>0.66</td><td>0.70</td><td>0.68</td><td>134</td></tr><tr><td>Comm.</td><td>0.62</td><td>0.54</td><td>0.58</td><td>0.63</td><td>0.54</td><td>0.58</td><td>0.65</td><td>0.53</td><td>0.58</td><td>382</td></tr><tr><td>Cond.</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>13</td></tr><tr><td>Conf-Q.</td><td>0.97</td><td>0.89</td><td>0.93</td><td>0.93</td><td>0.88</td><td>0.91</td><td>0.93</td><td>0.88</td><td>0.90</td><td>159</td></tr><tr><td>Contin.</td><td>0.40</td><td>0.52</td><td>0.45</td><td>0.41</td><td>0.51</td><td>0.45</td><td>0.39</td><td>0.45</td><td>0.42</td><td>321</td></tr><tr><td>Contr.</td><td>0.90</td><td>0.66</td><td>0.76</td><td>0.87</td><td>0.68</td><td>0.76</td><td>0.91</td><td>0.74</td><td>0.81</td><td>68</td></tr><tr><td>Corr.</td><td>0.73</td><td>0.71</td><td>0.72</td><td>0.77</td><td>0.72</td><td>0.75</td><td>0.81</td><td>0.69</td><td>0.74</td><td>403</td></tr><tr><td>Elab.</td><td>0.78</td><td>0.76</td><td>0.77</td><td>0.77</td><td>0.76</td><td>0.76</td><td>0.73</td><td>0.78</td><td>0.75</td><td>720</td></tr><tr><td>Expl.</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>16</td></tr><tr><td>Narr.</td><td>0.82</td><td>0.82</td><td>0.82</td><td>0.80</td><td>0.81</td><td>0.80</td><td>0.81</td><td>0.81</td><td>0.81</td><td>693</td></tr><tr><td>QAP.</td><td>0.85</td><td>0.78</td><td>0.81</td><td>0.83</td><td>0.80</td><td>0.81</td><td>0.83</td><td>0.81</td><td>0.82</td><td>290</td></tr><tr><td>QElab.</td><td>0.48</td><td>0.39</td><td>0.43</td><td>0.41</td><td>0.37</td><td>0.39</td><td>0.44</td><td>0.39</td><td>0.42</td><td>38</td></tr><tr><td>Resu.</td><td>0.90</td><td>0.91</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.90</td><td>1765</td></tr><tr><td>Sequ.</td><td>0.50</td><td>0.17</td><td>0.25</td><td>1.00</td><td>0.17</td><td>0.29</td><td>0.00</td><td>0.00</td><td>0.00</td><td>6</td></tr><tr><td>Weighted Average</td><td>0.80</td><td>0.79</td><td>0.79</td><td>0.80</td><td>0.79</td><td>0.79</td><td>0.79</td><td>0.78</td><td>0.79</td><td></td></tr></table>

Table 11: Performance of the constructed SDRT parsers. The “Supports” column indicates the number of instances for each relation in the test data. The Relation labels are shown in abbreviated form. The best F1-scores among the models are highlighted in bold.

<table><tr><td rowspan="2">Relation</td><td colspan="3">EN (Llama-3-base)</td><td colspan="3">JA (Llama-3-base)</td><td colspan="3">JA (Swallow-base)</td><td rowspan="2"></td></tr><tr><td>Prec.</td><td>Recall</td><td>F1</td><td>Prec.</td><td>Recall</td><td>F1</td><td>Prec.</td><td>Recall</td><td>F1 Supports</td></tr><tr><td>Ackn.</td><td>0.64</td><td>0.58</td><td>0.61</td><td>0.63</td><td>0.61</td><td>0.62</td><td>0.65</td><td>0.66</td><td>0.65</td><td>173</td></tr><tr><td>Alte.</td><td>0.75</td><td>0.60</td><td>0.67</td><td>0.71</td><td>0.50</td><td>0.59</td><td>0.83</td><td>0.50</td><td>0.62</td><td>10</td></tr><tr><td>Back.</td><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td><td>0.11</td><td>0.20</td><td>0.00</td><td>0.00</td><td>0.00</td><td>9</td></tr><tr><td>Clari-Q.</td><td>0.44</td><td>0.58</td><td>0.50</td><td>0.39</td><td>0.54</td><td>0.46</td><td>0.46</td><td>0.54</td><td>0.50</td><td>48</td></tr><tr><td>Comm.</td><td>0.58</td><td>0.62</td><td>0.60</td><td>0.58</td><td>0.63</td><td>0.60</td><td>0.58</td><td>0.62</td><td>0.60</td><td>222</td></tr><tr><td>Cond.</td><td>0.71</td><td>0.42</td><td>0.53</td><td>0.82</td><td>0.75</td><td>0.78</td><td>0.86</td><td>0.50</td><td>0.63</td><td>12</td></tr><tr><td>Contin.</td><td>0.27</td><td>0.30</td><td>0.29</td><td>0.23</td><td>0.28</td><td>0.25</td><td>0.34</td><td>0.38</td><td>0.36</td><td>69</td></tr><tr><td>Contr.</td><td>0.53</td><td>0.47</td><td>0.50</td><td>0.47</td><td>0.40</td><td>0.43</td><td>0.52</td><td>0.47</td><td>0.49</td><td>60</td></tr><tr><td>Corr.</td><td>0.50</td><td>0.30</td><td>0.38</td><td>0.36</td><td>0.17</td><td>0.24</td><td>0.38</td><td>0.26</td><td>0.31</td><td>23</td></tr><tr><td>Elab.</td><td>0.50</td><td>0.51</td><td>0.51</td><td>0.48</td><td>0.44</td><td>0.46</td><td>0.43</td><td>0.45</td><td>0.44</td><td>100</td></tr><tr><td>Expl.</td><td>0.40</td><td>0.38</td><td>0.39</td><td>0.36</td><td>0.38</td><td>0.37</td><td>0.39</td><td>0.36</td><td>0.38</td><td>55</td></tr><tr><td>Narr.</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>8</td></tr><tr><td>Para.</td><td>0.50</td><td>0.26</td><td>0.34</td><td>0.45</td><td>0.22</td><td>0.29</td><td>0.50</td><td>0.26</td><td>0.34</td><td>23</td></tr><tr><td>QAP.</td><td>0.80</td><td>0.83</td><td>0.81</td><td>0.82</td><td>0.81</td><td>0.81</td><td>0.77</td><td>0.79</td><td>0.78</td><td>273</td></tr><tr><td>QElab.</td><td>0.58</td><td>0.42</td><td>0.49</td><td>0.58</td><td>0.49</td><td>0.53</td><td>0.54</td><td>0.46</td><td>0.50</td><td>84</td></tr><tr><td>Resu.</td><td>0.67</td><td>0.64</td><td>0.65</td><td>0.63</td><td>0.60</td><td>0.61</td><td>0.65</td><td>0.58</td><td>0.61</td><td>145</td></tr><tr><td>Sequ.</td><td>0.82</td><td>0.75</td><td>0.78</td><td>0.82</td><td>0.75</td><td>0.78</td><td>0.78</td><td>0.73</td><td>0.75</td><td>124</td></tr><tr><td>Weighted Average</td><td>0.62</td><td>0.60</td><td>0.60</td><td>0.61</td><td>0.59</td><td>0.59</td><td>0.60</td><td>0.59</td><td>0.59</td><td></td></tr></table>

Table 12: Performance of the constructed SDRT parsers on STAC. The “Supports” column indicates the number of instances for each relation in the test data. The Relation labels are shown in abbreviated form. The best F1-scores among the models are highlighted in bold.

<table><tr><td></td><td colspan="2">Llama-3</td><td>Swallow-3</td><td colspan="6">Llama-3.1</td><td colspan="4">Swallow-3.1</td><td colspan="5">GPT-5.5</td><td rowspan="2"></td></tr><tr><td></td><td colspan="3">Llamipa</td><td colspan="2">Oshot</td><td colspan="2">1shot</td><td colspan="2">4shot</td><td>Oshot 1shot</td><td colspan="2">4shot</td><td>0shot</td><td colspan="2">1shot</td><td colspan="2">4shot</td><td></td></tr><tr><td>Relation</td><td>EN</td><td>JA</td><td>JA</td><td>EN</td><td>JA</td><td>EN</td><td>JA</td><td>EN</td><td>JA</td><td>JA</td><td>JA</td><td>JA</td><td>EN</td><td>JA</td><td>EN</td><td>JA</td><td>EN</td><td>JA</td><td>Support</td></tr><tr><td>Ackn.</td><td>0.86</td><td>0.86</td><td>0.85</td><td>0.56</td><td>0.56</td><td>0.47</td><td>0.43</td><td>0.40</td><td>0.43</td><td>0.51</td><td>0.36</td><td>0.15</td><td>0.55</td><td>0.56</td><td>0.76</td><td>0.76</td><td>0.78</td><td>0.78</td><td>792</td></tr><tr><td>Alte.</td><td>0.90</td><td>0.81</td><td>0.90</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.32</td><td>0.42</td><td>0.00</td><td>0.00</td><td>0.13</td><td>0.68</td><td>0.76</td><td>0.88</td><td>0.65</td><td>0.88</td><td>0.88</td><td>24</td></tr><tr><td>Clari-Q.</td><td>0.69</td><td>0.67</td><td>0.68</td><td>0.06</td><td>0.08</td><td>0.22</td><td>0.24</td><td>0.29</td><td>0.27</td><td>0.01</td><td>0.19</td><td>0.08</td><td>0.44</td><td>0.40</td><td>0.60</td><td>0.58</td><td>0.56</td><td>0.60</td><td>134</td></tr><tr><td>Comm.</td><td>0.58</td><td>0.58</td><td>0.58</td><td>0.16</td><td>0.16</td><td>0.28</td><td>0.23</td><td>0.25</td><td>0.12</td><td>0.19</td><td>0.14</td><td>0.03</td><td>0.33</td><td>0.34</td><td>0.45</td><td>0.47</td><td>0.52</td><td>0.49</td><td>382</td></tr><tr><td>Cond.</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.20</td><td>0.22</td><td>0.19</td><td>0.22</td><td>0.29</td><td>0.00</td><td>13</td></tr><tr><td>Conf-Q.</td><td>0.93</td><td>0.91</td><td>0.90</td><td>0.04</td><td>0.07</td><td>0.58</td><td>0.56</td><td>0.49</td><td>0.56</td><td>0.01</td><td>0.55</td><td>0.14</td><td>0.76</td><td>0.78</td><td>0.88</td><td>0.89</td><td>0.88</td><td>0.90</td><td>159</td></tr><tr><td>Contin.</td><td>0.45</td><td>0.45</td><td>0.42</td><td>0.07</td><td>0.04</td><td>0.06</td><td>0.06</td><td>0.11</td><td>0.07</td><td>0.06</td><td>0.04</td><td>0.03</td><td>0.15</td><td>0.11</td><td>0.24</td><td>0.26</td><td>0.41</td><td>0.40</td><td>321</td></tr><tr><td>Contr.</td><td>0.76</td><td>0.76</td><td>0.81</td><td>0.07</td><td>0.03</td><td>0.11</td><td>0.10</td><td>0.00</td><td>0.07</td><td>0.04</td><td>0.10</td><td>0.00</td><td>0.52</td><td>0.55</td><td>0.69</td><td>0.77</td><td>0.78</td><td>0.73</td><td>68</td></tr><tr><td>Corr.</td><td>0.72</td><td>0.75</td><td>0.74</td><td>0.02</td><td>0.03</td><td>0.09</td><td>0.08</td><td>0.08</td><td>0.07</td><td>0.03</td><td>0.08</td><td>0.01</td><td>0.26</td><td>0.22</td><td>0.47</td><td>0.49</td><td>0.60</td><td>0.60</td><td>403</td></tr><tr><td>Elab.</td><td>0.77</td><td>0.76</td><td>0.75</td><td>0.35</td><td>0.32</td><td>0.40</td><td>0.37</td><td>0.39</td><td>0.41</td><td>0.30</td><td>0.38</td><td>0.12</td><td>0.59</td><td>0.55</td><td>0.70</td><td>0.72</td><td>0.74</td><td>0.73</td><td>720</td></tr><tr><td>Expl.</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.02</td><td>0.00</td><td>0.04</td><td>0.02</td><td>0.05</td><td>0.01</td><td>0.04</td><td>0.00</td><td>0.17</td><td>0.10</td><td>0.20</td><td>0.15</td><td>0.19</td><td>0.16</td><td>16</td></tr><tr><td>Narr.</td><td>0.82</td><td>0.80</td><td>0.81</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.11</td><td>0.17</td><td>0.68</td><td>0.68</td><td>693</td></tr><tr><td>QAP.</td><td>0.81</td><td>0.81</td><td>0.82</td><td>0.05</td><td>0.03</td><td>0.43</td><td>0.41</td><td>0.42</td><td>0.46</td><td>0.03</td><td>0.37</td><td>0.11</td><td>0.65</td><td>0.63</td><td>0.79</td><td>0.78</td><td>0.81</td><td>0.79</td><td>290</td></tr><tr><td>QElab.</td><td>0.43</td><td>0.39</td><td>0.42</td><td>0.05</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.11</td><td>0.10</td><td>0.05</td><td>0.12</td><td>0.16</td><td>0.17</td><td>38</td></tr><tr><td>Resu.</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.38</td><td>0.35</td><td>0.54</td><td>0.50</td><td>0.50</td><td>0.52</td><td>0.30</td><td>0.51</td><td>0.12</td><td>0.49</td><td>0.51</td><td>0.67</td><td>0.73</td><td>0.85</td><td>0.85</td><td>1765</td></tr><tr><td>Sequ.</td><td>0.25</td><td>0.29</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>6</td></tr><tr><td>Weighted Average</td><td>0.79</td><td>0.79</td><td>0.79</td><td>0.26</td><td>0.24</td><td>0.35</td><td>0.32</td><td>0.33</td><td>0.33</td><td>0.22</td><td>0.30</td><td>0.09</td><td>0.42</td><td>0.41</td><td>0.57</td><td>0.60</td><td>0.73</td><td>0.72</td><td></td></tr></table>

Table 13: F1-scores for Llamipa and prompt-based methods, evaluated on a subset of gold relations from the MSDC corpus (cf. Appendix E). "Support" represents the number of instances per relation in the test data, and discourse relations are abbreviated.

<table><tr><td></td><td colspan="2">Llama-3</td><td>Swallow-3</td><td colspan="6">Llama-3.1</td><td colspan="4">Swallow-3.1</td><td colspan="5">GPT-5.5</td><td rowspan="2"></td></tr><tr><td></td><td colspan="3">Llamipa</td><td colspan="2">0shot</td><td colspan="2">1shot</td><td colspan="2">4shot</td><td>0shot 1shot</td><td colspan="2">4shot</td><td>0shot</td><td colspan="2">1shot</td><td colspan="2">4shot</td><td></td></tr><tr><td>Relation</td><td>EN</td><td>JA</td><td>JA</td><td>EN</td><td>JA</td><td>EN</td><td>JA</td><td>EN</td><td>JA</td><td>JA</td><td>JA</td><td>JA</td><td>EN</td><td>JA</td><td>EN</td><td>JA</td><td>EN</td><td>JA</td><td>Support</td></tr><tr><td>Ackn.</td><td>0.86</td><td>0.86</td><td>0.85</td><td>0.56</td><td>0.56</td><td>0.47</td><td>0.43</td><td>0.40</td><td>0.43</td><td>0.51</td><td>0.36</td><td>0.15</td><td>0.55</td><td>0.56</td><td>0.76</td><td>0.76</td><td>0.78</td><td>0.78</td><td>792</td></tr><tr><td>Alte.</td><td>0.90</td><td>0.81</td><td>0.90</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.32</td><td>0.42</td><td>0.00</td><td>0.00</td><td>0.13</td><td>0.68</td><td>0.76</td><td>0.88</td><td>0.65</td><td>0.88</td><td>0.88</td><td>24</td></tr><tr><td>Clari-Q.</td><td>0.69</td><td>0.67</td><td>0.68</td><td>0.06</td><td>0.08</td><td>0.22</td><td>0.24</td><td>0.29</td><td>0.27</td><td>0.01</td><td>0.19</td><td>0.08</td><td>0.44</td><td>0.40</td><td>0.60</td><td>0.58</td><td>0.56</td><td>0.60</td><td>134</td></tr><tr><td>Comm.</td><td>0.54</td><td>0.54</td><td>0.54</td><td>0.15</td><td>0.14</td><td>0.26</td><td>0.21</td><td>0.23</td><td>0.11</td><td>0.18</td><td>0.13</td><td>0.03</td><td>0.31</td><td>0.32</td><td>0.42</td><td>0.43</td><td>0.48</td><td>0.46</td><td>434</td></tr><tr><td>Cond.</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.17</td><td>0.27</td><td>0.16</td><td>0.18</td><td>0.24</td><td>0.10</td><td>17</td></tr><tr><td>Conf-Q.</td><td>0.93</td><td>0.91</td><td>0.90</td><td>0.04</td><td>0.07</td><td>0.58</td><td>0.56</td><td>0.49</td><td>0.56</td><td>0.01</td><td>0.55</td><td>0.14</td><td>0.76</td><td>0.78</td><td>0.88</td><td>0.89</td><td>0.88</td><td>0.90</td><td>159</td></tr><tr><td>Contin.</td><td>0.45</td><td>0.45</td><td>0.42</td><td>0.07</td><td>0.04</td><td>0.06</td><td>0.06</td><td>0.11</td><td>0.07</td><td>0.06</td><td>0.04</td><td>0.03</td><td>0.15</td><td>0.11</td><td>0.24</td><td>0.26</td><td>0.41</td><td>0.40</td><td>321</td></tr><tr><td>Contr.</td><td>0.76</td><td>0.76</td><td>0.81</td><td>0.07</td><td>0.03</td><td>0.11</td><td>0.10</td><td>0.00</td><td>0.07</td><td>0.04</td><td>0.10</td><td>0.00</td><td>0.52</td><td>0.55</td><td>0.69</td><td>0.77</td><td>0.78</td><td>0.73</td><td>68</td></tr><tr><td>Corr.</td><td>0.72</td><td>0.75</td><td>0.74</td><td>0.02</td><td>0.03</td><td>0.09</td><td>0.08</td><td>0.08</td><td>0.07</td><td>0.03</td><td>0.08</td><td>0.01</td><td>0.26</td><td>0.22</td><td>0.47</td><td>0.49</td><td>0.60</td><td>0.60</td><td>405</td></tr><tr><td>Elab.</td><td>0.77</td><td>0.76</td><td>0.75</td><td>0.35</td><td>0.32</td><td>0.40</td><td>0.37</td><td>0.39</td><td>0.41</td><td>0.30</td><td>0.38</td><td>0.12</td><td>0.59</td><td>0.55</td><td>0.70</td><td>0.72</td><td>0.74</td><td>0.73</td><td>720</td></tr><tr><td>Expl.</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.02</td><td>0.00</td><td>0.04</td><td>0.02</td><td>0.05</td><td>0.01</td><td>0.04</td><td>0.00</td><td>0.17</td><td>0.10</td><td>0.20</td><td>0.15</td><td>0.19</td><td>0.16</td><td>16</td></tr><tr><td>Narr.</td><td>0.79</td><td>0.78</td><td>0.78</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.11</td><td>0.17</td><td>0.68</td><td>0.68</td><td>739</td></tr><tr><td>QAP.</td><td>0.81</td><td>0.81</td><td>0.82</td><td>0.05</td><td>0.03</td><td>0.43</td><td>0.41</td><td>0.42</td><td>0.46</td><td>0.03</td><td>0.37</td><td>0.11</td><td>0.65</td><td>0.63</td><td>0.79</td><td>0.78</td><td>0.81</td><td>0.79</td><td>290</td></tr><tr><td>QElab.</td><td>0.43</td><td>0.39</td><td>0.42</td><td>0.05</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.11</td><td>0.10</td><td>0.05</td><td>0.12</td><td>0.16</td><td>0.17</td><td>38</td></tr><tr><td>Resu.</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.38</td><td>0.35</td><td>0.54</td><td>0.50</td><td>0.50</td><td>0.52</td><td>0.30</td><td>0.51</td><td>0.12</td><td>0.49</td><td>0.51</td><td>0.67</td><td>0.73</td><td>0.85</td><td>0.85</td><td>1765</td></tr><tr><td>Sequ.</td><td>0.25</td><td>0.29</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>6</td></tr><tr><td>Weighted Average</td><td>0.78</td><td>0.78</td><td>0.78</td><td>0.25</td><td>0.24</td><td>0.34</td><td>0.32</td><td>0.32</td><td>0.33</td><td>0.21</td><td>0.30</td><td>0.09</td><td>0.41</td><td>0.41</td><td>0.57</td><td>0.60</td><td>0.72</td><td>0.72</td><td></td></tr></table>

Table 14: F1-scores for Llamipa and prompt-based methods on the MSDC corpus, evaluated across all gold relations. "Support" represents the number of instances per relation in the test data, and discourse relations are abbreviated.