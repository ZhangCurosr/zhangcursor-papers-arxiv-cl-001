# Reading the News: Adapting Large Language Models to Swedish Journalism Through Continued Pre-Training

Lukas Borggren<sup>1,2</sup>, Jenny Kunz<sup>1</sup>, Marco Kuhlmann<sup>1</sup>

<sup>1</sup>Linköping University

<sup>2</sup>Bonnier News

firstname.lastname@liu.se

## Abstract

Large language models are increasingly capable in general, but their utility can remain modest in niche or understudied areas. One approach to address this limitation is to specialise existing models through additional training on target-domain corpora. In this work, we investigate such continued pre-training for adapting large language models to Swedish journalism, using a high-quality dataset that we curate from millions of news articles. To evaluate the adaptation efficacy, we also construct a novel domain-specific benchmark that covers six editorial tasks. Through full and parameterefficient fine-tuning across two model sizes, we find that continued pre-training yields benefits in the target domain, but only when paired with experience replay to mitigate forgetting. We observe consistent enhancements in the models’ generation quality and factual knowledge, but not their proficiency in discriminative tasks. Exploring a training-free method to facilitate instruction following, we see further improvements, but exclusively for models trained with low-rank adaptation. Crucially, we demonstrate the importance of targeted evaluation in the adaptation process, as an existing Swedish benchmark largely fails to capture the models’ indomain performance gains.

## 1 Introduction

News text has historically been near-ubiquitous in natural language processing via resources like the WSJ Penn Treebank (Marcus et al., 1993), Reuters corpora (Lewis et al., 2004), and CNN/Daily Mail (Hermann et al., 2015). In the current paradigm of generative large language models (LLMs), complex problems like fake news detection (Tong et al., 2025) and personalised headline generation (Wan et al., 2026) are studied. However, journalism has rarely been treated as a target domain for LLM adaptation; to the best of our knowledge, merely one study has investigated it (Yao et al., 2024). Other work has created foundational LLMs for journalism, but focuses on pre-training from scratch for high-resource languages (Wang et al., 2023; Wu et al., 2023). Such training demands vast amounts of compute and data, posing a barrier to model development for lower-resource languages.

Continued pre-training (CPT) on specialised corpora is commonly used to adapt general-purpose LLMs to domains like finance (Li et al., 2023), law (Colombo et al., 2024), and biomedicine (Labrak et al., 2024). It is similarly effective for improving models on resource-constrained languages, including the Scandinavian languages Danish (Zhang et al., 2025), Norwegian (Samuel et al., 2025), Icelandic (Gogoulou et al., 2024), and Faroese (Kunz et al., 2026). For Swedish, the remaining major Scandinavian language, CPT has been explored, but with limited methodological details (AI Sweden, 2024), or for relatively small models (Glocker et al., 2025; Kunz, 2026b). Instead, efforts have mainly targeted from-scratch pre-training (Norlund and Stenbom, 2021; Ekgren et al., 2022, 2024; Kalpakchi and Boye, 2023; Silo AI, 2024).

This paper aims to jointly address the understudied adaptation of LLMs to journalism and Swedish. By initially curating a high-quality corpus from millions of Swedish news articles, we can subsequently specialise models via CPT. Addressing the scarcity of evaluation resources, we introduce a domain-specific benchmark covering six editorial tasks to gauge the effectiveness of adaptation. Our results show that CPT increases models’ generation capabilities and parametric knowledge, but hurts their performance on discriminative tasks. Moreover, we find the data-mixture composition to be crucial for successful adaptation, observing negative results when omitting experience replay. In our experiments, low-rank adaptation is competitive with full fine-tuning, and drastically more compatible with instruct vectors to improve instruction following. Notably, an existing Swedish benchmark fails to reflect the broadly positive in-domain effects of CPT, illustrating the need for targeted evaluation. Overall, we demonstrate that CPT is a viable approach for adapting LLMs to Swedish journalism, with the potential to enable publishers to develop and retain control over their editorial tools. We also provide insights and a blueprint for researchers and practitioners seeking to specialise models for other languages and domains.

## 2 Background and Related Work

CPT is a canonical approach to enhancing a model’s knowledge and capabilities in some target area (Gururangan et al., 2020). A core challenge is to adapt a model without largely losing its original abilities, that is, to avoid catastrophic forgetting (Wu et al., 2025).

## 2.1 Data Curation

Experience replay (Chaudhry et al., 2019) is commonly combined with CPT, mitigating forgetting by including a proportion as small as 1% of previously seen examples during training (Scialom et al., 2022). Additionally, replaying data may even enhance target-domain performance (Ibrahim et al., 2024; Parmar et al., 2024; Ke et al., 2025). If a base model’s pre-training distribution is unknown, replay data like code and English text are typically sampled from openly available corpora (Chen et al., 2023; Etxaniz et al., 2024; Dorkin et al., 2026). The choice of such proxy data influences CPT efficacy, with higher-quality sources yielding larger in-domain gains (Wang et al., 2025). Training on a well-curated subset rather than the full dataset can improve results further (Xie et al., 2024; Guo et al., 2025b; Nag et al., 2025). For pre-training in general, deduplicating data increases efficiency, reduces data leakage, causes less memorisation, and can ultimately boost performance (Lee et al., 2022; Penedo et al., 2024).

## 2.2 Constrained Learning

A complementary remedy for catastrophic forgetting is to restrain CPT to avoid excessive overwriting of knowledge (Guo et al., 2025a). This regularising effect can be achieved by architecturally constraining parameter updates (Ke et al., 2022, 2023; Nayak et al., 2026) or carefully configuring learning rate dynamics (Gupta et al., 2023; Parmar et al., 2024). Parameter-efficient finetuning (PEFT) methods such as low-rank adaptation (LoRA) (Hu et al., 2022) may be used to this end (Nag et al., 2025; Pezeshkpour and Hruschka, 2025; Kunz et al., 2026). Functionally restricting updates, LoRA shifts the balance of CPT to reduce forgetting at the expense of learning (Biderman et al., 2024; Elhady et al., 2025), sometimes to the extent of obstructing learning altogether (Tejaswi et al., 2024). Upscaling methods like LLaMA Pro (Wu et al., 2024), SOLAR (Kim et al., 2024), and SCALE (Lee et al., 2026) serve a similar purpose by introducing parameters to expand a model’s width or depth, while keeping its pre-trained backbone frozen during CPT.

## 2.3 Instruction Following

Naively performing CPT can effectively erase a model’s instruction-following capabilities (Jindal et al., 2024). To counteract this, a CPT corpus may be augmented with instruction-like or readingcomprehension data (Shi and Lipani, 2023; Cheng et al., 2024a,b; Rodríguez et al., 2025). Given the scarcity of specialised instruction data, several training-free methods instead combine a CPTadapted model with an instruction-tuned version of its base model, relying on either model merging (Siriwardhana et al., 2024; Ueda et al., 2026) or model editing akin to task vectors (Ilharco et al., 2023). The latter subtracts the original pre-trained model from its instruction-tuned descendant in parameter space and adds the resulting instruct vector to the adapted model (Huang et al., 2024; Lin et al., 2025; Tanwar et al., 2025).

## 2.4 Targeted Evaluation

The effectiveness of CPT is commonly assessed using existing domain-specific benchmarks (Hirano and Imajo, 2024; Ying et al., 2025; Luo et al., 2026). For journalism, although general evaluation frameworks exist (Nishal et al., 2024), previous work relies on either English financial benchmarks (Wu et al., 2023) or Chinese tasks (Wang et al., 2023; Li et al., 2024; Yao et al., 2024). In Swedish, there exist merely three evaluation tasks related to journalism: lead paragraph generation (Monsen and Jönsson, 2021), as well as summarisation and search engine-optimised (SEO) headline generation (Eide, 2024). For Swedish in general, the only existing benchmark collections are the Swedish portion of EuroEval (Nielsen, 2023) and Superlim (Berdicevskis et al., 2023), the latter primarily catered to encoder models.

## 3 Method

Drawing on work reviewed in Section 2, we create a domain-specific corpus and evaluation benchmark, and develop a blueprint for CPT adaptation.

## 3.1 BonCorpus: Compiling Pre-Training Data

We source the pre-training corpus from 21.7M Swedish articles published by Bonnier News publications from 1991 up until 2026. The publications include daily, evening, financial, and local newspapers, as well as lifestyle, sports, and trade magazines. Although individual texts are generally of high quality, the collection as a whole is noisy in several respects. For example, older articles originally stored in now-obsolete formats have been imperfectly migrated over time, leaving artefacts such as links in the plain-text body. Moreover, publications frequently cross-publish content, resulting in many near-duplicate articles. To alleviate these issues, we implement an extensive preprocessing pipeline. For a detailed description, see Appendix A.

First, we parse and aggregate articles to standardise their format across publications. We then apply Unicode normalisation and resolve encoding inconsistencies before using regular expressions to remove extraneous text spans. To discard lowquality articles, we develop a broad set of filters based on text statistics, fastText (Joulin et al., 2017) language-identification scores, and metadata such as tags and article types. Thereafter, we perform exact and fuzzy deduplication, using MinHash LSH (Broder, 1997) to identify candidate pairs for the latter. After calculating and thresholding the Levenshtein distance for each pair, we construct a graph with the retained pairs as edges and their constituent articles as nodes. We compute the graph’s connected components to group similar articles and finally select only the longest one from each group. The full pipeline reduces the document count by nearly a third, yielding BonCorpus: 14.6M articles totalling 8.5B tokens under the tokeniser of our base model family, which is specified in Section 4.

## 3.2 BonEval: Constructing Evaluation Tasks

Due to the dearth of evaluation resources for Swedish journalism, we construct the BonEval benchmark by repurposing editorial content into evaluation tasks. See Appendix B for more details on this process. BonEval comprises six tasks spanning three categories representative of editorial LLM application: generative tasks entailing text production, discriminative tasks targeting information extraction, and knowledge-intensive tasks involving factual accuracy. The six tasks are:

1. Headline: headline generation based on the remainder of the article text.

2. Lead: lead-paragraph generation based on the article body.

3. Summary: bullet-point summarisation of the full article text.

4. Topic: multi-class news topic classification of the full article text. The 19 labels extend the top-level terms in the Media Topics taxonomy (IPTC, 2026).

5. Entity: salient named entity recognition (NER) on the full article text. Unlike standard NER, only persons, organisations, and locations central to the article are targeted.

6. Quiz: multiple-choice question answering based on general-knowledge quizzes. They span 15 categories and range from highly specific Swedish themes to globally known subjects.

For all tasks except Quiz, we source evaluation examples from BonCorpus and subsequently exclude the corresponding articles from the training data. To minimise potential overlap with our base models’ original pre-training data, while still covering major quadrennial events such as elections and sports championships, we only consider articles published from 2022 onward. For Headline, Lead, Topic, and Entities, we use stratified sampling to obtain 8,192 examples per task, approximately balanced across publications. Summary and Quiz each come from a single publication, and using all available data results in 100 and 11,960 examples, respectively.

## 3.3 BonLM: Adapting Models

To create the domain-specialised BonLM, we perform CPT with BonCorpus as the primary training data. However, because the corpus exclusively comprises Swedish editorial content, it introduces a marked distribution shift relative to existing models’ pre-training data. To mitigate forgetting and potentially improve adaptation, we therefore employ experience replay to construct several data mixtures. Specifically, we sample code and English text from Common Corpus, a high-quality collection of uncopyrighted and openly licensed data (Langlais et al., 2026). Although editorial text in pre-training corpora can improve general languagemodelling performance (de la Rosa et al., 2025), it is unknown whether general-domain text can improve performance on editorial tasks. To investigate this, we also sample Swedish text from Common Corpus for our mixtures. Omitting the article-specific parsing and cleaning steps, we apply our pre-processing pipeline to all replay data.

We curate the data mixtures by diluting BonCorpus with every possible combination of the three replay types. Guided by prior work (Fujii et al., 2024; Messmer et al., 2025; Samuel et al., 2025), we sample each included type to constitute 10% of the tokens in the resulting mixture. This yields eight mixtures of varying sizes, labelled by concatenating abbreviations for their constituents: BonCorpus (B), code (C), English (E), and Swedish (S). The full BonCorpus accounts for 100% of the tokens in the smallest mixture, B, and 70% in the largest, BCES. We also study the impact of PEFT methods on CPT by comparing LoRA and LLaMA Pro against full fine-tuning (FFT). Finally, we explore the effects of adding an instruct vector (IV) to each model.

## 4 Experimental Setup

Following Chen et al. (2025), we conduct initial experiments with smaller models to derive a training recipe. As starting points for CPT, we use the 3B and 8B base models from Mistral’s Ministral 3 family (Liu et al., 2026). We defer details of our training setup to Appendix C.

## 4.1 Evaluation

We assess the effectiveness of CPT with BonEval. To validate our benchmark, we compare its results to those obtained on three existing tasks. These supplementary tasks are similar to ours but smaller in scale: article summarisation and SEO headline generation (Eide, 2024), as well as question answering about Sweden-related facts (Kunz, 2026a). Additionally, we evaluate models on seven Swedish tasks from EuroEval: SweReC for sentiment classification, SUC 3.0 for NER, ScaLA for linguistic acceptability, MultiWikiQA for reading comprehension, SweDN for summarisation, MMLU for knowledge, and HellaSwag for common-sense reasoning (EuroEval, 2026). Among these, SweDN partially overlaps with BonCorpus. For comparability, we implement BonEval following Euro-

Eval’s methodology, using few-shot prompting, multiple evaluation rounds, and the same primary metrics (Nielsen et al., 2025). Specifically, we evaluate Headline, Lead, and Summary with CHRF3++ (Popovic´, 2017); Topic and Quiz with Matthews correlation coefficient (MCC); and Entity with micro-averaged $\mathrm { F _ { 1 } }$ score. See Appendix D for implementation details.

## 4.2 Initial Experiments

For each of the eight data mixtures, we run CPT on Ministral 3 3B for 8,000 steps, approximating one full epoch over B; the other mixtures are larger and thus subsampled. Based on the average BonEval score across tasks, we select the best-performing mixture for subsequent experiments. Thereafter, we perform CPT with LoRA and LLaMA Pro, configuring each method to approximately render equal numbers of trainable parameters. For the resulting models, we examine the effect of adding unweighted IVs. Since LLaMA Pro alters the model architecture, there is no straightforward way to apply IV, and preliminary experimentation yielded poor outcomes. We also explored adding analogously computed reasoning vectors, but they generally underperformed their instruct counterparts.

## 4.3 Final Training

To obtain BonLM 8B, we run CPT on Ministral 3 8B using the best recipe from the initial experiments. We set the number of training steps to the equivalent of four epochs, a duration shown to be effective for CPT in prior work (Muennighoff et al., 2023; Xue et al., 2023; Guo et al.,

<table><tr><td>Model</td><td>Average</td></tr><tr><td>Ministral 3 3B Base</td><td>39.73</td></tr><tr><td>+ BonCorpus (B)</td><td>36.98</td></tr><tr><td>+ Code (BC)</td><td>39.99</td></tr><tr><td>+ English (BE)</td><td>40.01</td></tr><tr><td>+ Swedish (BS)</td><td>40.06</td></tr><tr><td>+ Code + English (BCE)</td><td>40.01</td></tr><tr><td>+ Code + Swedish (BCS)</td><td>41.34</td></tr><tr><td>+ English + Swedish (BES)</td><td>39.02</td></tr><tr><td>+ Code + English + Swedish (BCES)</td><td>41.48</td></tr><tr><td>LoRA LLaMA Pro</td><td>41.32</td></tr><tr><td></td><td>41.17</td></tr><tr><td>Full fine-tuning (FFT) + instruct vector (IV) LoRA + IV</td><td>39.07 42.27</td></tr></table>

Table 1: Average BonEval scores from 3B experiments. The best mixture (BCES) is used for the results in the last four rows. Underline denotes section-best and bold overall best. See full results in Table 11, Appendix E.

<table><tr><td>Model</td><td>SweReC</td><td>SUC 3.0</td><td>ScaLA</td><td>MultiWikiQA</td><td>SweDN</td><td>MMLU</td><td>HellaSwag</td><td>Average</td></tr><tr><td>GPT-SW3 6.7B</td><td>10.44</td><td>26.72</td><td>9.84</td><td>69.47</td><td>30.62</td><td>2.92</td><td>2.05</td><td>21.72</td></tr><tr><td>Llama SW3 8B</td><td>79.90</td><td>36.02</td><td>8.81</td><td>72.22</td><td>32.23</td><td>17.02</td><td>5.64</td><td>35.98</td></tr><tr><td>Apertus 8B</td><td>79.11</td><td>45.74</td><td>34.34</td><td>74.05</td><td>33.39</td><td>41.54</td><td>32.54</td><td>48.67</td></tr><tr><td colspan="9">Ministral 3 8B</td></tr><tr><td>Base</td><td>80.19</td><td>66.44</td><td>51.70</td><td>75.71</td><td>35.30</td><td>58.10</td><td>43.68</td><td>58.73</td></tr><tr><td>Instruct</td><td>77.53</td><td>60.51</td><td>50.62</td><td>74.28</td><td>35.31</td><td>56.22</td><td>57.27</td><td>58.82</td></tr><tr><td>Reasoning</td><td>77.05</td><td>65.21</td><td>47.77</td><td>79.44</td><td>31.16</td><td>49.96</td><td>64.27</td><td>59.26</td></tr><tr><td colspan="9">BonLM 8B</td></tr><tr><td>FFT</td><td>80.01</td><td>50.64</td><td>39.61</td><td>70.53</td><td>31.94</td><td>30.62</td><td>17.66</td><td>45.86</td></tr><tr><td> $\mathrm { F F T } + \mathrm { I V }$ </td><td>79.07</td><td>45.84</td><td>22.54</td><td>70.04</td><td>36.03</td><td>20.09</td><td>12.48</td><td>40.87</td></tr><tr><td>LoRA</td><td>80.55</td><td>66.86</td><td>59.85</td><td>69.25</td><td>35.03</td><td>53.34</td><td>36.82</td><td>57.39</td></tr><tr><td> $_ { \mathrm { L o R A + I V } }$ </td><td>78.27</td><td>62.27</td><td>54.24</td><td>70.05</td><td>36.27</td><td>47.18</td><td>52.06</td><td>57.19</td></tr></table>

Table 2: Results for BonLM 8B and baselines on Swedish EuroEval tasks. Micro-averaged $\mathrm { F _ { 1 } }$ is reported for SUC 3.0, token-averaged $\mathrm { F _ { 1 } }$ for MultiWikiQA, and CHRF3++ for SweDN. For all other tasks, MCC is reported. Underline denotes section-best and bold overall best.

2025b). Thereafter, we evaluate the resulting model on all tasks and compare it with similarly sized baselines: the base, instruct, and reasoning versions of Ministral 3 8B; GPT-SW3 6.7B, pre-trained from scratch on Scandinavian-language data (Ekgren et al., 2024); Llama SW3 8B, adapted to the Scandinavian languages via CPT (AI Sweden, 2024); and Apertus 8B, pre-trained on multilingual data (Hernández-Cano et al., 2026).

## 5 Results and Analysis

The main experimental results are presented in Tables 1, 2, and 3, with additional details provided in Appendix E.

## 5.1 Impact of Data-Mixture Composition

Table 1 reports average BonEval scores from the initial 3B experiments. Overall, CPT improves upon the base model for all mixtures except B and BES. B exhibits the greatest degradation, likely reflecting catastrophic forgetting caused by the mixture’s homogeneous, BonCorpus-only composition. Conversely, BCES includes all text types and achieves the highest score, narrowly outperforming BCS. These observations corroborate the findings reviewed in Section 2.1 that replaying code and English text enhances target-domain performance. Furthermore, the results suggest that replaying general-domain text benefits performance on editorial tasks. The task-level breakdown in Table 11, Appendix E, show that adding Swedish text to any mixture consistently raises scores on Lead and especially Quiz. We surmise that the gain on Quiz arises because Common Corpus includes knowledge-dense sources such as Swedish Wikipedia, which cover detailed facts that complement those typically found in news articles. For the remaining experiments, we perform CPT with the BCES mixture.

## 5.2 PEFT and Instruct-Vector Compatibility

Among the PEFT methods in Table 1, LoRA slightly outperforms LLaMA Pro despite having fewer trainable parameters, as detailed in Appendix C.1. Although this result diverges from the findings of Wu et al. (2024), later work has reported better target-domain adaptation with LoRA than with LLaMA Pro (Lee et al., 2026). Adding an IV to the LoRA model further improves performance, yielding the highest average BonEval score. By contrast, adding an IV to the FFT model impairs performance, contrary to prior work described in Section 2.3. To examine LoRA’s regularising effect, we compute the mean relative Euclidean distance between the adapted and base-model parameters, obtaining 0.08 for LoRA and 0.18 for FFT. We hypothesise that this smaller drift partly explains LoRA’s compatibility with IV, as the resulting model remains closer in parameter space to the model that originally underwent instruction tuning. Why IV is counterproductive for the FFT model remains unclear. Because prior studies have focused primarily on Llama and Qwen models, IV compatibility may vary across model families, but further investigation is needed to confirm this. Given that FFT outperforms PEFT without IV, and LoRA with IV performs best overall, we train one BonLM 8B variant using each method.

<table><tr><td rowspan="2">Model</td><td colspan="7">BonEval</td><td colspan="3">Supplementary</td></tr><tr><td>Headline</td><td>Lead</td><td>Summary</td><td>Topic</td><td>Entity</td><td>Quiz</td><td>Average</td><td>SvD</td><td>AB</td><td>Facts</td></tr><tr><td>GPT-SW3 6.7B</td><td>17.70</td><td>22.78</td><td>36.11</td><td>0.49</td><td>27.50</td><td>6.00</td><td>18.43</td><td>15.96</td><td>33.87</td><td>1.69</td></tr><tr><td>Llama SW3 8B</td><td>19.35</td><td>29.13</td><td>40.51</td><td>68.28</td><td>26.17</td><td>61.30</td><td>40.79</td><td>18.57</td><td>36.17</td><td>23.54</td></tr><tr><td>Apertus 8B</td><td>18.61</td><td>23.09</td><td>40.93</td><td>70.95</td><td>22.81</td><td>61.05</td><td>39.57</td><td>21.09</td><td>40.07</td><td>18.38</td></tr><tr><td colspan="9">Ministral 3 8B</td><td></td><td></td></tr><tr><td>Base</td><td>18.35</td><td>25.79</td><td>42.49</td><td>72.84</td><td>29.97</td><td>60.12</td><td>41.59</td><td>20.08</td><td>41.88</td><td>20.62</td></tr><tr><td>Instruct</td><td>29.28</td><td>32.03</td><td>46.27</td><td>71.93</td><td>23.20</td><td>57.76</td><td>43.41</td><td>30.41</td><td>45.10</td><td>25.21</td></tr><tr><td>Reasoning</td><td>21.36</td><td>27.68</td><td>47.93</td><td>71.06</td><td>20.67</td><td>52.91</td><td>40.27</td><td>11.86</td><td>42.86</td><td>19.27</td></tr><tr><td colspan="9">BonLM 8B</td><td></td><td></td></tr><tr><td>FFT</td><td>21.58</td><td>29.88</td><td>43.76</td><td>72.40</td><td>28.34</td><td>68.86</td><td>44.14</td><td>21.81</td><td>39.80</td><td>40.82</td></tr><tr><td>FFT + IV</td><td>24.32</td><td>31.24</td><td>43.26</td><td>42.28</td><td>28.38</td><td>61.98</td><td>38.58</td><td>22.79</td><td>40.29</td><td>32.41</td></tr><tr><td>LoRA</td><td>21.29</td><td>31.07</td><td>44.35</td><td>68.93</td><td>29.81</td><td>71.36</td><td>44.47</td><td>23.85</td><td>43.60</td><td>42.24</td></tr><tr><td> $_ { \mathrm { L o R A + I V } }$ </td><td>29.76</td><td>35.01</td><td>49.88</td><td>70.54</td><td>23.99</td><td>66.95</td><td>46.02</td><td>30.14</td><td>47.18</td><td>38.56</td></tr></table>

Table 3: Results for BonLM 8B and baselines on BonEval and supplementary tasks Svenska Dagbladet (SvD) SEO Headline, Aftonbladet (AB) Summary, and SwedishFacts. Micro-averaged $\mathrm { F _ { 1 } }$ is reported for Entity, and MCC for Topic, Quiz, and SwedishFacts. For all other tasks, CHRF3++ is reported.

## 5.3 Mixed Results on EuroEval

The results for BonLM 8B and the baselines on the Swedish EuroEval tasks are shown in Table 2. Ministral Base outperforms the three other baselines across all tasks. It also surpasses its post-trained descendants on all tasks except reading comprehension, summarisation, and common-sense reasoning. On average, both BonLMs underperform their base model, plausibly due to CPT-induced catastrophic forgetting. Similarly, LoRA’s substantial outperformance of FFT may stem from the method’s regularising effect described in Section 2.2. For individual tasks, CPT causes drastic regressions on the LLM-translated MMLU and HellaSwag, and a more modest decline on the LLM-synthesised MultiWikiQA. Contrastively, BonLM LoRA narrowly beats the base model on the remaining tasks, all of which are natively Swedish.

## 5.4 Improved Performance on BonEval

Table 3 displays the results on BonEval and the supplementary tasks for BonLM 8B and the baselines. Again, Ministral Base generally outperforms the other baselines, although Llama SW3 and Apertus achieve higher scores on Headline and Quiz. BonLM FFT and LoRA both outperform all baselines on average and on most individual tasks. Unlike in the initial 3B experiments and in prior work (Biderman et al., 2024; Tejaswi et al., 2024), LoRA is generally superior to FFT. The effect of IV mirrors that observed for the 3B models, and LoRA + IV yields the highest average score. At the task level, CPT consistently improves the generative Headline, Lead, and Summary but slightly degrades the discriminative Topic and Entity. We conjecture that classification and NER are comparatively domain-agnostic, as there may be no uniquely journalistic way to perform them. If so, retaining the base model’s general capabilities likely benefits performance on these tasks. As with the 3B models, Quiz exhibits the largest absolute gains from CPT. This suggests that news articles contain culturally rich information and that LLMs can be imbued with such knowledge through CPT.

The supplementary-task results support the validity of BonEval, with CPT improving the base model’s generative abilities and factual knowledge. A minor discrepancy is Ministral Instruct’s strong performance on SEO headline generation. More broadly, Table 3 reveals a consistent trend: instruction tuning increases generation quality. Across all five generative tasks, Instruct outperforms Base, and LoRA + IV surpasses LoRA. The modest results for Reasoning are expected since BonEval does not target reasoning capabilities. Nevertheless, prolonged post-training of Ministral notably reduces scores on Topic, Entity, and Quiz.

## 5.5 Value of Targeted Evaluation

In summary, CPT improves overall performance on BonEval but degrades it on EuroEval. We partly attribute this disparity to BonLM acquiring domainspecific capabilities at the expense of general ones. For example, CPT on predominantly editorial content unsurprisingly decreases the models’ commonsense reasoning performance, as measured by HellaSwag. However, machine-translated benchmarks can exhibit cultural biases (Singh et al., 2025), overestimate target-language performance (Kuulmets and Fishel, 2023), and miss nuances captured by native evaluations (Chen et al., 2024). This raises the possibility that the LLM-synthesised portion of EuroEval does not accurately reflect a model’s proficiency in Swedish. On knowledge tasks specifically, CPT drives substantial gains on the native Quiz and SwedishFacts but causes regressions on the translated MMLU. Importantly, EuroEval does not fully capture the in-domain benefits of CPT evident in BonEval, such as stronger generative capabilities and greater factual knowledge. This underscores the importance of language- and domain-specific benchmarks for guiding model adaptation.

## 6 Conclusion and Future Work

This paper presents the first comprehensive study of adapting LLMs to Swedish journalism. By detailing the curation of BonCorpus, which spans 35 years of editorial content, we highlight the challenges involved in processing ostensibly clean articles. We further introduce BonEval, a domainspecific benchmark comprising six editorial tasks. Our results reaffirm that CPT is effective for domain adaptation, but only when coupled with carefully constructed experience-replay mixtures. Across two model sizes, CPT consistently improves performance on generative and knowledgeintensive tasks but not on discriminative tasks. We also find that IV compatibility depends on the adaptation method: IV generally benefits models trained with LoRA but not those trained with FFT. Crucially, our findings demonstrate the importance of integrating targeted evaluation into the adaptation process, as EuroEval’s Swedish tasks fail to reflect the broader in-domain trends revealed by BonEval.

We conclude that journalism is a promising domain for LLM research and application. In future work, we plan to investigate the downstream impact of CPT by fine-tuning BonLM for editorial use cases such as headline generation and stylistic correction. We also recognise the limitations of automatic evaluation and aggregate metrics. Going forward, we intend to collaborate with journalists to design human-evaluation protocols that better assess the real-world utility of LLMs in newsrooms.

## Limitations

For business and legal reasons, we cannot release proprietary data, code, or models, which reduces the reproducibility of our study. Computational constraints also restrict our adaptation experiments to a single model family, limiting the generalisability of our findings. Because Ministral already has relatively strong Swedish-language capabilities, experiments with another model family could yield different conclusions. Moreover, because the original pre-training data are undisclosed, we cannot assess potential overlap with BonCorpus or BonEval, nor its effects on our results. We also provide limited insight into the downstream effects of CPT since we do not fine-tune BonLM for specific editorial applications. Finally, our exclusive reliance on automatic evaluation precludes any assertive claims about real-world utility.

## Ethical Considerations

Because our research artefacts will not be released, the general risk of misuse by external actors is limited. However, BonLM is intended for internal deployment as part of Bonnier News’ editorial tools. It would be subject to strict usage policies but nonetheless affect the work of reporters and editors and, ultimately, the news presented to the public. Although adapting existing LLMs is more efficient than training them from scratch, it still requires substantial computational resources, as detailed in Appendix C. Yet we view a strong base model as an investment that can improve computational efficiency in future work. Finally, copyright is an existential concern for publishers, and we have carefully considered it throughout the data-sourcing process.

## Acknowledgments

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation (KAW), and by TrustLLM funded by Horizon Europe GA 101135671. Computational resources were provided on the Berzelius system funded by KAW and operated by the National Academic Infrastructure for Supercomputing in Sweden (NAISS). We are grateful to Hans Hjelm at Bonnier News for his support and guidance throughout the project.

## References

AI Sweden. 2024. AI-Sweden-Models/Llama-3-8B. Model repository. Accessed: 2026-06-16.

Aleksandrs Berdicevskis, Gerlof Bouma, Robin Kurtz, Felix Morger, Joey Öhman, Yvonne Adesam, Lars Borin, Dana Dannélls, Markus Forsberg, Tim Isbister, Anna Lindahl, Martin Malmsten, Faton Rekathati, Magnus Sahlgren, Elena Volodina, Love Börjeson, Simon Hengchen, and Nina Tahmasebi. 2023. Superlim: A Swedish Language Understanding Evaluation Benchmark. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 8137–8153, Singapore. Association for Computational Linguistics.

Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, and John Patrick Cunningham. 2024. LoRA Learns Less and Forgets Less. Transactions on Machine Learning Research.

Andrei Z. Broder. 1997. On the resemblance and containment of documents. In Proceedings. Compression and Complexity of SEQUENCES 1997, pages 21–29.

Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K. Dokania, Philip H. S. Torr, and Marc’Aurelio Ranzato. 2019. On Tiny Episodic Memories in Continual Learning. Computing Research Repository, arXiv:1902.10486. Version 4.

Jie Chen, Zhipeng Chen, Jiapeng Wang, Kun Zhou, Yutao Zhu, Jinhao Jiang, Yingqian Min, Wayne Xin Zhao, Zhicheng Dou, Jiaxin Mao, Yankai Lin, Ruihua Song, Jun Xu, Xu Chen, Rui Yan, Zhewei Wei, Di Hu, Wenbing Huang, and Ji-Rong Wen. 2025. Towards Effective and Efficient Continual Pretraining of Large Language Models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5779–5795, Vienna, Austria. Association for Computational Linguistics.

Pinzhen Chen, Simon Yu, Zhicheng Guo, and Barry Haddow. 2024. Is It Good Data for Multilingual Instruction Tuning or Just Bad Multilingual Evaluation for Large Language Models? In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 9706–9726, Miami, Florida, United States. Association for Computational Linguistics.

Zeming Chen, Alejandro Hernández Cano, Angelika Romanou, Antoine Bonnet, Kyle Matoba, Francesco Salvi, Matteo Pagliardini, Simin Fan, Andreas Köpf, Amirkeivan Mohtashami, Alexandre Sallinen, Alireza Sakhaeirad, Vinitra Swamy, Igor Krawczuk, Deniz Bayazit, Axel Marmet, Syrielle Montariol, Mary-Anne Hartley, Martin Jaggi, and Antoine Bosselut. 2023. MEDITRON-70B: Scaling Medical Pretraining for Large Language Models. Computing Research Repository, arXiv:2311.16079. Version 1.

Daixuan Cheng, Yuxian Gu, Shaohan Huang, Junyu Bi, Minlie Huang, and Furu Wei. 2024a. Instruction Pre-Training: Language Models are Supervised Multitask Learners. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 2529–2550, Miami, Florida, United States. Association for Computational Linguistics.

Daixuan Cheng, Shaohan Huang, and Furu Wei. 2024b. Adapting Large Language Models via Reading Comprehension. In Twelfth International Conference on Learning Representations.

Pierre Colombo, Telmo Pires, Malik Boudiaf, Rui Melo, Dominic Culver, Etienne Malaboeuf, Gabriel Hautreux, Johanne Charpentier, and Michael Desa. 2024. SaulLM-54B & SaulLM-141B: Scaling Up Domain Adaptation for the Legal Domain. In Advances in Neural Information Processing Systems, volume 37, pages 129672–129695.

Javier de la Rosa, Vladislav Mikhailov, Lemei Zhang, Freddy Wetjen, David Samuel, Peng Liu, Rolv-Arild Braaten, Petter Mæhlum, Magnus Breder Birkenes, Andrey Kutuzov, Tita Enstad, Hans Christian Farsethås, Svein Arne Brygfjeld, Jon Atle Gulla, Stephan Oepen, Erik Velldal, Wilfred Østgulen, Lilja Øvrelid, and Aslak Sira Myhre. 2025. The Impact of Copyrighted Material on Large Language Models: A Norwegian Perspective. In Proceedings ofthe Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies, pages 544–560, Tallinn, Estonia. University of Tartu Library.

Hantian Ding, Zijian Wang, Giovanni Paolini, Varun Kumar, Anoop Deoras, Dan Roth, and Stefano Soatto. 2024. Fewer Truncations Improve Language Modeling. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pages 11030–11048.

Aleksei Dorkin, Taido Purason, Emil Kalbaliyev, Hele-Andra Kuulmets, Marii Ojastu, Mark Fišel, Tanel Alumäe, Eleri Aedmaa, Krister Kruusmaa, and Kairit Sirts. 2026. EstLLM: Enhancing Estonian Capabilities in Multilingual LLMs via Continued Pretraining and Post-Training. Computing Research Repository, arXiv:2603.02041. Version 1.

Simen Eide. 2024. Schibsted Text Tasks: Introducing New Datasets for Norwegian and Swedish Language Models. Blog post. Accessed: 2026-06-16.

Ariel Ekgren, Amaru Cuba Gyllensten, Evangelia Gogoulou, Alice Heiman, Severine Verlinden, Joey Öhman, Fredrik Carlsson, and Magnus Sahlgren. 2022. Lessons Learned from GPT-SW3: Building the First Large-Scale Generative Language Model for Swedish. In Proceedings of the Thirteenth Conference on Language Resource and Evaluation, pages 3509–3518.

Ariel Ekgren, Amaru Cuba Gyllensten, Felix Stollenwerk, Joey Öhman, Tim Isbister, Evangelia

Gogoulou, Fredrik Carlsson, Judit Casademont, and Magnus Sahlgren. 2024. GPT-SW3: An Autoregressive Language Model for the Scandinavian Languages. In Proceedings ofthe 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, pages 7886–7900.

Ahmed Elhady, Eneko Agirre, and Mikel Artetxe. 2025. Emergent Abilities of Large Language Models under Continued Pre-training for Language Adaptation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 32174–32186, Vienna, Austria. Association for Computational Linguistics.

Julen Etxaniz, Oscar Sainz, Naiara Perez, Itziar Aldabe, German Rigau, Eneko Agirre, Aitor Ormazabal, Mikel Artetxe, and Aitor Soroa. 2024. Latxa: An Open Language Model and Evaluation Suite for Basque. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14952–14972, Bangkok, Thailand. Association for Computational Linguistics.

EuroEval. 2026. EuroEval Datasets: Swedish. Dataset documentation. Accessed: 2026-06-16.

Kazuki Fujii, Taishi Nakamura, Mengsay Loem, Hiroki Iida, Masanari Ohi, Kakeru Hattori, Hirai Shota, Sakae Mizuki, Rio Yokota, and Naoaki Okazaki. 2024. Continual Pre-Training for Cross-Lingual LLM Adaptation: Enhancing Japanese Language Capabilities. In First Conference on Language Modeling.

Kevin Glocker, Kätriin Kukk, Romina Oji, Marcel Bollmann, Marco Kuhlmann, and Jenny Kunz. 2025. Grow Up and Merge: Scaling Strategies for Efficient Language Adaptation. Computing Research Repository, arXiv:2512.10772. Version 1.

Evangelia Gogoulou, Timothée Lesort, Magnus Boman, and Joakim Nivre. 2024. Continual Learning Under Language Shift. In Text, Speech, and Dialogue: 27th International Conference, pages 71–84.

Haiyang Guo, Fanhu Zeng, Fei Zhu, Jiayi Wang, Xukai Wang, Jingang Zhou, Hongbo Zhao, Wenzhuo Liu, Shijie Ma, Xu-Yao Zhang, and Cheng-Lin Liu. 2025a. Continual Learning for Generative AI: From LLMs to MLLMs and Beyond. Computing Research Repository, arXiv:2506.13045. Version 4.

Yiduo Guo, Jie Fu, Huishuai Zhang, and Dongyan Zhao. 2025b. Efficient Domain Continual pretraining by Mitigating the Stability Gap. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 32850–32870, Vienna, Austria. Association for Computational Linguistics.

Kshitij Gupta, Benjamin Thérien, Adam Ibrahim, Mats Leon Richter, Quentin Gregory Anthony, Eugene Belilovsky, Irina Rish, and Timothée Lesort. 2023. Continual Pre-Training of Large Language

Models: How to re-warm your model? In Workshop on Efficient Systemsfor Foundation Models.

Suchin Gururangan, Ana Marasovic, Swabha Swayam-´ dipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. 2020. Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8342–8360, Online. Association for Computational Linguistics.

Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. 2015. Teaching Machines to Read and Comprehend. In Advances in Neural Information Processing Systems, volume 28, pages 1693–1701.

Alejandro Hernández-Cano, Alexander Hägele, Allen Hao Huang, Angelika Romanou, Antoni-Joan Solergibert, Barna Pásztor, Bettina Messmer, Dhia Garbaya, Eduard Frank Durech, Ido Hakimi,<sup>ˇ</sup> Juan Garcia Giraldo, Mete Ismayilzada, Negar Foroutan, Skander Moalla, Tiancheng Chen, Vinko Sabolcec, Yixuan Xu, Michael Aerni, Badr AlKha-ˇ missi, and 83 others. 2026. Apertus: Democratizing Open and Compliant LLMs for Global Language Environments. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 46877–46955, San Diego, California, United States. Association for Computational Linguistics.

Masanori Hirano and Kentaro Imajo. 2024. Construction of Domain-Specified Japanese Large Language Model for Finance Through Continual Pre-Training. In 2024 16th IIAI International Congress on Advanced Applied Informatics, pages 273–279.

Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli, Qingquan Song, Shao Tang, Siyu Zhu, Steven Shimizu, Shivam Sahni, Haowen Ning, Yanning Chen, and Zhipeng Wang. 2025. Liger-Kernel: Efficient Triton Kernels for LLM Training. In Championing Open-source Development in ML Workshop.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In Tenth International Conference on Learning Representations.

Shih-Cheng Huang, Pin-Zu Li, Yu-chi Hsu, Kuang-Ming Chen, Yu Tung Lin, Shih-Kai Hsiao, Richard Tsai, and Hung-yi Lee. 2024. Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 10943–10959, Bangkok, Thailand. Association for Computational Linguistics.

Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats Leon Richter, Quentin Gregory Anthony, Eugene Belilovsky, Timothée Lesort, and Irina Rish. 2024. Simple and Scalable Strategies to Continually

Pre-train Large Language Models. Transactions on Machine Learning Research.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2023. Editing Models with Task Arithmetic. In Eleventh International Conference on Learning Representations.

International Press Telecommunications Council. 2026. IPTC NewsCodes: Media Topics. Controlled vocabulary. Accessed: 2026-06-16.

Ishan Jindal, Chandana Badrinath, Pranjal Bharti, Lakkidi Vinay, and Sachin Dev Sharma. 2024. Balancing Continuous Pre-Training and Instruction Fine-Tuning: Optimizing Instruction-Following in LLMs. Computing Research Repository, arXiv:2410.10739. Version 1.

Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. 2017. Bag of Tricks for Efficient Text Classification. In Proceedings ofthe 15th Conference ofthe European Chapter ofthe Association for Computational Linguistics: Volume 2, Short Papers, pages 427–431, Valencia, Spain. Association for Computational Linguistics.

Dmytro Kalpakchi and Johan Boye. 2023. SweCTRL-Mini: a data-transparent Transformer-based large language model for controllable text generation in Swedish. Computing Research Repository, arXiv:2304.13994. Version 3.

Zixuan Ke, Yifei Ming, Xuan-Phi Nguyen, Caiming Xiong, and Shafiq Joty. 2025. Demystifying Domainadaptive Post-training for Financial LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 31033– 31059, Suzhou, China. Association for Computational Linguistics.

Zixuan Ke, Yijia Shao, Haowei Lin, Tatsuya Konishi, Gyuhak Kim, and Bing Liu. 2023. Continual Pretraining of Language Models. In Eleventh International Conference on Learning Representations.

Zixuan Ke, Yijia Shao, Haowei Lin, Hu Xu, Lei Shu, and Bing Liu. 2022. Adapting a Language Model While Preserving its General Knowledge. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10177– 10188, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Sanghoon Kim, Dahyun Kim, Chanjun Park, Wonsung Lee, Wonho Song, Yunsu Kim, Hyeonwoo Kim, Yungi Kim, Hyeonju Lee, Jihoo Kim, Changbae Ahn, Seonghoon Yang, Sukyung Lee, Hyunbyung Park, Gyoungjin Gim, Mikyoung Cha, Hwalsuk Lee, and Sunghun Kim. 2024. SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling. In Proceedings ofthe 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track), pages 23–35,

Mexico City, Mexico. Association for Computational Linguistics.

Achintya Kundu, Rhui Dih Lee, Laura Wynter, Raghu Kiran Ganti, and Mayank Mishra. 2024. Enhancing Training Efficiency Using Packing with Flash Attention. Computing Research Repository, arXiv:2407.09105. Version 6.

Jenny Kunz. 2026a. A Diagnostic Benchmark for Sweden-Related Factual Knowledge. In Proceedings ofthe Fifteenth Language Resources and Evaluation Conference, pages 5328–5334.

Jenny Kunz. 2026b. Preferences for Idiomatic Language are Acquired Slowly — and Forgotten Quickly: A Case Study on Swedish. Transactions of the Association for Computational Linguistics, 14:1266– 1285.

Jenny Kunz, Iben Nyholm Debess, and Annika Simonsen. 2026. Family Matters: Language Transfer and Merging for Adapting Small LLMs to Faroese. Computing Research Repository, arXiv:2510.00810. Version 2.

Hele-Andra Kuulmets and Mark Fishel. 2023. Translated Benchmarks Can Be Misleading: the Case of Estonian Question Answering. In Proceedings of the 24th Nordic Conference on Computational Linguistics, pages 710–716, Tórshavn, Faroe Islands. University of Tartu Library.

Yanis Labrak, Adrien Bazoge, Emmanuel Morin, Pierre-Antoine Gourraud, Mickael Rouvier, and Richard Dufour. 2024. BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains. In Findings of the Association for Computational Linguistics: ACL 2024, pages 5848– 5864, Bangkok, Thailand. Association for Computational Linguistics.

Pierre-Carl Langlais, Pavel Chizhov, Catherine Arnett, Carlos Rosas Hinostroza, Mattia Nee, Eliot Krzysztof Jones, Irène Girard, David Mach, Anastasia Stasenko, and Ivan P. Yamshchikov. 2026. Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training. In Fourteenth International Conference on Learning Representations.

Jin-woo Lee, Junhwa Choi, Bongkyu Hwang, Jinho Choo, Bogun Kim, Jeongseon Yi, Joonseok Lee, DongYoung Jung, Jaeseon Park, Kyoungwon Park, and Suk-hoon Jung. 2026. SCALE: Upscaled Continual Learning of Large Language Models. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 41006–41020, San Diego, California, United States. Association for Computational Linguistics.

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. 2022. Deduplicating Training Data Makes Language Models Better. In Proceedings ofthe 60th Annual Meeting ofthe Association

for Computational Linguistics (Volume 1: Long Papers), pages 8424–8445, Dublin, Ireland. Association for Computational Linguistics.

David D. Lewis, Yiming Yang, Tony G. Rose, and Fan Li. 2004. RCV1: A New Benchmark Collection for Text Categorization Research. Journal of Machine Learning Research, 5:361–397.

Jiangtong Li, Yuxuan Bian, Guoxuan Wang, Yang Lei, Dawei Cheng, Zhijun Ding, and Changjun Jiang. 2023. CFGPT: Chinese Financial Assistant with Large Language Model. Computing Research Repository, arXiv:2309.10654. Version 2.

Miao Li, Ming-Bin Chen, Bo Tang, ShengbinHou ShengbinHou, Pengyu Wang, Haiying Deng, Zhiyu Li, Feiyu Xiong, Keming Mao, Cheng Peng, and Yi Luo. 2024. NewsBench: A Systematic Evaluation Framework for Assessing Editorial Capabilities of Large Language Models in Chinese Journalism. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9993–10014, Bangkok, Thailand. Association for Computational Linguistics.

Pin-Jie Lin, Rishab Balasubramanian, Fengyuan Liu, Nikhil Kandpal, and Tu Vu. 2025. Efficient Model Development through Fine-tuning Transfer. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 2617– 2636, Suzhou, China. Association for Computational Linguistics.

Alexander H. Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sadé, Alan Jeffares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, Alexandre Sablayrolles, Amélie Héliou, Amos You, Andy Ehrenberg, Andy Lo, Anton Eliseev, Antonia Calvi, Avinash Sooriyarachchi, Baptiste Bout, and 101 others. 2026. Ministral 3. Computing Research Repository, arXiv:2601.08584. Version 1.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled Weight Decay Regularization. In Seventh International Conference on Learning Representations.

Yizhen Luo, Jiahuan Zhang, Siqi Fan, Kai Yang, Massimo Hong, Yushuai Wu, Mu Qiao, and Zaiqing Nie. 2026. BioMedGPT: An Open Multimodal Large Language Model for BioMedicine. IEEE Journal of Biomedical and Health Informatics, 30(2):981–992.

Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. 1993. Building a Large Annotated Corpus of English: The Penn Treebank. Computational Linguistics, 19(2):313–330.

Bettina Messmer, Vinko Sabolcec, and Martin Jaggi.ˇ 2025. Enhancing Multilingual LLM Pretraining with Model-Based Data Selection. In Advances in Neural Information Processing Systems, volume 38, pages 70037–70073.

Julius Monsen and Arne Jönsson. 2021. A Method for Building Non-English Corpora for Abstractive Text Summarization. In CLARIN Annual Conference Proceedings, pages 82–85.

Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. 2023. Scaling Data-Constrained Language Models. In Advances in Neural Information Processing Systems, volume 36, pages 50358–50376.

Arijit Nag, Soumen Chakrabarti, Animesh Mukherjee, and Niloy Ganguly. 2025. Efficient Continual Pretraining of LLMs for Low-resource Languages. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 3: Industry Track), pages 304–317, Albuquerque, New Mexico, United States. Association for Computational Linguistics.

Nikhil Shivakumar Nayak, Krishnateja Killamsetty, Ligong Han, Abhishek Bhandwaldar, Prateek Chanda, Kai Xu, Oleg Silkin, Mustafa Eyceoz, Hao Wang, Aldo Pareja, and Akash Srivastava. 2026. Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning. In Fourteenth International Conference on Learning Representations.

Dan Saattrup Nielsen. 2023. ScandEval: A Benchmark for Scandinavian Natural Language Processing. In Proceedings of the 24th Nordic Conference on Computational Linguistics, pages 185–201, Tórshavn, Faroe Islands. University of Tartu Library.

Dan Saattrup Nielsen, Kenneth Enevoldsen, and Peter Schneider-Kamp. 2025. Encoder vs Decoder: Comparative Analysis of Encoder and Decoder Language Models on Multilingual NLU Tasks. In Proceedings of the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies, pages 561–572, Tallinn, Estonia. University of Tartu Library.

Sachita Nishal, Charlotte Li, and Nicholas Diakopoulos. 2024. Domain-Specific Evaluation Strategies for AI in Journalism. Computing Research Repository, arXiv:2403.17911. Version 1.

Tobias Norlund and Agnes Stenbom. 2021. Building a Swedish Open-Domain Conversational Language Model. In Proceedings of the 23rd Nordic Conference on Computational Linguistics, pages 357–366, Reykjavik, Iceland (Online). Linköping University Electronic Press.

Jupinder Parmar, Sanjev Satheesh, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. 2024. Reuse, Don’t Retrain: A Recipe for Continued Pretraining of Language Models. Computing Research Repository, arXiv:2407.07263. Version 1.

Guilherme Penedo, Hynek Kydlícek, Loubna B. Al-ˇ lal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. 2024.

The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. In Advances in Neural Information Processing Systems, volume 37, pages 30811–30849.

Pouya Pezeshkpour and Estevam Hruschka. 2025. Learning Beyond the Surface: How Far Can Continual Pre-Training with LoRA Enhance LLMs Domain-Specific Insight Learning? Computing Research Repository. Version 1.

Maja Popovic. 2017. ´ chrF++: words helping character n-grams. In Proceedings of the Second Conference on Machine Translation, pages 612–618, Copenhagen, Denmark. Association for Computational Linguistics.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. ZeRO: Memory optimizations Toward Training Trillion Parameter Models. In Proceedings ofthe International Conferencefor High Performance Computing, Networking, Storage and Analysis, pages 1–16.

Pablo Rodríguez, Silvia Paniagua Suárez, Pablo Gamallo, and Susana Sotelo Docio. 2025. Continued Pretraining and Interpretability-Based Evaluation for Low-Resource Languages: A Galician Case Study. In Findings of the Association for Computational Linguistics: ACL 2025, pages 4622–4637, Vienna, Austria. Association for Computational Linguistics.

David Samuel, Vladislav Mikhailov, Erik Velldal, Lilja Øvrelid, Lucas Georges Gabriel Charpentier, Andrey Kutuzov, and Stephan Oepen. 2025. Small Languages, Big Models: A Study of Continual Training on Languages of Norway. In Proceedings ofthe Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies, pages 573–608, Tallinn, Estonia. University of Tartu Library.

Thomas Scialom, Tuhin Chakrabarty, and Smaranda Muresan. 2022. Fine-tuned Language Models are Continual Learners. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6107–6122, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. 2024. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. In Advances in Neural Information Processing Systems, volume 37, pages 68658–68685.

Zhengxiang Shi and Aldo Lipani. 2023. Don’t Stop Pretraining? Make Prompt-based Fine-tuning Powerful Learner. In Advances in Neural Information Processing Systems, volume 36, pages 5827–5849.

Silo AI. 2024. Viking 7B/13B/33B: Sailing the Nordic Seas of Multilinguality. Blog post. Accessed: 2026- 06-16.

Shivalika Singh, Angelika Romanou, Clémentine Fourrier, David Ifeoluwa Adelani, Jian Gang Ngui, Daniel Vila-Suero, Peerat Limkonchotiwat, Kelly Marchisio, Wei Qi Leong, Yosephine Susanto, Raymond Ng, Shayne Longpre, Sebastian Ruder, Wei-Yin Ko, Antoine Bosselut, Alice Oh, Andre Martins, Leshem Choshen, Daphne Ippolito, and 4 others. 2025. Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18761–18799, Vienna, Austria. Association for Computational Linguistics.

Shamane Siriwardhana, Mark McQuade, Thomas Gauthier, Lucas Atkins, Fernando Fernandes Neto, Luke Meyers, Anneketh Vij, Tyler Odenthal, Charles Goddard, Mary MacCarthy, and Jacob Solawetz. 2024. Domain Adaptation of Llama3-70B-Instruct through Continual Pre-Training and Model Merging: A Comprehensive Evaluation. Computing Research Repository, arXiv:2406.14971. Version 1.

Eshaan Tanwar, Deepak Nathani, William Yang Wang, and Tanmoy Chakraborty. 2025. Understanding the Effects of Domain Finetuning on LLMs. Computing Research Repository, arXiv:2510.09359. Version 1.

Atula Tejaswi, Nilesh Gupta, and Eunsol Choi. 2024. Exploring Design Choices for Building Language-Specific LLMs. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 10485–10500, Miami, Florida, United States. Association for Computational Linguistics.

Zhao Tong, Yimeng Gu, Huidong Liu, Qiang Liu, Shu Wu, Haichao Shi, and Xiao-Yu Zhang. 2025. Generate First, Then Sample: Enhancing Fake News Detection with LLM-Augmented Reinforced Sampling. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 24276–24290, Vienna, Austria. Association for Computational Linguistics.

Kentaro Ueda, François Portet, Hirohiko Suwa, and Keiichi Yasumoto. 2026. Merging Continual Pretraining Models for Domain-Specialized LLMs: A Case Study in Finance. In Proceedings of the Fifteenth Language Resources and Evaluation Conference, pages 10114–10129.

Jiajing Wan, Samia Touileb, Lubos Steskal, and Lilja Øvrelid. 2026. Personalizing News Headlines with Retrieval-Augmented Generation. In Proceedings ofthe Second Workshop on Customizable NLP: Progress and Challenges in Customizing NLP for a Domain, Application, Group, or Individual, pages 55– 67, San Diego, California, United States. Association for Computational Linguistics.

Shumin Wang, Yuexiang Xie, Bolin Ding, Jinyang Gao, and Yanyong Zhang. 2025. Language Adaptation of Large Language Models: An Empirical Study on LLaMA2. In Proceedings ofthe 31st International Conference on Computational Linguistics,

pages 7195–7208, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Zhonghao Wang, Zijia Lu, Bo Jin, and Haiying Deng. 2023. MediaGPT : A Large Language Model For Chinese Media. Computing Research Repository, arXiv:2307.10930. Version 2.

Chengyue Wu, Yukang Gan, Yixiao Ge, Zeyu Lu, Jiahao Wang, Ye Feng, Ying Shan, and Ping Luo. 2024. LLaMA Pro: Progressive LLaMA with Block Expansion. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6518–6537, Bangkok, Thailand. Association for Computational Linguistics.

Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023. BloombergGPT: A Large Language Model for Finance. Computing Research Repository, arXiv:2303.17564. Version 2.

Tongtong Wu, Trang Vu, Linhao Luo, and Gholamreza Haffari. 2025. Continual Learning of Large Language Models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing: Tutorial Abstracts, pages 16–17, Suzhou, China. Association for Computational Linguistics.

Yong Xie, Karan Aggarwal, and Aitzaz Ahmad. 2024. Efficient Continual Pre-training for Building Domain Specific Large Language Models. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 10184–10201, Bangkok, Thailand. Association for Computational Linguistics.

Fuzhao Xue, Yao Fu, Wangchunshu Zhou, Zangwei Zheng, and Yang You. 2023. To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis. In Advances in Neural Information Processing Systems, volume 36, pages 59304–59322.

Shunyu Yao, Qingqing Ke, Kangtong Li, Qiwei Wang, and Jie Hu. 2024. News GPT: A Large Language Model for Reliable and Hallucination-Controlled News Generation. In Proceedings ofthe 2024 3rd International Symposium on Robotics, Artificial Intelligence and Information Engineering, pages 113–119.

Yizhou Ying, Geng Zhang, Cui Danxin, Chengyu Du, Guanglei Yue, Sihang Jiang, Jiaqing Liang, Yifei Fu, Hailin Hu, and Yanghua Xiao. 2025. Data-Efficient Selection via Grammatical Complexity in Continual Pre-training of Domain-Specific LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 22055–22069, Suzhou, China. Association for Computational Linguistics.

Mike Zhang, Max Müller-Eberstein, Elisa Bassignana, and Rob van der Goot. 2025. SnakModel: Lessons Learned from Training an Open Danish Large Language Model. In Proceedings ofthe Joint 25th Nordic Conference on Computational Linguistics and

11th Baltic Conference on Human Language Technologies, pages 812–825, Tallinn, Estonia. University of Tartu Library.

## A Training Data

This section describes selected stages of the preprocessing pipeline in greater detail.

## A.1 Cleaning

We clean the articles by omitting extraneous fields during source-format parsing and by applying regular-expression-based filters. In particular, we remove text spans that are peripheral to the main article content, such as links, advertisements, reader notices, and author bylines. Additionally, we remove artefacts erroneously retained as plain text, such as JavaScript code and HTML tags. By contrast, we preserve elements that provide additional context, including fact boxes, tables, and lists.

## A.2 Filtering

To remove low-quality or otherwise unsuitable articles, we apply a broad set of text- and metadatabased filters to the raw corpus. We design these filters using a combination of domain knowledge and exploratory analysis. When articles of a known category cannot be identified from metadata alone, we instead develop text-based statistics and regular expressions to detect them. By analysing articles flagged as candidates for removal, we iteratively tune the filters to balance false positives and false negatives. Additionally, by examining articles with extreme values for various textual properties, such as unusually high newline counts, we identify and remove previously unknown categories of lowquality content. For example, some general textquality filters require each article to

• have a headline;

• have body text;

• contain between 2 and 2,000 newline characters, inclusive, counting those following the headline or lead paragraph;

• contain at least 20 words;

• contain at least 17 distinct words;

• contain at least 200 characters;

• have a type–token ratio of at least 0.2; and

• be classified as Swedish by fastText<sup>1</sup> with a score greater than 0.9.

Although articles are sourced from Swedish publications, some are written in other languages to serve specific readerships; using the fastText filter, we remove content in languages such as English, Arabic, and Russian. The largest category of excluded material consists of template-based articles that are automatically generated from structured data, typically covering sports results, real estate transactions, and weather forecasts. In addition, we discard large numbers of roundup articles consisting primarily of links to previously published content on a particular topic. Other article types we remove include republished press releases and public records, as well as personal notices such as birthday greetings, wedding announcements, and obituaries.

## A.3 Deduplication

We perform document-level deduplication on normalised text obtained by lowercasing and removing punctuation and diacritics. Following Penedo et al. (2024), we construct MinHash signatures using 112 hash functions and apply LSH for candidate generation, but lower the similarity threshold to 0.25 to increase recall. For each candidate pair, we compute the normalised Levenshtein distance and treat the articles as duplicates if it is less than 0.2.

## B Evaluation Data

When constructing BonEval, we apply task-specific filters to obtain suitable and representative evaluation examples. Across all tasks sampled from BonCorpus, we exclude articles longer than 1,024 tokens. For each task, we apply the following additional criteria:

1. Headline: Headlines contain 10–120 characters, and input texts contain more than 200 characters. We exclude headlines beginning with “LIST: . . . ” or similar prefixes.

2. Lead: Leads contain 30–500 characters, and input texts contain more than 200 characters. Each lead must also be shorter than half its corresponding body text. We exclude leads containing bullet points of any kind.

3. Summary: Summaries contain 200–800 characters and comprise three to five bullet points. We normalise their formatting.

4. Topic: Articles are originally labelled with one to five topics, exactly one of which is top-level and serves as the target. To simplify the underlying hierarchical multi-label classification problem, which contains thousands of topics, we restrict the label set to the 19 top-level topics shown in Table 8.

5. Entity: Articles have one to five entity labels, each of which is mentioned within the first 90% of the text. We exclude labels likely to reflect annotation errors, such as external news outlets derived from attribution phrases like “according to . . . ”. Additionally, for geographically nested location labels, we retain only the more specific one; for example, we keep Stockholm rather than Sweden when both are present.

6. Quiz: Each question has exactly four answer choices. We discard quizzes with an Englishlanguage theme or that rely on images or audio. Because the quizzes date back to 2009, we also attempt to exclude strongly time-dependent content, including quizzes in the category “Current affairs” and questions whose answers change over time, such as “How many prime ministers has the United Kingdom had?”. The quizzes span the following 15 categories:

(a) Words and language

(b) Current affairs (removed)

(c) Around Sweden

(d) Around the world

(e) Animals and nature

(f) Science

(g) Food and drink

(h) Sport and games

(i) Books

(j) Film and TV

(k) Music

(l) Culture

(m) Politics and society

(n) History

(o) Miscellaneous

## C Training Setup

Across all experiments, we use NVIDIA DGX H200 nodes and consume a total of 6,500 GPUhours. We use the AdamW optimiser (Loshchilov and Hutter, 2019) and the hyperparameters reported in Table 4. To improve computational and memory efficiency, we use DeepSpeed ZeRO-2 (Rajbhandari et al., 2020), FlashAttention-3 (Shah et al., 2024), and Liger kernels (Hsu et al., 2025).

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td> $\beta _ { 1 }$ </td><td>0.9 0.95</td></tr><tr><td> $\beta _ { 2 }$  €</td><td>1e-8</td></tr><tr><td>Weight decay Schedule</td><td>0.1 Cosine</td></tr><tr><td>Linear warmup Max. learning rate</td><td>10% 5e-5</td></tr><tr><td>Min. learning rate Dropout</td><td>5e-6 0.0</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Sequence length</td><td>16,384</td></tr><tr><td>Global batch size Precision</td><td>64 bfloat16</td></tr></table>

Table 4: Training configuration for all experiments.

Depending on the model size and number of GPUs, we use activation checkpointing and gradient accumulation as needed to achieve the target global batch size. Because article lengths vary substantially, we pack examples up to the maximum sequence length so that each optimisation step is based on roughly the same number of tokens. To minimise truncation, which can degrade model performance, we use a best-fit-decreasing packing strategy (Ding et al., 2024). As the packed sequences still vary slightly in length, we flatten each batch into a single sequence to eliminate padding inefficiencies (Kundu et al., 2024).

## C.1 PEFT Configurations

We configure LoRA and LLaMA Pro to have comparable numbers of trainable parameters for the 3B model: 395M and 466M, respectively. For LLaMA Pro, we insert one additional layer after every sixth layer of the base model. Following Biderman et al. (2024), we apply LoRA to all transformer modules with a rank of 256, an α of 512, and no dropout.

## D Evaluation Setup

We implement BonEval and run the evaluations with the EuroEval library. For tasks outside BonEval, we rely on EuroEval’s existing implementations. To parse model outputs, we use log probabilities for Topic and Quiz and structured generation for Entity.

We evaluate each model over 10 runs, randomly sampling few-shot examples from the training split for each run. During development, we evaluate on validation splits rather than test splits. Following Nielsen et al. (2025), we heuristically determine the number of few-shot examples for each task based on context length. Rather than targeting 1,000 tokens, we select the number of examples such that no input exceeds 4,096 tokens. We cap this number at 12, guided by the existing task implementations.<sup>2</sup> The resulting number of few-shot examples for each task is reported in Table 5. If IV is applied to a model, we consider it instructiontuned. Task prompts for base and instruction-tuned models are shown in Tables 6 and 7, respectively; English translations are provided in Tables 8 and 9, respectively. For Quiz, we use EuroEval’s default multiple-choice prompts.

<table><tr><td>Task</td><td>Train</td><td>Val</td><td>Test</td><td>Few-Shot</td></tr><tr><td>Headline</td><td>1,024</td><td>256</td><td>8,192</td><td>3</td></tr><tr><td>Lead</td><td>1,024</td><td>256</td><td>8,192</td><td>3</td></tr><tr><td>Summary</td><td>32</td><td>32</td><td>100</td><td>3</td></tr><tr><td>Topic</td><td>1,024</td><td>256</td><td>8,192</td><td>3</td></tr><tr><td>Entity</td><td>1,024</td><td>256</td><td>8,192</td><td>3</td></tr><tr><td>Quiz</td><td>1,024</td><td>256</td><td>11,960</td><td>12</td></tr></table>

Table 5: Number of examples for BonEval tasks.

## E Complete Evaluation Results

Complete evaluation results for the 3B experiments are presented in Tables 10 and 11, while those for BonLM 8B and the baselines are presented in Tables 12 and 13.

<table><tr><td>Task</td><td>Prompt Prefix</td><td>Prompt Template</td></tr><tr><td>Headline</td><td>Följande är artiklar med tillhörande rubriker.</td><td>Artikel: {lead + body} Rubrik: {headline}</td></tr><tr><td>Lead</td><td>Följande är artiklar med tillhörande ingresser.</td><td>Artikel: {body} Ingress: {lead}</td></tr><tr><td>Summary</td><td>Följande är artiklar med tillhörande sammanfattningar i punktform.</td><td>Artikel: {article} Sammanfattning: {summary}</td></tr><tr><td>Topic</td><td>tillhörande frågor och svar om deras ämnen.</td><td>Fråga: Vilket ämne beskriver bäst artikelns innehåll? Svarsalternativ: a. Arbete b. Brott, lag och rätt c. Ekonomi, näringsliv och finans d. Hälsa och sjukvård e. Klimat och miljö f. Konflikter och krig g. Kultur och nöje h. Kungligt i. Livsstil och fritid j. Olyckor och katastrofer k. Personligt 1. Politik m. Religion och tro n. Samhälle o. Skola och utbildning p. Sport</td></tr><tr><td>Entity</td><td>Följande är artiklar med tillhörande namngivna entiteter i JSON-format som är centrala för innehållet.</td><td>Artikel: {article} Namngivna entiteter: {  $" { \mathsf { p e r s o n } } " : \ [ \dots ] ,$   $" \mathsf { p l a t s } " : \ [ \dots ]$   $" o r g a n i s a t i o n " : ~ [ \dots ]$  }</td></tr><tr><td>Quiz</td><td>Följande är flervalsfrågor (med svar).</td><td>Fråga: {question} Svarsalternativ: a. {choice_a} b. {choice_b} c. {choice_c}</td></tr></table>

Table 6: BonEval prompts for base models.

<table><tr><td>Task</td><td>Instruction Prompt</td></tr><tr><td>Headline</td><td>Artikel: {lead + body} Skriv en rubrik till artikeln ovan.</td></tr><tr><td>Lead</td><td>Artikel: {body} Skriv en ingress till artikeln ovan.</td></tr><tr><td>Summary</td><td>Artikel: {article} Skriv en sammanfattning i punktform av artikeln ovan. Artikel: {article}</td></tr><tr><td></td><td>Fråga: Vilket ämne beskriver bäst artikelns innehåll? Svarsalternativ: a. Arbete b. Brott, lag och rätt c. Ekonomi, näringsliv och finans d. Hälsa och sjukvård e. Klimat och miljö f. Konflikter och krig g. Kultur och nöje h. Kungligt i. Livsstil och fritid j. Olyckor och katastrofer k. Personligt 1. Politik m. Religion och tro n. Samhälle o. Skola och utbildning p. Sport</td></tr><tr><td></td><td>Artikel: {article} Identifiera de namngivna entiteter som är centrala för artikelns innehåll. Svara i JSON-format med nycklarna person, plats och organisation, där värdena är listor över de namngivna entiterna av den typen, precis som de</td></tr><tr><td>Quiz</td><td>förekommer i artikeln. Fråga: {question} Besvara frågan ovan med a, b, c eller d, och inget annat.</td></tr></table>

Table 7: BonEval prompts for instruction-tuned models.

<table><tr><td>Task</td><td>Prompt Prefix</td><td>Prompt Template</td></tr><tr><td>Headline</td><td>The following are articles with associated headlines.</td><td>Article: {lead + body} Headline: {headline}</td></tr><tr><td>Lead</td><td>The following are articles with associated leads.</td><td>Article: {body} Lead: {lead}</td></tr><tr><td>Summary</td><td>The following are articles with associated bullet-point summaries.</td><td>Article: {article} Summary: {summary}</td></tr><tr><td>Topic</td><td>The following are articles with associated questions and answers about their topics.</td><td>Article: {article} Question: Which topic best describes the content of the article? Choices: a. Labour b. Crime, law and justice c. Economy, business and finance d. Health e. Environment f. Conflict, war and peace Arts, culture, entertainment and media h. Royalty i. Lifestyle and leisure j. Disaster, accident and emergency incident k. Human interest 1. Politics and government</td></tr><tr><td></td><td>The following are articles with associated named entities in JSON format that are central to the content.</td><td>Article: {article} Named entities: {  $" { \sf p e r s o n } " : \quad [ . . . ] ,$   $" \mathrm { l o c a t i o n " : ~ } [ \dots ] ,$   $" o r g a n i s a t i o n " : ~ [ \dots ]$ </td></tr><tr><td>Quiz</td><td>The following are multiple choice questions (with answers).</td><td>} Question: {question} Choices: a. {choice_a} b. {choice_b}</td></tr></table>

Table 8: Translated BonEval prompts for base models.

<table><tr><td>Task</td><td>Instruction Prompt</td></tr><tr><td>Headline</td><td>Article: {lead + body} Write a headline to the above article.</td></tr><tr><td>Lead</td><td>Article: {body} Write a lead to the above article.</td></tr><tr><td>Summary</td><td>Article: {article} Write a bullet-point summary of the above article.</td></tr><tr><td>Topic</td><td>Article: {article} Question: Which topic best describes the content of the article? Choices:</td></tr><tr><td></td><td>c. Economy, business and finance d. Health e. Environment f. Conflict, war and peace g. Arts, culture, entertainment and media h. Royalty i. Lifestyle and leisure j. Disaster, accident and emergency incident</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Entity</td><td></td></tr><tr><td></td><td>p. Sport</td></tr><tr><td></td><td>m. Religion</td></tr><tr><td></td><td></td></tr><tr><td></td><td>k. Human interest</td></tr><tr><td></td><td></td></tr><tr><td></td><td>1. Politics and government</td></tr><tr><td></td><td></td></tr><tr><td></td><td>n. Society</td></tr><tr><td></td><td>o. Education</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>q. Traffic and vehicles</td></tr><tr><td></td><td>r. Science and technology</td></tr><tr><td></td><td>s. Weather</td></tr><tr><td></td><td>Answer the above question by replying with</td></tr><tr><td></td><td>a, b, c, d, e, f, g, h, i, j, k, 1, m, n, o, p, q, r, or s, and nothing else.</td></tr><tr><td></td><td>Article: {article}</td></tr><tr><td></td><td>Identify the named entities central to the content of</td></tr><tr><td></td><td>the article. Respond in JSON format with the keys</td></tr><tr><td></td><td>person, location, and organisation, where the values</td></tr><tr><td></td><td>are lists of the named entities of that type, exactly</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>as they appear in the article.</td></tr><tr><td>Quiz</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

Table 9: Translated BonEval prompts for instruction-tuned models.

<table><tr><td>Model</td><td>SweReC</td><td>SUC 3.0</td><td>ScaLA</td><td>MultiWikiQA</td><td>SweDN</td><td>MMLU</td><td>HellaSwag</td><td>Average</td></tr><tr><td colspan="9">Ministral 3 3B</td></tr><tr><td>Base</td><td>77.61</td><td>55.79</td><td>38.86</td><td>75.41</td><td>34.70</td><td>49.04</td><td>26.02</td><td>51.06</td></tr><tr><td>Instruct</td><td>75.23</td><td>39.66</td><td>33.50</td><td>70.94</td><td>32.66</td><td>42.32</td><td>48.18</td><td>48.93</td></tr><tr><td>Reasoning</td><td>74.17</td><td>44.48</td><td>29.58</td><td>78.11</td><td>22.62</td><td>26.10</td><td>35.57</td><td>44.37</td></tr><tr><td colspan="9">Data Mixtures</td></tr><tr><td>B</td><td>78.22</td><td>31.41</td><td>28.22</td><td>65.83</td><td>30.77</td><td>21.24</td><td>5.80</td><td>37.35</td></tr><tr><td>BC</td><td>75.09</td><td>41.52</td><td>28.30</td><td>71.09</td><td>32.34</td><td>24.70</td><td>10.03</td><td>40.44</td></tr><tr><td>BE</td><td>75.68</td><td>39.46</td><td>22.84</td><td>72.28</td><td>33.12</td><td>24.07</td><td>8.21</td><td>39.38</td></tr><tr><td>BCE</td><td>78.62</td><td>47.04</td><td>25.50</td><td>72.41</td><td>33.58</td><td>26.78</td><td>7.20</td><td>41.59</td></tr><tr><td>BS</td><td>77.30</td><td>41.92</td><td>25.04</td><td>69.16</td><td>35.11</td><td>26.31</td><td>13.80</td><td>41.24</td></tr><tr><td>BCS</td><td>79.60</td><td>47.24</td><td>35.09</td><td>68.33</td><td>34.94</td><td>27.31</td><td>11.74</td><td>43.46</td></tr><tr><td>BES</td><td>79.25</td><td>45.77</td><td>33.25</td><td>69.28</td><td>34.99</td><td>28.61</td><td>10.26</td><td>43.06</td></tr><tr><td>BCES</td><td>78.85</td><td>47.60</td><td>32.31</td><td>70.43</td><td>33.46</td><td>28.90</td><td>11.49</td><td>43.29</td></tr><tr><td colspan="9">PEFT Methods</td></tr><tr><td>LoRA</td><td>77.39</td><td>55.46</td><td>33.50</td><td>75.18</td><td>34.37</td><td>44.46</td><td>20.30</td><td>48.67</td></tr><tr><td>LLaMA Pro</td><td>78.04</td><td>54.17</td><td>34.09</td><td>74.73</td><td>34.48</td><td>47.22</td><td>20.92</td><td>49.09</td></tr><tr><td colspan="9">Instruct Vectors</td></tr><tr><td> $\mathrm { \Delta } + \mathrm { \Delta } \mathrm { \Gamma } \mathrm { \Delta } \mathrm { V }$ </td><td>75.23</td><td>27.13</td><td>7.33</td><td>54.50</td><td>27.85</td><td>7.58</td><td>4.21</td><td>29.12</td></tr><tr><td> $\mathrm { B C } + \mathrm { I V }$ </td><td>77.10</td><td>36.87</td><td>9.77</td><td>72.08</td><td>32.24</td><td>10.97</td><td>6.24</td><td>35.04</td></tr><tr><td> $\mathrm { B E } + \mathrm { I V }$ </td><td>74.11</td><td>33.89</td><td>10.23</td><td>72.28</td><td>30.79</td><td>9.69</td><td>4.28</td><td>33.61</td></tr><tr><td> $\mathrm { B C E } + \mathrm { I V }$ </td><td>77.63</td><td>50.36</td><td>12.82</td><td>69.90</td><td>33.24</td><td>16.57</td><td>9.44</td><td>38.57</td></tr><tr><td> $\mathrm { B S } + \mathrm { I V }$ </td><td>77.83</td><td>44.04</td><td>15.27</td><td>71.25</td><td>33.74</td><td>13.42</td><td>9.40</td><td>37.85</td></tr><tr><td> $\mathrm { B C S } + \mathrm { I V }$ </td><td>74.58</td><td>42.88</td><td>14.63</td><td>72.20</td><td>35.87</td><td>13.26</td><td>6.58</td><td>37.14</td></tr><tr><td> $\mathrm { B E S } + \mathrm { I V }$ </td><td>69.12</td><td>40.72</td><td>26.07</td><td>69.68</td><td>32.18</td><td>17.40</td><td>7.82</td><td>37.57</td></tr><tr><td> $\mathrm { B C E S + I V }$   $_ { \mathrm { L o R A + I V } }$ </td><td>75.71</td><td>44.60 46.00</td><td>23.91 32.76</td><td>72.87 71.57</td><td>34.03 34.79</td><td>17.49 37.71</td><td>9.38 34.86</td><td>39.71 48.00</td></tr><tr><td></td><td>78.31</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 10: Results from 3B experiments on Swedish EuroEval tasks. Micro-averaged $\mathrm { F _ { 1 } }$ is reported for SUC 3.0, token-averaged $\mathrm { F _ { 1 } }$ for MultiWikiQA, and CHRF3++ for SweDN. For all other tasks, MCC is reported. Underline denotes section-best and bold overall best.

<table><tr><td rowspan="2">Model</td><td colspan="7">BonEval</td><td colspan="3">Supplementary</td></tr><tr><td>Headline</td><td>Lead</td><td>Summary</td><td>Topic</td><td>Entity</td><td>Quiz</td><td>Average</td><td>SvD</td><td>AB</td><td>Facts</td></tr><tr><td>Ministral 3 3B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>18.62</td><td>26.24</td><td>42.28</td><td>70.39</td><td>29.80</td><td>51.07</td><td>39.73</td><td>19.94</td><td>40.72</td><td>15.81</td></tr><tr><td>Instruct</td><td>26.38</td><td>31.28</td><td>45.87</td><td>71.80</td><td>23.15</td><td>47.80</td><td>41.05</td><td>27.96</td><td>44.15</td><td>16.64</td></tr><tr><td>Reasoning</td><td>21.19</td><td>28.80</td><td>46.08</td><td>68.45</td><td>23.49</td><td>33.66</td><td>36.94</td><td>14.70</td><td>39.59</td><td>10.15</td></tr><tr><td>Data Mixtures</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>B</td><td>20.61</td><td>25.37</td><td>39.42</td><td>52.49</td><td>23.68</td><td>60.30</td><td>36.98</td><td>17.56</td><td>34.15</td><td>31.25</td></tr><tr><td>BC</td><td>20.48</td><td>22.63</td><td>40.83</td><td>67.61</td><td>28.40</td><td>59.97</td><td>39.99</td><td>20.02</td><td>37.92</td><td>30.61</td></tr><tr><td>BE</td><td>20.54</td><td>25.03</td><td>41.04</td><td>67.59</td><td>26.38</td><td>59.46</td><td>40.01</td><td>18.97</td><td>38.26</td><td>30.10</td></tr><tr><td>BS</td><td>19.81</td><td>27.28</td><td>42.87</td><td>62.87</td><td>26.11</td><td>61.41</td><td>40.06</td><td>19.98</td><td>38.20</td><td>31.65</td></tr><tr><td>BCE</td><td>20.48</td><td>25.86</td><td>40.77</td><td>65.33</td><td>29.05</td><td>58.58</td><td>40.01</td><td>18.92</td><td>40.19</td><td>32.02</td></tr><tr><td>BCS</td><td>20.98</td><td>26.74</td><td>43.42</td><td>68.35</td><td>27.65</td><td>60.89</td><td>41.34</td><td>19.76</td><td>40.21</td><td>33.74</td></tr><tr><td>BES</td><td>20.26</td><td>27.63</td><td>40.80</td><td>60.92</td><td>23.73</td><td>60.80</td><td>39.02</td><td>17.73</td><td>40.30</td><td>30.78</td></tr><tr><td>BCES</td><td>20.73</td><td>26.83</td><td>42.24</td><td>70.44</td><td>29.28</td><td>59.40</td><td>41.48</td><td>20.46</td><td>39.48</td><td>32.93</td></tr><tr><td>PEFT Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoRA</td><td>19.95</td><td>27.24</td><td>43.98</td><td>69.32</td><td>30.47</td><td>56.97</td><td>41.32</td><td>20.95</td><td>41.12</td><td>24.74</td></tr><tr><td>LLaMA Pro</td><td>20.24</td><td>26.47</td><td>43.05</td><td>71.85</td><td>30.30</td><td>55.09</td><td>41.17</td><td>19.97</td><td>40.10</td><td>23.57</td></tr><tr><td>Instruct Vectors</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { B } + \mathrm { I V }$ </td><td>23.15</td><td>28.41</td><td>39.57</td><td>1.48</td><td>23.66</td><td>43.48</td><td>26.62</td><td>17.89</td><td>29.10</td><td>14.56</td></tr><tr><td> $\mathrm { B C } + \mathrm { I V }$ </td><td>25.61</td><td>29.06</td><td>41.34</td><td>47.30</td><td>25.66</td><td>48.36</td><td>36.22</td><td>25.74</td><td>38.25</td><td>18.26</td></tr><tr><td> $\mathrm { B E } + \mathrm { I V }$ </td><td>25.24</td><td>32.62</td><td>40.86</td><td>42.60</td><td>21.61</td><td>44.31</td><td>34.54</td><td>24.85</td><td>39.14</td><td>18.27</td></tr><tr><td> $\mathrm { B C E } + \mathrm { I V }$ </td><td>24.99</td><td>32.76</td><td>43.20</td><td>46.74</td><td>25.42</td><td>49.19</td><td>37.05</td><td>19.87</td><td>40.91</td><td>25.88</td></tr><tr><td> $\mathrm { B S } + \mathrm { I V }$ </td><td>25.38</td><td>30.79</td><td>41.77</td><td>7.64</td><td>23.93</td><td>54.35</td><td>30.64</td><td>20.88</td><td>35.57</td><td>25.97</td></tr><tr><td> $\mathrm { B C S } + \mathrm { I V }$ </td><td>24.00</td><td>31.79</td><td>44.56</td><td>16.38</td><td>23.34</td><td>49.76</td><td>31.64</td><td>23.92</td><td>41.66</td><td>22.28</td></tr><tr><td> $\mathrm { B E S } + \mathrm { I V }$ </td><td>23.48</td><td>31.34</td><td>41.06</td><td>31.18</td><td>21.38</td><td>51.23</td><td>33.28</td><td>20.35</td><td>38.32</td><td>24.38</td></tr><tr><td> $\mathrm { B C E S + I V }$ </td><td>25.09</td><td>31.86</td><td>44.72</td><td>55.34</td><td>26.06</td><td>51.36</td><td>39.07</td><td>20.08</td><td>40.75</td><td>27.66</td></tr><tr><td> $_ { \mathrm { L o R A + I V } }$ </td><td>26.06</td><td>31.55</td><td>48.20</td><td>69.33</td><td>25.10</td><td>53.36</td><td>42.27</td><td>30.14</td><td>45.60</td><td>22.45</td></tr></table>

Table 11: Results from 3B experiments on BonEval and supplementary tasks Svenska Dagbladet (SvD) SEO Headline, Aftonbladet (AB) Summary, and SwedishFacts. Micro-averaged $\mathrm { F _ { 1 } }$ is reported for Entity, and MCC for Topic, Quiz, and SwedishFacts. For all other tasks, CHRF3++ is reported. Underline denotes section-best and bold overall best.

$$
1 0 . 4 4 \pm 5 . 5 2 / 1 3 . 1 7 \pm 4 . 0 4
$$

$$
7 9 . 9 0 \pm 0 . 8 6 / 7 5 . 3 6 \pm 1 . 9 5
$$

$$
2 6 . 7 2 \pm 2 . 5 8 / 2 0 . 2 9 \pm 2 . 6 5
$$

$$
3 6 . 0 2 \pm 2 . 1 6 / 2 6 . 6 7 \pm 2 . 2 2
$$

$$
9 . 8 4 \pm 1 . 7 2 / 5 0 . 8 3 \pm 3 . 0 1
$$

$$
7 9 . 1 1 \pm 0 . 4 2 / 7 5 . 5 6 \pm 1 . 4 0
$$

$$
8 . 8 1 \pm 2 . 2 6 / 5 3 . 3 4 \pm 0 . 9 7
$$

$$
4 5 . 7 4 \pm 2 . 0 2 / 3 1 . 9 9 \pm 2 . 8 6
$$

$$
3 4 . 3 4 \pm 4 . 2 8 / 6 3 . 4 1 \pm 4 . 6 1
$$

$$
8 0 . 1 9 \pm 0 . 8 9 / 7 7 . 6 1 \pm 1 . 6 8
$$

$$
6 6 . 4 4 \pm 2 . 7 3 / 5 6 . 1 3 \pm 5 . 1 4
$$

$$
7 7 . 5 3 \pm 1 . 4 0 / 7 7 . 7 2 \pm 1 . 6 0
$$

$$
5 1 . 7 0 \pm 3 . 1 8 / 7 2 . 7 5 \pm 2 . 5 7
$$

$$
6 0 . 5 1 \pm 1 . 8 0 / 3 7 . 4 9 \pm 3 . 1 2
$$

$$
5 0 . 6 2 \pm 1 . 6 2 / 7 3 . 5 8 \pm 1 . 5 3
$$

$$
7 7 . 0 5 \pm 1 . 2 3 / 7 8 . 5 8 \pm 0 . 9 6
$$

$$
6 5 . 2 1 \pm 2 . 3 5 / 4 6 . 8 4 \pm 5 . 9 2
$$

$$
4 7 . 7 7 \pm 2 . 1 1 / 7 3 . 3 8 \pm 1 . 1 3
$$

$$
8 0 . 0 1 \pm 1 . 1 9 / 7 8 . 6 8 \pm 0 . 8 3
$$

$$
5 0 . 6 4 \pm 2 . 5 4 / 3 9 . 6 9 \pm 4 . 4 6
$$

$$
\mathrm { F F T } + \mathrm { I V }
$$

$$
3 9 . 6 1 \pm 3 . 3 5 / 6 8 . 0 7 \pm 2 . 4 9
$$

$$
7 9 . 0 7 \pm 1 . 5 1 / 7 9 . 9 6 \pm 0 . 9 0
$$

$$
4 5 . 8 4 \pm 3 . 3 3 / 3 0 . 8 5 \pm 3 . 4 5
$$

$$
8 0 . 5 5 \pm 0 . 7 1 / 7 9 . 6 8 \pm 1 . 1 0
$$

$$
2 2 . 5 4 \pm 2 . 0 8 / 5 2 . 4 9 \pm 3 . 1 0
$$

$$
_ { \mathrm { L o R A + I V } }
$$

$$
6 6 . 8 6 \pm 2 . 4 0 / 5 7 . 7 0 \pm 4 . 6 6
$$

$$
7 8 . 2 7 \pm 0 . 9 9 / 7 9 . 7 4 \pm 0 . 7 2
$$

$$
5 9 . 8 5 \pm 2 . 3 5 / 7 9 . 3 5 \pm 1 . 4 5
$$

$$
6 2 . 2 7 \pm 1 . 7 3 / 3 7 . 5 8 \pm 3 . 2 6
$$

$$
5 4 . 2 4 \pm 2 . 1 2 / 7 6 . 6 8 \pm 1 . 1 6
$$

$$
2 . 9 2 \pm 0 . 7 8 / 2 4 . 3 8 \pm 0 . 7 4
$$

$$
2 . 0 5 \pm 1 . 2 6 / 2 5 . 9 1 \pm 0 . 8 0
$$

$$
1 7 . 0 2 \pm 1 . 2 4 / 3 7 . 2 7 \pm 0 . 8 7
$$

$$
5 . 6 4 \pm 1 . 4 4 / 2 9 . 1 5 \pm 1 . 0 2
$$

$$
4 1 . 5 4 \pm 1 . 0 4 / 5 5 . 8 1 \pm 0 . 8 5
$$

$$
3 2 . 5 4 \pm 1 . 8 0 / 4 8 . 6 1 \pm 1 . 5 2
$$

$$
5 8 . 1 0 \pm 0 . 9 0 / 6 8 . 4 4 \pm 0 . 6 7
$$

$$
4 3 . 6 8 \pm 3 . 1 9 / 5 6 . 1 6 \pm 2 . 7 4
$$

$$
5 6 . 2 2 \pm 1 . 3 2 / 6 6 . 9 3 \pm 1 . 0 1
$$

$$
5 7 . 2 7 \pm 1 . 2 2 / 6 7 . 8 8 \pm 0 . 9 2
$$

$$
4 9 . 9 6 \pm 1 . 6 1 / 6 2 . 3 9 \pm 1 . 2 0
$$

$$
6 4 . 2 7 \pm 1 . 5 7 / 7 3 . 0 0 \pm 1 . 2 7
$$

$$
3 0 . 6 2 \pm 0 . 9 4 / 4 7 . 9 3 \pm 0 . 7 4
$$

$$
\mathrm { F F T } + \mathrm { I V }
$$

$$
1 7 . 6 6 \pm 2 . 1 7 / 3 7 . 3 1 \pm 1 . 6 6
$$

$$
2 0 . 0 9 \pm 2 . 1 1 / 3 9 . 4 9 \pm 2 . 0 8
$$

$$
1 2 . 4 8 \pm 3 . 4 6 / 3 2 . 9 2 \pm 3 . 2 2
$$

$$
5 3 . 3 4 \pm 0 . 9 1 / 6 4 . 8 0 \pm 0 . 6 3
$$

$$
_ { \mathrm { L o R A + I V } }
$$

$$
3 6 . 8 2 \pm 2 . 9 6 / 5 0 . 7 5 \pm 2 . 6 2
$$

$$
4 7 . 1 8 \pm 1 . 5 0 / 5 9 . 8 6 \pm 1 . 1 6
$$

$$
5 2 . 0 6 \pm 1 . 6 0 / 6 3 . 8 0 \pm 1 . 2 0
$$

$$
6 9 . 4 7 \pm 1 . 9 5 / 5 4 . 2 8 \pm 2 . 5 2
$$

$$
3 0 . 6 2 \pm 1 . 0 9 / 3 2 . 8 9 \pm 1 . 2 6
$$

$$
7 2 . 2 2 \pm 1 . 7 4 / 5 6 . 1 6 \pm 2 . 1 5
$$

$$
3 2 . 2 3 \pm { 0 . 8 6 } / { 3 3 . 1 4 } \pm { 1 . 0 3 }
$$

$$
7 4 . 0 5 \pm 1 . 9 9 / 5 5 . 9 3 \pm 2 . 6 6
$$

$$
3 3 . 3 9 \pm 0 . 8 5 / 3 5 . 0 3 \pm 0 . 9 8
$$

$$
7 5 . 7 1 \pm 2 . 8 9 / 5 9 . 7 1 \pm 3 . 4 5
$$

$$
3 5 . 3 0 \pm 1 . 1 3 / 3 7 . 3 0 \pm 1 . 4 5
$$

$$
7 4 . 2 8 \pm 1 . 7 5 / 5 6 . 3 2 \pm 2 . 0 7
$$

$$
3 5 . 3 1 \pm 0 . 2 0 / 3 8 . 5 4 \pm 0 . 2 0
$$

$$
7 9 . 4 4 \pm 1 . 9 3 / 6 4 . 0 1 \pm 2 . 5 4
$$

$$
3 1 . 1 6 \pm 1 . 2 8 / 3 7 . 3 9 \pm 1 . 1 2
$$

$$
7 0 . 5 3 \pm 1 . 4 0 / 5 2 . 9 2 \pm 1 . 3 1
$$

$$
3 1 . 9 4 \pm 1 . 8 5 / 3 2 . 8 9 \pm 2 . 1 1
$$

$$
\mathrm { F F T } + \mathrm { I V }
$$

$$
7 0 . 0 4 \pm 1 . 8 9 / 5 6 . 3 6 \pm 2 . 2 1
$$

$$
3 6 . 0 3 \pm 0 . 8 3 / 3 8 . 2 0 \pm 1 . 0 5
$$

$$
6 9 . 2 5 \pm 2 . 2 7 / 5 1 . 6 7 \pm 2 . 6 8
$$

$$
3 5 . 0 3 \pm 1 . 1 4 / 3 6 . 8 4 \pm 1 . 3 1
$$

$$
_ { \mathrm { L o R A + I V } }
$$

$$
7 0 . 0 5 \pm 2 . 0 9 / 5 0 . 2 3 \pm 2 . 6 1
$$

$$
3 6 . 2 7 \pm 0 . 3 3 / 3 9 . 5 7 \pm 0 . 3 6
$$

Table 12: Results for BonLM 8B and baselines on Swedish EuroEval tasks, showing primary and secondary metrics with 95% confidence intervals. MCC / macro-averaged F<sub>1</sub> are reported for SweRec and ScaLA, micro-averaged $\mathrm { F _ { 1 } }$ without / with MISC for SUC 3.0, MCC / accuracy for MMLU and HellaSwag, token-averaged $\mathrm { F _ { 1 } }$ / exact match for MultiWikiQA, and CHRF3++ / CHRF4++ for SweDN.

<table><tr><td>Model</td><td>Headline</td><td>Lead</td><td>Summary</td></tr><tr><td>GPT-SW3 6.7B</td><td> $1 7 . 7 0 \pm 0 . 9 8 / 1 7 . 8 5 \pm 1 . 0 6$ </td><td> $2 2 . 7 8 \pm 1 . 7 9 / 2 3 . 3 6 \pm 2 . 0 2$ </td><td> $3 6 . 1 1 \pm 1 . 9 3 / 3 7 . 1 7 \pm 2 . 1 3$ </td></tr><tr><td>Llama SW3 8B</td><td> $1 9 . 3 5 \pm 0 . 9 7 / 1 9 . 4 1 \pm 1 . 0 1$ </td><td> $2 9 . 1 3 \pm 1 . 3 3 / 2 9 . 8 6 \pm 1 . 3 9$ </td><td> $4 0 . 5 1 \pm 2 . 0 8 / 4 0 . 9 0 \pm 2 . 3 0$ </td></tr><tr><td>Apertus 8B</td><td> $1 8 . 6 1 \pm 0 . 8 8 / 1 8 . 6 6 \pm 0 . 9 2$ </td><td> $2 3 . 0 9 \pm 2 . 4 2 / 2 3 . 2 2 \pm 2 . 5 9$ </td><td> $4 0 . 9 3 \pm 2 . 2 3 / 4 1 . 6 0 \pm 2 . 4 5$ </td></tr><tr><td colspan="4">Ministral 3 8B</td></tr><tr><td>Base</td><td> $1 8 . 3 5 \pm 0 . 7 4 / 1 8 . 4 0 \pm 0 . 7 7$ </td><td> $2 5 . 7 9 \pm 2 . 2 2 / 2 6 . 0 7 \pm 2 . 4 1$ </td><td> $4 2 . 4 9 \pm 3 . 0 6 / 4 3 . 0 1 \pm 3 . 4 4$ </td></tr><tr><td>Instruct</td><td> $2 9 . 2 8 \pm 0 . 2 1 / 3 1 . 0 2 \pm 0 . 2 9$ </td><td> $3 2 . 0 3 \pm { 0 . 8 0 } / { 3 3 . 1 9 } \pm { 1 . 0 4 }$ </td><td> $4 6 . 2 7 \pm 0 . 4 5 / 4 8 . 7 4 \pm 0 . 4 3$ </td></tr><tr><td>Reasoning</td><td> $2 1 . 3 6 \pm 1 . 5 0 / 2 1 . 5 8 \pm 1 . 5 8$ </td><td> $2 7 . 6 8 \pm 1 . 7 1 / 2 7 . 9 2 \pm 1 . 8 8$ </td><td> $4 7 . 9 3 \pm 1 . 3 9 / 4 9 . 3 1 \pm 1 . 6 8$ </td></tr><tr><td colspan="4">BonLM 8B</td></tr><tr><td>FFT</td><td> $2 1 . 5 8 \pm 1 . 2 5 / 2 1 . 6 9 \pm 1 . 3 0$ </td><td> $2 9 . 8 8 \pm 2 . 5 5 / 3 0 . 4 4 \pm 2 . 7 5$ </td><td> $4 3 . 7 6 \pm 3 . 0 9 / 4 4 . 2 0 \pm 3 . 3 6$ </td></tr><tr><td> $\mathrm { F F T } + \mathrm { I V }$ </td><td> $2 4 . 3 2 \pm 1 . 6 8 / 2 4 . 5 6 \pm 1 . 7 7$ </td><td> $3 1 . 2 4 \pm 1 . 4 9 / 3 1 . 7 1 \pm 1 . 6 9$ </td><td> $4 3 . 2 6 \pm 2 . 3 0 / 4 3 . 5 3 \pm 2 . 4 9$ </td></tr><tr><td> $\operatorname { L o R A }$ </td><td> $2 1 . 2 9 \pm 0 . 9 1 / 2 1 . 3 8 \pm 0 . 9 4$ </td><td> $3 1 . 0 7 \pm 1 . 9 0 / 3 2 . 0 1 \pm 2 . 1 4$ </td><td> $4 4 . 3 5 \pm 2 . 9 4 / 4 5 . 0 3 \pm 3 . 3 0$ </td></tr><tr><td> $_ { \mathrm { L o R A + I V } }$ </td><td> $2 9 . 7 6 \pm 0 . 5 6 / 3 0 . 7 3 \pm 0 . 6 6$ </td><td> $3 5 . 0 1 \pm 0 . 3 8 / 3 6 . 7 9 \pm 0 . 6 2$ </td><td> $4 9 . 8 8 \pm 0 . 8 4 / 5 1 . 3 2 \pm 1 . 0 5$ </td></tr><tr><td colspan="4"></td></tr><tr><td>Model</td><td>Topic</td><td>Entity</td><td>Quiz</td></tr><tr><td>GPT-SW3 6.7B</td><td> $0 . 4 9 \pm 0 . 2 0 / 0 . 2 8 \pm 0 . 0 6$ </td><td> $2 7 . 5 0 \pm 1 . 4 5 / 2 7 . 5 0 \pm 1 . 4 5$ </td><td> $6 . 0 0 \pm 1 . 1 1 / 2 8 . 7 2 \pm 1 . 0 3$ </td></tr><tr><td>Llama SW3 8B</td><td> $6 8 . 2 8 \pm 0 . 7 6 / 4 7 . 7 3 \pm 0 . 7 8$ </td><td> $2 6 . 1 7 \pm 0 . 6 4 / 2 6 . 1 7 \pm 0 . 6 4$ </td><td> $6 1 . 3 0 \pm 0 . 6 9 / 7 0 . 8 3 \pm 0 . 5 4$ </td></tr><tr><td>Apertus 8B</td><td> $7 0 . 9 5 \pm 0 . 6 1 / 5 1 . 7 5 \pm 0 . 8 4$ </td><td> $2 2 . 8 1 \pm 2 . 5 8 / 2 2 . 8 1 \pm 2 . 5 8$ </td><td> $6 1 . 0 5 \pm 0 . 6 2 / 7 0 . 7 4 \pm 0 . 4 9$ </td></tr><tr><td colspan="4">Ministral 3 8B</td></tr><tr><td>Base</td><td> $7 2 . 8 4 \pm 0 . 7 6 / 5 9 . 8 4 \pm 1 . 3 6$ </td><td> $2 9 . 9 7 \pm 0 . 9 6 / 2 9 . 9 7 \pm 0 . 9 6$ </td><td> $6 0 . 1 2 \pm 0 . 5 6 / 6 9 . 9 6 \pm 0 . 4 3$ </td></tr><tr><td>Instruct</td><td> $7 1 . 9 3 \pm 1 . 0 2 / 5 9 . 3 9 \pm 1 . 2 5$ </td><td> $2 3 . 2 0 \pm 0 . 2 8 / 2 3 . 2 0 \pm 0 . 2 8$ </td><td> $5 7 . 7 6 \pm 0 . 5 5 / 6 8 . 0 8 \pm 0 . 4 3$ </td></tr><tr><td>Reasoning</td><td> $7 1 . 0 6 \pm 1 . 1 7 / 5 5 . 4 6 \pm 1 . 2 0$ </td><td> $2 0 . 6 7 \pm 1 . 2 4 / 2 0 . 6 7 \pm 1 . 2 4$ </td><td> $5 2 . 9 1 \pm 0 . 4 3 / 6 4 . 5 0 \pm 0 . 3 3$ </td></tr><tr><td colspan="4">BonLM 8B</td></tr><tr><td>FFT</td><td> $7 2 . 4 0 \pm 0 . 5 8 / 5 3 . 9 2 \pm 1 . 2 8$ </td><td> $2 8 . 3 4 \pm 1 . 0 7 / 2 8 . 3 4 \pm 1 . 0 7$ </td><td> $6 8 . 8 6 \pm 0 . 3 3 / 7 6 . 5 9 \pm 0 . 2 5$ </td></tr><tr><td> $\mathrm { F F T } + \mathrm { I V }$ </td><td> $4 2 . 2 8 \pm 9 . 3 0 / 2 9 . 8 2 \pm 6 . 4 2$ </td><td> $2 8 . 3 8 \pm 0 . 8 2 / 2 8 . 3 8 \pm 0 . 8 2$ </td><td> $6 1 . 9 8 \pm 0 . 6 2 / 7 1 . 3 4 \pm 0 . 5 1$ </td></tr><tr><td> $\operatorname { L o R A }$ </td><td> $6 8 . 9 3 \pm 0 . 7 7 / 5 7 . 4 9 \pm 0 . 9 5$ </td><td> $2 9 . 8 1 \pm 0 . 5 7 / 2 9 . 8 1 \pm 0 . 5 7$ </td><td> $7 1 . 3 6 \pm 0 . 3 5 / 7 8 . 4 9 \pm 0 . 2 7$ </td></tr><tr><td> $_ { \mathrm { L o R A + I V } }$ </td><td> $7 0 . 5 4 \pm 1 . 5 3 / 5 8 . 2 1 \pm 1 . 4 8$ </td><td> $2 3 . 9 9 \pm 0 . 2 7 / 2 3 . 9 9 \pm 0 . 2 7$ </td><td> $6 6 . 9 5 \pm 0 . 6 1 / 7 5 . 0 9 \pm 0 . 4 7$ </td></tr></table>

Table 13: Results for BonLM 8B and baselines on BonEval, showing primary and secondary metrics with 95% confidence intervals. Micro-averaged $\mathrm { F _ { 1 } }$ without / with MISC are reported for Entity, MCC / macro-averaged $\mathrm { F _ { 1 } }$ for Topic, and MCC / accuracy for Quiz. For all other tasks, CHRF3++ / CHRF4++ is reported.