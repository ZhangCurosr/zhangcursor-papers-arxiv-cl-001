# MAPLE: A Multi-Axis Parametric Knowledge Extraction Lens for LLM Evaluation

Minsuh Joo<sup>1</sup>, Youna Kim<sup>2</sup>, Choonghyun Park<sup>2</sup>,

Soo Kyung Kim<sup>13</sup>, Hyunsoo Cho<sup>13∗</sup>

<sup>1</sup>Department of AI, Ewha Womans University <sup>2</sup> Seoul National University <sup>3</sup>Institute for Multiscale Matter and Systems, Ewha Womans University {judyjoo21, chohyunsoo}@ewha.ac.kr

## Abstract

LLMs absorb vast knowledge from training data, yet their inherently probabilistic nature offers no guarantee that this knowledge will surface in their outputs. To bridge this gap, Knowledge Extraction Enhancement (KEE) techniques such as activation steering, decoding adjustments, and prompting modify inference, potentially reinforcing existing predictions or surfacing less accessible answers. Yet their evaluations centered on aggregate accuracy, which obscures diverse effects by collapsing recovered failures and disrupted successes into a single score. We introduce MAPLE, a unified framework for characterizing how KEE techniques affect parametric knowledge extraction. Its diagnostic metrics disentangle these effects at the surface, probability, and beyond-sampling levels by measuring answer recovery and disruption, gold-token rank shifts on unrecovered failures, and recovery of answers missed by fixed-budget sampling. We evaluate six KEE techniques spanning decoding, activation steering, and prompting across three model families: OLMo2, Qwen2.5, and Llama3. Our analysis reveals distinct effects and trade-offs across techniques, highlighting the value of MAPLE for understanding and comparing KEE beyond aggregate accuracy.

## 1 Introduction

LLMs serve as vast repositories of factual knowledge, yet their outputs are not a direct readout of what they have learned (Petroni et al., 2019; Jiang et al., 2020). Unlike traditional knowledge bases that return explicitly stored entries, LLMs construct answers from parametric representations through probabilistic inference. A common implicit assumption is that a model knows a fact if it appeared in the pretraining data or if the model answers it correctly. However, prior work shows that both criteria are insufficient, as facts vary considerably in the strength of their encoding and the reliability of their recall (Kandpal et al., 2023; Mallen et al., 2023). The inference pathway from input prompts through hidden representations and token distributions to generated answers (x → h → p → y) therefore offers multiple points of intervention for improving knowledge extraction. Various inference-time techniques, which we group under Knowledge Extraction Enhancement (KEE), intervene at different stages of this pathway. Prompting methods modify the input context to elicit relevant knowledge (Sun et al., 2023; Yu et al., 2023), activation steering methods shift internal representations (Li et al., 2023a; Rimsky et al., 2024), and decoding methods recalibrate output distributions (Chuang et al., 2024; Li et al., 2023b). Despite their methodological differences, these techniques share the goal of improving parametric knowledge extraction without updating the underlying model’s parameters.

![](images/5f8e57698c5bed1f4f75960b42dcc57ab367e7622d3e9f3aaaba2ec19c08898d.jpg)  
Figure 1: The overview of MAPLE.

Their evaluation, however, remains fragmented, as KEE methods typically reuse benchmarks designed for different objectives, with heterogeneous task formats and metrics (Mallen et al., 2023; Lin et al., 2022; Yang et al., 2018; Thorne et al., 2018). Following these benchmarks’ original emphasis on task performance, KEE evaluations often focus on aggregate accuracy rather than on how interventions change individual predictions (Kwiatkowski et al., 2019; Joshi et al., 2017; Mallen et al., 2023). For instance, two methods can achieve the same accuracy while one largely preserves baseline answers and the other recovers many failures but disrupts many successes. These limitations call for a benchmark that supports consistent comparisons across KEE techniques and diagnoses their effects beyond aggregate accuracy.

We introduce MAPLE, a unified framework for diagnosing how KEE techniques change factual knowledge extraction relative to a model’s own baseline (Figure 1). MAPLE combines a standard ized benchmark with multi-axis diagnostic metrics, providing a common basis for comparing KEE techniques. At the output level, we track correctness changes for each question, separately quantifying the recovery of baseline failures and the disrup tion of baseline successes. This separation reveals how much a technique recovers and what it sac rifices, exposing trade-offs that a single accuracy score can conceal. Beyond these correctness transitions, we also examine questions that remain incorrectly answered by tracking how the rank of the first gold token shifts between baseline and KEE technique. This probability level reveals whether the gold token is ranked higher, equal or lower un der the KEE technique, which is invisible at the surface level. Moreover, prior evaluations rarely ask whether a technique recovers answers beyond what the model already reaches through its own sampling. Since sampling requires no intervention, a KEE technique offers a real benefit when it recovers answers that sampling alone cannot reach. We further improve the reliability of our benchmark by verifying the presence of the queried facts in pretraining data, reducing the risk of conflating a lack of prior exposure with the failure of a KEE technique. To this end, we construct the dataset from a pre-cutoff Wikipedia dump and retain only facts whose sentences appear verbatim in the pretraining corpus of OLMo2 (OLMo et al., 2024), for which training data is publicly available. Although corpus presence does not guarantee successful encoding or extractability, this verification establishes that OLMo2 had an opportunity to learn each fact, and it serves as a proxy for the other models that share the same Wikipedia source.

We evaluate six KEE techniques across three model families (OLMo2, Qwen2.5, and Llama3)

using MAPLE. This unified comparison reveals distinct patterns that aggregate accuracy alone cannot explain. Among the evaluated techniques, prompting methods provide the strongest recovery, including answers missed by fixed-budget sampling, but also disrupt previously correct responses. ITI remains close to the baseline across the diagnostic axes, whereas the evaluated decoding methods and CAA alter gold-token ranks on unrecovered failures without consistently improving them. The contrast between ITI and CAA further shows that methods acting at the same stage of inference need not produce similar effects. Together, these findings demonstrate the value of comparing KEE techniques under a shared diagnostic framework, rather than treating aggregate accuracy as a complete account of their effects on knowledge extraction.

Our contributions are threefold:

• A standardized factual-recall benchmark that provides a common evaluation task across KEE families, with verified pretrainingcorpus presence for OLMo2.

• A multi-axis diagnostic protocol that separates answer recovery and disruption, goldtoken rank shifts on unrecovered failures, and recovery relative to fixed-budget sampling.

• A unified empirical analysis across six techniques and three model families, revealing differences within and across KEE families that are obscured by aggregate accuracy.

## 2 Related Work

KEE Techniques. Numerous methods aim to improve parametric knowledge extraction without additional training, but they span diverse tasks and methodological approaches. Decoding techniques recalibrate output distributions using contrastive signals from amateur counterparts, such as smaller models (Contrastive Decoding; Li et al., 2023b), early layers (DoLa; Chuang et al., 2024), masked attention heads (DeCoRe; Gema et al., 2025), or activation entropy (Activation Decoding; Chen et al., 2024). Steering techniques shift internal activations toward truthful directions, using steering vectors obtained from linear probes (ITI; Li et al., 2023a), contrastive examples (CAA; Rimsky et al., 2024), or input-adaptive semantics (SADI; Wang et al., 2025). Prompting techniques take a two-stage approach, first eliciting the model’s implicit knowledge as explicit text, then conditioning the final generation on this self-produced context, using either demonstrations (RECITE; Sun et al., 2023) or zero-shot instructions (GenRead; Yu et al., 2023).

Knowledge and Factuality Benchmarks. Existing benchmarks used to evaluate KEE techniques were originally designed for other purposes: TriviaQA (Joshi et al., 2017) is constructed from openweb and Wikipedia sources; Natural Questions (Kwiatkowski et al., 2019) is built from Google search queries; HotpotQA (Yang et al., 2018) measures multi-hop reasoning; TruthfulQA (Lin et al., 2022) evaluates susceptibility to human misconceptions; FACTOR (Muhlgay et al., 2024) and FEVER (Thorne et al., 2018) assess the capability to choose the factually correct option from explicitly given candidates, rather than generating the answer from the model’s own knowledge; and PopQA (Mallen et al., 2023) focuses narrowly on long-tail entities rather than balanced knowledge distributions.

## 3 MAPLE

MAPLE comprises two components: an exposurecontrolled benchmark (§3.1) and a set of diagnostic metrics (§3.2) that characterize how a KEE technique changes model predictions.

## 3.1 Benchmark Construction

We construct our benchmark through a three-stage pipeline. We first extract knowledge triplets, the atomic units of factual knowledge, from a precutoff Wikipedia dump. We then apply multi-stage filtering to retain triplets that are valid and unambiguous while preserving broad knowledge coverage. Finally, we verify the presence of each remaining triplet in OLMo2’s open pretraining corpus to further strengthen its reliability by reducing the risk of conflating a lack of prior exposure with ineffective knowledge extraction.

Stage 1: Knowledge Triplet Extraction. Following Shi et al. (2024); Zhang et al. (2025); Fu et al. (2025), we assume that Wikipedia data predating a model’s training cutoff has been seen during the model’s pretraining. As KEE techniques aim to recover knowledge acquired during pretraining, we leverage this assumption by constructing our benchmark from a March 2022 Wikipedia dump, ensuring temporal precedence over the training cutoffs of all evaluated models. Temporal precedence alone does not guarantee that a specific fact appeared in a model’s pretraining data, so we directly verify corpus presence in Stage 3. Since a single text can encompass multiple facts of heterogeneous extraction difficulty, document- or passage-level evaluation is too coarse. We therefore decompose the unstructured corpora into atomic (subject, relation, object) triplets using REBEL (Cabot and Navigli, 2021), a seq2seq relation-extraction model. Finally, to ensure broad topical coverage, we map document entities to Wikidata’s instance of properties, organizing them into a custom taxonomy of ten domains (e.g., science, history, law). Table 4 shows the full domain categories with representative instance of examples. Before extraction, we remove list-style articles, HTML artifacts, and documents under 100 words from the source corpus.

Stage 2: Multi-stage Filtering. Raw triplets from Stage 1 often contain noise and exhibit skewed distributions. To ensure validity, we map entities and relations to Wikidata identifiers using Wikimapper and verify their existence via the Wikidata API. We discard instances with domain mismatches via LLM-based filtering<sup>\*</sup>; specifically, we remove overly generic triplets that do not reflect their document’s assigned domain. For quality control, we eliminate triplets containing uninformative relations (e.g., opposite of, different from) or direct lexical overlap between the subject and the object. Crucially, we restrict our benchmark strictly to single-answer triplets using the Wikidata SPARQL API and LLM-based classification; multi-answer cases introduce evaluation ambiguity by conflating a model’s lack of knowledge with its preference for an alternative valid entity. In order to label entity frequency, we utilize monthly Wikipedia pageviews<sup>†</sup> as a proxy for pretraining exposure. Using percentile thresholds, we assign each entity to a head, torso, or tail bin, incorporating 20% buffer zones between adjacent bins (10% head, 40% torso, 10% tail, separated by 20% buffer margins) to ensure that frequency bins are separated by distinct margins rather than arbitrary cutoffs. Finally, to prevent specific relation dominance, we cap each relation at 300 instances, yielding a candidate pool that spans a broad frequency distribution. This pool consists of 3,254 triplets in a short-answer question format, which we further verify against pretraining data in Stage 3. See Appendix A for benchmark construction details.

Stage 3: Corpus Presence Filtering. Temporal precedence in Stage 1 does not establish whether individual facts appeared in pretraining data. We therefore filter the remaining triplets against OLMo-mix-1124 (OLMo et al., 2024), OLMo2’s publicly available pretraining corpus, through the following three steps.

Candidate sentences. We map each triplet $( s , r , o )$ from Section 3.1 to the Wikipedia document from which REBEL extracted it and split the document into sentences. A sentence becomes a candidate when it mentions both the subject and the object, where we also match their Wikidata aliases. We match the subject and object rather than the relation because REBEL outputs relations from a fixed set of about 200 labels, which rarely appear verbatim in the text.

Relation verification. A sentence that mentions both entities does not necessarily express the target relation, since the same pair can be linked by different relations. We therefore feed each candidate sentence back into REBEL and keep it as a supporting sentence only if REBEL re-extracts the same triplet $( s , r , o )$ with aliases taken into account. This step ensures that each supporting sentence expresses the target triplet rather than merely mentioning both entities.

Corpus presence. We query each supporting sentence against OLMo-mix-1124 with exact matching through the infini-gram API (Liu et al., 2024) and record its count. A triplet is retained if at least one of its supporting sentences appears in the corpus. Of the 3,254 candidate triplets, 1,303 pass all three steps and form our final benchmark. The filter is conservative, as a removed triplet may still appear in the corpus in a paraphrased form or in a sentence our pipeline does not recover.

## 3.2 Evaluation Metrics

We introduce three complementary evaluation metrics that systematically capture changes at the surface, probability, and beyond-sampling levels. Surface level computes Gain and Loss based on generated outputs of models. The probability level captures internal dynamics via Probability Rank Shift (PRS), which measures how a KEE technique moves the rank of the gold token on failures that remain unrecovered. The beyond-sampling level compares each technique with temperature sampling through the Shallow Gain Rate (SGR) and

Deep Gain Rate (DGR). Together, these three dimensions constitute a diagnostic evaluation protocol of MAPLE.

Surface Level. At the surface level, we evaluate the model using its actual generated outputs. Traditional accuracy merges the gain and loss resulting from interventions of KEE techniques into a single value. However, since they directly alter the model behavior, a metric that decouples improvement and degradation is required to isolate each effect. Thus, we define Gain and Loss as follows: Gain is the proportion of baseline failures that the KEE technique answers correctly (Eq 1); Loss is the proportion of baseline successes that the KEE technique answers incorrectly (Eq 2).

$$
\mathrm { G a i n } = { \frac { | S _ { \mathrm { K E E } } \cap F _ { \mathrm { b a s e } } | } { | F _ { \mathrm { b a s e } } | } }\tag{1}
$$

$$
\mathrm { L o s s } = \frac { | F _ { \mathrm { K E E } } \cap S _ { \mathrm { b a s e } } | } { | S _ { \mathrm { b a s e } } | }\tag{2}
$$

Here, $S _ { \mathrm { K E E } }$ and $F _ { \mathrm { K E E } }$ denote the sets of instances answered correctly and incorrectly under the KEE technique, respectively, while $S _ { \mathrm { b a s e } }$ and $F _ { \mathrm { b a s e } }$ represent those under the baseline model. By setting the denominators of Gain and Loss to $| F _ { \mathrm { b a s e } } |$ and $| S _ { \mathrm { b a s e } } |$ rather than the total dataset size, we measure recovery and degradation of correct answers.

Probability Level. Surface level evaluation only captures instances whose correctness flips between the baseline and a KEE technique. We therefore trace the internal dynamics at the probability level for the unflipped 0 → 0 instances that remain incorrect under both settings. We refer to these instances as unrecovered failures. Prior work evaluates factual knowledge by how highly a model ranks the gold token (Petroni et al., 2019). We extend this view from static presence to paired change and introduce Probability Rank Shift (PRS), which measures how a KEE technique moves the rank of the gold token on the same fact even when a failure does not turn into a success.

We first compute the probability of the full gold answer with teacher forcing and observe that once the first token is given, the remaining tokens become nearly deterministic and add little information beyond it. We therefore use only the first token, which is predicted under the same condition as free generation. Let $r _ { \mathrm { b a s e } }$ and r<sub>KEE</sub> denote the rank of the first gold token over the full vocabulary under the baseline and the KEE technique. We exclude instances where the gold token is ranked first under either setting, since their failure lies beyond the first token. Then, we assign each remaining instance to one of three categories:

$$
c = \left\{ \begin{array} { l l } { \mathrm { I m p r o v e d } } & { \mathrm { i f } r _ { \mathrm { K E E } } < r _ { \mathrm { b a s e } } } \\ { \mathrm { T i e d } } & { \mathrm { i f } r _ { \mathrm { K E E } } = r _ { \mathrm { b a s e } } } \\ { \mathrm { W o r s e n e d } } & { \mathrm { i f } r _ { \mathrm { K E E } } > r _ { \mathrm { b a s e } } } \end{array} \right.\tag{3}
$$

Improved, Tied and Worsened indicate that the gold token is ranked higher, equal or lower under the KEE technique than under the baseline. We then report PRS as the proportion of each category:

$$
\mathrm { P R S } _ { c } = \frac { N _ { c } } { N _ { \mathrm { I m p r o v e d } } + N _ { \mathrm { T i e d } } + N _ { \mathrm { W o r s e n e d } } }\tag{4}
$$

where $N _ { c }$ is the number of instances in category c ∈ {Improved, Tied, Worsened}.

Beyond-Sampling Level. Beyond-sampling level analysis asks whether a KEE technique recovers instances that the model fails to reach through its own sampling. As a point of comparison, we use temperature sampling at $\tau = 0 . 7 ( \tau _ { 0 . 7 } )$ and $\tau = 1 . 2 ( \tau _ { 1 . 2 } )$ , since changing the temperature alone can also yield gains, which allows us to test whether KEE gains reflect an actual effect beyond sampling noise. We first partition baseline failures by whether sampling reaches the correct answer, using an initial seed (Seed $\mathrm { A } = 4 2 )$ for each temperature. Shallow failures $\mathcal { D } _ { \mathrm { s h a l l o w } }$ are incorrect under greedy decoding but reached at least once within K samples, whereas deep failures $\mathcal { D } _ { \mathrm { d e e p } }$ are never reached within K samples. Shallow and deep thus refer to sampling recoverability for each model rather than to the objective difficulty of a fact. We set $K = 2 0$ by sweeping K from 1 to 30. At $K = 2 0$ , every model already identifies about 90% of the shallow failures found with 30 samples at both temperatures. To check that this choice does not drive our conclusions, we compare the ranking of methods at each K with the ranking at $K \ : = \ : 2 0$ using the Kendall rank correlation, which equals 1 when two rankings agree on the order of every pair of methods. The correlation stays at 0.89 or higher for both SGR and DGR at every K in Table 1, so the ranking of methods remains stable regardless of K. We then evaluate each method $m \in \{ \mathrm { K E E } , \tau _ { 0 . 7 } , \tau _ { 1 . 2 } \}$ on these subsets. For $\tau _ { 0 . 7 }$ and τ , we draw K new samples with an independent seed (Seed $\mathbf B = 0 )$ so that the samples used for partitioning are not reused for evaluation. Using the sizes of $\mathcal { D } _ { \mathrm { { s } } }$ shallow $\mathcal { D } _ { \mathrm { d e e p } }$ as denominators, we define the Shallow Gain Rate (SGR) and Deep Gain Rate (DGR) as Eq. 5 and Eq. 6:

<table><tr><td rowspan="2">K</td><td colspan="2">T0.7</td><td colspan="2">T1.2</td></tr><tr><td>SGR</td><td>DGR</td><td>SGR</td><td>DGR</td></tr><tr><td>10</td><td>0.915</td><td>0.941</td><td>0.988</td><td>0.898</td></tr><tr><td>15</td><td>0.927</td><td>0.969</td><td>0.958</td><td>0.956</td></tr><tr><td>20</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>25</td><td>0.983</td><td>0.991</td><td>0.972</td><td>0.950</td></tr><tr><td>30</td><td>0.956</td><td>0.987</td><td>0.931</td><td>0.950</td></tr></table>

Table 1: Mean Kendall rank correlation with the ranking at $K = 2 0$

$$
\mathrm { S G R } = \frac { | S _ { m } \cap \mathcal { D } _ { \mathrm { s h a l l o w } } | } { | \mathcal { D } _ { \mathrm { s h a l l o w } } | }\tag{5}
$$

$$
\mathrm { D G R } = \frac { | S _ { m } \cap \mathcal { D } _ { \mathrm { d e e p } } | } { | \mathcal { D } _ { \mathrm { d e e p } } | }\tag{6}
$$

where $S _ { m }$ is the set of instances recovered by method m. SGR measures how much a method recovers among failures that sampling can also recover, and DGR measures how much it recovers among failures that sampling cannot recover within K samples. A KEE technique that reaches a DGR above that of temperature sampling therefore recovers failures beyond what sampling alone provides.

## 4 Experiments

## 4.1 Settings

Models. To examine the effectiveness of KEE techniques across different model scales and families, we conduct experiments on OLMo2-7B/32B-Instruct, Qwen2.5-7B/32B-Instruct, and Llama3- 8B/70B-Instruct.

KEE Techniques. As discussed in Section 2, KEE techniques span a wide range of approaches, making it impractical to cover all existing methods. We therefore select two representative techniques from each of the three KEE families that are commonly used as baselines in prior work. From the decoding techniques, we include DeCoRe and Activation Decoding (AD); from the steering techniques, ITI and CAA; from the prompting techniques, RECITE and GenRead. In addition, we compare KEE techniques against temperature sampling, sampled at $\left( \tau _ { 0 . 7 } \right)$ and $\left( \tau _ { 1 . 2 } \right)$ . Detailed configurations for each technique are provided in Appendix B.

Generation Setting. For the no-intervention baseline and all KEE techniques, we use greedy decoding to eliminate sampling noise and ensure reproducibility. Using greedy decoding, any change in the top-1 prediction strictly reflects the intervention impact of the KEE technique, allowing us to precisely measure Gain and Loss. For temperature sampling, the final answer is determined via majority voting across 10 samples. See Appendix C for detailed prompts.

Evaluation & Correctness Metrics. We evaluate the techniques using the metrics introduced in Section 3.2 and accuracy (Acc). For correctness evaluation, we treat both the gold answer and the aliases as valid answers. After normalization— removing special characters and whitespace—a prediction is judged correct if it exactly matches a valid answer or if a valid answer appears as a substring of the prediction.

## 4.2 Main Results

Table 2 reports accuracy together with Gain and Loss for each KEE technique where the temperature sampling serves as the reference. Figure 2 shows PRS on unrecovered failures for each KEE technique. Each bar and its value give the averaged PRS over the six models. Range bars mark the minimum and maximum of that PRS across the six models. Table 3 reports SGR and DGR at $\tau = 1 . 2$ , which separate the gains of KEE technique on shallow failures that sampling reaches within K samples from those on deep failures that sampling never reaches. Full results and raw count breakdown are in Appendix D.

A Diagnostic Lens with Three Complementary Metrics. Each metric in Section 3.2 covers a blind spot that the others leave open. Gain and Loss decompose accuracy into the failures a technique recovers and the successes it breaks, recovering bidirectional shifts that a single accuracy score conflates. Since they register a change only when the answer flips, PRS complements them by tracing how the gold token’s rank moves on unrecovered failures. Raising the sampling temperature alone can also improve correctness without any intervention, so SGR and DGR compare each technique with temperature sampling to test whether its recovery merely reflects what sampling can recover or goes beyond it. Together the metrics capture correctness flips in both directions, the dynamics beneath unflipped failures, and the margin over a temperature sampling. Because all of them rely only on answer flips or ranks, they apply to every KEE technique and remain comparable across decoding, steering and prompting techniques that reshape the output distribution in different ways.

What the Unified Lens Reveals about KEE Techniques. While the three metrics capture distinct dimensions, they are fundamentally interconnected. Probability level dynamics are reflected at the surface level, and a robust signal at the surface level accompanies the recovery of deep failures beyond sampling. Prompting techniques such as RECITE and GenRead exemplify this end-to-end connection. By injecting relevant context directly into the input, they reshape the model’s probability distribution most decisively and leave only a small share of $\mathrm { P R S } _ { \mathrm { T i e d } }$ , which coincides with both the largest Gain and the strongest recovery of deep failures. Steering via ITI lies at the opposite end. It shows the largest share of $\mathrm { P R S } _ { \mathrm { T i e d } }$ across models, and this minimal reshaping is mirrored by small Gain alongside low Loss and DGR close to zero, keeping ITI close to the baseline at every level. Decodingbased techniques such as DeCoRe and AD and steering via CAA fall in between. They reshape the distribution more than ITI, yet this reshaping is not directed toward the gold token as $\mathrm { P R S } _ { \mathrm { W o r s e n e d } }$ exceeds $\mathrm { P R S } _ { \mathrm { I m p r o v e d } }$ on average. Correspondingly, their Gain remains comparable to that of temperature sampling in most models, and their knowledge recovery remains largely confined to shallow failures and rarely extends into deep ones.

## 5 Analysis

## 5.1 Surface Level

Recalling knowledge makes substantial interventions. In Table 2, RECITE and GenRead achieve the highest Gain across all models, while GenRead also incurs the highest Loss in every model. RE-CITE keeps a comparably high Gain whereas its Loss stays lower than GenRead’s in every model.

Decoding-based modifications produce comparatively weak interventions. In Table 2, decoding techniques provide weak interventions whose efficacy varies across model families. On Gain, they achieve 1.6–4.9× higher values than temperature sampling on the Llama3 family, but fall to 0.3–1.5× on the OLMo2 family and Qwen2.5-32B. On Loss, they remain comparable to temperature sampling and markedly lower than prompting techniques, except for a spike on Qwen2.5-7B where DeCoRe and AD reach 9.96% and 11.21%.

<table><tr><td>Category</td><td>Model</td><td>Acc↑</td><td>Gain↑</td><td>Loss↓</td><td>Model</td><td>Acc↑</td><td>Gain↑</td><td>Loss↓</td><td>Model</td><td>Acc↑</td><td>Gain↑</td><td>Loss↓</td></tr><tr><td></td><td>OLMo2-7B</td><td>46.89</td><td></td><td></td><td>Qwen2.5-7B</td><td>43.13</td><td></td><td></td><td>Llama3-8B</td><td>50.58</td><td></td><td></td></tr><tr><td rowspan="2">Sampling</td><td>+ T0.7</td><td>47.35</td><td>3.32</td><td>2.78</td><td> $+ \tau _ { 0 . 7 }$ </td><td>43.21</td><td>1.35</td><td>1.60</td><td> $+ \tau _ { 0 . 7 }$ </td><td>50.88</td><td>2.95</td><td>2.28</td></tr><tr><td> $+ \tau _ { 1 . 2 }$ </td><td>45.43</td><td>3.61</td><td>7.20</td><td> $+ \tau _ { 1 . 2 }$ </td><td>42.98</td><td>3.51</td><td>4.98</td><td> $+ \tau _ { 1 . 2 }$ </td><td>50.27</td><td>6.21</td><td>6.68</td></tr><tr><td rowspan="2">Decoding</td><td>+ DeCoRe</td><td>46.89</td><td>3.32</td><td>3.76</td><td> $+ \mathrm { D e C o R e }$ </td><td>41.29</td><td>4.32</td><td>9.96</td><td>+ DeCoRe</td><td>53.34</td><td>11.18</td><td>5.46</td></tr><tr><td>+ AD</td><td>46.12</td><td>1.88</td><td>3.76</td><td> $+ \textrm { A D }$ </td><td>40.45</td><td>3.78</td><td>11.21</td><td>+ AD</td><td>52.11</td><td>10.09</td><td>6.83</td></tr><tr><td rowspan="2">Steering</td><td>+ ITI</td><td>47.89</td><td>4.91</td><td>3.44</td><td>+ ITI</td><td>42.06</td><td>1.21</td><td>4.09</td><td>+ ITI</td><td>50.96</td><td>4.66</td><td>3.79</td></tr><tr><td>+ CAA</td><td>46.12</td><td>3.03</td><td>5.07</td><td> $+ \mathrm { C A A }$ </td><td>42.67</td><td>1.48</td><td>3.02</td><td>+ CAA</td><td>51.50</td><td>11.02</td><td>8.95</td></tr><tr><td rowspan="2">Prompting</td><td>+ RECITE</td><td>56.25</td><td>23.27</td><td>6.38</td><td>+ RECITE</td><td>50.96</td><td>20.65</td><td>9.07</td><td>+ RECITE</td><td>56.49</td><td>19.57</td><td>7.44</td></tr><tr><td>+ GenRead</td><td>52.65</td><td>19.94</td><td>10.31</td><td>+ GenRead</td><td>52.03</td><td>24.97</td><td>12.28</td><td>+ GenRead</td><td>58.63</td><td>28.88</td><td>12.29</td></tr><tr><td></td><td>OLMo2-32B</td><td>63.24</td><td></td><td>一</td><td>Qwen2.5-32B</td><td>56.10</td><td></td><td></td><td>Llama3-70B</td><td>67.84</td><td></td><td></td></tr><tr><td rowspan="2">Sampling</td><td>+ T0.7</td><td>62.24</td><td>3.13</td><td>3.40</td><td> $+ \tau _ { 0 . 7 }$ </td><td>56.10</td><td>1.40</td><td>1.09</td><td> $+ \tau _ { 0 . 7 }$ </td><td>67.77</td><td>1.67</td><td>0.90</td></tr><tr><td> $+ \tau _ { 1 . 2 }$ </td><td>63.01</td><td>6.89</td><td>4.37</td><td> $+ \tau _ { 1 , 2 }$ </td><td>56.49</td><td>3.32</td><td>1.92</td><td> $+ \tau _ { 1 , 2 }$ </td><td>67.92</td><td>3.10</td><td>1.36</td></tr><tr><td rowspan="2">Decoding</td><td>+ DeCoRe</td><td>63.55</td><td>4.80</td><td>2.31</td><td>+ DeCoRe</td><td>55.72</td><td>1.40</td><td>1.78</td><td>+ DeCoRe</td><td>68.53</td><td>5.49</td><td>1.58</td></tr><tr><td>+ AD</td><td>62.32</td><td>3.13</td><td>3.28</td><td> $+ \textrm { A D }$ </td><td>56.25</td><td>1.05</td><td>0.55</td><td>+ AD</td><td>68.30</td><td>8.11</td><td>3.17</td></tr><tr><td rowspan="2">Steering</td><td>+ ITI</td><td>63.39</td><td>2.92</td><td>1.46</td><td> $+ \ \mathrm { I T I }$ </td><td>55.56</td><td>1.57</td><td>2.19</td><td>+ ITI</td><td>67.92</td><td>2.86</td><td>1.24</td></tr><tr><td>+ CAA</td><td>62.70</td><td>5.85</td><td>4.25</td><td> $+ \mathrm { C A A }$ </td><td>56.41</td><td>2.10</td><td>1.09</td><td>+ CAA</td><td>67.77</td><td>8.59</td><td>4.19</td></tr><tr><td rowspan="2">Prompting</td><td>+ RECITE</td><td>69.38</td><td>26.93</td><td>5.95</td><td> $+ \mathrm { R E C I T E }$ </td><td>61.55</td><td>22.03</td><td>7.52</td><td>+ RECITE</td><td>70.76</td><td>22.43</td><td>6.33</td></tr><tr><td>+ GenRead</td><td>66.69</td><td>23.38</td><td>8.13</td><td> $+ { \mathrm { G e n R e a d } }$ </td><td>59.71</td><td>22.90</td><td>11.49</td><td>+ GenRead</td><td>73.83</td><td>32.70</td><td>6.67</td></tr></table>

Table 2: Gain/Loss (%) decomposition across model families and KEE techniques. Bold and underline indicate the best and second-best values, respectively. Alternating gray shading indicates technique category boundaries.

Activation steering behaviors diverge by specific methodology. CAA aligns more closely with decoding than with its fellow steering technique ITI on both Gain and Loss, exhibiting a similar modeldependent pattern. In Table 2, while CAA’s Gain and Loss stay comparable to temperature sampling on OLMo2 and Qwen2.5, they reach 1.8–5.1× and 1.3–4.7× higher levels on the Llama3 family, respectively. Notably, on Qwen2.5 family, CAA shows both the lowest Gain at 1.48% and 2.10% and the lowest Loss at 3.02% and 1.09% among all six models, suggesting its intervention signal is comparatively muted for this model family. ITI produces a weak intervention signal and stays close to the baseline in every model, with Gain between 1.21% and 4.91% and consistently low Loss among KEE techniques across six models. Since prior decoding and steering literature has been validated predominantly on Llama (Gema et al., 2025; Rimsky et al., 2024), our findings put a spotlight on the evaluation of cross-family generalization as a critical direction for future KEE research.

## 5.2 Probability Level

Unrecovered failures are reshaped without moving toward the gold answer. For decoding techniques and CAA, although they seem ineffective at the surface level with low Gain, most of their unrecovered failures still change the gold rank.

![](images/d72ee5fe2d364aeae8c8342f05383e2d1cb1aac804b379e968a5d14cec9ccb3b.jpg)  
Figure 2: Probability Rank Shift of the gold token on unrecovered failures for KEE techniques, averaged over six models. Range bars show the minimum and maximum across models. Per-model results are in Figure 12.

Only 29.7%, 34.7% and 33.3% of these failures remain Tied for DeCoRe, AD and CAA in Figure 2. This reshaping, however, is not directed toward the gold token. On average, $\mathrm { P R S } _ { \mathrm { W o r s e n e d } }$ exceeds $\mathrm { P R S } _ { \mathrm { I m p r o v e d } }$ for all three techniques, and the gap is pronounced in some models. For instance, on Llama3-70B, AD shows a similar gap at 46.7% against 13.1% in Table 15. These results indicate that most unchanged cases are still affected, but the change rarely brings the gold answer closer to the top.

ITI leaves most unrecovered failures untouched. ITI shows the largest $\mathrm { P R S } _ { \mathrm { T i e d } }$ at 62.5% in Figure 2, ranging from 53.0% to 72.2% across models in Table 15. Its $\mathrm { P R S } _ { \mathrm { I m p r o v e d } }$ and $\mathrm { P R S } _ { \mathrm { W o r s e n e d } }$ are nearly balanced at 18.2% and 19.3%, so the few changes it makes carry no consistent direction. This minimal reshaping is consistent with its small Gain and low Loss at the surface level in Table 2.

<table><tr><td>Category</td><td>Model</td><td>SGR↑</td><td>DGR↑</td><td>Model</td><td>SGR↑</td><td>DGR↑</td><td>Model</td><td>SGR↑</td><td>DGR↑</td></tr><tr><td></td><td>OLMo2-7B</td><td></td><td>1</td><td>Qwen2.5-7B</td><td>一</td><td>一</td><td>Llama3-8B</td><td></td><td>一</td></tr><tr><td>Sampling</td><td>+ T1.2</td><td>16.06</td><td>0.42</td><td>+ T1.2</td><td>18.67</td><td>0.17</td><td>+ T1.2</td><td>15.72</td><td>0.24</td></tr><tr><td rowspan="2">Decoding</td><td>+ DeCoRe</td><td>9.63</td><td>0.42</td><td>+ DeCoRe</td><td>16.00</td><td>1.35</td><td>+ DeCoRe</td><td>24.02</td><td>4.10</td></tr><tr><td>+ AD</td><td>5.96</td><td>0.00</td><td>+ AD</td><td>15.33</td><td>0.85</td><td>+ AD</td><td>20.96</td><td>4.10</td></tr><tr><td rowspan="2">Steering</td><td>+ ITI</td><td>14.68</td><td>0.42</td><td>+ ITI</td><td>6.00</td><td>0.00</td><td>+ ITI</td><td>10.04</td><td>1.69</td></tr><tr><td>+ CAA</td><td>9.63</td><td>0.00</td><td>+ CAA</td><td>6.67</td><td>0.17</td><td>+ CAA</td><td>23.58</td><td>4.10</td></tr><tr><td rowspan="2">Prompting</td><td>+ RECITE</td><td>51.83</td><td>10.13</td><td>+ RECITE</td><td>53.33</td><td>12.35</td><td>+ RECITE</td><td>41.48</td><td>7.47</td></tr><tr><td>+ GenRead</td><td>44.50</td><td>8.65</td><td>+ GenRead</td><td>60.00</td><td>16.07</td><td>+ GenRead</td><td>52.84</td><td>15.66</td></tr><tr><td></td><td>OLMo2-32B</td><td></td><td>–</td><td>Qwen2.5-32B</td><td>I</td><td>一</td><td>Llama3-70B</td><td></td><td>1</td></tr><tr><td rowspan="2">Sampling</td><td>+ T1.2</td><td>13.07</td><td>0.00</td><td>+ T1.2</td><td>14.55</td><td>0.22</td><td>+ T1.2</td><td>16.33</td><td>0.00</td></tr><tr><td>+ DeCoRe</td><td>10.55</td><td>0.71</td><td>+ DeCoRe</td><td>7.27</td><td>0.00</td><td>+ DeCoRe</td><td>18.37</td><td>1.56</td></tr><tr><td rowspan="2">Decoding Steering</td><td>+ AD</td><td>7.04</td><td>0.36</td><td>+ AD</td><td>5.45</td><td>0.00</td><td>+ AD</td><td>26.53</td><td>2.49</td></tr><tr><td>+ ITI</td><td>7.04</td><td>0.00</td><td>+ ITI</td><td>8.18</td><td>0.00</td><td>+ ITI</td><td>11.22</td><td>0.31</td></tr><tr><td rowspan="2">Prompting</td><td>+ CAA</td><td>14.07</td><td>0.00</td><td>+ CAA</td><td>10.91</td><td>0.00</td><td>+ CAA</td><td>29.59</td><td>2.18</td></tr><tr><td>+ RECITE</td><td>48.74</td><td>11.43</td><td>+ RECITE</td><td>51.82</td><td>14.94</td><td>+ RECITE</td><td>46.94</td><td>14.95</td></tr><tr><td rowspan="2"></td><td>+ GenRead</td><td>43.22</td><td>9.29</td><td>+ GenRead</td><td>41.82</td><td>18.40</td><td>+ GenRead</td><td>59.18</td><td>24.61</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: SGR/DGR (%) across model families and KEE techniques at $\tau { = } 1 . 2$ . Bold and underline indicate the best and second-best values, respectively. Alternating gray shading indicates technique category boundaries.

Context injection strongly reshapes the output distributions. Both prompting techniques show the smallest $\mathrm { P R S } _ { \mathrm { T i e d } }$ at about 7.8% for RECITE and 5.0% for GenRead in Figure 2. However, their consistency differs. GenRead consistently pushes the gold token downward, with $\mathrm { P R S } _ { \mathrm { W o r s e n e d } }$ ranging from 75.1% to 89.1% across models. In contrast, RECITE exhibits less consistency across model families, with its $\mathrm { P R S } _ { \mathrm { I m p r o v e d } }$ ranging from 14.8% on OLMo2-32B to 44.3% on Llama3-70B in Table 15.

## 5.3 Beyond-Sampling Level

Weak interventions mostly recover shallow failures. In Tables 3 and 12, decoding techniques and CAA demonstrate gains that are largely concentrated in shallow failures rather than deep ones. On OLMo2 and Qwen2.5 families, these interventions generally perform comparably to or lag behind $\tau _ { 0 . 7 }$ and $\tau _ { 1 . 2 } .$ . For instance, OLMo2-7B’s AD achieves an SGR of 5.96% and a DGR of 0.00%, whereas $\tau _ { 1 . 2 }$ reaches 16.06% and 0.42% in Table 3. In contrast, the Llama3 family retains a clear advantage at both temperatures. Llama3-8B’s CAA achieves a DGR of 4.10%, far outperforming $\tau _ { 1 . 2 }$ at 0.24%. These interventions primarily recover shallow failures within the reach of temperature sampling rather than addressing deep failures that lie beyond sampling, with an exception on Llama3 where a substantial gap remains. ITI also stays at or below 2.06% DGR in every model at both temperatures.

Prompting unlocks deep failures robustly across temperatures. RECITE and GenRead decisively outperform $\tau _ { 0 . 7 }$ and $\tau _ { 1 . 2 }$ on both SGR and DGR. For instance, on OLMo2-32B, prompting techniques achieve a DGR of up to 15.76% at $\tau = 0 . 7$ and 11.43% at $\tau = 1 . 2 .$ compared to 0.00% from both $\tau _ { 0 . 7 }$ and $\tau _ { 1 . 2 }$ in Tables 3 and 12.

## 6 Conclusion

We introduce MAPLE, a unified framework for evaluating KEE techniques that comprises a benchmark of facts verified to be present in OLMo2’s pretraining corpus and a diagnostic protocol spanning surface, probability, and beyond-sampling level metrics. Across three model families and six KEE techniques, MAPLE reveals behaviors that aggregated accuracy cannot surface. Decoding techniques and CAA reshape the output distribution without moving it toward the gold answer and largely remain within the reach of temperature sampling, ITI stays close to the baseline at every level, and prompting techniques recover the most failures beyond sampling but also incur the largest Loss. By capturing these distinct behaviors across complementary axes, our framework offers a fine-grained lens to dissect the trade-offs of KEE techniques and a multi-axis standard to guide the development of more effective and stable methods for future LLMs.

## Acknowledgements

This research was supported by grants from the National Research Foundation of Korea (NRF), funded by the Korean government (MSIT and MOE: RS-2025-16063688; MSIT: RS-2025-02215813, RS-2026-25491306, RS-2025- 25442252).

## Limitations

While our framework provides a systematic diagnostic lens for KEE techniques, several limitations remain.

Corpus Verification. We verify that every benchmark fact has a supporting sentence that appears verbatim in OLMo2’s pretraining corpus and from which the same triplet is re-extracted as described in Stage 3 of Section 3.1. This verification establishes presence in pretraining data rather than encoding in model parameters. Encoding strength varies continuously with exposure (Kandpal et al., 2023; Mallen et al., 2023), so a fact present in the corpus may still be encoded only weakly or not at all. The verification is also limited to OLMo2, whose pretraining corpus is public. For Qwen2.5 and Llama3, we rely on the facts being drawn from a Wikipedia dump that predates their training cutoffs, which makes their presence likely but not verified.

Model & KEE Technique Coverage. Our benchmark, built from a March 2022 Wikipedia dump, applies only to models whose pretraining post-dates this cutoff. Due to computational constraints, we evaluate six models from three families, restricting the transferability of our findings to other families. Similarly, we evaluate two representative techniques from each of the three KEE technique categories (decoding, steering, prompting) rather than the full space of methods. Our framework, however, is model- and technique-agnostic and can be readily extended along both axes. We leave further investigation of other models and KEE techniques as future work.

Scope of Knowledge. Although our benchmark covers a wide range of domains, frequency levels, and relations, it only consists of factual triplets with a single answer, drawn from Wikipedia. There are different types of knowledge relevant to our work, such as ambiguous multi-answer facts or non-English knowledge. We do not cover these in this paper as they are out of our scope. Extending the benchmark to these settings will be an interesting direction for future work.

## References

Pere-Lluís Huguet Cabot and Roberto Navigli. 2021. Rebel: Relation extraction by end-to-end language generation. In Findings of the association for computational linguistics: EMNLP 2021, pages 2370–2381.

Shiqi Chen, Miao Xiong, Junteng Liu, Zhengxuan Wu, Teng Xiao, Siyang Gao, and Junxian He. 2024. Incontext sharpness as alerts: An inner representation perspective for hallucination mitigation. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 7553–7567. PMLR.

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R Glass, and Pengcheng He. 2024. Dola: Decoding by contrasting layers improves factuality in large language models. In International Conference on Learning Representations, volume 2024, pages 54158–54183.

Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, and Tao Jiang. 2025. Mia-tuner: Adapting large language models as pre-training text detector. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 27295–27303.

Aryo Pradipta Gema, Chen Jin, Ahmed Abdulaal, Tom Diethe, Philip Alexander Teare, Beatrice Alex, Pasquale Minervini, and Amrutha Saseendran. 2025. DeCoRe: Decoding by contrasting retrieval heads to mitigate hallucinations. In Findings ofthe Association for Computational Linguistics: EMNLP 2025, pages 10003–10039, Suzhou, China. Association for Computational Linguistics.

Zhengbao Jiang, Frank F Xu, Jun Araki, and Graham Neubig. 2020. How can we know what language models know? Transactions of the Association for Computational Linguistics, 8:423–438.

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings ofthe 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611.

Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. 2023. Large language models struggle to learn long-tail knowledge. In International conference on machine learning, pages 15696–15707. PMLR.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, and 1 others. 2019. Natural questions: a

benchmark for question answering research. Transactions ofthe Associationfor Computational Linguistics, 7:453–466.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023a. Inferencetime intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36:41451–41530.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori B Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2023b. Contrastive decoding: Open-ended text generation as optimization. In Proceedings ofthe 61st annual meeting ofthe association for computational linguistics (volume 1: Long papers), pages 12286–12312.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long papers), pages 3214–3252.

Jiacheng Liu, Sewon Min, Luke Zettlemoyer, Yejin Choi, and Hannaneh Hajishirzi. 2024. Infini-gram: Scaling unbounded n-gram language models to a trillion tokens. In First Conference on Language Modeling.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: Long papers), pages 9802–9822.

Dor Muhlgay, Ori Ram, Inbal Magar, Yoav Levine, Nir Ratner, Yonatan Belinkov, Omri Abend, Kevin Leyton-Brown, Amnon Shashua, and Yoav Shoham. 2024. Generating benchmarks for factuality evaluation of language models. In Proceedings ofthe 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 49–66.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, and 1 others. 2024. 2 olmo 2 furious. arXiv preprint arXiv:2501.00656.

Fabio Petroni, Tim Rocktäschel, Sebastian Riedel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and Alexander Miller. 2019. Language models as knowledge bases? In Proceedings ofthe 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 2463–2473.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1:

Long Papers), pages 15504–15522, Bangkok, Thailand. Association for Computational Linguistics.

Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. 2024. Detecting pretraining data from large language models. In International Conference on Learning Representations, volume 2024, pages 51826–51843.

Zhiqing Sun, Xuezhi Wang, Yi Tay, Yiming Yang, and Denny Zhou. 2023. Recitation-augmented language models. In The Eleventh International Conference on Learning Representations.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. Fever: a large-scale dataset for fact extraction and verification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809–819.

Weixuan Wang, JINGYUAN YANG, and Wei Peng. 2025. Semantics-adaptive activation intervention for llms via dynamic steering vectors. In International Conference on Learning Representations, volume 2025, pages 79334–79351.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380.

Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang. 2023. Generate rather than retrieve: Large language models are strong context generators. In The Eleventh International Conference on Learning Representations.

Jingyang Zhang, Jingwei Sun, Eric Yeats, Yang Ouyang, Martin Kuo, Jianyi Zhang, Hao Yang, and Hai Li. 2025. Min-k%++: Improved baseline for pre-training data detection from large language models. In International Conference on Learning Representations, volume 2025, pages 64845–64862.

Domain Classification Prompt Template   
You are a domain relevance classifier for   
knowledge graph triplets.   
Given a triplet (subject, relation, object),   
determine if it belongs to the <DOMAIN> domain.   
<DOMAIN> includes: <SCOPE>.   
Rule:   
Classify as "<DOMAIN>" if any of the following   
is true:   
- The subject or object is a <DOMAIN>-related   
entity.   
- The triplet describes a <DOMAIN>-related   
fact.   
Classify as "non-<DOMAIN>" only if none of the   
above conditions are met.   
Respond with ONLY "<DOMAIN>" or "non-<DOMAIN>".  
Figure 3: System prompt for the domain classifier.

## A Benchmark Construction Details

We organize source entities into ten categories; the category set and representative instance of (P31) types are listed in Table 4.

## A.1 Domain Classification Prompt

We classify each triplet into one of ten domains using an LLM-based relevance classifier. All ten domains share an identical system prompt template; only the domain name (i.e., <DOMAIN>) and its entity set of in-domain scope (i.e., <SCOPE>) differ. The shared template is shown in Figure 3, where <DOMAIN> and <SCOPE> are instantiated per domain using the values listed in Table 5.

## A.2 Answer Cardinality Filtering Prompt

To restrict the benchmark to triplets with a single valid answer, we filter each candidate triplet with an LLM-based cardinality classifier. The classifier labels a triplet as SINGLE, MULTIPLE, or UNSURE, and only SINGLE triplets are retained. The system prompt is shown in Figure 4, and the few-shot demonstrations supplied to the classifier are listed in Table 6.

## A.3 Benchmark Statistics

Statistics of our benchmark are shown in Figure 5.

## B Experiments Details

We present the configurations we applied to implement KEE techniques. Unless otherwise noted, hyperparameters follow the recommendations of each method’s original paper.

```jsonl
Answer Cardinality Classification Prompt
Template
You are given a factoid question.
Determine whether this question has exactly one
valid answer, or multiple valid answers.
- SINGLE: The question has exactly one correct
answer.
- MULTIPLE: The question has more than one valid
answer. This includes:
(a) Factually multiple answers: the entity
genuinely has multiple valid values (e.g., a
drug treats multiple diseases, a person held
multiple positions, a country borders multiple
neighbors).
(b) Temporally variable answers: the
answer changes over time, so multiple answers
have been correct at different points (e.g.,
headquarters that relocated, capitals that
changed, leaders who were replaced).
(c) Perspectival ambiguity: the answer
depends on which context (e.g., classification
system, hierarchical system) is used.
- UNSURE: You are not confident enough to judge,
or you do not know enough about the subject.
Rule:
- Even if one answer is most famous or commonly
cited, if other valid answers exist, classify
as MULTIPLE.
Output JSON:
{"cardinality2": "SINGLE" or "MULTIPLE" or
"UNSURE", "reason": str}
```

Figure 4: System prompt for the answer cardinality classifier.

## B.1 Decoding Methods

DeCoRe. We set num\_retrieval\_head = 10 for all models, following the original paper, which reports that performance on factuality benchmarks degrades as more retrieval heads are masked (Gema et al., 2025). As a post-hoc check in our setting, Table 7 shows that performance is nearly identical between 10 and 20 heads and degrades as more heads are masked.

Activation Decoding. The original paper (Chen et al., 2024) adopts the layer 26 for a 32-layer model—i.e., 81.25% of the total depth—for its main experiments. We therefore set info\_layer to the layer at 81.25% of each model’s total depth, as listed in Table 8. Also, we set α = 0.5, which controls the degree of entropy.

## B.2 Steering Methods

ITI. For ITI, we follow the default setting of the original paper (Li et al., 2023a), using K=48 heads and an intervention strength of α=15. As in the original work, we split TruthfulQA into training and validation sets to identify the top-K heads, and then apply interventions on these heads at inference time on our benchmark.

<table><tr><td>Category</td><td>Instance of (P31)</td></tr><tr><td>Culture</td><td>literary work, museum, art movement, heritage site, cultural center</td></tr><tr><td>Education</td><td>school, university, college, campus, faculty</td></tr><tr><td>Health</td><td>hospital, anatomical structure, medical treatment, disease, symptom or sign</td></tr><tr><td>History</td><td>battle, war, dynasty, historical period, invasion</td></tr><tr><td>Law</td><td>legal case, prison, court, criminal case, decision of the Supreme Court of the United States</td></tr><tr><td>Politics</td><td>political organization, electoral unit, public election, constituency of the European Parliament, executive branch</td></tr><tr><td>Religion</td><td>mythological Greek character, religious community, Catholic religious institute, Jewish organization, Islamic term</td></tr><tr><td>Science</td><td>protein, mutant, spiral galaxy, chemical element, scientific theory</td></tr><tr><td>Society</td><td>voluntary association, ethnic group, clan, human population, women&#x27;s organization</td></tr><tr><td>Sports</td><td>roller derby league, amateur football club, esports team, championship, rugby league venue</td></tr></table>

Table 4: Categories with representative instance of (P31) types.

CAA. For CAA, we follow the original paper’s procedure for selecting the intervention layer per model, using $\Delta p ,$ the change in the probability of the ground-truth answer’s first token before and after intervention, as the selection criterion. Since aggregating over a larger number of contrast pairs reduces noise in the steering vector (Rimsky et al., 2024), we perform layer selection on all 3,254 facts prior to Corpus presence filtering. To prevent data leakage, we split this pool into two folds. Within each fold, 80% is used to construct the steering vector and the remaining 20% to select the optimal layer, which is then applied at inference on the other fold. As a result, every benchmark fact is evaluated with a steering vector and layer derived without it. We set the steering multiplier to 1.0. The selected layers for each model are listed in Table 9.

## B.3 Prompting Methods

RECITE. Figure 6 and Figure 7 show the prompts for stage 1 (recitation) and stage 2 (answer generation), respectively. Following the original paper’s setting (Sun et al., 2023), we run stage 1 and stage 2 for 20 times and choose the final answer with majority voting.

GenRead. Figure 8 and Figure 9 show the prompts for stage 1 (background generation) and stage 2 (answer generation), respectively.

## C Benchmark Prompt Details

Figure 10 shows the generation prompt used for the baseline, and Table 10 shows the five few-shot demonstrations used for each short-answer question.

<table><tr><td>Domain</td><td>In-domain scope (&lt;SCoPE&gt;)</td></tr><tr><td>Culture</td><td>literature, books, authorship, visual arts, paintings, sculpture, architecture, performing arts, dance, museums, libraries, collections of cultural works, monuments, cultural heritage, cultural institutions, festivals, and directly related cultural concepts</td></tr><tr><td></td><td>Education schools, universities, colleges, faculties, academic departments, school districts, educational institutions, academic programs, degrees, curricula, teachers, students, education systems, and directly related educational concepts</td></tr><tr><td>Health</td><td>diseases, symptoms, anatomy, organs, mental health, bacteria, viruses, drugs, vaccines, medicines, treatments, antibiotics, medical procedures, diagnosis, healthcare institutions, hospitals, healthcare organizations, doctors, nurses, medical specialties, and any entity or fact closely associated with these topics</td></tr><tr><td>History</td><td>historical events, wars, battles, conflicts, military, empires, dynasties, reigns, historical periods, historical rulers, archaeological periods, and directly related historical concepts</td></tr><tr><td>Law</td><td>legal cases, court decisions, laws, legal concepts, crimes, courts, judicial institutions, judges, legal offices, constitu- tions, and directly related legal concepts</td></tr><tr><td>Politics</td><td>political parties, elections, governments, parliaments, ministries, presidents, prime ministers, politicians, legislation, political institutions, protests, revolution, political ideologies, diplomacy, international relations, and directly related political concepts</td></tr><tr><td>Religion</td><td>religions, religious organizations, churches, temples, mosques, monasteries, religious buildings, deities, mythologi- cal figures, mythology, rituals, doctrines, saints, sacred places, and directly related religious concepts</td></tr><tr><td>Science</td><td>biology, species, taxonomy, genes, astronomy, planets, stars, galaxies, chemistry, chemical elements, compounds, reactions, earth science, scientific theories, scientific laws, scientific methods, scientists, research institutions, and directly related scientific concepts</td></tr><tr><td>Society</td><td>social groups, ethnic groups, demographic groups, communities, tribes, non-profit organizations, social movements, social issues, and directly related societal concepts</td></tr><tr><td>Sports</td><td>sports teams, athletes in sports roles, leagues, tournaments, seasons, matches, races, competitions, sporting events, stadiums, arenas, sports venues, coaches, and directly related sports concepts</td></tr></table>

Table 5: Per-domain instantiation of the prompt template in Figure 3. Each row provides the <DOMAIN> name and the <SCOPE> string substituted into the shared template.

## D Additional Results

Full results for raw count of Table 2 are in Table 11. Full results for per-model and raw count breakdown of Figure 2 are in Figure 12 and Table 15. Full results for raw count of Tables 3 and 12 are in Tables 13 and 14, respectively.

## E Use of AI assistants

AI assistants like ChatGPT and Claude Code were intermittently used to generate template code and rephrase sentences in the paper, etc. However, no complete paper sections or code scripts have been generated by an AI assistant. All generated content has been inspected and verified by the authors.

<table><tr><td>Subject</td><td>Relation</td><td>Object</td><td>Label</td><td>Reason</td></tr><tr><td>Italy</td><td>capital</td><td>Rome</td><td>SINGLE</td><td>Italy has exactly one capital, Rome, which has not changed.</td></tr><tr><td>aspirin</td><td>medical condition treated pain</td><td></td><td>MULTIPLE</td><td>Aspirin treats multiple conditions including pain, fever, inflammation, and cardiovascu- lar disease.</td></tr><tr><td>Microsoft</td><td>headquarters location</td><td>Redmond</td><td>MULTIPLE</td><td>The headquarters of Microsoft has changed over time (Albuquerque, then Redmond), so the answer depends on the time period.</td></tr><tr><td>Winston Churchill</td><td>conflict</td><td>World War II</td><td>MULTIPLE</td><td>Churchill participated in multiple conflicts including the Second Boer War, World War I, and World War II.</td></tr><tr><td>amygdala</td><td>part of</td><td>limbic system</td><td>SINGLE</td><td>The amygdala is part of exactly one brain system: the limbic system.</td></tr><tr><td>Kaba-class destroyer country</td><td></td><td>Japan</td><td>UNSURE</td><td>Not confident enough about this specific naval vessel class to determine whether it was operated by one or multiple countries.</td></tr></table>

Table 6: Few-shot demonstrations provided to the answer cardinality classifier. Each demonstration pairs an input triplet with its gold label and rationale.  
![](images/e680b18592de311a7aab942a663e43843fb4dc908b1dbe30d6ff75d4223416eb.jpg)  
Figure 5: Distribution of the 1,303 triplets across domain, frequency bin, and relation, showing broad knowledge coverage along all three axes. Bins are defined by frequency percentiles rather than the targeted proportions; the tail bin is smaller (7.4%) because low-frequency entities yield fewer triplets under our relation and domain constraints. The relation chart shows the top-20 of 97 unique types.

<table><tr><td rowspan="2">Model</td><td colspan="5">num_retrieval_head</td></tr><tr><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td></tr><tr><td>Llama3-8B-Instruct</td><td>0.5236</td><td>0.5265</td><td>0.5178</td><td>0.5170</td><td>0.5167</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>0.4017</td><td>0.4028</td><td>0.3981</td><td>0.3899</td><td>0.3748</td></tr></table>

Table 7: DeCoRe accuracy on our benchmark as num\_retrieval\_head varies from 10 to 50. Unlike the main results, which use greedy decoding, accuracy here is the mean per-fact correctness over N = 10 sampled trials at temperature $T = 0 . 7$

<table><tr><td>Model info_layer</td></tr><tr><td>OLMo2-7B-Instruct 26/32</td></tr><tr><td>OLMo2-32B-Instruct 52 / 64</td></tr><tr><td>Qwen2.5-7B-Instruct 23 / 28</td></tr><tr><td>Qwen2.5-32B-Instruct 52 /64</td></tr><tr><td>Llama3-8B-Instruct 26 /32</td></tr><tr><td>Llama3-70B-Instruct 65 / 80</td></tr></table>

Table 8: Per-model info\_layer for Activation Decoding.

<table><tr><td rowspan="2">Model</td><td colspan="2">intervention_layer</td></tr><tr><td>Fold 1</td><td>Fold 2</td></tr><tr><td>OLMo2-7B-Instruct</td><td>19</td><td>11</td></tr><tr><td>OLMo2-32B-Instruct</td><td>48</td><td>43</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>17</td><td>19</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>47</td><td>40</td></tr><tr><td>Llama3-8B-Instruct</td><td>9</td><td>18</td></tr><tr><td>Llama3-70B-Instruct</td><td>39</td><td>39</td></tr></table>

Table 9: Per-fold intervention\_layer selected for CAA and applied to the other.

![](images/1ce7ea8990e1723d2a0a368200ec95cce3684154df13bdfa9d83abe4e02f101c.jpg)

Figure 6: RECITE Stage 1 prompt. Five demonstrations of this form are provided in the system message.  
![](images/f73b837e3eccc86a4788bdf32b034fe518f7bed874e0d85321a4d0074dc9fcc8.jpg)

Figure 7: RECITE Stage 2 prompt.  
![](images/b67ba96a4932a5beea76ee308be7dfd1e22efcdf49f988a1f1f608c7d40b5ded.jpg)  
Figure 8: GenRead Stage 1 prompt (zero-shot).

![](images/257a33db1f1bcb46acc24530b8a53a32e673ac902bc8c80290337c85f3856172.jpg)  
Figure 9: GenRead Stage 2 prompt.

![](images/3d2448d87ba93f527f67d751667ec592e882c4557557ecebc8ffa1698624ea5f.jpg)  
Figure 10: Prompt template for generating the output.

<table><tr><td>Question</td><td>Answer</td></tr><tr><td>What religious order was Koenigsbruck Abbey associated with?</td><td>Cistercian</td></tr><tr><td>Which Olympic Games did John 1908 Summer Olympics Garrels participate in?</td><td></td></tr><tr><td>Which dynasty was Thutmose III eighteenth dynasty part of?</td><td></td></tr><tr><td>Who founded World Book and UNESCO Copyright Day?</td><td></td></tr><tr><td>Which organization is Llanelli Welsh Rugby Union Scarlets affiliated with?</td><td></td></tr></table>

Table 10: Five few-shot demonstrations used for the output generation.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Metric</td><td colspan="2">Decoding</td><td colspan="2">Steering</td><td colspan="2">Prompting</td><td colspan="2">Temperature Sampling</td><td rowspan="2">Denominator</td></tr><tr><td>DeCoRe</td><td>AD</td><td>ITI</td><td>CAA</td><td>RECITE</td><td>GenRead</td><td>τ = 0.7</td><td>τ = 1.2</td></tr><tr><td rowspan="2">OLMo2-7B</td><td>Gain↑</td><td>23</td><td>13</td><td>34</td><td>21</td><td>161</td><td>138</td><td>23</td><td>25</td><td>692</td></tr><tr><td>Loss↓</td><td>23</td><td>23</td><td>21</td><td>31</td><td>39</td><td>63</td><td>17</td><td>44</td><td>611</td></tr><tr><td rowspan="2">OLMo2-32B</td><td>Gain↑</td><td>23</td><td>15</td><td>14</td><td>28</td><td>129</td><td>112</td><td>15</td><td>33</td><td>479</td></tr><tr><td>Loss↓</td><td>19</td><td>27</td><td>12</td><td>35</td><td>49</td><td>67</td><td>28</td><td>36</td><td>824</td></tr><tr><td rowspan="2">Qwen2.5-7B</td><td>Gain↑</td><td>32</td><td>28</td><td>9</td><td>11</td><td>153</td><td>185</td><td>10</td><td>26</td><td>741</td></tr><tr><td>Loss↓</td><td>56</td><td>63</td><td>23</td><td>17</td><td>51</td><td>69</td><td>9</td><td>28</td><td>562</td></tr><tr><td rowspan="2">Qwen2.5-32B</td><td>Gain↑</td><td>8</td><td>6</td><td>9</td><td>12</td><td>126</td><td>131</td><td>8</td><td>19</td><td>572</td></tr><tr><td>Loss↓</td><td>13</td><td>4</td><td>16</td><td>8</td><td>55</td><td>84</td><td>8</td><td>14</td><td>731</td></tr><tr><td rowspan="2">Llama3-8B</td><td>Gain↑</td><td>72</td><td>65</td><td>30</td><td>71</td><td>126</td><td>186</td><td>19</td><td>40</td><td>644</td></tr><tr><td>Loss↓</td><td>36</td><td>45</td><td>25</td><td>59</td><td>49</td><td>81</td><td>15</td><td>44</td><td>659</td></tr><tr><td rowspan="2">Llama3-70B</td><td>Gain↑</td><td>23</td><td>34</td><td>12</td><td>36</td><td>94</td><td>137</td><td>7</td><td>13</td><td>419</td></tr><tr><td>Loss↓</td><td>14</td><td>28</td><td>11</td><td>37</td><td>56</td><td>59</td><td>8</td><td>12</td><td>884</td></tr></table>

Table 11: Raw counts for Table 2.

<table><tr><td>Category</td><td>Model</td><td>SGR↑</td><td>DGR↑</td><td>Model</td><td>SGR↑</td><td>DGR↑</td><td>Model</td><td>SGR↑</td><td>DGR↑</td></tr><tr><td></td><td>OLMo2-7B</td><td>-</td><td>1</td><td>Qwen2.5-7B</td><td>一</td><td>一</td><td>Llama3-8B</td><td>-</td><td>-</td></tr><tr><td>Sampling</td><td>+ T0.7</td><td>16.15</td><td>0.18</td><td>+ T0.7</td><td>11.11</td><td>0.00</td><td>+ T0.7</td><td>14.47</td><td>0.00</td></tr><tr><td rowspan="2">Decoding</td><td>+ DeCoRe</td><td>16.15</td><td>0.36</td><td>+ DeCoRe</td><td>24.44</td><td>1.54</td><td>+ DeCoRe</td><td>32.70</td><td>4.12</td></tr><tr><td>+ AD</td><td>10.00</td><td>0.00</td><td>+ AD</td><td>23.33</td><td>1.08</td><td>+ AD</td><td>27.67</td><td>4.33</td></tr><tr><td rowspan="2">Steering</td><td>+ ITI</td><td>17.69</td><td>1.96</td><td>+ ITI</td><td>6.67</td><td>0.46</td><td>+ ITI</td><td>12.58</td><td>2.06</td></tr><tr><td>+ CAA</td><td>15.38</td><td>0.18</td><td>+ CAA</td><td>11.11</td><td>0.15</td><td>+ CAA</td><td>29.56</td><td>4.95</td></tr><tr><td rowspan="2">Prompting</td><td>+ RECITE</td><td>55.38</td><td>15.84</td><td>+ RECITE</td><td>60.00</td><td>15.21</td><td>+ RECITE</td><td>47.80</td><td>10.31</td></tr><tr><td>+ GenRead</td><td>47.69</td><td>13.52</td><td>+ GenRead</td><td>62.22</td><td>19.82</td><td>+ GenRead</td><td>60.38</td><td>18.56</td></tr><tr><td></td><td>OLMo2-32B</td><td></td><td></td><td>Qwen2.5-32B</td><td></td><td></td><td>Llama3-70B</td><td></td><td>一</td></tr><tr><td rowspan="2">Sampling</td><td>+ T0.7</td><td>11.54</td><td>0.00</td><td>+ T0.7</td><td>15.87</td><td>0.00</td><td>+ T0.7</td><td>6.78</td><td>0.00</td></tr><tr><td>+ DeCoRe</td><td>16.15</td><td>0.57</td><td>+ DeCoRe</td><td>12.70</td><td>0.00</td><td>+ DeCoRe</td><td>25.42</td><td>2.22</td></tr><tr><td rowspan="2">Decoding Steering</td><td>+ AD</td><td>11.54</td><td>0.00</td><td>+ AD</td><td>9.52</td><td>0.00</td><td>+ AD</td><td>30.51</td><td>4.44</td></tr><tr><td>+ ITI</td><td>10.77</td><td>0.00</td><td>+ ITI</td><td>11.11</td><td>0.39</td><td>+ ITI</td><td>16.95</td><td>0.56</td></tr><tr><td rowspan="2">Prompting</td><td>+ CAA</td><td>20.77</td><td>0.29</td><td>+ CAA</td><td>17.46</td><td>0.20</td><td>+ CAA</td><td>35.59</td><td>4.17</td></tr><tr><td>+ RECITE</td><td>56.92</td><td>15.76</td><td>+ RECITE</td><td>55.56</td><td>17.88</td><td>+ RECITE</td><td>54.24</td><td>17.22</td></tr><tr><td></td><td>+ GenRead</td><td>53.08</td><td>12.32</td><td>+ GenRead</td><td>50.79</td><td>19.45</td><td>+ GenRead</td><td>62.71</td><td>27.78</td></tr></table>

Table 12: SGR/DGR (%) across model families and KEE techniques at τ=0.7. Bold and underline indicate the best and second-best values, respectively. Alternating gray shading indicates technique category boundaries.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Metric</td><td rowspan="2">Temp. 0.7</td><td colspan="2">Decoding</td><td colspan="2">Steering</td><td colspan="2">Prompting</td><td rowspan="2">Denominator</td></tr><tr><td>DeCoRe</td><td>AD</td><td>ITI</td><td>CAA</td><td>RECITE</td><td>GenRead</td></tr><tr><td rowspan="2">OLMo2-7B</td><td>SGR↑</td><td>21</td><td>21</td><td>13</td><td>23</td><td>20</td><td>72</td><td>62</td><td>130</td></tr><tr><td>DGR↑</td><td>1</td><td>2</td><td>0</td><td>11</td><td>1</td><td>89</td><td>76</td><td>562</td></tr><tr><td rowspan="2">OLMo2-32B</td><td>SGR↑</td><td>15</td><td>21</td><td>15</td><td>14</td><td>27</td><td>74</td><td>69</td><td>130</td></tr><tr><td>DGR↑</td><td>0</td><td>2</td><td>0</td><td>0</td><td>1</td><td>55</td><td>43</td><td>349</td></tr><tr><td rowspan="2">Qwen2.5-7B</td><td>SGR↑</td><td>10</td><td>22</td><td>21</td><td>6</td><td>10</td><td>54</td><td>56</td><td>90</td></tr><tr><td>DGR↑</td><td>0</td><td>10</td><td>7</td><td>3</td><td>1</td><td>99</td><td>129</td><td>651</td></tr><tr><td rowspan="2">Qwen2.5-32B</td><td>SGR↑</td><td>10</td><td>8</td><td>6</td><td>7</td><td>11</td><td>35</td><td>32</td><td>63</td></tr><tr><td>DGR↑</td><td>0</td><td>0</td><td>0</td><td>2</td><td>1</td><td>91</td><td>99</td><td>509</td></tr><tr><td rowspan="2">Llama3-8B</td><td>SGR↑</td><td>23</td><td>52</td><td>44</td><td>20</td><td>47</td><td>76</td><td>96</td><td>159</td></tr><tr><td>DGR↑</td><td>0</td><td>20</td><td>21</td><td>10</td><td>24</td><td>50</td><td>90</td><td>485</td></tr><tr><td rowspan="2">Llama3-70B</td><td>SGR↑</td><td>4</td><td>15</td><td>18</td><td>10</td><td>21</td><td>32</td><td>37</td><td>59</td></tr><tr><td>DGR↑</td><td>0</td><td>8</td><td>16</td><td>2</td><td>15</td><td>62</td><td>100</td><td>360</td></tr></table>

Table 13: Raw counts for Table 12.

![](images/cfaef929baa3b51c9c8930aa06441899c9d4c35dbe016cc4d16bcbac7ae7784b.jpg)  
(a)

![](images/22bc5c5cc473ee0d56414d91c7d7a3950f395b944d0ceb2a065ee8eb2f322143.jpg)  
(b)  
Figure 11: Proportion of shallow failures identified with K samples relative to those identified with 30 samples, $| \mathcal { D } _ { \mathrm { s h a l l o w } } ( K ) | / | \mathcal { D } _ { \mathrm { s h a l l o w } } ( 3 0 ) |$ , for K from 1 to 30. (a) The left panel shows temperature sampling at $\tau = 0 . 7$ and (b) the right panel shows $\tau = 1 . 2 .$ . The dotted line marks 90%. Coverage rises steeply for small K and saturates afterward, and at $K = 2 0$ every model reaches about 90% or more at both temperatures. We therefore use $K = 2 0$ as a balance between coverage and sampling cost.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Metric</td><td rowspan="2">Temp. 1.2</td><td colspan="2">Decoding</td><td colspan="2">Steering</td><td colspan="2">Prompting</td><td rowspan="2">Denominator</td></tr><tr><td>DeCoRe</td><td>AD</td><td>ITI</td><td>CAA</td><td>RECITE</td><td>GenRead</td></tr><tr><td rowspan="2">OLMo2-7B</td><td>SGR↑</td><td>35</td><td>21</td><td>13</td><td>32</td><td>21</td><td>113</td><td>97</td><td>218</td></tr><tr><td>DGR↑</td><td>2</td><td>2</td><td>0</td><td>2</td><td>0</td><td>48</td><td>41</td><td>474</td></tr><tr><td rowspan="2">OLMo2-32B</td><td>SGR↑</td><td>26</td><td>21</td><td>14</td><td>14</td><td>28</td><td>97</td><td>86</td><td>199</td></tr><tr><td>DGR↑</td><td>0</td><td>2</td><td>1</td><td>0</td><td>0</td><td>32</td><td>26</td><td>280</td></tr><tr><td rowspan="2">Qwen2.5-7B</td><td>SGR↑</td><td>28</td><td>24</td><td>23</td><td>9</td><td>10</td><td>80</td><td>90</td><td>150</td></tr><tr><td>DGR↑</td><td>1</td><td>8</td><td>5</td><td>0</td><td>1</td><td>73</td><td>95</td><td>591</td></tr><tr><td rowspan="2">Qwen2.5-32B</td><td>SGR↑</td><td>16</td><td>8</td><td>6</td><td>9</td><td>12</td><td>57</td><td>46</td><td>110</td></tr><tr><td>DGR↑</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>69</td><td>85</td><td>462</td></tr><tr><td rowspan="2">Llama3-8B</td><td>SGR↑</td><td>36</td><td>55</td><td>48</td><td>23</td><td>54</td><td>95</td><td>121</td><td>229</td></tr><tr><td>DGR↑</td><td>1</td><td>17</td><td>17</td><td>7</td><td>17</td><td>31</td><td>65</td><td>415</td></tr><tr><td rowspan="2">Llama3-70B</td><td>SGR↑</td><td>16</td><td>18</td><td>26</td><td>11</td><td>29</td><td>46</td><td>58</td><td>98</td></tr><tr><td>DGR↑</td><td>0</td><td>5</td><td>8</td><td>1</td><td>7</td><td>48</td><td>79</td><td>321</td></tr></table>

Table 14: Raw counts for Table 3.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Category</td><td colspan="2">Decoding</td><td colspan="2">Steering</td><td colspan="2">Prompting</td></tr><tr><td>DeCoRe</td><td>AD</td><td>ITI</td><td>CAA</td><td>RECITE</td><td>GenRead</td></tr><tr><td rowspan="6">OLMo2-7B</td><td>n00</td><td>669</td><td>679</td><td>658</td><td>671</td><td>531</td><td>554</td></tr><tr><td>Nexc</td><td>102</td><td>91</td><td>74</td><td>88</td><td>61</td><td>88</td></tr><tr><td>Ninc</td><td>567</td><td>588</td><td>584</td><td>583</td><td>470</td><td>466</td></tr><tr><td>Improved</td><td>28.7 (163)</td><td>25.7 (151)</td><td>20.0 (117)</td><td>36.4 (212)</td><td>18.1 (85)</td><td>17.8 (83)</td></tr><tr><td>Tied</td><td>23.8 (135)</td><td>39.5 (232)</td><td>60.8 (355)</td><td>25.4 (148)</td><td>6.4 (30)</td><td>3.9 (18)</td></tr><tr><td>Worsened</td><td>47.4 (269)</td><td>34.9 (205)</td><td>19.2 (112)</td><td>38.3 (223)</td><td>75.5 (355)</td><td>78.3 (365)</td></tr><tr><td rowspan="6">OLMo2-32B</td><td>n00</td><td>456</td><td>464</td><td>465</td><td>451</td><td>350</td><td>367</td></tr><tr><td>Nexc</td><td>100</td><td>96</td><td>79</td><td>82</td><td>60</td><td>83</td></tr><tr><td> $n _ { \mathrm { i n c } }$ </td><td>356</td><td>368</td><td>386</td><td>369</td><td>290</td><td>284</td></tr><tr><td>Improved</td><td>28.4 (101)</td><td>22.3 (82)</td><td>15.5 (60)</td><td>35.5 (131)</td><td>14.8 (43)</td><td>16.5 (47)</td></tr><tr><td>Tied</td><td>32.3 (115)</td><td>40.2 (148)</td><td>69.4 (268)</td><td>33.9 (125)</td><td>6.6 (19)</td><td>6.3 (18)</td></tr><tr><td>Worsened</td><td>39.3 (140)</td><td>37.5 (138)</td><td>15.0 (58)</td><td>30.6 (113)</td><td>78.6 (228)</td><td>77.1 (219)</td></tr><tr><td rowspan="6">Qwen2.5-7B</td><td>n00</td><td>709</td><td>713</td><td>732</td><td>730</td><td>588</td><td>556</td></tr><tr><td>Nexc</td><td>113</td><td>108</td><td>102</td><td>107</td><td>209</td><td>95</td></tr><tr><td>Ninc</td><td>596</td><td>605</td><td>630</td><td>623</td><td>379</td><td>461</td></tr><tr><td>Improved</td><td>28.5 (170)</td><td>29.1 (176)</td><td>20.5 (129)</td><td>23.6 (147)</td><td>40.9 (155)</td><td>22.1 (102)</td></tr><tr><td>Tied</td><td>15.9 (95)</td><td>17.0 (103)</td><td>53.0 (334)</td><td>31.0 (193)</td><td>3.2 (12)</td><td>2.8 (13)</td></tr><tr><td>Worsened</td><td>55.5 (331)</td><td>53.9 (326)</td><td>26.5 (167)</td><td>45.4 (283)</td><td>55.9 (212)</td><td>75.1 (346)</td></tr><tr><td rowspan="6">Qwen2.5-32B</td><td>n00</td><td>564</td><td>566</td><td>563</td><td>560</td><td>446</td><td>441</td></tr><tr><td>Nexc</td><td>89</td><td>84</td><td>80</td><td>83</td><td>182</td><td>74</td></tr><tr><td> $n _ { \mathrm { i n c } }$ </td><td>475</td><td>482</td><td>483</td><td>477</td><td>264</td><td>367</td></tr><tr><td>Improved</td><td>21.1 (100)</td><td>25.5 (123)</td><td>23.4 (113)</td><td>31.9 (152)</td><td>33.3 (88)</td><td>9.3 (34)</td></tr><tr><td>Tied Worsened</td><td>36.6 (174)</td><td>45.2 (218)</td><td>53.8 (260)</td><td>43.4 (207)</td><td>3.4 (9)</td><td>1.6 (6)</td></tr><tr><td></td><td>42.3 (201)</td><td>29.3 (141)</td><td>22.8 (110)</td><td>24.7 (118)</td><td>63.3 (167)</td><td>89.1 (327)</td></tr><tr><td rowspan="6">Llama3-8B</td><td>n00</td><td>572</td><td>579</td><td>614</td><td>573</td><td>518</td><td>458</td></tr><tr><td>Nexc Ninc</td><td>131</td><td>127</td><td>103</td><td>133</td><td>169</td><td>101</td></tr><tr><td></td><td>441</td><td>452</td><td>511</td><td>440</td><td>349</td><td>357</td></tr><tr><td>Improved</td><td>39.0 (172)</td><td>33.2 (150)</td><td>17.4 (89)</td><td>38.4 (169)</td><td>41.0 (143)</td><td>18.8 (67)</td></tr><tr><td>Tied Worsened</td><td>25.6 (113)</td><td>25.9 (117)</td><td>65.6 (335)</td><td>22.7 (100)</td><td>11.2 (39)</td><td>5.9 (21)</td></tr><tr><td></td><td>35.4 (156)</td><td>40.9 (185)</td><td>17.0 (87)</td><td>38.9 (171)</td><td>47.9 (167)</td><td>75.4 (269)</td></tr><tr><td rowspan="6">Llama3-70B</td><td>n00</td><td>396</td><td>385</td><td>407</td><td>383</td><td>325</td><td>282</td></tr><tr><td> $n _ { \mathrm { e x c } }$ </td><td>128</td><td>126</td><td>119</td><td>125</td><td>203</td><td>97</td></tr><tr><td> $n _ { \mathrm { i n c } }$ </td><td>268</td><td>259</td><td>288</td><td>258</td><td>122</td><td>185</td></tr><tr><td>Improved</td><td>22.8 (61)</td><td>13.1 (34)</td><td>12.5 (36)</td><td>27.9 (72)</td><td>44.3 (54)</td><td>14.6 (27)</td></tr><tr><td>Tied</td><td>43.7 (117)</td><td>40.2 (104)</td><td>72.2 (208)</td><td>43.4 (112)</td><td>16.4 (20)</td><td>9.2 (17)</td></tr><tr><td>Worsened</td><td>33.6 (90)</td><td>46.7 (121)</td><td>15.3 (44)</td><td>28.7 (74)</td><td>39.3 (48)</td><td>76.2 (141)</td></tr></table>

Table 15: Per-model PRS on unrecovered failures. Among the $n _ { 0 0 }$ unrecovered failures, $n _ { \mathrm { e x c } }$ are excluded because the gold token is ranked first by either the baseline or the KEE technique, and for RECITE this includes ranking first in any of its 20 recitation paths. The remaining $n _ { \mathrm { i n c } } = N _ { \mathrm { I m p r o v e d } } + N _ { \mathrm { T i e d } } + N _ { \mathrm { W o r s e n e d } }$ failures are categorized by Eq. 3, and each category reports PRS<sub>c</sub> in percent followed by $N _ { c }$ in parentheses. For RECITE, each failure is assigned the most frequent category across its 20 recitation paths.

![](images/23b244fc4923185e944541650e3f7bcab4ba4c713e6c4ced49b5cee08abdc741.jpg)  
Figure 12: Per-model PRS on unrecovered failures in a bar graph.