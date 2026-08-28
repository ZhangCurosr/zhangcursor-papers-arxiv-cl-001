# Instruction Quality Matters: Refining Instructions for Effective Preference Learning

Seohyeong Lee<sup>1</sup> Hwaran Lee<sup>1,</sup>\* Buru Chang<sup>2,</sup>\*

<sup>1</sup>Sogang University <sup>2</sup>Korea University

{seohyeong, hwaranlee}@sogang.ac.kr buru\_chang@korea.ac.kr

## Abstract

Preference learning optimizes models using response pairs, yet the informativeness of these pairs is fundamentally shaped by the instructions from which they are generated. We identify instruction quality as a hidden bottleneck in preference learning: low-quality or ambiguous instructions restrict the responsequality distribution, limiting strong chosen responses and weakening preference signals. Through Best- and Worst-of-N analyses, we show that instruction quality constrains both the ceiling and floor of sampled response quality. Motivated by this observation, we introduce an instruction-refinement pipeline that selects weak instructions using reward signals and revises them with rubric-guided LLM feedback, improving preference data without discarding examples. Across offline and online preference learning settings, experiments on multiple models and benchmarks show broad alignment improvements over original data and alternative data-improvement strategies. Further analyses indicate that instruction refinement raises achievable response quality and complements response-centric preference data curation. Overall, instruction quality emerges as a key factor governing how informative preference signals are formed for LLM alignment. Code is available at: https://github.com/ 01choco/instruction-refinement/

## 1 Introduction

Preference learning (Christiano et al., 2017) is a central paradigm for aligning Large Language Models (LLMs) with human preferences. Despite advances in optimization methods (Meng et al., 2024; Liu et al., 2024), its effectiveness remains highly dependent on the quality of preference data (Zhou et al., 2023a). Existing work has primarily examined preference data quality through response quality and annotation reliability, showing that weak responses or noisy labels can substantially degrade alignment (Pan et al., 2025; Lee et al., 2025b; Chowdhury et al.; Gao et al., 2024). This view, however, treats the instruction as a fixed input, overlooking its role in shaping preference signals.

![](images/b93ef0e7df978b3f026cf5d26a5d2311c5287f284693eca7a34782c464f4ca29.jpg)  
Figure 1: Data example from the UltraFeedback (Cui et al., 2024). Ambiguous instructions can produce invalid responses, leading to weak preference learning signals. Refining such instructions yields high-quality outputs that provide stronger signals.

Preference pairs are sampled from a response distribution induced by the instruction. When an instruction is ambiguous, incomplete, or underspecified, this distribution may contain few high-quality responses and many trivial failures, making the resulting preference data less informative. As illustrated in Figure 1, such cases appear in preference datasets and can induce invalid or low-quality responses. Because this candidate response pool is constrained upstream, response refinement or filtering can only partially mitigate the problem.

We first test this upstream effect by analyzing the response distribution from which preference pairs are constructed. Using Best-of-N and Worst-of-N analyses, which approximate the response-quality ceiling and floor under a fixed sampling budget, we examine how instruction quality shapes sampled responses. We find clear gaps between low- and high-quality instructions: high-quality instructions improve not only the best sampled responses but also the worst sampled responses. These gaps indicate that low-quality instructions restrict useful candidate coverage, reducing both strong positive examples and informative preference contrasts for DPO-style optimization (see Section 3.1 and Figure 2). Thus, while preference learning depends on response quality (Pan et al., 2025), the response quality available for learning can itself be constrained by instruction quality.

To mitigate these upstream effects on preferencepair construction, we refine instructions before generating candidate responses. Our pipeline selects weak instructions using reward signals and revises them with rubric-guided LLM feedback, improving preference data without discarding examples. Rather than treating refinement as generic rewriting, the rubrics provide structured diagnostic feedback that guides revisions toward instructions that elicit reliable and comparable candidate responses. Unlike inference-time prompt engineering, our approach targets the training data from which preference-learning signals are derived.

We evaluate instruction refinement on UltraFeedback (Cui et al., 2024) across offline and online preference learning. In offline experiments, refinement improves alignment across multiple backbones, objectives, and benchmarks, including MT-Bench (Zheng et al., 2023), Evol-Instruct (Xu et al., 2023), and AlpacaEval (Li et al., 2023). In online experiments, even one refinement step per datageneration round improves iterative preferencelearning pipelines such as SPA (Kim et al., 2025). Across settings, refinement yields broad alignment gains of up to 8%p, indicating that instruction quality remains a limiting factor even as models and preference data co-evolve.

Further analyses show that refinement improves instruction-quality scores and response rewards, raises the Best-of-N response-quality ceiling, and remains robust across threshold choices. A rubric ablation further shows that rubric-guided feedback outperforms generic instruction rewriting, indicating that structured diagnostic feedback is important for effective refinement. These results support the view that instruction refinement improves preference learning by making the candidate response pool more informative.

Our contributions are summarized as follows:

• We identify instruction quality as an upstream factor in preference-signal formation and show that it shapes response-quality ceilings and floors, affecting the informativeness of pairwise preference data.

• We introduce an instruction refinement pipeline that improves weak instructions using reward signals and rubric-guided LLM feedback, without discarding data.

• We validate the approach across offline and online preference learning, showing broad alignment gains and analyses that support its effect on response-quality distributions.

## 2 Related Work

## 2.1 Preference Data Curation

Prior work has recognized the importance of data quality in preference learning and proposed methods for improving preference pairs in both offline and online settings. In offline preference learning, existing approaches typically identify informative or reliable preference pairs based on response-level or pair-level signals, such as reward margins (Deng et al., 2025), response quality (Lee et al., 2025b), or annotation consistency (Lee et al., 2025a). In online preference learning, prior work improves alignment by iteratively generating responses, constructing preference pairs, and updating the model while reducing noise in preference labels or refining response quality (Kim et al., 2025; Pan et al., 2025; Dong et al., 2025). In contrast, these methods largely treat instructions as fixed inputs, whereas we study instruction quality as an upstream factor in preference data informativeness.

## 2.2 Preference Data Refinement

Our work is methodologically related to selfrefinement and broader data-improvement methods. Self-Refine (Madaan et al., 2023), Reflexion (Shinn et al., 2023), and recent refinement pipelines (Cayir et al., 2025) use natural language feedback, reflection, or LLM-based judges to improve responses, agent behavior, or dataset quality. However, these methods primarily refine responses while leaving instructions fixed.

Regarding instruction-based selection methods, R.I.P. (Yu et al., 2025) shows that prompt quality affects model training by selecting highquality prompts for instruction tuning. Li et al. (2024) select instruction-tuning examples based on instruction-following difficulty. In contrast, we focus on preference data curation by refining instructions for preference learning to elicit clearer preference signals. This perspective is also related to recent work showing that reward models and instruction-tuned models can be sensitive to input or instruction variations (Wu et al., 2025; Yan et al., 2024). While these studies target model robustness, we refine low-quality instructions to construct more informative preference data.

![](images/9fe51c6dcf757d1222f04fb01ddc6e6687e2b685d5c467c148d35b62913ba794.jpg)  
(a) Best-of-N Reward Plot

![](images/c7ee8d901d6e1f15ef0cf3cd8347b7e73ae93b9c7ccfe0ebb08698088492fc39.jpg)  
(b) Worst-of-N Reward Plot

![](images/84c5a435d71164f9030528c0321882311b53ab040dcce488f1b0c4f0d39b9066.jpg)

![](images/dd73f61062ffb797f5b3d9e39824d2d2ff40eee7a554d0096c1af6d1ed8e9e2a.jpg)  
(d) $\mathrm { \mathbf { x } _ { h i g h } }$ Response Distribution  
✅ Our Goal : ⇧ Chosen Response Reward & ⇩ Chosen-Rejected Reward Gap  
Figure 2: Instruction quality shapes response distributions and preference-pair informativeness. Left: (a) Best-of-N reward shows that high-quality instructions raise the response-quality ceiling. (b) Worst-of-N reward shows that they also raise the response-quality floor. Right: (c): Reward score distributions of chosen $y ^ { + }$ and rejected $y ^ { - }$ responses paired with low-quality instructions $x _ { l o w } .$ (d): Those of high-quality instructions $x _ { h i g h }$

## 3 Instruction Quality and Refinement

## 3.1 Motivating Analysis: Instruction Quality Shapes Response Distributions

Preference pairs are constructed from instructionconditioned candidate responses. Ambiguous or underspecified instructions may yield few highquality candidates and many trivial failures, weakening preference data before optimization begins.

To examine this effect, we first conduct Best-of-N and Worst-of-N analyses on 1.5K UltraFeedback (Cui et al., 2024) instructions, partitioned into High, Mid, and Low quality groups using rubricbased instruction scores. For each instruction, we generate multiple responses with LLaMA3-8B and score them with a reward model. Best-of-N approximates the response-quality ceiling under a fixed sampling budget, while Worst-of-N approximates the response-quality floor.

We use reward-model scores as explicit proxies for response quality, rather than as the rewards directly optimized by DPO. DPO instead induces an implicit reward $\begin{array} { r } { r _ { \theta } ( x , y ) = \beta \log \frac { \pi _ { \theta } ( y | x ) } { \pi _ { \mathrm { r e f } } ( y | x ) } } \end{array}$ Therefore, our interpretation assumes that explicit reward-model scores preserve the relevant ordering and margin trends of the implicit rewards that emerge during preference optimization.

As shown in Figure 2, high-quality instructions consistently achieve higher Best-of-N and Worst-of-N rewards than low-quality instructions. This indicates that low-quality instructions both lower the response-quality ceiling and weaken the response-quality floor; they reduce access to strong candidate responses while increasing the prevalence of trivial low-quality responses.

We next connect these ceiling and floor effects to preference-gradient informativeness. For a preference tuple $( x , y ^ { + } , y ^ { - } )$ , the DPO gradient can be written using the preference margin $u \theta$ and likelihood contrast $d _ { \theta } \colon$

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { L } _ { \mathrm { D P O } } = - \beta \mathbb { E } \left[ \sigma ( - u _ { \theta } ) d _ { \theta } \right] , } \\ & { \quad \quad \quad d _ { \theta } = \nabla _ { \theta } \log \pi _ { \theta } ( y ^ { + } | x ) - \nabla _ { \theta } \log \pi _ { \theta } ( y ^ { - } | x ) , } \\ & { \quad \quad \quad u _ { \theta } = \beta \left[ \log \frac { \pi _ { \theta } ( y ^ { + } | x ) } { \pi _ { \mathrm { r e f } } ( y ^ { + } | x ) } - \log \frac { \pi _ { \theta } ( y ^ { - } | x ) } { \pi _ { \mathrm { r e f } } ( y ^ { - } | x ) } \right] . } \end{array}
$$

Here, $d _ { \theta }$ encodes the gradient direction favoring $y ^ { + }$ over $y ^ { - }$ , while $\sigma ( - u _ { \theta } )$ decays as the current policy $\pi _ { \theta }$ already separates the pair (i.e., as $u _ { \theta }$ grows during training). A higher response-quality ceiling increases the chance that $y ^ { + }$ provides a high-quality target for optimization, enabling the positive term in $d _ { \theta }$ to learn from stronger targets. A higher response-quality floor makes rejected responses less degenerate. When $y ^ { - }$ is trivially poor, the chosen–rejected contrast can become overly easy: as training progresses, such pairs are likely to attain a large implicit margin $u _ { \theta }$ , leading to a small DPO gradient weight $\sigma ( - u _ { \theta } )$ . By reducing degenerate negatives, a higher floor helps preserve nontrivial preference contrasts, making preference pairs informative for optimization (Houliston et al., 2024). Thus, ceiling and floor improvements are complementary: the former provides stronger positive examples, while the latter reduces degenerate negatives without eliminating meaningful chosen– rejected gaps. This motivates instruction refinement before candidate generation; Appendix A provides a DPO-based interpretation.

Algorithm 1 Instruction Refinement Loop   
Require: Instruction Set {X}, Reward model $R ( \cdot )$ , Thresh  
old τ, Max iterations T   
Ensure: Refined On-Policy Dataset $\mathcal { D } = ( X ^ { \prime } , Y ^ { + } , Y ^ { - } )$   
1: Initialize $\mathcal { D } ^ { \prime }  \emptyset$   
2: for Each instruction $X \in \{ \mathcal { X } \}$ do   
3: Generate $( Y _ { 1 } , Y _ { 2 } )$ from the target model given X   
4: Compute rewards $R _ { 1 }  R ( X , Y _ { 1 } ) , R _ { 2 }  R ( X , Y _ { 2 } )$   
5: $t  { \bar { 0 } }$   
6: while min $R _ { 1 } , R _ { 2 } ) < \tau$ and $t < T$ do   
7: Generate rubric-based feedback for X   
8: Refine instruction $X  X ^ { \prime }$   
9: Regenerate $( Y _ { 1 } , Y _ { 2 } )$ and recompute $( R _ { 1 } , R _ { 2 } )$   
10: $t \gets t + 1$   
11: end while   
12: Add refined tuple $( X , Y ^ { + } , Y ^ { - } )$ to D   
13: end for   
14: return D

## 3.2 Refining Instructions for Preference Data

We view a preference dataset as tuples $( X , Y ^ { + } , Y ^ { - } )$ , where X is an instruction and $Y ^ { + } , Y$ are chosen and rejected responses. Instead of refining responses after generation, we refine X before constructing preference pairs, aiming to improve the candidate response pool from which preference signals are formed.

Algorithm 1 summarizes the refinement procedure. For each instruction X, we generate onpolicy responses $( Y _ { 1 } , Y _ { 2 } )$ with the target model and compute reward scores $R _ { i } \ : = \ : R ( X , Y _ { i } )$ . If min $( R _ { 1 } , R _ { 2 } ) < \tau .$ , where τ is a minimum-reward threshold, we select X for refinement, treating the low minimum reward as a proxy for lower-tail failures induced by the instruction. Selected instructions are revised using rubric-based feedback; after each revision, responses and rewards are regenerated. The loop repeats until min $( R _ { 1 } , R _ { 2 } ) \geq \tau$ or the maximum number of iterations is reached<sup>1</sup>.

This feedback–refinement loop follows the spirit of Self-Refine (Madaan et al., 2023), but applies refinement to instructions in preference data rather than to generated responses, improving training data without discarding examples.

## 3.3 Evaluation and Refinement Criteria

We use rubric-based evaluation as structured diagnostic feedback for instruction refinement in preference data. Reward scores identify instructions that induce weak candidates, while rubrics diagnose why they fail and how they should be revised. Specifically, the rubrics assess whether an instruction is likely to elicit reliable and comparable candidate responses for preference learning, covering clarity, specificity, completeness, safety, answerability, conciseness, and format consistency. Detailed definitions, motivations, and supporting studies are provided in Appendix B.1.

Scoring and Feedback Generation. For each instruction X, an LLM-as-a-judge (Zheng et al., 2023) evaluates all rubrics and produces structured feedback ${ \mathcal F } .$ . For each rubric, the evaluator outputs a $_ { 1 - 5 }$ score and a short natural-language explanation. The scores provide a quantitative estimate of instruction quality, while the explanations identify the instruction-level deficiencies that should be revised. We average the rubric scores to obtain an overall instruction-quality score.

Refinement. The refiner takes the original instruction X and feedback $\mathcal { F }$ as input, and outputs a revised instruction $X ^ { \prime }$ that addresses the diagnosed deficiencies. The full feedback and refinement prompts are provided in Appendix B.2

## 4 Experiments

To test whether improving instruction quality improves preference learning, we evaluate instruction refinement in both offline and online settings. The offline setting refines a fixed preference dataset, whereas the online setting refines instructions during iterative on-policy data generation.

## 4.1 Offline Preference Learning

Experimental Setup. We conduct offline experiments on 32K instructions randomly sampled from UltraFeedback (Cui et al., 2024). For each instruction, we generate on-policy responses with the target model, apply three iterations of instruction refinement, and then construct the final preference dataset. We evaluate two target backbones, LLaMA3-8B (Grattafiori et al., 2024) and Mistral-7B (Jiang et al., 2023b), initialized from SFT checkpoints, and train them with DPO (Rafailov et al., 2023) and SimPO (Meng et al., 2024).

We use GPT-4o for feedback and refinement. ArmoRM (Wang et al., 2024) is used both to score responses during refinement and to assign final preference labels. The refinement threshold is set to τ = 0.13, corresponding to approximately the top 30% of the initial on-policy reward distribution. We evaluate on MT-Bench (Zheng et al., 2023), Evol-Instruct (Xu et al., 2023), and AlpacaEval (Li et al., 2023); implementation details are provided in Appendix C.

<table><tr><td colspan="2">Method</td><td colspan="5">DPO</td><td colspan="5">SimPO</td></tr><tr><td>Backbone</td><td>Data Type</td><td>MT-Bench WR Score</td><td>WR</td><td>Evol-Instruct Score</td><td>AlpacaEval 2.0 WR</td><td>LC WR</td><td>MT-Bench WR Score</td><td>WR</td><td>Evol-Instruct Score</td><td>WR</td><td>AlpacaEval 2.0 LC WR</td></tr><tr><td rowspan="3">Llama3-8b</td><td>SFT</td><td>27.8</td><td>4.61</td><td>23.4 5.80</td><td>3.95</td><td>7.49</td><td>27.8 4.61</td><td>23.4</td><td>5.80</td><td>3.95</td><td>7.49</td></tr><tr><td>Original</td><td>39.7 4.91</td><td></td><td>33.3 6.15</td><td>5.91</td><td>11.90</td><td>47.5 5.26</td><td>40.8</td><td>6.53</td><td>8.65</td><td>15.90</td></tr><tr><td>Refined</td><td>47.8 5.38</td><td>35.8</td><td>6.39</td><td>8.86</td><td>18.78</td><td>48.1 5.56</td><td>44.5</td><td>6.62</td><td>9.19</td><td>20.30</td></tr><tr><td rowspan="3">Mistral-7b</td><td>SFT</td><td>|21.6 4.04</td><td>16.3</td><td>4.87</td><td>4.30</td><td>8.25</td><td>21.6 4.04</td><td>16.3</td><td>4.87</td><td>4.30</td><td>8.25</td></tr><tr><td>Original</td><td>35.6</td><td>5.03</td><td>31.0 6.35</td><td>7.80</td><td>12.95</td><td>24.1 4.18</td><td>12.4</td><td>5.56</td><td>8.09</td><td>16.83</td></tr><tr><td>Refined</td><td>43.4</td><td>4.96</td><td>33.3 6.24</td><td>8.21</td><td>14.60</td><td>28.8 4.43</td><td>19.5</td><td>5.72</td><td>9.18</td><td>17.40</td></tr></table>

Table 1: Evaluation results on MT-Bench, Evol-Instruct, and AlpacaEval. For MT-Bench and Evol-Instruct, each cell reports the pairwise win rate and single-answer score; for AlpacaEval, each cell reports the win rate and length-controlled win rate. The highest values for each backbone are highlighted in bold.

<table><tr><td rowspan="2">Method</td><td colspan="2">MT-Bench</td><td colspan="2">Evol-Instruct</td><td colspan="2">AlpacaEval</td></tr><tr><td>WR</td><td>Score</td><td>WR</td><td>Score</td><td>WR</td><td>LC WR</td></tr><tr><td>SFT</td><td>27.8</td><td>4.61</td><td>23.4</td><td>5.80</td><td>3.55</td><td>7.15</td></tr><tr><td>DPO (Original Dataset)</td><td>39.7</td><td>4.91</td><td>33.3</td><td>6.15</td><td>5.91</td><td>11.90</td></tr><tr><td>Prompt Engineering</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CoT Prompting (Wei et al., 2022)</td><td>30.6</td><td>4.68</td><td>28.2</td><td>5.98</td><td>4.82</td><td>9.42</td></tr><tr><td>Paraphrasing (Zhou et al., 2022)</td><td>33.8</td><td>4.80</td><td>30.3</td><td>5.95</td><td>4.82</td><td>9.75</td></tr><tr><td>Response Refinement</td><td>38.1</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Self-Refine (Madaan et al., 2023)</td><td></td><td>4.82</td><td>31.2</td><td>5.94</td><td>7.32</td><td>11.81</td></tr><tr><td>Data Selection</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Filtering (Original)</td><td>34.1</td><td>4.76</td><td>28.0</td><td>5.90</td><td>4.70</td><td>9.00</td></tr><tr><td>Filtering (Refined)</td><td>31.6 29.7</td><td>4.57 4.71</td><td>31.2</td><td>6.03</td><td>4.93</td><td>10.09</td></tr><tr><td>R.I.P (Yu et al., 2025)</td><td></td><td></td><td>23.9</td><td>6.03</td><td>4.55</td><td>8.63</td></tr><tr><td>Refined Dataset</td><td>47.8</td><td>5.38</td><td>35.8</td><td>6.39</td><td>8.86</td><td>18.78</td></tr></table>

Table 2: Results on MT-Bench, Evol-Instruct, and AlpacaEval across dataset improvement strategies. WR: win rate; Score: average score.

Experimental Results. Table 1 reports offline results for DPO and SimPO on both backbones. Refined datasets yield broad gains over original datasets across objectives, backbones, and benchmarks, with especially clear improvements on AlpacaEval LC win rate. These results indicate that improving instruction quality can strengthen preference data without changing the preference-learning objective. We further demonstrate on Llama-3-8B feedback and refiner and a different reward model <sup>2</sup> setting in Appendices D.1 and D.2, respectively.

Baseline Comparisons. We compare our methods to other baseline methods: Prompt rewriting includes CoT prompting (Wei et al., 2022) and generic paraphrasing (Zhou et al., 2022); response refinement uses Self-Refine (Madaan et al., 2023); and data selection includes reward-based filtering and R.I.P. (Yu et al., 2025), which select examples using response-level signals such as reward, length, and reward gap. These baselines modify the instruction surface, generated responses, or dataset composition, whereas our method refines weak instructions to improve preference data without discarding examples.

As shown in Table 2, instruction refinement achieves the strongest overall performance among all compared interventions. The gains over CoT prompting and paraphrasing indicate that generic rewriting is not sufficient, while gains over Self-Refine and data selection show that repairing weak instructions provides benefits beyond responselevel refinement or filtering. These results support instruction quality as a distinct axis of preference data improvement.

Refined Dataset Analysis. We examine whether refinement changes task intent or reduces task difficulty. Human evaluation shows that 88.4% of refined instructions preserve the core task intent (see Appendix F.1). We also compare Instruction-Following Difficulty (IFD) scores (Li et al., 2024) and find only a small change, with a slight increase after refinement. This suggests that the gains are unlikely to come simply from making instructions easier to follow (Appendix E.4).

## 4.2 Online Preference Learning

Experimental Setups. In online preference learning, the model generates preference data and is then updated on those data, allowing the policy and data distribution to co-evolve. This setting tests whether instruction quality continues to limit newly generated data as the model improves over iterations.

We follow online preference learning pipelines from SPA (Kim et al., 2025) and iterative DPO, where labels are obtained from model-generated responses using either implicit rewards or an external reward model; SPA further smooths smallgap preference signals. We integrate instruction refinement by applying one refinement step to all instructions before each response-generation round, without threshold-based selection. The underlying online preference-learning algorithm is otherwise unchanged, isolating the effect of improving instruction quality during iterative data generation.

<table><tr><td rowspan="2">Method</td><td colspan="2">MT-Bench</td><td colspan="2">Evol-Instruct</td><td colspan="2">AlpacaEval 2.0</td></tr><tr><td>Win Rate</td><td>Score</td><td>Win Rate</td><td>Score</td><td>Win Rate</td><td>LC Win Rate</td></tr><tr><td>SFT</td><td>27.8</td><td>4.61</td><td>23.4</td><td>5.8</td><td>3.95</td><td>7.49</td></tr><tr><td>DPO</td><td>39.4</td><td>5.12</td><td>37.4</td><td>6.05</td><td>7.73</td><td>12.33</td></tr><tr><td>SPA (iter 3) + w/ instruction refinement</td><td>50.6 55.9 (↑ 5.3%p)</td><td>5.48 5.49 (↑ 0.01)</td><td>55.0 55.0 (−)</td><td>6.91 7.24 (↑ 0.33)</td><td>22.89 23.60 (↑0.71%p)</td><td>21.52 21.81 (↑ 0.29%p)</td></tr><tr><td>Reward Model (iter 3) + w/ instruction refinement</td><td>47.8 47.8 (-)</td><td>5.40 5.33 (↓ 0.07)</td><td>47.0 44.3 (↓ 2.7%p)</td><td>6.39 6.61 (↑ 0.22)</td><td>11.78 11.51 (↓0.26%p)</td><td>19.92 20.62 (↑ 0.69%p)</td></tr><tr><td>Implicit Reward (iter 3) + w/ instruction refinement</td><td>51.9 54.4 (↑2.5%p)</td><td>5.68 5.49 (↓ 0.19)</td><td>52.3 57.6 (↑ 5.3%p)</td><td>6.78 6.91 (↑ 0.13)</td><td>18.40 25.53 (↑ 7.14%p)</td><td>24.62 25.03 (↑ 0.41%p)</td></tr></table>

Table 3: Evaluation results on MT-Bench, Evol-Instruct, and AlpacaEval 2.0 for LLaMA-3-8B models trained with SPA (iter 3), reward model annotation, and implicit reward annotation. Bold indicates the better result within each method, and arrows indicate the change compared to the non-refined counterpart.

<table><tr><td rowspan="2">Model</td><td colspan="6">DPO</td><td colspan="6">SimPO</td></tr><tr><td colspan="2">MT-Bench Pairwise Single</td><td colspan="2">Evol-Instruct Pairwise</td><td colspan="2">AlpacaEval WR</td><td colspan="2">MT-Bench Pairwise</td><td colspan="2">Evol-Instruct Pairwise</td><td colspan="2">AlpacaEval LC WR</td></tr><tr><td>SFT</td><td>27.8%</td><td>4.61</td><td>23.4%</td><td>Single 5.80</td><td>3.55</td><td>LC WR 7.15</td><td>27.8%</td><td>Single 4.61</td><td>23.4%</td><td>Single 5.80</td><td>WR 3.55</td><td>7.15</td></tr><tr><td>Original-OSSAT</td><td></td><td>5.11</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Refined-OSSAT</td><td>41.9% 40.9%</td><td>5.24</td><td>29.8% 36.5%</td><td>6.32 6.26</td><td>7.90 7.99</td><td>11.71 13.86</td><td>37.5% 44.7%</td><td>5.26 5.07</td><td>26.4% 32.3%</td><td>6.41 6.48</td><td>8.75 8.37</td><td>12.32 13.31</td></tr></table>

Table 4: Cross-reward-model evaluation results on MT-Bench, Evol-Instruct, and AlpacaEval. For MT-Bench and Evol-Instruct, each cell reports the pairwise win rate and single-answer score, while for AlpacaEval, it reports the win rate and length-controlled win rate. The highest values in each column are highlighted in bold.

Following the SPA setup, we use UltraFeedback and initialize the policy from a LLaMA3-8B SFT checkpoint. Training starts with DPO on 3.3% gold-labeled data, followed by three online iterations with 8K, 20K, and 30K on-policy samples. PairRM (Jiang et al., 2023a) is used for rewardmodel preference annotation.

Experimental Results. Table 3 shows that instruction refinement improves most online preference learning variants relative to training with original instructions. The gains are strongest for SPA at iteration 3, where all benchmarks improve, and AlpacaEval LC win rate improves consistently across SPA, reward-model, and implicit-reward variants. These gains are notable because online learning continually updates the policy that generates new data; nevertheless, weak instructions remain a bottleneck in the evolving data-generation process.

The gains are broad but not uniform across all metrics. MT-Bench score slightly decreases for reward-model and implicit-reward variants despite improved or maintained win rates. A case study in Appendix G.2 suggests that such drops often occur in tasks requiring strict output formats, indicating that online instruction refinement remains task-dependent. Additional intermediate-iteration results are provided in Appendix E.2.

## 5 Discussion

## 5.1 Cross-Reward Model Experiment

In Section 4.1, we use ArmoRM both to select instructions for refinement and to construct preference labels. To examine whether the observed gains are specific to this reward model, we conduct a cross-reward-model experiment. Specifically, we use ArmoRM to select instructions for refinement, while using OSSAT Reward Model (OpenAssistant, 2023) as reward model to assign preference labels. Using the resulting datasets, we train models with the same DPO and SimPO pipelines and compare Original-OSSAT with Refined-OSSAT.

The results in Table 4 show that Refined-OSSAT generally performs better than Original-OSSAT under the OSSAT-RM annotation setting. While the gains are not uniform across all metrics, the overall trend suggests that refined-instruction data remains effective even when preference labels are assigned by a different reward model. This suggests that the benefit of our method is not specific to the bias of a single reward model.

Reward Model Human Evaluation. To further examine reward-model dependency, we conducted a small-scale human evaluation on 30 instruction– response-pair tuples, evenly split between examples selected and not selected for refinement. Each tuple was annotated by three evaluators, with the final decision determined by majority vote.

![](images/5436af01561bb3b20792fd53006d72046cbec777585f0a89365a64d5df970615.jpg)  
(a) Best-of-N Reward Plot

![](images/45573e46622a25f01e54f869c1ec5efc9521443401b6154c149e255943d238e5.jpg)  
(b) Worst-of-N Reward Plot

Figure 3: Best-of-N and Worst-of-N analysis showing that instruction refinement raises the response quality ceiling across high and low instruction quality groups, with more pronounced improvements for lower-quality instructions.  
![](images/49a22e921900cf87e33989c4c31a54fc2536fa7847121aeb7e05ffe923731a2d.jpg)  
Figure 4: Sensitivity analysis of the minimum-reward threshold $\tau$ for instruction refinement. Refinement consistently improves over the non-refined baseline, with optimal performance at moderate thresholds.

The reward-model-based refinement decision agreed with the human majority judgment in 76.7% of cases. For preference labeling, the reward-model label agreed with the human majority choice in 71.4% of cases. These results suggest that the reward-model signals used for both instruction selection and preference-label construction are reasonably aligned with human judgments. Nevertheless, the imperfect agreement with human judgments may be attributed in part to known reward-model biases toward factors such as response length, style, or surface-level fluency (Bu et al., 2025). Therefore, in our pipeline, the rewardmodel objective should be interpreted as a proxy for human preference rather than a direct substitute. Details of the annotation protocol and prompt are provided in Appendix F.2.

## 5.2 BoN & WoN Analysis

Figure 3 shows how response quality changes after instruction refinement under a fixed sampling budget using Best-of-N (BoN) and Worst-of-N (WoN) evaluation. Across two instruction quality groups (High-X, Low-X), instruction refinement consistently shifts both BoN and WoN curves upward, indicating a higher achievable reward ceiling as well as an improved reward floor. The effect is most pronounced for the Low-X group. Before refinement, low-quality instructions exhibit a lower average reward even with increased sampling, reflecting a tight upper bound imposed by poor instruction quality. After refinement, the BoN and WoN curves for Low-X rise significantly, suggesting that instruction improvement can alleviate limitations induced by low-quality instructions. Overall, these results suggest that improving instruction quality can raise achievable response quality while also improving the lower tail of the responsequality distribution under a fixed sampling budget.

## 5.3 Threshold Analysis

In the offline setting, we select instructions for refinement by thresholding the minimum response reward. A higher threshold makes refinement stricter, filtering more low-quality instructions but increasing risk of over-constrained data. To analyze sensitivity to the threshold τ , we vary $\tau \in$ {0.05, 0.08, 0.10, 0.13, 0.15} under the same setup as Section 4.1. Figure 4 reports AlpacaEval LC win rates for DPO and SimPO across thresholds.

Across most values of $\tau ,$ instruction refinement consistently outperforms the baseline without refinement, indicating robust improvements in alignment performance. Performance generally increases as the threshold becomes more selective, suggesting that refining lower-reward instructions yields more informative preference data. DPO performs best around τ = 0.13 (refining ∼30% of instructions) and remains robust at higher thresholds. In contrast, SimPO degrades at higher thresholds, likely due to overly aggressive refinement that reduces response diversity and weakens preference signals. Overall, instruction refinement is broadly robust, with moderate thresholds yielding the most consistent gains. However, the degradation of SimPO at higher thresholds highlights a practical limitation, as overly aggressive refinement can reduce response diversity and weaken preference signals. Additional sensitivity analyses are provided in Appendix E.3.

<table><tr><td>Method</td><td colspan="2">MT-Bench WR Score</td><td colspan="2">Evol-Instruct WR Score</td><td colspan="2">AlpacaEval WR LC WR</td></tr><tr><td>SFT</td><td>27.8</td><td>4.61</td><td>23.4</td><td>5.80</td><td>3.55</td><td>7.15</td></tr><tr><td>DPO (Original Dataset)</td><td>39.7</td><td>4.91</td><td>33.3</td><td>6.15</td><td>5.91</td><td>11.90</td></tr><tr><td>Refined Dataset</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o Rubric-Based Prompt</td><td>47.8</td><td>5.52</td><td>36.2</td><td>6.54</td><td>8.60</td><td>16.23</td></tr><tr><td>w/ Rubric-Based Prompt</td><td>47.5</td><td>5.60</td><td>41.7</td><td>6.70</td><td>9.92</td><td>19.54</td></tr></table>

Table 5: Rubric ablation results across evaluation benchmarks. WR: win rate; Score: average score.

<table><tr><td>∆ Rubric Score</td><td colspan="2">w/o Rubric ∆ Avg. ∆ Min.</td><td colspan="2">w/ Rubric</td></tr><tr><td>Clarity</td><td>0.048</td><td>0.047</td><td>∆ Avg. 0.064</td><td>∆ Min. 0.077</td></tr><tr><td>Specificity</td><td>-0.041</td><td>-0.029</td><td>-0.008</td><td>0.015</td></tr><tr><td>Completeness</td><td>-0.028</td><td>-0.017</td><td>-0.032</td><td>-0.005</td></tr><tr><td>Safety</td><td>0.062</td><td>0.057</td><td>0.091</td><td>0.091</td></tr><tr><td>Answerability</td><td>-0.002</td><td>0.009</td><td>0.056</td><td>0.078</td></tr><tr><td></td><td>0.097</td><td>0.087</td><td>0.089</td><td></td></tr><tr><td>Conciseness</td><td></td><td></td><td></td><td>0.086</td></tr><tr><td>Format Consistency</td><td>0.016</td><td>0.018</td><td>0.003</td><td>0.014</td></tr></table>

Table 6: Pearson correlation between rubric score gains and average/minimum reward score gains, with and without rubric-based feedback.

## 5.4 Rubric Effect Analysis

Rubric Ablation. Although our method uses GPT-4o to refine instructions, the observed gains may partly come from general rewriting by a strong foundation model. To disentangle these effects, we conduct a rubric ablation in which GPT-4o refines the instructions without access to rubric-based feedback. Both settings refine the entire dataset once for a controlled comparison. In this setting, GPT-4o rewrites the instructions using the same guidance but without access to rubric-based feedback. This design isolates the contribution of rubrics while controlling for the refine model and prompt. As shown in Table 5, rubric-based feedback yields improvements over rubric-free refinement on all metrics except the MT-Bench win rate. This indicates that rubric-guided refinement provides robust gains beyond general instruction rewriting by GPT-4o, suggesting that our seven rubric dimensions help produce instructions that are more beneficial for preference learning.

Rubric-Reward Correlation. We further examine whether rubric-based feedback better aligns instruction-level improvements with downstream reward gains than rubric-free feedback. For each rubric dimension, we compute the correlation between rubric score gains and response reward gains from the original to the refined instruction. We consider two reward gain metrics: average reward gain and minimum reward gain. Since each instruction has two responses, average reward is their mean score, while minimum reward is the lower score. This dimension-level analysis identifies which rubric improvements are most closely associated with downstream reward gains.

Table 6 presents the results. Rubric-based refinement shows stronger positive correlations for safety, answerability, and clarity than rubric-free refinement, suggesting that rubric-based feedback makes improvements in these dimensions more aligned with downstream reward gains. By contrast, specificity and completeness show near-zero or negative correlations with reward gains. This suggests that these dimensions are less directly associated with reward increases, although they may capture quality aspects not fully reflected by the reward model. Overall, the results indicate that rubric-based feedback better aligns instruction improvements with response reward gains.

Additional Analyses. We provide additional analyses to further examine the validity and generality of instruction refinement. First, we conduct a human evaluation to assess whether refinement preserves the original task intent. The results show that most refined instructions retain the core task, suggesting that the observed gains are not primarily driven by task drift or distributional replacement (Appendix F.1). We also evaluate refinement on a medical-domain preference dataset and observe improvements on the QA-style medical benchmark (UltraMedical-Preference (Zhang et al., 2024)), indicating that instruction refinement can also benefit preference learning in domain-specific settings (Appendix D.3).

## 6 Conclusion

We show that instruction quality can limit achievable response quality and, consequently, preference learning performance. Through Best-of-N analysis, we provide evidence that low-quality instructions impose a ceiling on response quality under a fixed sampling budget. We further show that improving instruction quality strengthens preference signals without discarding data, in both offline and online settings. Across multiple algorithms and benchmarks, instruction refinement consistently improves preference data quality and alignment performance. Overall, these results suggest that instruction quality is an important factor in determining the effectiveness of preference learning, beyond response quality and optimization objectives.

## Limitation

Our study has several limitations. First, our approach relies on a reward model to identify and refine low-quality instructions, and its effectiveness may depend on the model’s calibration and biases. Second, aggressive refinement can degrade performance in some settings, indicating sensitivity to threshold selection. Third, instruction refinement introduces additional computational cost due to iterative feedback and rewriting, although it is generally more cost-efficient than collecting new preference data. Finally, our study is limited to text-only language models, and extending this approach to multimodal settings remains an important direction for future work.

## References

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, and 1 others. 2022. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073.

Yuyan Bu, Liangyu Huo, Yi Jing, and Qing Yang. 2025. Beyond excess and deficiency: Adaptive length bias mitigation in reward models for rlhf. In Findings of the association for computational linguistics: NAACL 2025, pages 3091–3098.

Derin Cayir, Renjie Tao, Rashi Rungta, Kai Sun, Sean Chen, Haidar Khan, Minseok Kim, Julia Reinspach, and Yue Liu. 2025. Refine-n-judge: Curating highquality preference chains for llm-fine-tuning. arXiv preprint arXiv:2508.01543.

Sayak Ray Chowdhury, Anush Kini, and Nagarajan Natarajan. Provably robust dpo: Aligning language

models with noisy feedback. In Forty-first International Conference on Machine Learning.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2024. ULTRAFEEDBACK: Boosting language models with scaled AI feedback. In Forty-first International Conference on Machine Learning.

Xun Deng, Han Zhong, Rui Ai, Fuli Feng, Zheng Wang, and Xiangnan He. 2025. Less is more: Improving llm alignment via preference data selection. arXiv preprint arXiv:2502.14560.

Qingxiu Dong, Li Dong, Xingxing Zhang, Zhifang Sui, and Furu Wei. 2025. Self-boosting large language models with synthetic preference data. In The Thirteenth International Conference on Learning Representations.

Yang Gao, Dana Alon, and Donald Metzler. 2024. Impact of preference noise on the alignment performance of generative language models. arXiv preprint arXiv:2404.09824.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Sam Houliston, Alizée Pace, Alexander Immer, and Gunnar Rätsch. 2024. Uncertainty-penalized direct preference optimization. arXiv preprint arXiv:2410.20187.

Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. 2023a. Llm-blender: Ensembling large language models with pairwise ranking and generative fusion. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 14165–14178.

Dongsheng Jiang, Yuchen Liu, Songlin Liu, Jin’e Zhao, Hao Zhang, Zhen Gao, Xiaopeng Zhang, Jin Li, and Hongkai Xiong. 2023b. From clip to dino: Visual encoders shout in multi-modal large language models. arXiv preprint arXiv:2310.08825.

Dongyoung Kim, Kimin Lee, Jinwoo Shin, and Jaehyung Kim. 2025. Spread preference annotation: Direct preference judgment for efficient LLM alignment. In The Thirteenth International Conference on Learning Representations.

Olivia Kim. 2025. Detail matters: Measuring the impact of prompt specificity on reasoning in large language models. arXiv preprint arXiv:2512.02246.

JoonHo Lee, JuYoun Son, Juree Seok, Wooseok Jang, and Yeong-Dae Kwon. 2025a. Preference consistency matters: Enhancing preference learning in language models with automated self-curation of training corpora. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 12150–12169.

Seohyeong Lee, Eunwon Kim, Hwaran Lee, and Buru Chang. 2025b. Dataset cartography for large language model alignment: Mapping and diagnosing preference data. arXiv preprint arXiv:2505.23114.

Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, and Jing Xiao. 2024. From quantity to quality: Boosting llm performance with self-guided data selection for instruction tuning. In Proceedings of the 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7602–7635.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca\_eval.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, and 1 others. 2022. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110.

Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, and 1 others. 2024. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522.

Yixin Liu, Kejian Shi, Alexander Richard Fabbri, Yilun Zhao, Peifeng Wang, Chien-Sheng Wu, Shafiq Joty, and Arman Cohan. 2025. Reife: Re-evaluating instruction-following evaluation. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 12247–12287.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, and 1 others. 2023. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36:46534–46594.

Saaduddin Mahmud, Mason Nakamura, and Shlomo Zilberstein. 2025. Maple: A framework for active preference learning guided by large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 27518–27528.

Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. Simpo: Simple preference optimization with a reference-free reward. Advances in Neural Information Processing Systems, 37:124198–124235.

OpenAssistant. 2023. OpenAssistant/reward-modeldeberta-v3-base. Hugging Face Model Card.

Yu Pan, Zhongze Cai, Huaiyang Zhong, Guanting Chen, and Chonghuan Wang. 2025. What matters in data for DPO? In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. 2022. Red teaming language models with language models. arXiv preprint arXiv:2202.03286.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652.

Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. 2024. Interpretable preferences via multi-objective reward modeling and mixture-ofexperts. In EMNLP.

Jiaming Wang, Yunke Zhao, Peng Ding, Jun Kuang, Zongyu Wang, Xuezhi Cao, and Xunliang Cai. 2025. Ask, fail, repeat: Meeseeks, an iterative feedback benchmark for llms’ multi-turn instruction-following ability. arXiv preprint arXiv:2504.21625.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824– 24837.

Zhaofeng Wu, Michihiro Yasunaga, Andrew Cohen, Yoon Kim, Asli Celikyilmaz, and Marjan Ghazvininejad. 2025. rewordbench: Benchmarking and improving the robustness of reward models with transformed inputs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 3383–3409.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin

Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244.

Tianyi Yan, Fei Wang, James Y Huang, Wenxuan Zhou, Fan Yin, Aram Galstyan, Wenpeng Yin, and Muhao Chen. 2024. Contrastive instruction tuning. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 10288–10302.

Ping Yu, Weizhe Yuan, Olga Golovneva, Tianhao Wu, Sainbayar Sukhbaatar, Jason E Weston, and Jing Xu. 2025. Rip: Better models by survival of the fittest prompts. In International Conference on Machine Learning, pages 73350–73374. PMLR.

Kaiyan Zhang, Sihang Zeng, Ermo Hua, Ning Ding, Zhang-Ren Chen, Zhiyuan Ma, Haoxin Li, Ganqu Cui, Biqing Qi, Xuekai Zhu, Xingtai Lv, Hu Jinfang, Zhiyuan Liu, and Bowen Zhou. 2024. Ultramedical: Building specialized generalists in biomedicine. Preprint, arXiv:2406.03949.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, and 1 others. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand. Association for Computational Linguistics.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, and 1 others. 2023a. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023b. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. 2022. Large language models are human-level prompt engineers. In The eleventh international conference on learning representations.

He Zhu, Yifan Ding, Yicheng Tao, Zhiwen Ruan, Yixia Li, Wenjia Zhang, Yun Chen, and Guanhua Chen. 2025. Fanno: Augmenting high-quality instruction data with open-sourced llms only. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 17633–17653.

## A Motivating Analysis Details

This matters for preference learning because both tails affect pairwise data informativeness. Let $F _ { x } ( t ) = { \mathrm { P r } } _ { Y \sim \mu _ { x } } [ R ( x , Y ) \leq t ]$ be the responsequality distribution induced by instruction x, where $R ( x , y )$ is a reward-based proxy for response quality. With K sampled candidates, the probability of obtaining a high-quality response above $\tau _ { h }$ is $1 - F _ { x } ( \tau _ { h } ) ^ { \bar { K } }$ , while the probability of obtaining a trivial response below $\tau _ { l }$ is $\begin{array} { r } { 1 - ( 1 - F _ { x } ( \tau _ { l } ) ) ^ { K } } \end{array}$ Thus, improving the upper tail increases coverage of strong chosen candidates, while reducing lowertail mass decreases degenerate rejected candidates that can form overly easy preference pairs. This motivates instruction refinement as an upstream intervention before candidate response generation.

We introduce our motivation by presenting evidence from Best-of-N (BoN) and Worst-of-N (WoN) experiments, showing that instruction quality constrains both the upper and lower tails of response quality. BoN selects the highest-reward response among N generated candidates, providing an estimate of the maximum attainable response quality, while WoN selects the lowestreward response, characterizing the quality floor of responses induced by an instruction. We partition 1.5K sampled instructions from the UltraFeedback dataset into three quality groups (High, Mid, and Low) based on our instruction quality assessment, which scores each instruction on seven criteria and uses the average score to assign a quality tier (refer to Section 3.3), and generate multiple responses per instruction using LLaMA3-8B (Grattafiori et al., 2024).

As shown in Figure $2 ( \mathrm { a } ) .$ , BoN reward improves with increasing $N .$ , but a substantial gap persists between high- and low-quality instruction groups even at large sampling budgets. At N = 16, highquality instructions achieve a mean BoN reward of 0.150, compared to 0.128 for low-quality instructions, yielding a gap of 0.0218 $( p < 0 . 0 0 0 1 )$ This suggests that instruction quality constrains the observed upper tail of response quality under the sampled candidate distribution. Figure 2(b) further shows that the same trend holds for WoN: highquality instructions also yield higher worst-case rewards than low-quality instructions. At $N = 1 6$ high-quality instructions achieve a mean WoN reward of 0.092, compared to 0.063 for low-quality instructions, corresponding to a gap of 0.0297 $( p < 0 . 0 0 0 1 )$ . Together, these results suggest that high-quality instructions improve both the upper and lower tails of the sampled response-quality distribution.

To interpret these results, let $\mu _ { x } ( y ) = \mu ( y \mid x )$ denote the candidate response distribution induced by instruction x. For each sampled response $Y _ { i } \sim$ $\mu _ { x }$ , let $R ^ { * } ( x , Y _ { i } )$ denote its latent response quality. Given N sampled candidates $Y _ { 1 } , \dots , Y _ { N } \sim \mu _ { x } ,$ the BoN and WoN experiments can be viewed as measuring the upper and lower order statistics of this quality distribution:

$$
\begin{array} { r } { R _ { \operatorname* { m a x } } ^ { ( N ) } ( x ) = \underset { i \in [ N ] } { \operatorname* { m a x } } R ^ { * } ( x , Y _ { i } ) , } \\ { R _ { \operatorname* { m i n } } ^ { ( N ) } ( x ) = \underset { i \in [ N ] } { \operatorname* { m i n } } R ^ { * } ( x , Y _ { i } ) . } \end{array}\tag{1}
$$

The observed BoN gap indicates that high-quality instructions induce candidate distributions with better upper tails. The observed WoN gap further indicates that they also raise the lower tail, reducing the likelihood of very poor sampled responses.

Preference data are constructed from these sampled candidates by selecting a chosen response $Y ^ { + }$ and a rejected response $Y ^ { - }$ <sup>−</sup>. Let

$$
r ^ { + } = R ^ { * } ( x , Y ^ { + } ) , \qquad r ^ { - } = R ^ { * } ( x , Y ^ { - } )\tag{2}
$$

denote the latent qualities of the chosen and rejected responses.

Coverage Enables Direct Gradients. When an instruction is low quality, the induced candidate distribution may rarely contain strong responses. For a high-quality threshold $\tau ,$ suppose that lowquality instructions assign negligible probability mass to high-quality responses:

$$
\operatorname* { P r } _ { Y \sim \mu _ { x _ { \mathrm { l o w } } } } \left[ R ^ { * } ( x _ { \mathrm { l o w } } , Y ) \geq \tau \right] \approx 0 .\tag{3}
$$

Since $Y ^ { + }$ is selected from a finite set of candidates sampled from $\mu _ { x _ { \mathrm { l o w } } }$ , the probability of obtaining a high-quality chosen response is also small:

$$
\operatorname* { P r } \left[ R ^ { * } ( x _ { \mathrm { l o w } } , Y ^ { + } ) \ge \tau \ | \ x _ { \mathrm { l o w } } \right] \approx 0 .\tag{4}
$$

For a preference tuple $( x , y ^ { + } , y ^ { - } )$ , the DPO gradient (Rafailov et al., 2023) can be written as

$$
\begin{array} { r } { \nabla _ { \theta } \mathcal { L } _ { \mathrm { D P O } } = - \beta \mathbb { E } _ { ( x , y ^ { + } , y ^ { - } ) \sim \mathcal { D } } \left[ \sigma ( - u _ { \theta } ) d _ { \theta } \right] , } \end{array}\tag{5}
$$

where

$$
u _ { \theta } = \beta \left[ \log \frac { \pi _ { \theta } ( y ^ { + } \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { + } \mid x ) } - \log \frac { \pi _ { \theta } ( y ^ { - } \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { - } \mid x ) } \right] _ { }\tag{6}
$$

and

$$
d _ { \theta } = \nabla _ { \theta } \log \pi _ { \theta } ( y ^ { + } \mid x ) - \nabla _ { \theta } \log \pi _ { \theta } ( y ^ { - } \mid x ) .\tag{7}
$$

The first term in $d _ { \theta }$ increases the likelihood of the chosen response. Therefore, if high-quality chosen responses are rarely observed, DPO receives little direct gradient signal for increasing the likelihood of high-quality responses. (Pan et al., 2025) In contrast, high-quality instructions make such chosen responses more likely to appear in preference data, yielding more useful positive updates.

Higher Rejected Floors Yield More Informative Pairs. Instruction quality also affects the rejected response because $Y ^ { - }$ is selected from candidates sampled from the same response distribution $\mu _ { x }$ . A higher-quality lower tail increases the typical value of $r ^ { - }$ <sup>−</sup>, thereby reducing the quality gap $r ^ { + } - r$ between chosen and rejected responses. When a large quality gap between $r ^ { + }$ and $r ^ { \cdot }$ <sup>−</sup> is reflected in the learned preference margin $u _ { \theta } ,$ , we have

$$
u _ { \theta } \gg 0 \Rightarrow \sigma ( - u _ { \theta } ) \approx 0 .\tag{8}
$$

Such pairs have a large quality gap between the chosen and rejected responses, making them easy to distinguish and less informative for gradient updates (Houliston et al., 2024). In contrast, when the quality gap $r ^ { + } - r ^ { - }$ is smaller and reflected in $u _ { \theta } ,$ , the resulting margin can be smaller (Lee et al., 2025b):

$$
u _ { \theta } \gg 0 \Rightarrow \sigma ( - u _ { \theta } ) \approx 0 .
$$

Thus, improving instruction quality can provide more direct positive updates toward high-quality chosen responses while maintaining informative gradients from pairs with smaller chosen–rejected quality gaps.

## B Experimental Details

## B.1 Instruction Evaluation Rubric Details

We survey prior studies on active preference learning and preference dataset selection to identify which aspects of instructions degrade preference data quality, and derive seven instruction refinement rubrics from this analysis.

• Clarity assesses whether the instruction clearly specifies what the model is expected to do without ambiguity. Prior work shows that ambiguous instructions can lead models to misidentify the intended action, motivating consistency in interpretation (Wang et al., 2025).

• Specificity evaluates whether the instruction provides sufficiently concrete requirements, as vague or underspecified instructions have been shown to degrade model performance (Zhou et al., 2023b; Kim, 2025). Together, these criteria encourage instructions to state requirements explicitly and precisely.

• Completeness measures whether all necessary information and constraints to answer an instruction are provided. Prior work shows that missing context or incomplete task definitions are associated with failures independent of model capability, highlighting issues in instruction design (Wang et al., 2025; Liang et al., 2022). Providing a complete specification is therefore essential for reliable response generation.

• Safety evaluates whether the instruction may elicit harmful or unsafe outputs. Previous studies report that unsafe instructions can induce harmful behaviors or generate dangerous information (Bai et al., 2022; Perez et al., 2022). We explicitly assess instruction safety to ensure refinement remains within acceptable response boundaries.

• Answerability examines whether the instruction admits a well-defined and reasonably answerable response. Several data collection studies identify answerability as a core criterion for high-quality instruction data, noting that unanswerable or ill-posed instructions lead to failures regardless of model strength (Mahmud et al., 2025; Zhu et al., 2025). We therefore treat answerability as a first-class criterion in instruction evaluation.

• Conciseness assesses whether the instruction avoids unnecessary information that may distract the model from core requirements. Prior work shows that excessive constraints or irrelevant details can impair generation quality (Liu et al., 2023). This criterion ensures that instructions convey only information essential for the task.

• Format Consistency evaluates whether the desired output format is clearly specified and consistently illustrated, for example through few-shot examples. Inconsistent or underspecified formats have been shown to cause instruction-following failures (Zhou et al., 2023b; Liu et al., 2025).

For each rubric dimension, we use a 5-point Likert-style scale. This choice is motivated by prior human and AI evaluation practices in natural language generation. In particular, recent AIfeedback annotation pipelines such as UltraFeedback (Cui et al., 2024) use 1–5 scalar scores with detailed score documentation for each aspect to reduce variability and subjectivity in annotation standards. Furthermore, because our refinement procedure is methodologically inspired by Self-Refine (Madaan et al., 2023), we follow its use of bounded, multi-dimensional 1–5 scoring prompts with textual feedback.

## B.2 Feedback and Refinement Prompt

As described in section 3.3, we include the prompts used as input to the LLM-as-a-judge for feedback generation and instruction refinement in Figure 9 and 10. These prompts are adapted from prior Self-Refine (Madaan et al., 2023), with newly defined rubric criteria for instruction-level refinement. The prompts are designed to robustly elicit both rubricbased scores and natural language feedback, and we strictly constrain the output format and provide few-shot examples to ensure consistent and accurate refinement. Especially, to preserve the original intent during refinement, we explicitly impose the constraints to preserve the intent of the original on the instructions.

## B.3 Benchmarks

For both main experimental settings, LLM alignment is evaluated using three benchmarks—MT-Bench (Zheng et al., 2023), Evol-Instruct (Xu et al., 2023), and AlpacaEval 2.0 (Li et al., 2023). These benchmarks adopt an LLM-as-a-judge evaluation paradigm and report Win Rate, defined as the proportion of cases in which the target model’s response is preferred over that of a comparison model. For MT-Bench and Evol-Instruct, we additionally report single score evaluations that quantify absolute response quality. For AlpacaEval, we report both the standard Win Rate and the lengthcontrolled Win Rate to mitigate biases arising from response length.

MT-Bench (Zheng et al., 2023) and Evol-Instruct (Xu et al., 2023) are alignment benchmarks designed to evaluate instruction-following capabilities under different levels of interaction and complexity. MT-Bench focuses on multi-turn dialogue and instruction execution, assessing a model’s ability to maintain coherent and natural conversations across two-turn interactions. In contrast, Evol-Instruct addresses the bias of existing benchmarks toward relatively simple instructions by automatically generating instruction data with diverse difficulty levels using LLMs, resulting in a more challenging evaluation set.

Both benchmarks adopt the same automatic evaluation protocol based on the llm-judge framework (Zheng et al., 2023), with GPT-4o-2024-11-20 serving as the judge model. For each instruction, responses generated by the target model are compared against those produced by GPT-3.5-turbo to assign win/lose/tie outcomes, enabling the computation of win rates. In addition, the judge produces scalar quality scores for individual responses, and we report the averaged single score for each benchmark.

AlpacaEval (Li et al., 2023) is an LLM-based benchmark that evaluates general instructionfollowing performance. Each example consists of an instruction paired with two responses: one generated by the target model and the other by a reference model, GPT-4-1106-preview. Judging is conducted by GPT-4o-mini-2024-07-18, which compares the two responses and assigns a win/lose/tie label according to how well each fulfills the instruction. We report the win rate of the target model relative to the reference model.

## C Training Details

In this section, we provide the training setup used in our experiments for reproducibility, including the basic configuration and hyperparameter settings. In the main experiments, both DPO and SimPO training were conducted, and we implemented both using the LLaMA-Factory (Zheng et al., 2024) codebase.

## C.1 DPO and SimPO Training Details

Offline Settings. In the offline setting, we conduct experiments using hyperparameters that achieved the best performance identified through random hyperparameter search.

For DPO, the learning rate is selected from $[ 2 e \mathrm { ~ - ~ } 6 , 1 e \mathrm { ~ - ~ } 6 , 5 e \mathrm { ~ - ~ } 7 , 6 e \mathrm { ~ - ~ } 7 ]$ and β is chosen from [0.01, 0.05, 0.1]. For SimPO, hyperparameter tuning is performed over learning rates $[ 3 e - 6 , 2 e - 6 , 1 e - 6 , 3 e - 7 , 5 e - 7 , 6 e - 7 ] , \beta \mathrm { v a l } .$ ues [1, 2, 2.5], and γ values [1.0, 1.6]. In LLaMA3- 8B SimPO, learning rate range is $[ 3 e \mathrm { ~ - ~ } 6 , 2 e \mathrm { ~ - ~ }$ $6 , 1 e - 6 , 3 e - 7 , 5 e - 7 , 6 e - 7 ]$ , and in Mistral-7B $\mathrm { S i m P O , [ 3 e - 7 , 5 e - 7 , 2 e - 6 , 1 e - 5 , 2 e - 5 ] } ,$ . The optimal hyperparameters were selected separately for each model backbone. In our experiments, for LLaMA-3-8B, DPO training was conducted with β = 0.01 and a learning rate of 1e − 6, while SimPO used $\beta = 1$ , a learning rate of 2e − 6, and γ = 1.0. For Mistral-7B, DPO was trained with $\beta = 0 . 0 1$ and a learning rate of 2e − 6. SimPO employed β = 1, a learning rate of 2e − 6, and $\gamma = 1 . 0 .$

<table><tr><td colspan="2">Method</td><td colspan="5">DPO</td><td colspan="5">SimPO</td></tr><tr><td>Backbone</td><td>Data Type</td><td colspan="2">MT-Bench WR Score</td><td colspan="2">Evol-Instruct WR Score</td><td colspan="2">AlpacaEval WR LC WR WR</td><td colspan="2">MT-Bench Evol-Instruct Score WR Score</td><td colspan="2">AlpacaEval WR LC WR</td></tr><tr><td rowspan="3">Llama3-8b</td><td>SFT</td><td>27.8</td><td>4.61</td><td>23.4 5.80</td><td>3.95</td><td>7.49</td><td>27.8</td><td>4.61 23.4</td><td>5.80</td><td>3.95</td><td>7.49</td></tr><tr><td>Original</td><td>45.0</td><td>5.57</td><td>40.6 6.60</td><td>8.74</td><td>17.59</td><td>45.6</td><td>5.45</td><td>38.1 6.63</td><td>7.08</td><td>14.92</td></tr><tr><td>Refined</td><td>51.6</td><td>5.53</td><td>36.0 6.63</td><td>9.26</td><td>19.00</td><td>48.8 5.39</td><td>41.1</td><td>6.64</td><td>8.87</td><td>17.07</td></tr></table>

Table 7: Evaluation results of prompt refinement experiment with the Feedback & Refine module, conducted under the same experimental setting as Table 1. The highest values for each backbone are highlighted in bold.

The LLaMA-3-8B model was trained for 3 epochs, while the Mistral-7B model was trained for 1 epoch. Both models were trained with a batch size of 16. All experiments were conducted using two NVIDIA H200 GPUs.

Online Settings. In the online setting, we followed the SPA (Kim et al., 2025) experiment configuration and used the same hyperparameters. In this experiment, we employed a LLaMA-3-8B backbone and performed one epoch of DPO training per iteration with a learning rate of 1e − 5 and $\beta = 0 . 0 1$ . We also utilize two NVIDIA H200 GPUs in online setting.

## C.2 Feedback and Refinement Module Training.

In feedback and refinement module training, we adopted supervised fine-tuning in LLaMA-3-8B model. We utilized GPT-generated 16k feedback data and refinement data, which produce in Ultra-Feedback offline experiment in Section 4.1. SFT was conducted with learning rate of 1e − 4, and we train model for 3 epoch with batch size 16.

## D Additional Experiments

## D.1 Feedback & Refinement Module Experiment

To demonstrate that refinement is possible by replacing the foundation model with a module trained via supervised fine-tuning (SFT), we conducted module experiments under the same settings as in Section 4.1.

Experimental Setups. As for the fine-tuning LLaMA3-8B, we use another unseen instructions in the UltraFeedback and collect the feedback and refinement results of them from GPT-4o. Similarly, we performed three iteration of instruction refinement. Models were trained on the dataset before refinement and on the refined dataset, and their performance was subsequently compared.

Experimental Results. Table 7 shows the offline preference learning results, where utilized a feedback module and refiner module trained via SFT instead of GPT-4o. Consistent with previous results, table also demonstrates that refined data leads to superior training performance across nearly all benchmarks. This demonstrates that a trained refiner module can effectively polish data as an alternative to a foundation model.

## D.2 Alternative Reward Model Experiment

We have demonstrated that our framework works with two different reward models—ArmoRM in the offline setting and PairRM in the online setting. To further verify that our approach does not depend on a specific reward model, we conduct additional experiments with an alternative reward model, OpenAssistant/reward-model-deberta-v3- large-v2, a general-purpose reward model trained on diverse datasets such as WebGPT and HH-RLHF. All experimental settings follow those described in Section 4.1, and we conduct DPO and SimPO on LLaMA-3-8B.

<table><tr><td colspan="2">Method</td><td colspan="6">DPO</td><td colspan="6">SimPO</td></tr><tr><td rowspan="2">Backbone</td><td rowspan="2">Data Type</td><td colspan="2">MT-Bench</td><td colspan="2">Evol-Instruct</td><td colspan="2">AlpacaEval WRLC WR</td><td colspan="2">MT-Bench</td><td colspan="2">Evol-Instruct</td><td colspan="2">AlpacaEval LC WR</td></tr><tr><td>WR</td><td>Score</td><td>WR</td><td>Score</td><td></td><td></td><td>WR</td><td>Score</td><td>WR</td><td>Score</td><td>WR</td><td></td></tr><tr><td rowspan="3">Llama3-8b</td><td>SFT</td><td>0.278</td><td>4.61</td><td>0.234</td><td>5.80</td><td>3.55</td><td>7.15</td><td>0.278</td><td>4.61</td><td>0.234</td><td>5.80</td><td>3.55</td><td>7.15</td></tr><tr><td>Original</td><td>0.466</td><td>5.47</td><td>0.374</td><td>6.39</td><td>9.43</td><td>16.56</td><td>0.466</td><td>5.23</td><td>0.401</td><td>6.56</td><td>3.61</td><td>8.07</td></tr><tr><td>Refined</td><td>0.453</td><td>5.22</td><td>0.383</td><td>6.47</td><td>9.38</td><td>17.20</td><td>0.466</td><td>5.30</td><td>0.436</td><td>6.49</td><td>6.75</td><td>13.84</td></tr></table>

Table 8: Results with an alternative reward model on MT-Bench, Evol-Instruct, and AlpacaEval. For MT-Bench and Evol-Instruct, conducted under the same experimental setting as Table 1. The highest values for each backbone are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td colspan="6">MMLU-med</td><td rowspan="2">MedQA</td><td>MedMCQA Acc.</td><td>PubMedQA</td></tr><tr><td>Clinical Knowledge</td><td>Medical Genetics</td><td>Anatomy</td><td>Professional Medicine</td><td>College Biology</td><td>College Medicine</td><td>Acc.</td><td>Acc.</td></tr><tr><td>SFT</td><td>0.592</td><td>0.710</td><td>0.556</td><td>0.592</td><td>0.618</td><td>0.578</td><td>0.515</td><td>0.486</td><td>0.022</td></tr><tr><td>DPO (Original)</td><td>0.660</td><td>0.620</td><td>0.593</td><td>0.607</td><td>0.681</td><td>0.566</td><td>0.534</td><td>0.496</td><td>0.196</td></tr><tr><td>Refined</td><td>0.694</td><td>0.680</td><td>0.593</td><td>0.588</td><td>0.681</td><td>0.578</td><td>0.537</td><td>0.502</td><td>0.254</td></tr></table>

Table 9: Medical-domain evaluation results on MMLU-med, MedQA, and PubMedQA. The highest values in each column are highlighted in bold.

Table 8 reports the results. Overall, refining instructions yields improvements on multiple benchmarks and settings under the alternative reward model. In particular, clearer improvements are observed on Evol-Instruct and AlpacaEval, while some MT-Bench metrics show smaller gains or slight decreases. These results suggest that the proposed approach is not tied to a single reward model configuration, while also indicating that the magnitude and consistency of the improvements are affected by the reward signal.

## D.3 Alternative Dataset Experiment

Since our main experiments are conducted on UltraFeedback, the observed performance gains could be specific to the data characteristics of UltraFeedback. To examine this possibility, we conduct an additional DPO experiment with LLaMA-3-8B using TsinghuaC3I/UltraMedical-Preference, a medical-domain preference dataset. We use 33K examples corresponding to 30% of the full 110K UltraMedical-Preference dataset. Following the setup in Section 4, we train LLaMA-3-8B-SFT with DPO on original and refined dataset. For instruction refinement, we use GPT-4o with the same configuration as in our main experiments, applying a refinement threshold of 0.13 for two iterations.

The results in Table 9 show an overall positive trend from instruction refinement, while the effect varies across MMLU-med. The improvements are particularly clear on QA-style benchmarks, where the refined model achieves higher scores on MedQA, MedMCQA, and PubMedQA.

![](images/bc8cecb6b603a2608369addf1e53892d33b0c45203e21d7511a3ce8a296a8c26.jpg)

![](images/c01440894726bca81c726ef2937e4983dc1bbf054e55d232458633b56ebf0a4d.jpg)  
Figure 5: Distribution of feedback scores and minimum response rewards before and after instruction refinement, showing consistent improvements in both instruction quality and lower-bound response quality.

These results suggest that our pipeline can also provide benefits on preference datasets from different domains.

## E Additional Analysis

## E.1 Instruction–Response Quality Analysis

To analyze the relationship between instruction quality and response quality, we present a scatter plot in Figure 5 that compares the distributions of feedback scores and minimum reward scores before and after instruction refinement. Each point represents an instruction, with the horizontal axis denoting the average rubric-based feedback score and the vertical axis indicating the minimum reward score across sampled responses.

After refinement, we observe a clear upward shift along both axes. The average minimum reward score rises from $\mu _ { y } = 0 . 1 0 0$ to $\mu _ { y } = 0 . 1 1 4$ which corresponds to 11.65% of the effective reward range of ArmoRM scores on UltraFeedback.

This observation suggests that refinement reduces the prevalence of low-quality generations. In other words, refined instructions are more likely to elicit consistently high-quality responses, rather than exhibiting unstable generation behavior. This distributional shift helps explain the observed improvements in both offline and online preference learning performance. These patterns suggest that high-quality instructions are associated with reliable response quality and strong preference signals.

<table><tr><td rowspan="2">Method</td><td colspan="2">MT-Bench</td><td colspan="2">Evol-Instruct</td><td colspan="2">AlpacaEval 2.0</td></tr><tr><td>Win Rate</td><td>Score</td><td>Win Rate</td><td>Score</td><td>Win Rate</td><td>LC Win Rate</td></tr><tr><td>SFT</td><td>27.8</td><td>4.61</td><td>23.4</td><td>5.8</td><td>3.95</td><td>7.49</td></tr><tr><td>DPO</td><td>39.4</td><td>5.12</td><td>37.4</td><td>6.05</td><td>7.73</td><td>12.33</td></tr><tr><td>SPA (iter 2) + w/ instruction refinement</td><td>53.1 54.7 (↑ 1.6%p)</td><td>5.56 5.59 (↑ 0.03)</td><td>56.0 55.7 (↓ 0.3%p)</td><td>6.46 6.89 (↑ 0.43)</td><td>19.14 20.80 (↑ 1.66%p)</td><td>20.27 20.82 (↑ 0.55%p)</td></tr><tr><td>Reward Model (iter 2) + w/ instruction refinement</td><td>43.1 45.3 (↑ 2.2%p)</td><td>5.19 5.18 (↓ 0.01)</td><td>40.8 42.9 (↑ 2.1%p)</td><td>6.41 6.33 (↓ 0.08)</td><td>9.19 9.01 (↓ 0.18%p)</td><td>15.49 15.62 (↑ 0.14%p)</td></tr><tr><td>Implicit Reward (iter 2) + w/ instruction refinement</td><td>53.1 50.9 (↓ 2.2%p)</td><td>5.49 5.36 (↓ 0.13)</td><td>48.4 55.5 (↑ 7.1%p)</td><td>6.54 6.61 (↑ 0.07)</td><td>14.27 21.90 (↑ 7.63%p)</td><td>21.38 22.84 (↑ 1.46%p)</td></tr></table>

Table 10: Evaluation results on MT-Bench, Evol-Instruct, and AlpacaEval 2.0 for LLaMA-3-8B models trained with SPA (iter 2). Bold indicates the better result within each method, and arrows indicate the change compared to the non-refined counterpart.

![](images/726a2f3897965128553e9f5fe8805dfbd8b2accbf2b78beccef52769e4eff634.jpg)

![](images/fb74242be979ef20b68dc8008f7f154edb3c49115e67f778e344032b73c40099.jpg)

![](images/b1437b24e089ef68a8b47d29e81b8fbfbceaedb1be29c7897bf901e5b0d2eb4c.jpg)  
Figure 6: AlpacaEval LC win rate result per iteration of online preference learning. Model capability increases as train iteration repeat, showing major increase in SPA + prompt refinement.

## E.2 Online Experiment Analysis

To provide a more detailed analysis of the SPA experiments presented in the main paper (Section 4.2), this section reports the results at iteration 2 and presents performance gain across iterations, illustrating how model performance evolves as iterative preference learning progresses.

Table 10 summarizes the results of online preference learning with two iterations of SPA, comparing baseline training with and without instruction refinement. Across most settings, incorporating instruction refinement consistently improves or maintains performance over the non-refined counterparts, demonstrating its effectiveness as a complementary mechanism in online preference learning. Notably, SPA with instruction refinement yields consistent gains across most benchmarks, while reward-model-based and implicit-reward-based settings also benefit from refinement, particularly on Evol-Instruct and AlpacaEval. These results indicate that refining queries improves the quality of on-policy preference data, leading to more informative learning signals during online training.

Figure 6 further illustrates performance trends across training iterations. In most settings—SPA and implicit-reward-based training—instruction refinement accelerates performance improvements as training progresses. These results show that instruction refinement enhances the effectiveness of online preference learning by improving data quality throughout training, rather than only at initialization.

## E.3 Threshold Sensitivity Analysis

Figure 7 reports the sensitivity of instruction refinement performance to the minimum-reward threshold τ across different evaluation benchmarks for both DPO and SimPO. Across MT-Bench, Evol-Instruct, and AlpacaEval, moderate threshold values consistently lead to improved performance compared to the non-refined baseline, indicating that selectively refining low-quality queries effectively enhances preference learning. In particular, both methods achieve their best or near-best performance around $\tau = 0 . 1 – 0 . 1 3 .$ , suggesting that filtering and refining approximately the lower portion of queries strikes a balance between removing unstable instructions and preserving informative preference signals. When the threshold is set too high (e.g., τ = 0.15), performance sometimes degrades, implying that excessive refinement may over-constrain instructions or reduce preference signal effectivity. Overall, these results confirm that instruction refinement is robust to threshold choice within a reasonable range, while highlighting the importance of moderate filtering for stable and effective preference optimization.

![](images/ab66a19b006fded7124a94e83464b3c906be8483733167dcf8ea631b619ac778.jpg)

Figure 7: τ Sensitivity Analysis Result in MT-Bench (Zheng et al., 2023), Evol-Instruct (Xu et al., 2023), and AlpacaEval 2.0 (Li et al., 2023).
<table><tr><td rowspan="2">Dataset</td><td colspan="4">MT-Bench</td><td colspan="4">Evol-Instruct</td></tr><tr><td>Overall</td><td>Low-IFD</td><td>Mid-IFD</td><td>High-IFD</td><td>Overall</td><td>Low-IFD</td><td>Mid-IFD</td><td>High-IFD</td></tr><tr><td>Original</td><td>4.913</td><td>4.302</td><td>5.056</td><td>4.925</td><td>6.147</td><td>6.342</td><td>6.208</td><td>6.411</td></tr><tr><td>Refined</td><td>5.381</td><td>5.037</td><td>5.717</td><td>5.396</td><td>6.390</td><td>6.205</td><td>6.417</td><td>6.548</td></tr></table>

Table 11: Evaluation results across different IFD difficulty levels on MT-Bench (Zheng et al., 2023) and Evol-Instruct (Xu et al., 2023). Each cell reports the average score. The highest value in each column is highlighted in bold.

## E.4 IFD Score Analysis

IFD Score Experiment. Task difficulty and instruction quality can be closely related, making it non-trivial to determine whether improved downstream performance stems from better instruction quality or from a reduction in task difficulty. In particular, if instruction refinement reduces task difficulty, the resulting performance gains may be confounded with difficulty reduction rather than reflecting improvements in instruction quality alone.

To examine whether our refinement process reduces task difficulty, we conducted an additional analysis using Instruction-Following Difficulty (IFD) scores (Li et al., 2024). We use IFD as a proxy for instruction-following difficulty, with higher scores indicating greater difficulty under the model used to compute IFD. For an instruction– answer pair (Q, A), the IFD score is defined as:

$$
\operatorname { I F D } _ { \theta } ( Q , A ) = { \frac { s _ { \theta } ( A \mid Q ) } { s _ { \theta } ( A ) } }\tag{9}
$$

We computed IFD scores for each instance before and after instruction refinement using the Llama-3-Base-8B-SFT model and compared the resulting paired scores. After refinement, the mean IFD score increased from 0.6423 to 0.6823, yielding a mean paired difference of +0.0406. In addition, 54.38% of the instances had higher IFD scores after refinement. The difference was limited in magnitude, and the increase in IFD suggests that the refined instructions were not easier but marginally more difficult to follow. Therefore, the observed improvement in model performance is unlikely to be attributable to reduced task difficulty in the refined data.

<table><tr><td>Category</td><td>Original instruction</td><td>Refined instruction</td></tr><tr><td>Major Shift</td><td>How to use corkscrew to chop the vegeta- bles?</td><td>Ensure safety by securely holding a knife to chop carrots, onions, and bell peppers on a stable cutting board.</td></tr><tr><td>Minor Shift</td><td>Create a sentence using five synonyms</td><td>Use five synonyms for “happy&quot; in a single sentence.</td></tr></table>

Table 12: Examples of major and minor instruction shifts between original and refined instructions.

![](images/a8221e738e2b48c59487b1c963e4970d38ed25ab2c1aa8bbd744f590664f7264.jpg)

![](images/824170f475ff8ffe60d768493ed88e33738fd10afd67ffedfb4d1797106e077f.jpg)

![](images/3ce1e43dd808d312ac6ff73a9b0718d8e4ac1f72cd9581ca37a3a9949de5adfc.jpg)  
Figure 8: Performance across refinement iterations. Win rate generally improves from SFT to later refinement iterations across MT-Bench, Evol-Instruct, and AlpacaEval.

Benchmark Evaluation. To further examine whether the downstream gains are driven by easier data, we conducted an additional analysis on the evaluation benchmark. If the observed improvements were mainly driven by easier training data, the refined model would be expected to show degraded performance on evaluation prompts with higher IFD scores. We therefore grouped the eval uation prompts into low-, middle-, and high-IFD groups based on their IFD scores and compared model performance within each group.

The results in Table 11 show that the gains from instruction-quality refinement were not confined to low-difficulty evaluation prompts. On both Evol-Instruct and MT-Bench, the refined model outperformed the original model across most IFD-based difficulty groups, including prompts with relatively high IFD scores. This suggests that instructionquality refinement remains effective not only for easier prompts but also for more difficult instructions. Therefore, model performance gains are unlikely to be explained solely by task difficulty.

## E.5 Effect of Refinement Iterations on Performance

To examine how performance changes with repeated refinement, we evaluate models trained on instructions refined over multiple iterations. We report the win rate for MT-Bench and Evol-Instruct, and the length-controlled win rate for AlpacaEval. As shown in Figure 8, performance generally improves from SFT to later refinement iterations across MT-Bench, Evol-Instruct, and AlpacaEval. The gains are most pronounced in the early refinement stages, while later iterations show smaller improvements or signs of saturation. This suggests that iterative refinement can further improve downstream performance, but its marginal benefit diminishes after a small number of iterations.

## F Human Evaluation

To further strengthen the robustness of the refinement process, we conducted two small-scale human evaluations. The first evaluation was designed to verify whether task drift occurred during refinement. In this evaluation, human annotators directly compared the instructions before and after refinement and assessed whether the underlying task was preserved. The second evaluation was conducted to further examine the alignment between the reward model and human preferences. Human evaluators were asked to review the instruction and the corresponding model responses, assess their preference between the responses, and determine whether instruction refinement was necessary.

## F.1 Task/Distribution Shift Human Evaluation

To verify that intent preservation is achieved in practice, we conduct a human evaluation on task/intent drift. We quantitatively evaluate intent and task preservation between the original and refined instructions. The evaluation is conducted on 60 randomly selected instructions, with each instruction assessed by three human annotators (12 annotators in total, each evaluating 20 instructions). Annotators are shown the original and refined instructions in random order and asked to judge whether they are semantically equivalent, exhibit a minor shift, or exhibit a major shift.

• Same intent : The core task and output type remain unchanged, with only changes in expression.

• Minor shift : The core task is preserved, but some constraints or output formats are modified.

• Major shift : The task type or underlying objective is altered.

Differences in instruction length, wording, or style are not considered intent drift. The results show 66.7% Same Intent, 21.7% Minor Shift, 8.3% Major Shift, and 3.3% Tie. Overall, 88.4% of samples preserve the core task (Same Intent + Minor Shift), indicating that intent preservation is largely maintained despite occasional shifts.

We also conduct case studies of minor and major intent shifts in table 12. In major shift example, instruction asks how to cut vegetables with a corkscrew involves using a tool in a dangerous and unintended way. Following safety and answerability criteria, the refiner shifts the intent to providing safety guidance and requesting the use of appropriate tools. In minor shift example, instruction requests to generate sentences using five synonyms is unanswerable without a target word. The refiner adds a reference word (e.g., happy), introducing a minor shift while preserving the main task and input/output format. These cases illustrate that even when intent shifts occur, they can lead to improved preference data quality. The instructions given to human evaluators are provided in Figure 11.

## F.2 Reward Model Human Evaluation

To assess how well our reward-model-based pipeline aligns with human preferences, we conducted a small-scale human evaluation on 30 randomly sampled instruction–response-pair tuples. The evaluation set consisted of 15 samples selected by our pipeline as requiring refinement and 15 samples not selected for refinement. Each tuple was annotated by three human evaluators, and the final decision was determined by majority vote. (9 annotators in total, each evaluating 10 instructions)

<table><tr><td rowspan=1 colspan=1>Original instruction: How long is the needle inside themonkey&#x27;s ear?</td></tr><tr><td rowspan=1 colspan=1>Original Response 1: The needle inside the monkey&#x27;sear is 2cm long.Original Response 2: The needle is 2.5cm long and itis inserted into the monkey&#x27;s ear. [...]</td></tr><tr><td rowspan=1 colspan=1>Feedback (Evaluator): Clarity: 1/5, Specificity: 1/5,Completeness: 1/5, Safety: 3/5, Answerability: 1/5,Conciseness: 3/5, FormatConsistency: 4/5. &quot;Unclearcontext, lacks details, unanswerable without additionalinformation.1</td></tr><tr><td rowspan=1 colspan=1>Refined instruction: Provide the typical length of nee-dles used in veterinary procedures for monkeys’ ears,specifying the context and purpose of the procedure.</td></tr><tr><td rowspan=1 colspan=1>Refined Response 1: The typical length ranges from1 to 1.5 inches. These needles are usually used for earcanal biopsies or ear canal washes to diagnose [...]Refined Response 2: The typical length is 1 inch (2.5cm). This needle length is used for blood collection,injection, or fluid removal [...]</td></tr></table>

Table 13: Case study of instruction refinement.

To approximate the conditions of our actual pipeline, annotators were asked to judge whether the instruction should be refined after inspecting the two associated responses. If either response revealed an issue attributable to the instruction, annotators were instructed to mark the instruction as requiring refinement. In addition, annotators assigned scores to the two responses and selected the better one, thereby producing a human chosen/rejected preference label. The detailed annotation prompt is provided in Figure 12.

The reward-model-based refinement decision agreed with the human majority judgment in 76.7% of cases. For preference labeling, the reward-model label agreed with the human majority choice in 71.4% of cases. These results indicate that both instruction selection and preference labeling are reasonably aligned with human judgments. However, reward models can still be biased by factors such as response length, style, and surface-level fluency. Thus, the reward model in our pipeline should be viewed as a reasonably accurate proxy for human preference, rather than a perfect substitute.

## G Case Study

## G.1 Refinement Case Study

Table 13 presents a qualitative case study illustrating the effect of instruction refinement. The original instruction is ambiguous and underspecified, asking for the length of a “needle inside the monkey’s ear” without providing any context, which leads to arbitrary and inconsistent responses despite surface-level confidence. The evaluator feedback identifies deficiencies in clarity and specificity, indicating that the instruction is fundamentally underspecified. After refinement, the instruction is revised to specify a realistic veterinary context and purpose, resulting in responses that are more concrete, medically plausible, and mutually consistent. This example illustrates how improving instruction quality can yield more reliable preference data.

## G.2 Degradation Case Study

We further analyzed cases where the refined model receives lower evaluation scores than the original model. These cases suggest that larger drops can occur in tasks where correctness or output format is critical, including probability reasoning and named entity extraction. Table 14 shows a representative degradation case in named entity extraction (NER). The task requires a JSON dictionary grouped into predefined entity types. While the original model follows the requested format, the refined model adds non-requested content and includes invalid or misclassified entities. This example illustrates that instruction refinement can occasionally be less stable for tasks requiring strict output formats. This pattern is less evident in open-ended generation tasks, where score changes are more likely to reflect judge-score variance or response-style differences.

```markdown
You are an evaluator. Given an Original instruction, evaluate the instruction using the criteria
below. Follow these STRICT rules:
1. Output must start with exactly ’Evaluation:’ on its own line.
2. You must include ALL 7 criteria in the following order: Clarity, Specificity, Completeness,
Safety, Answerability, Conciseness, FormatConsistency. 3. Each line must follow the format: *
<Criterion>: <digit 1-5>/5 - <one concise note>
4. Do NOT add any text before or after the evaluation block.
Output format:
Evaluation:
* Clarity: <1-5>/5 - <one-line note>
* Specificity: <1-5>/5 - <one-line note>
* Completeness: <1-5>/5 - <one-line note>
* Safety: <1-5>/5 - <one-line note>
* Answerability: <1-5>/5 - <one-line note>
* Conciseness: <1-5>/5 - <one-line note>
* FormatConsistency: <1-5>/5 - <one-line note>
### Few-shot Examples
Original instruction:
Write something about animals
Evaluation:
* Clarity: 2/5 - vague request
* Specificity: 2/5 - no length, no scope
* Completeness: 2/5 - missing output format
* Safety: 5/5 - safe
* Answerability: 4/5 - feasible but broad
* Conciseness: 3/5 - some redundancy
* FormatConsistency: 3/5 - unspecified output format
###
Original instruction:
Make a paragraph using that language
Evaluation:
* Clarity: 2/5 - unclear what "that language" refers to
* Specificity: 2/5 - format is defined (paragraph) but content is vague
* Completeness: 2/5 - missing target language or subject
* Safety: 5/5 - safe request
* Answerability: 2/5 - partially answerable but underspecified
* Conciseness: 4/5 - concise but incomplete
* FormatConsistency: 3/5 - loosely consistent but ambiguous wording
###
Original instruction:
{instruction}
```  
Figure 9: Prompt used for Feedback Generation.

![](images/242e8914530a02de985a4806f8b8a38ad38ffbe72fa0b6e4338f784276826bc1.jpg)  
Figure 10: Prompt used for Refinement Generation.

![](images/d165604cfd488fc56f6a646f4f372660501b23253205e7ff1cdcd046a953318b.jpg)  
Figure 11: Instructions provided to human evaluators for task shift evaluation.

![](images/513ed52f4932b26bc30dde805c4280bbe05aac37e5abb01e01b7ede5cfc93eb3.jpg)  
Figure 12: Instructions provided to human evaluators for response evaluation and instruction refinement.

Question: Identify the named entities (people, organizations, locations) mentioned in the given news article. Please generate a JSON dictionary that lists the named entities in three separate groups based on their entity types. The key is the type of entity and the value is a list of strings.

Article: Yesterday, Adamson Emerson, the CEO of Faraday, and Dieter Zetsche, the CEO of Daimler AG, announced plans to build a new Gigafactory in Berlin. The facility will be a joint venture between Faraday and Daimler, producing electric vehicles and battery packs for both companies, creating thousands of job opportunities in the region. Emerson and Zetsche stated that the strategic location of Berlin, coupled with its skilled workforce and strong infrastructure, makes it an ideal choice for expansion. The new Gigafactory aims to meet the growing demand for electric vehicles in Europe and contribute to a sustainable future. Volkswagen CEO Herbert Diess welcomed the news, saying greater collaboration will benefit the auto industry’s transition to e-mobility.

<table><tr><td>Model</td><td>Model Response</td><td>Score</td></tr><tr><td>Original</td><td>&quot;Person&quot;: [&quot;Adamson Emerson&quot;, &quot;Dieter Zetsche&quot;, &quot;Herbert Diess&quot;], &quot;Organization&quot;: [&quot;Faraday&quot;, &quot;Daimler AG&quot;, &quot;Volkswagen&quot;], &quot;Location&quot;: [&quot;Berlin&quot;]</td><td>9</td></tr><tr><td>Refined</td><td>Named Entities: - Adamson Emerson (person) - Dieter Zetsche (person) - Faraday (organization)</td><td>5</td></tr></table>

Table 14: Representative degradation case in a strict-format named entity extraction task. The refined model adds non-requested content and misclassifies entities, leading to a lower evaluation score.