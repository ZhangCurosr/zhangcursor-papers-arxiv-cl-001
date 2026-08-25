# Language Chain in Alignment: Cross-Lingual Ranking Preference Optimization

Seungyoon Lee, Minhyuk Kim, Jungseob Lee, Heuiseok Lim\* Department of Computer Science and Engineering, Korea University {dltmddbs100, mhkim0929, omanma1928, limhseok}@korea.ac.kr

## Abstract

The alignment of Large Language Models heavily relies on English-centric high-quality preference data, which often leads to suboptimal performance in other languages. In this paper, we propose Cross-Lingual Ranking Preference Optimization (CRPO), a novel framework that leverages robust preference knowledge from English to facilitate preference alignment in the target language. We design a hierarchical structure within parallel preference pairs across the target language and English to jointly optimize intra- and inter-lingual preferences, thereby enhancing language adaptation and output quality. Building on the LambdaLoss framework, CRPO goes beyond the binary comparison based optimization by providing a relative ranking signal across multiple candidate responses. Our experiments across five languages with varying resource scales demonstrate that CRPO consistently outperforms standard approaches in both instruction-following and knowledge utilization capability. Notably, the robust performance gains observed across various weighting schemes further validate the empirical effectiveness of our hierarchical design in a multilingual setup. Furthermore, our findings highlight that CRPO significantly improves both reward margins and the log-probability of desirable responses, contributing to a more stable preference manifold for cross-lingual alignment.

## 1 Introduction

Integrating human feedback through the alignment process has emerged as a critical paradigm for optimizing the instruction-following capabilities of Large Language Models (LLMs) and grounding their objective functions in human value systems (Stiennon et al., 2020; Askell et al., 2021; Hong et al., 2024; Meng et al., 2024). In particular, Direct Preference Optimization (DPO) (Rafailov et al., 2023) has streamlined the complex Reinforcement Learning from

![](images/a4bdaf2366f901b2899c8ccdc8f5c668372ae272b56c6935f70ba696f145e202.jpg)  
Figure 1: Example of model responses to multilingual prompt after alignment tuning.

Human Feedback (RLHF) (Ouyang et al., 2022) pipeline, effectively guiding models toward preferred responses while improving helpfulness and utility (Kirk et al., 2023; Pang et al., 2024; Tu et al., 2025). However, these alignment gains remain largely confined to English-centric contexts, leaving consistency and response accuracy in other languages as a formidable challenge. This disparity often leads to undesirable behavior, including input-output language inconsistency and degraded performance for other language queries (Lai et al., 2023a; Puttaparthi et al., 2023; Huang et al., 2023; Wu et al., 2024a; Marchisio et al., 2024).

Existing approaches to multilingual alignment typically rely on expensive human annotation or on integrating target language data into Englishdominant corpora (Lai et al., 2023b; Zhao et al., 2024; Ranaldi et al., 2024; Lai et al., 2024). However, these approaches are limited by their inability to leverage the model’s inherent English capabilities for cross-lingual alignment. By leaving this internalized English preference knowledge decoupled from the target language, they struggle to achieve effective cross-lingual alignment.

In this paper, we propose Cross-Lingual Ranking Preference Optimization (CRPO), a novel alignment framework that aligns the preference distribution of a target language by leveraging the model’s inherent preference knowledge in English. CRPO moves beyond conventional binary comparisons within a single language; instead, we adopt an alignment strategy based on a hierarchical ranking structure that integrates two core dimensions: language consistency and response quality. As illustrated in Figure 1, while LLMs often suffer from language confusion (e.g., responding in English to a non-English prompt) and existing approaches generate low-quality outputs, CRPO guides the model toward generating high-quality responses appropriate for the target language query.

To achieve this, we transplant the “Learning-to-Rank” (LTR) (Liu et al., 2009) philosophy, which optimizes relative importance among multiple candidates, into the cross-lingual alignment training. To be specific, we design a hierarchical signal based on parallel preference pairs between English and the target language. This design facilitates alignment of target language preferences by referring to well-defined preference distributions, while inducing the generation of target language responses rather than English.

Our experiments show that CRPO achieves substantial improvements in instruction-following and knowledge-intensive tasks. Furthermore, our comparative analysis of reward and log-likelihood distributions demonstrates that CRPO leads to a more accurate likelihood preference of winning and losing responses on a held-out test set, which in turn translates to a better policy model for the target language. Our contributions are as follows:

• We propose CRPO, a novel framework that optimizes the hierarchical priority between language and quality by integrating ranking principles into cross-lingual preference alignment.

• Through an analysis of reward and probability distributions, we find that CRPO more effectively steers the preference distribution of the target language toward a desirable alignment compared to existing methodologies.

• We demonstrate that our cross-lingual preference hierarchy structure itself, which leverages the relationship between English and the target language, induces significant performance gains in target language alignment.

## 2 Related Work

Enhancing the multilingual capabilities of LLMs is a critical endeavor to ensure that global users can benefit from technological advancements without language barriers (Costa-Jussà et al., 2022; Zhao et al., 2024; Huang et al., 2024; Li et al., 2024a). Most research primarily focused on machine translating English datasets or training on heterogeneous mixtures of multilingual corpora (Workshop et al., 2022; Muennighoff et al., 2023; Lai et al., 2023b; Gao et al., 2024; Kew et al., 2024). However, these approaches often require large-scale datasets or suffer from persistent language bias (Aggarwal et al., 2024; Li et al., 2024c; Lee et al., 2025).

Recent studies seeking to mitigate these limitations have increasingly explored extending alignment algorithms proven in English to multilingual contexts (Yang et al., 2024; Dang et al., 2024a). In prior work, Wu et al. (2024b) facilitates zeroshot cross-lingual transfer in reward model training, aiming to build robust multilingual reward systems. MAPO (She et al., 2024) targets improvement in the multilingual reasoning domain using reward signals derived from external translation models. Similarly, MPO (Zhao et al., 2025) directly optimizes the reward margin gap across languages to enhance safety. While effective, these approaches rely on external dependencies, such as online machine translation pipelines, and are limited to specific domains. Another line of work operates directly on the policy model using implicit binary comparisons (Yang et al., 2025; Lee et al., 2025), yet they remain constrained by binary optimization, failing to capture complex preference distributions.

Although some works have explored list-based ranking to move beyond binary comparisons, these studies primarily focus on improving quality or safety and are strictly limited to English-centric contexts (Yuan et al., 2023; Song et al., 2024; Liu et al., 2025b). In contrast, CRPO distinguishes itself by optimizing a cross-lingual hierarchical ranking structure that jointly models language and quality. By enabling simultaneous modeling of linguistic and quality hierarchies, CRPO leads to meaningful improvements in target language capabilities.

## 3 CRPO Framework

Our approach is established on the DPO framework, extending its pairwise alignment capabilities into a ranking paradigm specifically designed for crosslingual contexts.

## 3.1 Preliminary: Direct Preference Optimization

DPO (Rafailov et al., 2023) is a stable and efficient method to align LLMs with human preferences without the need for training an explicit reward model or using complex reinforcement learning pipelines such as PPO (Schulman et al., 2017):

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D P O } } ( \pi _ { \theta } ; \pi _ { \mathrm { r e f } } ) = - \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) \sim \mathcal { D } } } \\ & { \left[ \log \sigma \left( \beta \log \frac { \pi _ { \theta } ( y _ { w } | x ) } { \pi _ { \mathrm { r e f } } ( y _ { w } | x ) } - \beta \log \frac { \pi _ { \theta } ( y _ { l } | x ) } { \pi _ { \mathrm { r e f } } ( y _ { l } | x ) } \right) \right] } \end{array}\tag{1}
$$

where $\pi _ { \theta }$ is the policy model, $\pi _ { \mathrm { r e f } }$ is the reference policy given $( x , y _ { w } , y _ { l } )$ are preference pairs which are the prompt, chosen response, and rejected response from the preference dataset. By minimizing the given loss, the model increases the margin between the preferred response $y _ { w }$ relative to $y _ { l }$

## 3.2 LambdaLoss Ranking Objective

Given that our aim is to encourage cross-lingual adaptation alongside robust preference alignment, the binary formulation of DPO is inherently limited, as it cannot accommodate multiple optimization spheres. To address this, we draw inspiration from LambdaLoss (Wang et al., 2018), a type of Learning to Rank (LTR) (Liu et al., 2009) framework that enables joint optimization of multiple items across a global ranking structure.

This approach allows the model to optimize wellfounded ranking metrics, such as Discounted Cumulative Gain (DCG) (Burges et al., 2006), by weighting pairwise transitions based on their impact on the overall list. Specifically, for a prompt x and a set of candidate responses $\mathbf { y } = \{ y 1 , \dots , y _ { n } \}$ the training objective of our ranking scheme is defined as:

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { l a m b d a } } ( \pi _ { \theta } ; \pi _ { \mathrm { r e f } } , \beta ) = } \\ { \displaystyle \mathbb { E } _ { ( x , \mathbf { y } , \psi ) \sim \mathcal { D } } \left[ \sum _ { \psi _ { i } > \psi _ { j } } \Delta _ { i , j } \log ( 1 + e ^ { - ( s _ { i } - s _ { j } ) } ) \right] } \end{array}\tag{2}
$$

where $\begin{array} { r } { s _ { i } = \beta \log { \frac { \pi _ { \theta } \left( y _ { i } | x \right) } { \pi _ { \mathrm { r e f } } \left( y _ { i } | x \right) } } } \end{array}$ represents the implicit reward score of response $y _ { i }$ . The term $\Delta _ { i , j }$ is the Lambda weight, which determines the importance of the pair $( y _ { i } , y _ { j } )$ based on their positions in the ranking:

$$
\Delta _ { i , j } = | G _ { i } - G _ { j } | \cdot \left| \frac { 1 } { D ( \tau ( i ) ) } - \frac { 1 } { D ( \tau ( j ) ) } \right|\tag{3}
$$

In this formulation, $G$ is the gain function (typically $G _ { i } = 2 ^ { \psi _ { i } } - 1$ where $\psi _ { i }$ is the relevance label), and D is the rank discount function (typically $D ( \tau ( i ) ) = \log ( 1 + \tau ( i ) ) )$ . Here, $\tau ( i )$ denotes the rank position of $y _ { i }$ induced by the score $s .$

This objective can optimize the DCG metric by prioritizing the placement of high-relevance items at the top of the list by applying a logarithmic discount to lower ranks. Consequently, any rank inversion that displaces a desirable response from its superior position results in a significant reduction of the overall score, which allows the $\Delta _ { i , j }$ to impose a stringent penalty on misalignment that disrupts the intended preference order.

## 3.3 CRPO: Cross-Lingual Ranking Preference Optimization

Building upon the LambdaLoss framework, CRPO aims to induce both cross-lingual and intra-lingual alignment by jointly optimizing target language responses alongside semantically equivalent preference knowledge from English. For each training instance, we construct a quadruple of responses consisting of: $y _ { w } ^ { t } , y _ { l } ^ { t } , y _ { w } ^ { e }$ , and $y _ { l } ^ { e }$ , which are chosen and rejected responses in the target language and English, respectively. To ensure a consistent preference manifold is maintained across languages, we assign hierarchical relevance labels $( \psi )$ based on both response quality and language type among these four responses:

$$
\psi ( y _ { w } ^ { t } ) > \psi ( y _ { w } ^ { e } ) > \psi ( y _ { l } ^ { t } ) > \psi ( y _ { l } ^ { e } )\tag{4}
$$

Under this hierarchy, the model is trained to prioritize the preferred response in the target language, $y _ { w } ^ { t }$ , as the highest-ranked candidate. At the same time, CRPO provides comprehensive supervisory signals by aligning all pairs within the $\{ y ^ { t } , y ^ { e } \}$ set for a given target language prompt $x .$ . The model is compelled to jointly optimize both intra-lingual preferences $( ( y _ { w } ^ { t } , y _ { l } ^ { t } ) , ( y _ { w } ^ { e } , y _ { l } ^ { e } ) )$ and cross-lingual preference pairs, such as $( y _ { w } ^ { t } , y _ { l } ^ { e } )$ and $( y _ { w } ^ { e } , y _ { l } ^ { t } )$

Since the optimization of target language involves concurrent alignment with semantically equivalent English pairs, the model can leverage these English candidates as informative cues. Furthermore, prioritizing $y _ { w } ^ { t }$ provides training signals for maintaining input-output language consistency. Thus, the gain function with ψ in CRPO enables the model to inherit the logical consistency of English preferences while imposing a stringent quality constraint on responses within the target language. The gain functions for each response in CRPO are detailed in Appendix A.

Finally, we employ a negative loglikelihood (NLL) loss term for $y _ { w } ^ { t }$ in our loss to improve alignment performance during the instruction tuning process. The resulting loss function of CRPO is defined as:

$$
\mathcal { L } _ { \mathrm { C R P O } } = ( 1 - \alpha ) \cdot \mathcal { L } _ { \mathrm { N L L } } + \alpha \cdot \mathcal { L } _ { \mathrm { l a m b d a } } ( \pi _ { \theta } ; \pi _ { \mathrm { r e f } } , \beta )\tag{5}
$$

where α is a hyperparameter that balances the ranking-based alignment and the language modeling objective. While conventional DPO is restricted to binary comparisons of responses within the target language, CRPO promotes the precise mapping of hierarchies defined by both language and quality onto a unified optimization plane.

## 3.4 Weighting Schemes for CRPO

The flexibility of the LambdaLoss framework allows CRPO to incorporate various weighting functions for each optimizing different bounds of the ranking metric. We explore three schemes for $\Delta _ { i , j }$ derived from LambdaLoss to investigate the effectiveness in cross-lingual alignment.

LambdaRank optimizes a coarse upper bound of the nDCG metric. It defines the weight based on the absolute difference between the inverse discounts of the two positions, which is presented in Equation 3. This scheme provides a foundational gradient signal for ranking by considering the global positions of the responses in the list.

nDCG2 introduces a tighter bound for nDCG optimization. Unlike the global position-based weighting of LambdaRank, it focuses on the distance between the two items being compared:

$$
\begin{array} { l } { { \Delta _ { i , j } ^ { \mathrm { n D C G 2 } } = } } \\ { { \displaystyle | G _ { i } - G _ { j } | \cdot \left| \frac 1 { D ( | \tau ( i ) - \tau ( j ) | ) } - \frac 1 { D ( | \tau ( i ) - \tau ( j ) | + 1 ) } \right| } } \end{array}\tag{6}
$$

where $| \tau ( i ) - \tau ( j ) |$ represents the rank distance between the two responses. This scheme has been empirically shown to achieve superior performance

by providing more localized and precise gradients for rank inversion.

nDCG2++ is a hybrid scheme that combines the strengths of both nDCG2 and LambdaRank to maximize ranking performance:

$$
\Delta _ { i , j } ^ { \mathrm { n D C G 2 + + } } = \mu \cdot \Delta _ { i , j } ^ { \mathrm { n D C G 2 } } + \Delta _ { i , j }\tag{7}
$$

where $\mu$ is a hyperparameter that controls the contribution of the nDCG2 component. In our experiments, $\mu$ is set to 10 following the prior study (Wang et al., 2018).

While nDCG2++ is known to reach the strongest performance in traditional Information Retrieval tasks, we found that nDCG2 or LambdaRank alone can be sufficient for aligning LLM preference distributions across languages. To demonstrate the generality and theoretical robustness of the CRPO framework, we employ nDCG2 as our primary weighting scheme for the experiments.

## 4 Experimental Setup

## 4.1 Models and Training

Models We perform preference optimization with three different models, Llama-2-7B (Touvron et al., 2023), Llama-3-8B (Dubey et al., 2024), and Mistral-7B-v0.1 (Jiang et al., 2023). We employ five languages with varying resource availability: Chinese (zh), Indonesian (id), Korean (ko), Swahili (sw), and Bengali (bn). They were chosen to provide a mix of high-, medium-, and lowresource languages, typological and script diversity, while satisfying the practical constraints of available evaluation datasets.

Dataset To construct the preference dataset for each language, we sample 3,000 instances from the UltraFeedback dataset (Cui et al., 2024) and translate them into the respective target languages using gpt-5-chat. This procedure yields fully parallel target language pairs $( x _ { t } , y _ { w } ^ { t } , y _ { l } ^ { t } )$ that correspond directly to the original English preference pairs $( x _ { e } , y _ { w } ^ { e } , y _ { l } ^ { e } )$ . We randomly split the resulting dataset, including English pairs, into a 90% training set and a 10% test set.

Baseline We compare CRPO against two competitive baselines that aim to align cross-lingual preferences via monolithic strategy. We establish SFT+DPO as our strong baseline. In this setup, the model is optimized using response pairs in the language of the input. Since the training dataset encompasses both English and the target language, the model is trained on $( y _ { w } ^ { t } , y _ { l } ^ { t } )$ for $x ^ { t }$ and $( y _ { w } ^ { e } , y _ { l } ^ { e } )$ for $x ^ { e }$ , in the same batch.

As another method, CLO (Lee et al., 2025) is a variant of DPO that incorporates binary comparison between English and target language response pairs within the same batch. To be specific, CLO explicitly steers the model toward generating desirable and language-consistent responses by designating the target language response as chosen and the English response as rejected. Furthermore, it mitigates inherent English bias and facilitates alignment by restricting the NLL loss exclusively to target language responses.

Note that all baselines and models use identical hyperparameters and the same amount of parallel datasets to ensure a fair comparison. Further details about the training can be found in Appendix B.

## 4.2 Evaluation

AlpacaEval We employ AlpacaEval (Li et al., 2023) to evaluate models’ conversational ability across a diverse set of queries. AlpacaEval comprises 805 questions from 5 datasets and evaluates model performance through qualitative comparisons of reference responses and candidate model outputs. To support evaluation across multiple languages, we adopt the multilingual version of AlpacaEval established in prior work (Lee et al., 2025) as our primary evaluation suite. We report the win rate (WR) and length-controlled win rate (LC) against the SFT model. The LC metric is specifically designed to be robust against model verbosity (Dubois et al., 2024). Also, by including the instruction in the evaluation prompt, the model is penalized for responding in a language other than the target language. We use gpt-5-chat as a judge for assessment and also report multilingual Arena-Hard results in Appendix C.

Knowledge & Understanding We further evaluate the models’ capacity for knowledge utilization and linguistic comprehension. We use MMMLU (Multilingual Massive Multitask Language Understanding) (Hendrycks et al., 2021) to verify whether the trained model can leverage inherent world knowledge in target language contexts. MMMLU is a human-translated multilingual extension of the original MMLU benchmark, encompassing 57 subjects across STEM, humanities, and social science <sup>1</sup>. In addition, we employ Belebele (Bandarkar et al., 2024), a benchmark for multilingual reading comprehension. In this task, we can evaluate literacy and contextual understanding, as the model should select the most appropriate answer from four choices based on the short passage and question. Following standard practice, we report accuracy by treating the option with the highest overall log-likelihood as the model’s predicted answer (Gao et al., 2021).

In-depth Analysis To move beyond aggregate benchmark scores and gain deeper insights into the model’s internal mechanics, we conduct an intrinsic analysis. We investigate the reward distributions and log probabilities of the preference pairs in our test set to verify whether each method enables balanced alignment that reinforces desirable generative behaviors, rather than relying solely on suppressing suboptimal outputs.

Furthermore, we measure the quality distribution of generated responses using an external reward model. The objective of this analysis is to empirically demonstrate that the internal optimization of CRPO translates into measurable improvements in output quality, ensuring that the model’s preference manifold is robustly aligned with objective judgments in the target language.

## 5 Results and Analysis

In this section, we present the main results of our experiments, highlighting CRPO’s superior performance across various benchmarks in Section 5.1. Then in Section 5.2, we provide an in-depth understanding of the internal reward dynamics and generative behaviors that distinguish CRPO from other alignment methods. In addition, we employ an external reward model to objectively quantify improvements in absolute generation quality in Section 5.3. Finally, we conduct a study on various ranking schemes to investigate how different configurations of the hierarchical preference structure influence the effectiveness of cross-lingual alignment in Section 5.4.

## 5.1 Main Results

Overall Performance As shown in Table 1, all preference optimization methods provide improvements over the SFT model in most cases, with CRPO achieving the best results across languages. SFT+DPO, a standard approach for model alignment, also serves as a robust baseline in other languages. CLO exhibits comparable or slightly below performance to SFT+DPO, demonstrating the utility of a straightforward objective that suppresses the probability of English responses. Furthermore, we include the English performance of each case in Appendix D in terms of alignment tax.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">Chinese</td><td colspan="2">Indonesian</td><td colspan="2">Korean</td><td colspan="2">Swahili</td><td colspan="2">Bengali</td></tr><tr><td>LC (%)</td><td>WR (%)</td><td>LC (%)</td><td>WR (%)</td><td>LC (%)</td><td>WR (%)</td><td>LC (%)</td><td>WR (%)</td><td>LC (%)</td><td>WR (%)</td></tr><tr><td rowspan="3">Llama-2-7B</td><td>SFT+DPO</td><td>57.80</td><td>58.50</td><td>57.92</td><td>58.88</td><td>54.28</td><td>55.40</td><td>38.63</td><td>38.07</td><td>34.90</td><td>35.68</td></tr><tr><td>CLO</td><td>56.41</td><td>56.08</td><td>53.84</td><td>53.91</td><td>51.15</td><td>50.24</td><td>44.49</td><td>43.97</td><td>33.15</td><td>30.93</td></tr><tr><td>CRPO</td><td>64.18</td><td>64.34</td><td>63.83</td><td>64.47</td><td>63.02</td><td>63.97</td><td>44.07</td><td>43.72</td><td>37.52</td><td>36.77</td></tr><tr><td rowspan="3">Llama-3-8B</td><td>SFT+DPO</td><td>60.24</td><td>60.37</td><td>59.24</td><td>60.37</td><td>64.97</td><td>65.96</td><td>46.23</td><td>46.95</td><td>34.77</td><td>36.89</td></tr><tr><td>CLO</td><td>58.70</td><td>58.13</td><td>52.42</td><td>52.96</td><td>53.94</td><td>53.41</td><td>43.78</td><td>44.93</td><td>35.67</td><td>33.29</td></tr><tr><td>CRPO</td><td>62.45</td><td>62.36</td><td>67.97</td><td>68.40</td><td>68.45</td><td>68.81</td><td>61.84</td><td>62.17</td><td>53.86</td><td>55.65</td></tr><tr><td rowspan="3">Mistral-7B</td><td>SFT+DPO</td><td>55.65</td><td>56.14</td><td>55.98</td><td>58.50</td><td>51.49</td><td>52.85</td><td>27.95</td><td>27.33</td><td>24.63</td><td>24.59</td></tr><tr><td>CLO</td><td>55.18</td><td>55.03</td><td>51.29</td><td>51.98</td><td>48.14</td><td>48.32</td><td>35.15</td><td>35.40</td><td>19.98</td><td>18.01</td></tr><tr><td>CRPO</td><td>62.83</td><td>63.10</td><td>66.81</td><td>68.19</td><td>62.90</td><td>63.72</td><td>50.76</td><td>51.98</td><td>48.69</td><td>48.57</td></tr></table>

Table 1: AlpacaEval across five target languages. LC and WR denote length-controlled and raw win rate, respectively. The best results for each model are highlighted in bold.

<table><tr><td>Model</td><td>Lang</td><td>SFT+DPO</td><td>CLO</td><td>CRPO</td></tr><tr><td></td><td></td><td>MMMLU</td><td></td><td></td></tr><tr><td rowspan="3">Llama-2-7B</td><td>id</td><td>26.33</td><td>24.69</td><td>27.85</td></tr><tr><td>ko</td><td>26.20</td><td>28.57</td><td>31.07</td></tr><tr><td>SW</td><td>25.08</td><td>25.62</td><td>25.12</td></tr><tr><td rowspan="3">Llama-3-8B</td><td>id</td><td>41.69</td><td>36.41</td><td>45.94</td></tr><tr><td>ko</td><td>39.58</td><td>40.19</td><td>39.84</td></tr><tr><td>SW</td><td>36.02</td><td>27.48</td><td>37.03</td></tr><tr><td rowspan="3">Mistral-7B</td><td>id</td><td>37.36</td><td>31.38</td><td>37.47</td></tr><tr><td>ko</td><td>30.39</td><td>29.27</td><td>35.01</td></tr><tr><td>SW</td><td>30.57</td><td>28.32</td><td>31.78</td></tr><tr><td></td><td></td><td>Belebele</td><td></td><td></td></tr><tr><td rowspan="3">Llama-2-7B</td><td>id</td><td>33.11</td><td>36.55</td><td>36.66</td></tr><tr><td>ko</td><td>33.00</td><td>35.66</td><td>36.33</td></tr><tr><td>SW</td><td>29.22</td><td>28.22</td><td>29.22</td></tr><tr><td rowspan="3">Llama-3-8B</td><td>id</td><td>64.22</td><td>35.11</td><td>65.88</td></tr><tr><td>ko</td><td>57.55</td><td>59.44</td><td>68.66</td></tr><tr><td>SW</td><td>47.55</td><td>28.66</td><td>49.22</td></tr><tr><td rowspan="3">Mistral-7B</td><td>id</td><td>48.33</td><td>30.44</td><td>47.44</td></tr><tr><td>ko</td><td>49.66</td><td>35.77</td><td>51.77</td></tr><tr><td>SW</td><td>33.00</td><td>33.88</td><td>34.77</td></tr></table>

Table 2: MMMLU and Belebele evaluation results. We report the performance of zero-shot in MMMLU and one-shot in Belebele.

Robustness in Low-Resource Languages While most methods outperform SFT in highand mid-resource languages, a different tendency emerges in low-resource settings. Unlike CRPO, other methods exhibit notable performance degradation in low-resource scenarios. This trend is particularly pronounced for Llama-3-8B, which suffers from a collapse compared to the SFT, as evidenced by notably low WR in Bengali.

This stems from optimization instability and excessive parameter volatility during the preference alignment when the model’s linguistic grounding in the target language is insufficient (Kadyrbek et al., 2025; Lee et al., 2025). From this perspective, SFT+DPO, relying on isolated preference pairs in the target language, fails to establish a stable preference manifold in low-resource scenarios. Similarly, the performance drop in CLO shows that simply enforcing language-consistent responses through binary cross-lingual hierarchies is insufficient for robust alignment.

While CRPO also encounters adaptation challenges due to the inherent limitations of models, it consistently surpasses baselines and demonstrates remarkable robustness in most cases. Notably, CRPO achieves a significant performance gain in Swahili with Llama-3, scoring a WR of 62.17. This gain can be attributed to the use of relatively well-aligned English pairs as a logical anchor for aligning the target language. By realigning language quality and hierarchy, CRPO exhibits the most robust resistance to this alignment collapse and contributes to overall alignment stabilization.

Knowledge Utilization The validity of CRPO extends beyond conversational improvements, as evidenced in the knowledge utilization and linguistic comprehension areas. As shown in Table 2, CRPO achieves competitive performance across both benchmarks. Notably, the MMMLU score in Indonesian for Llama-3 exceeds the strongest baseline by over 4 points, while the Korean Belebele score reaches 68.66. These results demonstrate that the ranking structure in CRPO successfully anchors the model’s inherent knowledge system within the target language space, thereby achieving better alignment for complex reasoning tasks.

![](images/c5341054cded643f35f3f9cb2bdd31eef3eb69f150c07200cca913148ec68276.jpg)

![](images/f55c9aab76e58ca00e3ffafd76933e9aba167685059dc868c7cf4ef2602a6a31.jpg)

![](images/45db035ef60b66d315265ffcc9cd327ee40145ad2c24ed3ec1fc0a8a9433ebd9.jpg)  
(a) Reward Difference

![](images/999ad86df778a89116b54a1106e736c7acac1a0e74d32703e23a9ff9120f856d.jpg)

![](images/f1f11eaef71fd84567002435c026903d35dbe096f8436c2ebfe92e84a92ecd1b.jpg)

![](images/3451d6bde255599139e1654653aa624b6b58aa2a5ede3832f7331a1b6106ed1f.jpg)

![](images/cad24fca1885199b95659642b2c0a4fa111ea2c359965da3f7be9e3200f1014b.jpg)  
(b) Log Probability Distribution

![](images/503580b256d26b67011e005493db461f51a08337ad504835f620ac7588cc8388.jpg)

Figure 2: (a) Reward difference distribution and (b) Log-likelihood distribution on chosen responses $( y _ { w } )$ of Llama-3-8B trained with SFT (blue), CLO (orange), SFT+DPO (blue) and CRPO (red) across languages.
<table><tr><td>Model</td><td>Method</td><td>zh</td><td>id</td><td>k0</td><td>SW</td></tr><tr><td rowspan="4">Llama-2-7B</td><td>SFT</td><td>-1.749</td><td>-1.797</td><td>-2.870</td><td>-2.901</td></tr><tr><td>SFT+DPO</td><td>-1.227</td><td>-0.904</td><td>-2.121</td><td>-2.958</td></tr><tr><td>CLO</td><td>-1.539</td><td>-1.200</td><td>-2.855</td><td>-2.457</td></tr><tr><td>CRPO</td><td>-0.607</td><td>-0.936</td><td>-1.601</td><td>-2.175</td></tr><tr><td rowspan="3">Llama-3-8B</td><td>SFT SFT+DPO</td><td>0.344 1.506</td><td>0.208</td><td>-0.904</td><td>-0.487</td></tr><tr><td>CLO</td><td></td><td>1.585</td><td>0.869</td><td>-0.212</td></tr><tr><td>CRPO</td><td>0.613 1.738</td><td>1.034 1.883</td><td>-0.119 0.911</td><td>-0.531 1.122</td></tr><tr><td rowspan="4">Mistral-7B</td><td>SFT</td><td>-0.706</td><td>-0.808</td><td>-1.800</td><td>-0.667</td></tr><tr><td>SFT+DPO</td><td>0.255</td><td>-0.037</td><td>-0.719</td><td>-1.519</td></tr><tr><td>CLO</td><td>-0.096</td><td>-0.665</td><td>-1.237</td><td>-0.981</td></tr><tr><td>CRPO</td><td>0.508</td><td>0.805</td><td>-0.344</td><td>-0.040</td></tr></table>

Table 3: Average of reward scores from Skywork-Reward-V2-Qwen3-8B on generation outputs. The best results achieved across the methods are highlighted in bold, while the second-best results are underlined.

## 5.2 Reward and Probability Shifts

To investigate the impact of CRPO on the model’s preference manifold, we examine the internal dynamics of reward differences and generative logprobabilities. As illustrated in Figure 2, the magnitude of distribution shifts differs across methods regarding the SFT model as a default.

In Figure 2 (a), we find that regardless of the language, CRPO consistently achieves a large reward difference and shifts the distribution in a positive direction compared to the SFT. Likewise, the SFT+DPO approach exhibits a similar trend, though the magnitude of the shift is relatively smaller. In contrast, CLO yields almost identical reward difference to SFT, or in the case of Indonesian, even shows signs of degradation.

One of the primary risks in preference optimization is the improper focus on maximizing the reward margin, which often results in suppressing the probability of rejected responses rather than increasing the probability of chosen responses. This imbalance can undermine the language model’s inherent stability and fluency by failing to provide an explicit guide for target responses, potentially leading to degeneration (Meng et al., 2024; Hong et al., 2024). As shown in the comparison of Figure 2 (a) and (b), while other methods increase the reward margin, their log-likelihood distributions for chosen responses remain largely congruent with the SFT. This indicates that other approaches primarily fail to incentivize the generation of favored responses.

On the other hand, CRPO assigns higher values to both the log-likelihood of the chosen response and the reward difference across all languages than in other cases. This demonstrates that CRPO achieves better alignment in the target language by increasing the likelihood of the favored response. Further analysis of the reward distribution is provided in Appendix E.

![](images/d69f83f11b6eb9eb060d7735886f3875b02bdca5e370d450f160d0ec1a3b3fec.jpg)  
Figure 3: Win rate comparison of different weighting schemes within the CRPO framework.

## 5.3 External Quality Assessment

While win rates provide a measure of relative preference, they may not fully capture the absolute quality shift in the model’s generative distribution. To provide a more objective assessment of response quality, we use Skywork-Reward-V2- Qwen3-8B (Liu et al., 2025a), an external reward model known to support multilingual environments, to score the outputs of each method on the test set. Table 3 summarizes the average reward scores across diverse models and languages.

The results demonstrate that CRPO achieves superior reward magnitudes across nearly all configurations. A notable score escalation is observed in the Mistral-7B; in Indonesian (id), while the SFT+DPO reaches a score of -0.037, CRPO elevates the quality to a considerable positive margin of 0.805, outperforming SFT+DPO by a wide gap. This also substantiates our claim that anchoring target language alignment to established English preference structures can guide the model toward higher-quality regions of the generation manifold.

## 5.4 Study on Ranking Scheme

We investigate how the choice of weighting scheme in Equation 2 influences cross-lingual preference alignment. Figure 3 shows AlpacaEval results obtained by diverse weighting schemes within the same CRPO framework.

As shown in Figure 3, all three schemes consistently outperform both SFT+DPO and the uniform scheme, confirming that hierarchical ranking signals provide more informative supervision than pairwise comparisons. We can attribute the success across diverse configurations to CRPO’s hierarchical structural design. As evidence for this, assigning a uniform weight to all response pairs results in a notable performance drop. The degradation mainly arises from the lack of alignment between the languages. Removing hierarchical cues triggers an English bias, leading to a failure in linguistic consistency with the given prompt.

Moreover, LambdaRank and nDCG2 achieve strong performance across most settings. This suggests that global position-based signals and local distance-based signals can complement each other depending on the model and language. Given that our experiments are conducted under the nDCG2 scheme, the additional gains observed with alternative schemes further indicate that the potential performance ceiling of CRPO may be even higher.

## 6 Conclusion

In this paper, we propose CRPO, a new alignment framework designed to establish robust preference manifolds both within and across languages. By reformulating the target language adaptation and alignment task as a hierarchical ranking problem, CRPO benefits from the transfer of preference knowledge along the language chain. Our empirical results demonstrate that CRPO not only achieves consistent superiority over standard pairwise approaches but also enhances the generative likelihood of preferred responses, facilitating the generation of high-fidelity, precise responses in the target language. We believe that our approach shifts the paradigm of cross-lingual alignment from isolated binary comparisons to an integrated cross-lingual ranking perspective. Moving forward, we will investigate a dynamic gain adaptation strategy based on linguistic similarities and training stages. By mitigating the optimization complexity inherent in expansive ranking spaces, we expect to ensure robust cross-lingual alignment across an even wider array of language pairs.

## Limitations

Although we use the same amount of the dataset during the training, our proposed framework employs a hierarchical ranking structure that processes four response candidates in a single step. This inherently demands more computational resources during training. However, this overhead can be considered an unavoidable cost inherent to methods that leverage item-wise ranking, not only limited to the preference optimization field. Given the limitations of standard binary comparisons in optimizing both quality and consistency simultaneously, the performance gains achieved by CRPO demonstrate that this additional cost is a worthwhile trade-off for effective cross-lingual alignment.

Furthermore, our main experiments include evaluation results from the multilingual version of the AlpacaEval dataset, translated using gpt-4o in prior work (Lee et al., 2025), and the MMMLU dataset (Hendrycks et al., 2021), which was translated into the respective languages by professional human translators. Additionally, we employed the Belebele dataset (Bandarkar et al., 2024), which was constructed and reviewed by human experts during translation. While these datasets are suitable for measuring general model performance, they may not fully capture the model’s ability to respond appropriately to queries involving languagespecific characteristics or cultural contexts. Although this limitation may prevent a comprehensive evaluation of the model’s grasp of cultural nuances and localized content, we made every effort to utilize the highest-quality benchmarks available.

## Ethics Statement

In this research, we utilized only publicly available resources. All training and evaluation data were obtained from publicly authorized open-access repositories. Nevertheless, we recognize the potential for residual biases or harmful concepts in the English source data to be propagated across languages. To mitigate this, we carefully adhered to the copyright guidelines and terms of use for all original works, datasets, and language resources, including translated materials. We confirm that the collection, processing, and application of these resources raise no distinct ethical concerns.

## Acknowledgments

This research was supported by Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Education (NRF-2021R1A6A1A03045425). This work was supported by Institute for Information & communications Technology Promotion (IITP) grant funded by the Korea government (MSIT) (RS-2024-00398115, Research on the reliability and coherence of outcomes produced by Generative AI). This research was supported by Culture, Sports and Tourism R&D Program through the Korea Creative Content Agency grant funded by the Ministry of Culture, Sports and Tourism in 2026 (Project Name: Development of an AI Agent Integrating Korean Language Knowledge for Personalized Language Consultation Services, Project Number: RS-2026-25506607, Contribution Rate: 25%). This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) under the Leading Generative AI Human Resources Development (IITP-2026-RS-2026-25545512) grant funded by the Korea government (MSIT).

## References

Divyanshu Aggarwal, Ashutosh Sathe, Ishaan Watts, and Sunayana Sitaram. 2024. MAPLE: Multilingual evaluation of parameter efficient finetuning of large language models. In Findings of the Association for Computational Linguistics ACL 2024, pages 14824– 14867, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al. 2021. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861.

Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa. 2024. The belebele benchmark: a parallel reading comprehension dataset in 122 language variants. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 749–775, Bangkok, Thailand. Association for Computational Linguistics.

Christopher Burges, Robert Ragno, and Quoc Le. 2006. Learning to rank with nonsmooth cost functions. Advances in neural information processing systems, 19.

Marta R Costa-Jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe

Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. 2022. No language left behind: Scaling human-centered machine translation. arXiv preprint arXiv:2207.04672.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, et al. 2024. Ultrafeedback: Boosting language models with scaled ai feedback. In International Conference on Machine Learning, pages 9722–9744. PMLR.

John Dang, Arash Ahmadian, Kelly Marchisio, Julia Kreutzer, Ahmet Üstün, and Sara Hooker. 2024a. RLHF can speak many languages: Unlocking multilingual preference optimization for LLMs. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 13134– 13156, Miami, Florida, USA. Association for Computational Linguistics.

John Dang, Shivalika Singh, Daniel D’souza, Arash Ahmadian, Alejandro Salamanca, Madeline Smith, Aidan Peppin, Sungjin Hong, Manoj Govindassamy, Terrence Zhao, Sandra Kublik, Meor Amer, Viraat Aryabumi, Jon Ander Campos, Yi-Chern Tan, Tom Kocmi, Florian Strub, Nathan Grinsztajn, Yannis Flet-Berliac, Acyr Locatelli, Hangyu Lin, Dwarak Talupuru, Bharat Venkitesh, David Cairuz, Bowen Yang, Tim Chung, Wei-Yin Ko, Sylvie Shang Shi, Amir Shukayev, Sammie Bae, Aleksandra Piktus, Roman Castagné, Felipe Cruz-Salinas, Eddie Kim, Lucas Crawhall-Stein, Adrien Morisot, Sudip Roy, Phil Blunsom, Ivan Zhang, Aidan Gomez, Nick Frosst, Marzieh Fadaee, Beyza Ermis, Ahmet Üstün, and Sara Hooker. 2024b. Aya expanse: Combining research breakthroughs for a new multilingual frontier. Preprint, arXiv:2412.04261.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. 2024. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475.

Changjiang Gao, Hongda Hu, Peng Hu, Jiajun Chen, Jixing Li, and Shujian Huang. 2024. Multilingual pretraining and instruction tuning improve cross-lingual knowledge alignment, but only shallowly. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6101–6117, Mexico City, Mexico. Association for Computational Linguistics.

Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, et al. 2021. A framework for few-shot language model evaluation. Zenodo.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. Preprint, arXiv:2009.03300.

Jiwoo Hong, Noah Lee, and James Thorne. 2024. Orpo: Monolithic preference optimization without reference model. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 11170–11189.

Haoyang Huang, Tianyi Tang, Dongdong Zhang, Xin Zhao, Ting Song, Yan Xia, and Furu Wei. 2023. Not all languages are created equal in LLMs: Improving multilingual capability by cross-lingual-thought prompting. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 12365– 12394, Singapore. Association for Computational Linguistics.

Kaiyu Huang, Fengran Mo, Hongliang Li, You Li, Yuanchi Zhang, Weijian Yi, Yulong Mao, Jinchen Liu, Yuzhuang Xu, Jinan Xu, et al. 2024. A survey on large language models with multilingualism: Recent advances and new frontiers. arXiv preprint arXiv:2405.10936.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Nurgali Kadyrbek, Zhanseit Tuimebayev, Madina Mansurova, and Vítor Viegas. 2025. The development of small-scale language models for lowresource languages, with a focus on kazakh and direct preference optimization. Big Data and Cognitive Computing, 9(5):137.

Tannon Kew, Florian Schottmann, and Rico Sennrich. 2024. Turning English-centric LLMs into polyglots: How much multilinguality is needed? In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 13097–13124, Miami, Florida, USA. Association for Computational Linguistics.

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. 2023. Understanding the effects of rlhf on llm generalisation and diversity. arXiv preprint arXiv:2310.06452.

Viet Lai, Nghia Ngo, Amir Pouran Ben Veyseh, Hieu Man, Franck Dernoncourt, Trung Bui, and Thien Nguyen. 2023a. ChatGPT beyond English: Towards a comprehensive evaluation of large language models in multilingual learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 13171–13189, Singapore. Association for Computational Linguistics.

Viet Lai, Chien Nguyen, Nghia Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan Rossi, and Thien Nguyen. 2023b. Okapi: Instruction-tuned large language models in multiple languages with reinforcement learning from human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 318–327.

Wen Lai, Mohsen Mesgar, and Alexander Fraser. 2024. LLMs beyond English: Scaling the multilingual capability of LLMs with cross-lingual feedback. In Findings of the Association for Computational Linguistics: ACL 2024, pages 8186–8213, Bangkok, Thailand. Association for Computational Linguistics.

Jungseob Lee, Seongtae Hong, Hyeonseok Moon, and Heuiseok Lim. 2025. Cross-lingual optimization for language transfer in large language models. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15100–15119, Vienna, Austria. Association for Computational Linguistics.

Chong Li, Shaonan Wang, Jiajun Zhang, and Chengqing Zong. 2024a. Improving in-context learning of multilingual generative language models with crosslingual alignment. In Proceedings of the 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8058–8076.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. 2024b. From crowdsourced data to highquality benchmarks: Arena-hard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca\_eval.

Zihao Li, Yucheng Shi, Zirui Liu, Fan Yang, Ninghao Liu, and Mengnan Du. 2024c. Quantifying multilingual performance of large language models across languages. arXiv preprint arXiv:2404.11553.

Chris Yuhao Liu, Liang Zeng, Yuzhen Xiao, Jujie He, Jiacai Liu, Chaojie Wang, Rui Yan, Wei Shen, Fuxiang Zhang, Jiacheng Xu, et al. 2025a. Skywork-rewardv2: Scaling preference data curation via human-ai synergy. arXiv preprint arXiv:2507.01352.

Tianqi Liu, Zhen Qin, Junru Wu, Jiaming Shen, Misha Khalman, Rishabh Joshi, Yao Zhao, Mohammad Saleh, Simon Baumgartner, Jialu Liu, Peter J Liu, and Xuanhui Wang. 2025b. LiPO: Listwise preference optimization through learning-to-rank. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2404–2420, Al-

buquerque, New Mexico. Association for Computational Linguistics.

Tie-Yan Liu et al. 2009. Learning to rank for information retrieval. Foundations and Trends® in Information Retrieval, 3(3):225–331.

Kelly Marchisio, Wei-Yin Ko, Alexandre Berard, Théo Dehaze, and Sebastian Ruder. 2024. Understanding and mitigating language confusion in LLMs. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 6653– 6677, Miami, Florida, USA. Association for Computational Linguistics.

Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. Simpo: Simple preference optimization with a reference-free reward. Advances in Neural Information Processing Systems, 37:124198–124235.

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. 2023. Crosslingual generalization through multitask finetuning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15991–16111, Toronto, Canada. Association for Computational Linguistics.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Richard Yuanzhe Pang, Weizhe Yuan, He He, Kyunghyun Cho, Sainbayar Sukhbaatar, and Jason Weston. 2024. Iterative reasoning preference optimization. Advances in Neural Information Processing Systems, 37:116617–116637.

Poorna Chander Reddy Puttaparthi, Soham Sanjay Deo, Hakan Gul, Yiming Tang, Weiyi Shang, and Zhe Yu. 2023. Comprehensive evaluation of chatgpt reliability through multilingual inquiries. arXiv preprint arXiv:2312.10524.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741.

Leonardo Ranaldi, Giulia Pucci, and Andre Freitas. 2024. Empowering cross-lingual abilities of instruction-tuned large language models by translation-following demonstrations. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 7961–7973, Bangkok, Thailand. Association for Computational Linguistics.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Shuaijie She, Wei Zou, Shujian Huang, Wenhao Zhu, Xiang Liu, Xiang Geng, and Jiajun Chen. 2024. MAPO: Advancing multilingual reasoning through multilingual-alignment-as-preference optimization. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 10015–10027, Bangkok, Thailand. Association for Computational Linguistics.

Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang. 2024. Preference ranking optimization for human alignment. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 18990–18998.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008– 3021.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Songjun Tu, Jiahao Lin, Xiangyu Tian, Qichao Zhang, Linjing Li, Yuqian Fu, Nan Xu, Wei He, Xiangyuan Lan, Dongmei Jiang, et al. 2025. Enhancing llm reasoning with iterative dpo: A comprehensive empirical investigation. arXiv preprint arXiv:2503.12854.

Xuanhui Wang, Cheng Li, Nadav Golbandi, Michael Bendersky, and Marc Najork. 2018. The lambdaloss framework for ranking metric optimization. In Proceedings ofthe 27th ACM international conference on information and knowledge management, pages 1313–1322.

BigScience Workshop, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel´ Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, et al. 2022. Bloom: A 176bparameter open-access multilingual language model. arXiv preprint arXiv:2211.05100.

Suhang Wu, Jialong Tang, Baosong Yang, Ante Wang, Kaidi Jia, Jiawei Yu, Junfeng Yao, and Jinsong Su. 2024a. Not all languages are equal: Insights into multilingual retrieval-augmented generation. arXiv preprint arXiv:2410.21970.

Zhaofeng Wu, Ananth Balashankar, Yoon Kim, Jacob Eisenstein, and Ahmad Beirami. 2024b. Reuse your rewards: Reward model transfer for zero-shot crosslingual alignment. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 1332–1353.

Wen Yang, Junhong Wu, Chen Wang, Chengqing Zong, and Jiajun Zhang. 2024. Language imbalance driven rewarding for multilingual self-improving. arXiv preprint arXiv:2410.08964.

Wen Yang, Junhong Wu, Chen Wang, Chengqing Zong, and Jiajun Zhang. 2025. Implicit cross-lingual rewarding for efficient multilingual preference alignment. In Findings of the Association for Computational Linguistics: ACL 2025, pages 21125–21147, Vienna, Austria. Association for Computational Linguistics.

Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang. 2023. Rrhf: Rank responses to align language models with human feedback. Advances in Neural Information Processing Systems, 36:10935–10950.

Jun Zhao, Zhihao Zhang, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024. Llama beyond english: An empirical study on language capability transfer. arXiv preprint arXiv:2401.01055.

Weixiang Zhao, Yulin Hu, Yang Deng, Tongtong Wu, Wenxuan Zhang, Jiahe Guo, An Zhang, Yanyan Zhao, Bing Qin, Tat-Seng Chua, and Ting Liu. 2025. MPO: Multilingual safety alignment via reward gap optimization. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23564–23587, Vienna, Austria. Association for Computational Linguistics.

## A Gain Formulation in Cross-Lingual Ranking

Preference Hierarchy In our pilot study, we observed that language adaptation is more readily achieved than the fine-grained differentiation of response quality within the same language, which can easily overshadow quality optimization during training. To reflect this, we established the following preference hierarchy for our ranking-based objective: $\psi ( y _ { w } ^ { t } ) > \psi ( y _ { w } ^ { e } ) > \psi ( y _ { l } ^ { t } ) > \psi ( y _ { l } ^ { e } )$ . A critical design choice in this formulation is the relation $\psi ( y _ { w } ^ { e } ) > \psi ( y _ { l } ^ { t } )$ , which prioritizes a high-quality response in English over a lower-quality response in the target language. Furthermore, this provides a learning signal that encourages the model to generate desirable responses regardless of the language. Since gain adjustments across items in a ranking structure are interdependent rather than orthogonal, penalizing the English winner below the target language loser would otherwise heavily bias the learning signal toward language matching.

Balance between Two Dimensions This philosophy also led to our choice of gain values. In standard Information Retrieval, which adopts the LambdaLoss framework, the gain function typically follows an exponential discount: $G _ { i } = 2 ^ { \psi _ { i } } - 1$ (Wang et al., 2018). However, applying this to our dualdimensional objective can introduce an inductive bias.

<table><tr><td rowspan="2">Gain values (G)</td><td colspan="2">Chinese</td><td colspan="2">Korean</td><td colspan="2">Indonesian</td><td colspan="2">Swahili</td><td colspan="2">Bengali</td></tr><tr><td>WC (%)</td><td>LC (%)</td><td>WC (%)</td><td>LC (%)</td><td>WC (%)</td><td>LC (%)</td><td>WC (%)</td><td>LC (%)</td><td>WC (%)</td><td>LC (%)</td></tr><tr><td>7,3,1,0</td><td>60.50</td><td>60.75</td><td>66.46</td><td>66.04</td><td>67.20</td><td>66.86</td><td>66.46</td><td>66.35</td><td>54.16</td><td>52.54</td></tr><tr><td>9,5,7,4</td><td>60.00</td><td>60.39</td><td>66.21</td><td>66.11</td><td>65.22</td><td>65.52</td><td>67.14</td><td>67.43</td><td>53.98</td><td>53.79</td></tr><tr><td>9,7,5,4</td><td>62.36</td><td>62.46</td><td>68.82</td><td>68.46</td><td>68.41</td><td>67.98</td><td>62.17</td><td>61.85</td><td>55.65</td><td>53.87</td></tr></table>

Table 4: WR and LC according to different gain values.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">Chinese</td><td colspan="2">Indonesian</td><td colspan="2">Korean</td><td colspan="2">Swahili</td><td colspan="2">Bengali</td></tr><tr><td>LC (%)</td><td>WR (%)</td><td>LC (%)</td><td>WR (%)</td><td>LC (%)</td><td>WR (%)</td><td>LC (%)</td><td>WR (%)</td><td>LC (%)</td><td>WR (%)</td></tr><tr><td rowspan="3">Llama-2-7B</td><td>SFT+DPO</td><td>55.06</td><td>55.71</td><td>55.89</td><td>56.52</td><td>57.79</td><td>58.26</td><td>56.04</td><td>56.46</td><td>51.46</td><td>52.42</td></tr><tr><td>CLO</td><td>57.46</td><td>57.27</td><td>53.49</td><td>52.24</td><td>55.30</td><td>54.91</td><td>53.41</td><td>51.93</td><td>53.22</td><td>52.61</td></tr><tr><td>CRPO</td><td>62.33</td><td>62.36</td><td>60.41</td><td>60.62</td><td>60.46</td><td>60.87</td><td>60.43</td><td>60.00</td><td>61.60</td><td>61.99</td></tr><tr><td rowspan="3">Llama-3-8B</td><td>SFT+DPO</td><td>57.71</td><td>58.45</td><td>60.31</td><td>61.30</td><td>56.23</td><td>56.83</td><td>58.06</td><td>58.01</td><td>56.98</td><td>57.76</td></tr><tr><td>CLO</td><td>52.83</td><td>52.48</td><td>56.89</td><td>56.21</td><td>52.66</td><td>50.99</td><td>54.46</td><td>52.61</td><td>56.27</td><td>54.78</td></tr><tr><td>CRPO</td><td>56.73</td><td>57.20</td><td>64.24</td><td>64.35</td><td>59.19</td><td>59.07</td><td>60.53</td><td>60.06</td><td>57.43</td><td>57.64</td></tr><tr><td rowspan="3">Mistral-7B</td><td>SFT+DPO</td><td>61.30</td><td>65.59</td><td>58.44</td><td>61.06</td><td>63.31</td><td>69.69</td><td>51.73</td><td>54.88</td><td>47.65</td><td>55.47</td></tr><tr><td>CLO</td><td>51.84</td><td>54.47</td><td>52.05</td><td>51.61</td><td>58.19</td><td>61.80</td><td>50.37</td><td>49.81</td><td>47.97</td><td>45.83</td></tr><tr><td>CRPO</td><td>50.60</td><td>57.08</td><td>59.96</td><td>61.86</td><td>57.42</td><td>64.35</td><td>56.21</td><td>57.76</td><td>53.49</td><td>55.86</td></tr></table>

Table 5: English performance in AlpacaEval benchmark. The best results for each model are highlighted in bold.

Given our hierarchy, the standard formula yields $G _ { w } ^ { t } = 7 , G _ { w } ^ { e } = 3 , G _ { l } ^ { t } = 1$ , and $G _ { l } ^ { e } = 0$ . Although ranking optimization inherently assigns disproportionate weights between items, this exponential spacing exacerbates biased local gradients.

To resolve this and provide a more balanced optimization manifold, we manually set the gains to $G _ { w } ^ { t } \ = \ 9 , G _ { w } ^ { e } \ = \ 7 , G _ { l } ^ { t } \ = \ 5$ , and $G _ { l } ^ { e } \ = \ 4$ . In this setup, the primary intra-language quality gaps $( G _ { w } ^ { t } - G _ { l } ^ { t } = 4$ for target, $G _ { w } ^ { e } - G _ { l } ^ { e } = 3$ for English) remain strictly greater than the primary language gap $( G _ { w } ^ { t } - G _ { w } ^ { e } = 2 )$ . This constraint explicitly ensures that the LambdaLoss framework jointly optimizes both cross-lingual alignment and quality, preventing the simple language-matching signal from overwhelming the fine-grained quality optimization.

Empirical Validation To validate our gain formulation, we compare the performance of Llama-3-8B across different gain configurations, as presented in Table 4. The theoretical range of possible gains is infinite, but as it is computationally prohibitive to explore the entire space, our experiments are restricted to a set of representative fixed values.

The standard exponential setting (7, 3, 1, 0) generally yields suboptimal win rates, suggesting that disproportionate local gradients hinder effective intra-lingual alignment. Furthermore, we examine a configuration (9, 5, 7, 4) that explicitly violates our preference hierarchy by penalizing the English winner $( G = 5 )$ below the target language loser $( G = 7 )$ . This configuration also yields slightly lower performance than our hierarchy across most languages, including Chinese, Korean, Indonesian, and Bengali, demonstrating that heavily biasing the signal toward language matching compromises overall response quality. Interestingly, a different tendency appears in Swahili. We attribute this to the low-resource nature of Swahili, where an aggressive language-matching penalty might provide a temporary empirical advantage in enforcing the target script. Nevertheless, our proposed hierarchy configuration demonstrates greater robustness across diverse linguistic environments, justifying our balanced hierarchical design.

## B Implementation Details

Training We observe that hyperparameter tuning is crucial for achieving optimal performance of preference optimization methods. To ensure a fair comparison, we conduct thorough hyperparameter tuning across all methods in our experiments.

We conduct preliminary experiments to search for learning rate in $[ 5 e \mathrm { ~ - ~ } 7 , 8 e \mathrm { ~ - ~ } 6 , 1 e \mathrm { ~ - ~ } 5 ]$ and training epochs in [1, 2]. We find that an effective learning rate of 8e-6 and training for two epochs consistently yield stable results across all models and languages. Therefore, we fix these values for all preference optimization experiments and report results using the final checkpoint. Additionally, we set the maximum sequence length to 3072 and apply a linear learning rate schedule with a 10% warmup ratio. To balance the training objective, we use a weighting factor of α = 0.2 for CRPO and SFT+DPO, and 0.5 for CLO following the recommended value in the original paper. We set β to 0.1, the default value in DPO. All experiments are conducted under four NVIDIA A100 GPUs.

![](images/90f69a99312243d16d092f07528d0f78391c6232c7b93abda5303506706b45b1.jpg)  
(b) Reward distribution of y<sub>l</sub>  
Figure 4: Reward distribution of chosen response (a) and rejected response (b) of Llama-3-8B trained with SFT (blue), CLO (orange), SFT+DPO (blue) and CRPO (red) across languages.

Evaluation For generative evaluation, we maintain the same maximum sequence length as used during the training phase to ensure consistency. In the generation process, we use temperature as the primary hyperparameter in the decoding step and set it to 0.8 to balance the trade-off between coherence and diversity in the outputs.

## C Arena-Hard

To further substantiate our evaluation, we report the multilingual Arena-Hard (m-ArenaHard)<sup>2</sup> (Dang et al., 2024b) evaluation results in Table 6. The m-ArenaHard benchmark was created by translating the prompts from the original Arena-Hardv0.1 (Li et al., 2024b) test dataset to 22 languages. We conduct supplementary experiments on the three languages that overlap with those covered in our study. Since several queries in the dataset exceed the relatively short context length of Llama-2 (4096 tokens), Llama-2 is excluded from the evaluation. Similar to AlpacaEval, we report the win rate against the SFT model.

<table><tr><td>Model</td><td>Method</td><td>Chinese</td><td>Indonesian</td><td>Korean</td></tr><tr><td rowspan="3">Llama-3-8B</td><td>SFT+DPO</td><td>61.56</td><td>61.34</td><td>64.30</td></tr><tr><td>CLO</td><td>54.49</td><td>54.53</td><td>54.95</td></tr><tr><td>CRPO</td><td>62.15</td><td>61.87</td><td>68.81</td></tr><tr><td rowspan="3">Mistral-7B</td><td>SFT+DPO</td><td>54.62</td><td>54.34</td><td>55.39</td></tr><tr><td>CLO</td><td>55.49</td><td>53.34</td><td>53.02</td></tr><tr><td>CRPO</td><td>58.22</td><td>63.85</td><td>62.63</td></tr></table>

Table 6: Win rate in Arena-Hard benchmark.

## D Impact on Alignment Tax

In our main experiments, we focused on evaluating the target languages to demonstrate the effectiveness of cross-lingual alignment. However, since our approach leverages the model’s inherent English knowledge as an anchor to induce transfer to the target language, it is imperative to investigate how the alignment training on the target language affects the model’s English proficiency. To this end, we conduct an additional English evaluation on AlpacaEval to measure the alignment tax.

Table 5 presents the English performance of the models trained under each target language setting. Note that all methods also include English in their training data composition. As shown in the results, CRPO achieves higher English performance compared to the baselines in most cases, indicating that it successfully mitigates the alignment tax. To be specific, for the Llama-2-7B and Llama-3-8B models, CRPO achieves the highest WR across almost all target language configurations. This suggests that CRPO preserves the robustness of the model’s English capabilities while effectively performing cross-lingual alignment.

## E Reward Distribution Analysis

In addition to Figure 2, we also provide an analysis of the reward distribution of responses in Figure 4. As shown in Figure 4 (a), CRPO consistently yields higher rewards for favored responses compared to the other methods, whereas the reward distributions for the other methods do not exhibit a significant shift relative to SFT.

In Figure 4 (b), SFT+DPO and CRPO demonstrate comparable distributional shifts, which is the nature of DPO training that generally increases the reward of both chosen and rejected responses. Taken together, these findings indicate that the major reward difference achieved by CRPO, as observed in Figure 2, is primarily driven by elevating the rewards of desirable responses rather than solely relying on penalizing those of unfavored ones. This further substantiates our claim in Section 5.2 that the advantage of CRPO stems from promoting high-quality responses rather than solely from suppressing the probability of disfavored responses.

## F Comparison with Another Ranked Optimization

To investigate whether the effectiveness of CRPO can be attributed simply to the use of ranked preference structure, we further compare CRPO with PRO (Song et al., 2024), which constructs preferences sequentially from a ranked set of responses:

$$
\mathcal { L } _ { \mathrm { P R O } } = - \log \prod _ { k = 1 } ^ { n - 1 } \frac { \exp { \left( r _ { \pi } ( x , y _ { k } ) \right) } } { \sum _ { i = k } ^ { n } \exp { \left( r _ { \pi } ( x , y _ { i } ) \right) } } ,\tag{8}
$$

which is further combined with an NLL objective. Table 7 reports the AlpacaEval results across three target languages and two backbone models.

Despite also performing ranking-based preference optimization, PRO substantially underperforms CRPO. An inspection of generations from the PRO-trained models shows that they generally retain the ability to respond in the language of the input. This suggests that PRO can achieve the primary objective of language adaptation, but does not translate this adaptation into a comparable improvement in response quality. In contrast, CRPO improves both target-language and English AlpacaEval performance by a large margin.

<table><tr><td rowspan="2">Lang Model</td><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2">Target</td><td colspan="2">English</td></tr><tr><td>WR</td><td>LC</td><td>WR</td><td>LC</td></tr><tr><td rowspan="2">zh</td><td>Llama-3-8B</td><td>PRO CRPO</td><td>24.97 62.36</td><td>24.24 62.45</td><td>37.27 57.20</td><td>36.85 56.73</td></tr><tr><td>Mistral-7B</td><td>PRO CRPO</td><td>13.42 63.10</td><td>13.75 62.83</td><td>36.15 57.08</td><td>33.67 50.60</td></tr><tr><td rowspan="2">id</td><td>Llama-3-8B</td><td>PRO CRPO</td><td>23.11 68.40</td><td>24.79 67.97</td><td>44.47 64.35</td><td>43.60 64.24</td></tr><tr><td>Mistral-7B</td><td>PRO CRPO</td><td>13.29 68.19</td><td>12.42 66.81</td><td>29.81 61.86</td><td>30.52 59.96</td></tr><tr><td rowspan="2">SW</td><td>Llama-3-8B</td><td>PRO CRPO</td><td>11.43 62.17</td><td>11.78 61.84</td><td>37.64 60.06</td><td>36.17 60.53</td></tr><tr><td>Mistral-7B</td><td>PRO CRPO</td><td>6.58 51.98</td><td>5.11 50.76</td><td>10.62 57.76</td><td>12.15 56.21</td></tr></table>

Table 7: AlpacaEval results of PRO and CRPO under the same training budget.

We attribute this difference to how the ranked preference structure is exploited during optimization. PRO primarily constructs pairwise comparisons between a higher-ranked response and lowerranked alternatives. CRPO instead exploits the relative positions of responses throughout the ranked list and incorporates ranking-aware objectives to place greater emphasis on preference violations that affect the top of the ranking. Consequently, CRPO provides a more fine-grained learning signal for distinguishing responses with different quality levels rather than merely separating preferred from less-preferred responses.

This distinction is particularly important in our setting because language adaptation itself can already be encouraged by the NLL objective: the model can learn to generate in the target language without necessarily improving the quality of the generated content. The ranking-aware objective of CRPO complements this signal by explicitly preserving and promoting high-quality responses according to their positions in the ranked list. The considerable gap between PRO and CRPO therefore indicates that the gains of CRPO do not arise merely from introducing ranked preference optimization. Rather, how the ranked preference information is incorporated into the optimization objective is critical, further supporting the design of CRPO.