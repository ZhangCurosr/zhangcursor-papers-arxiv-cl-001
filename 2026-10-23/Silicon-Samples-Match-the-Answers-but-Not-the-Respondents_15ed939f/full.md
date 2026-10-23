# Silicon Samples Match the Answers but Not the Respondents

Yusuke Kumagae Hakuhodo DY Holdings Inc. yusuke.kumagae@hakuhodo.co.jp

Masahiro Kaneko MBZUAI Masahiro.Kaneko@mbzuai.ac.ae

## Abstract

Silicon sampling generates survey respondents by having a large language model (LLM) answer a survey as a person with given demo graphic attributes, and the resulting synthetic populations are increasingly used in place of human ones. Existing evaluations focus on individual-answer accuracy, per-question answer distributions, and correlations across questions. None of these compares the respondents themselves, so a generator whose respondents have combinations of answers found in no human respondent can still score well if it reproduces every answer distribution. We propose the respondent coverage gap (RCG)<sup>1</sup>, which represents each respondent as a point defined by their complete answer profile and measures how far each human respondent is from the nearest generated one, relative to the distance between human samples. On two US social surveys, all six LLMs we evaluate leave human respondents uncovered, and under our main settings, they fall behind a baseline that samples each answer independently from the human marginals. Across the prompting and inference settings we test, every LLM has a higher RCG than a Gaussian copula fitted to the human data.

## 1 Introduction

Large language models (LLMs) can act as a specified person and answer questions from that person’s standpoint (Zhang et al., 2018; Shanahan et al., 2023; Wang et al., 2024). Silicon sampling (Argyle et al., 2023) exploits this ability: it gives an LLM a combination of demographic attributes (e.g., female, in her thirties) as a persona, has it produce a full set of answers to a questionnaire (an answer profile) as a human respondent would, and repeats this over many personas to obtain a synthetic population. Such synthetic populations are increasingly used as substitutes for, or supplements to, human populations in opinion polls, market research, and social science experiments (Santurkar et al., 2023; Park et al., 2026; Horton et al., 2023; Aher et al., 2023). Typical uses include estimating the distribution of opinions, probing the reactions of groups that are hard to recruit, and piloting a survey before fielding it.

![](images/f694c2f365b75a66995d89a0cef8e50d1c124d1a8e25de547d17323869802596.jpg)  
Figure 1: Marginals reproduces each question’s answer distribution, and Copula additionally models pairwise dependence between questions. Top: answer profiles of human and generated respondents. Bottom: existing evaluation compares answers per question, where the three groups are identical (left); ours compares respondents, where humans are farther from the nearest generated respondent than from the nearest other human (right).

The quality of such a synthetic population has so far been evaluated through the distribution of answers to each question. Individual accuracy (Park et al., 2026; Hu et al., 2026) counts, question by question, whether the answer the LLM generates from a respondent’s attributes matches that respondent’s own answer. Marginal fidelity (Santurkar et al., 2023; Durmus et al., 2024) compares, for each question, the distribution of human answers with the distribution of answers generated from the same attributes. Correlational fidelity (Moon et al., 2024; Williams et al., 2026; Libovický, 2026; Ma et al., 2026) computes the correlation between answers to different questions separately on the human side and on the generated side, and compares the two. All of them aggregate answers along the axis of questions; none of them compares the respondent distribution, that is, how respondents are spread out as answer profiles.

Agreement in answer distributions does not guarantee agreement in respondent distributions. Even when the answer distributions for every question match, the synthetic population can consist of generated respondents whose answer profiles are absent from the human population; for a synthetic population to stand in for a human one, the respondent distributions must agree. We fit two generators to the World Values Survey (WVS) (Haerpfer et al., 2022): Marginals reproduces only the per-question answer distributions, and Copula additionally models pairwise dependence between questions. Per question, both are indistinguishable from the human respondents (Figure 1, bottom left). Per respondent, measuring the distance from each human respondent to the nearest generated one separates the two (Figure 1, bottom right; details in Appendix A): Marginals lies far from the human respondents, and Copula, despite modeling pairwise dependence, still falls short of the human reference. Per-question distributions and pairwise correlations do not fully characterize the respondent distribution.

In this paper, we propose the respondent coverage gap (RCG), a metric of how well a synthetic population covers the respondent distribution of a human population. We treat each answer profile as a point and measure, from each human point, the distance to the nearest generated point; this directly quantifies how close each human respondent is to a generated respondent. Because this distance depends on the number of questions and the size of the populations, we normalize it by the same distance measured among the human points themselves. The reference is another human sample rather than the human respondents themselves: a synthetic population should not copy the human respondents but resemble them as a population. A ratio of one means the synthetic population is as close to the human respondents as another human sample would be; the more it exceeds one, the farther the least covered human respondents are from any generated respondent.

Evaluating six LLMs (two proprietary models from OpenAI and Google and four open-weight models from 9B to 70B parameters) on two realworld surveys of US residents, we find that none of them cover the human respondents. Under our main settings, all fall short of Marginals, which reproduces nothing but the per-question answer distributions. Some settings bring individual models close to or below Marginals: using the full response history, increasing reasoning effort, or using highertier models such as gpt-5.4 and gemini-3.1-propreview. However, no tested setting yields a lower RCG than Copula, which additionally models pairwise dependence between questions. Calibrating the generated answer distributions to match the human ones, question by question, reduces RCG below that of Marginals, but not below that of Copula.

This failure of LLMs is invisible to existing metrics. Marginal fidelity does not separate generators that differ in respondent coverage, and an aggregate correlation metric assigns Marginals, which samples every question independently, a score on par with the split-half ceiling of the human data. The failure also takes model-specific forms: some models contract onto typical profiles, others drift off the support of the human population. These failures surface only when the respondent distribution is examined; by that standard, synthetic populations from the evaluated LLMs are not yet adequate substitutes for human ones.

## 2 Related Work

## 2.1 Silicon Sampling and Its Evaluation

Silicon sampling has been applied across domains such as social simulations (Park et al., 2023, 2026), economic experiments (Horton et al., 2023), social science experiments (Ashokkumar et al., 2026; Aher et al., 2023), and social surveys (Santurkar et al., 2023; Argyle et al., 2023; Ma et al., 2025).

Individual accuracy. This line of work scores, question by question, whether the generated answer matches the answer observed from the corresponding human respondent. Argyle et al. (2023) address the task of predicting vote choice and measure the correspondence with tetrachoric correlations. Park et al. (2026) ground LLM agents in interviews with Americans and predict their held-out answers to the

General Social Survey and the Big Five Inventory. Yet the scoring remains per-question. Park et al. (2026) themselves note that agents can predict a respondent’s answers well yet still misrepresent how attitudes covary across the sample, and defer sample-level correlational structure to future work.

Marginal fidelity. In contrast, this line of work compares answer distributions question by question, without pairing generated and human respondents. Santurkar et al. (2023) measure the 1- Wasserstein distance between model and human answer distributions over 1,498 Pew survey questions. Durmus et al. (2024) apply the same perquestion comparison across 60+ countries, scoring one minus the Jensen–Shannon distance. In all of these, the human reference is itself an aggregate distribution; no individual enters the comparison.

## 2.2 Beyond Marginal Evaluation

Both lines of work treat each question in isolation. A few studies (Moon et al., 2024; Williams et al., 2026; Libovický, 2026; Ma et al., 2026) take a step further and evaluate dependence across questions. To do so, they share a common two-step procedure: (1) computing inter-item associations separately within the human sample and within the generated sample, and (2) comparing the two association matrices as wholes. While these methods do capture dependence across questions, the comparison is made at the level of question pairs, not of the individual respondents that silicon sampling promises to substitute for. Williams et al. (2026) themselves defer individual-level structure to future work.

## 2.3 Comparing Generated and Human Respondents as Distributions

Outside silicon sampling, generated samples are compared with human ones as whole distributions, each sample being one point. Sajjadi et al. (2018) decompose this comparison into precision, how closely the generated samples resemble the human ones (fidelity), and recall, how well they span the variability of the human ones (diversity). Kynkäänniemi et al. (2019) and Naeem et al. (2020) estimate both with k-nearest-neighbor balls, and MAUVE (Pillutla et al., 2021) measures both for text as a divergence between the two distributions. These methods use discrete membership information (whether a sample lies inside a neighborhood ball or which quantization cell it occupies) rather than retaining its distance to the human samples. Prior work also compares synthetic-to-human nearest-neighbor distances with human-to-human distances. Alaa et al. (2022) use such a comparison in a binary authenticity test for each generated sample, whereas Lautrup et al. (2025) use a ratio of median distances in med.DCR to assess privacy risk. Our focus is instead on human respondents whose nearest generated respondent is too far away.

Silicon sampling has so far been evaluated along the axis of questions (Section 2.1 and Section 2.2); we instead compare distributions at the respondent level, treating each respondent’s complete answer profile as one sample.

## 3 Evaluation Framework

Let $q _ { 1 } , \ldots , q _ { M }$ be M questions. A respondent is represented as an answer vector $\pmb { u } = ( u _ { 1 } , \dots , u _ { M } )$ where $u _ { m }$ is the answer chosen for question $q _ { m }$ . We write H for a set of human answer vectors and $\mathcal { G }$ for a set of generated answer vectors.

## 3.1 Conditional Generation

We generate each respondent by conditioning an LLM on a set of demographic attributes (e.g., sex and age), and compare the generated respondents $\mathcal { G }$ against the human respondents H who share the same attributes. The LLM answers the questions one at a time, and the prompt for each question includes a few of that respondent’s earlier answers. This sequential format, also employed in prior work (Moon et al., 2024; Libovický, $2 0 2 6 ) , ^ { 2 }$ allows later answers to depend on earlier ones within the same respondent.<sup>3</sup>

## 3.2 Respondent Coverage Gap

Given a distance $d ( { \pmb u } , { \pmb v } )$ between answer vectors, let $\delta ( { \pmb u } ; S )$ be the distance from an answer vector u to its nearest neighbor in a set $\mathcal { S } . ^ { 4 }$ Our core idea is to examine the distribution of the nearestneighbor distance $\delta ( { \pmb u } ; { \mathcal { G } } )$ over the human respondents $\mathbf { \boldsymbol { u } } \in \mathcal { H }$ and, symmetrically, of $\delta ( { \pmb v } ; \mathcal { H } )$ over the generated respondents ${ \pmb v } \in \mathcal G$ . The former asks, for each human respondent, how close the nearest generated respondent is; the latter asks, for each generated respondent, how close the nearest human respondent is.

Taking the maximum of the former recovers the directed Hausdorff distance (Hausdorff, 1914):

$$
d _ { \mathscr { H } \to \mathscr { G } } = \operatorname* { m a x } _ { \pmb { u } \in \mathscr { H } } \delta ( \pmb { u } ; \mathscr { G } ) .\tag{1}
$$

We modify this distance in two ways. First, we replace the maximum with the 95th percentile, denoted $d _ { \mathcal { H }  \mathcal { G } } ^ { 9 5 }$ , for robustness to outliers (Huttenlocher et al., 1993). Second, we normalize it by the same quantity computed within the human respondents, $d _ { \mathcal { H }  \mathcal { H } } ^ { 9 5 } ,$ the 95th percentile of the distances from each human respondent to their nearest other human respondent; this makes the ratio comparable across different choices of H.

We define the respondent coverage gap (RCG) and its reverse-direction companion, the $o f f -$ support ratio (OSR), which share this human-side quantity as the denominator:

$$
\operatorname { R C G } ( \mathcal { H } , \mathcal { G } ) = \frac { d _ { \mathcal { H }  \mathcal { G } } ^ { 9 5 } } { d _ { \mathcal { H }  \mathcal { H } } ^ { 9 5 } } ,\tag{2}
$$

$$
\mathrm { O S R } ( \mathcal { H } , \mathcal { G } ) = \frac { d _ { \mathcal { G }  \mathcal { H } } ^ { 9 5 } } { d _ { \mathcal { H }  \mathcal { H } } ^ { 9 5 } } .\tag{3}
$$

Since $\delta ( \pmb { u } ; S )$ shrinks as S grows, we use equally sized H and G in our main evaluation (Section 4.2 and Appendix D). RCG measures the extent to which human respondents are left uncovered by the generated ones; OSR reverses the direction of the numerator while keeping the same denominator, and thus measures the extent to which generated respondents lie outside the support of the human respondents. We use RCG as our main metric and OSR as a diagnostic when analyzing failure modes (Section 5.2).

Both ratios are expected to be near one when G is an independent sample of human respondents. RCG > 1 indicates undercoverage of human respondents, whereas OSR > 1 indicates generated respondents unusually far from the human sample (off-support). A low OSR is not necessarily better: when RCG > 1 but OSR < 1, generated respondents stay close to some human respondents while leaving others poorly covered (contraction). Figure 2 illustrates these patterns: the off-support and contraction examples have similar RCG values (2.17 and 2.06), but their OSR values lie on opposite sides of one (2.24 and 0.55).

Human respondents Generated respondents  
![](images/d74c7d5a636865f913d020203e20c24c28c57beba8dcd7e532242ebdec6f7cd1.jpg)  
Figure 2: Toy examples of the human-like, off-support, and contraction patterns on two-dimensional synthetic data (50 human and 50 generated respondents, Euclidean distance). The panels illustrate representative configurations, not an exhaustive classification of RCG and OSR values.

OSR has a similar form to med.DCR (Lautrup et al., 2025), but addresses the opposite concern (Section 2.3). For privacy, a synthetic record that is too close to a human record poses a risk. Accordingly, med.DCR divides the median syntheticto-real nearest-neighbor distance by its real-toreal counterpart, with values below one indicating greater concern. In contrast, OSR uses the 95th percentile to identify generated respondents that are unusually far from the human sample, with values above one indicating off-support generation.

Retaining the distances also sets our ratios apart from the density and coverage of Naeem et al. (2020), which test each nearest-neighbor distance against a threshold (Section 2.3). Our ratios stay informative no matter how far apart the two sets are: for a generated set that lies outside all human neighborhood balls, the coverage of Naeem et al. (2020) is zero however far away it lies, whereas $d _ { \mathcal { H }  \mathcal { G } } ^ { 9 5 }$ keeps growing with the distance. Our ratios also do not require the neighborhood size k that sets that threshold; the only remaining choice is the aggregation percentile, whose effect we examine in Appendix H.

## 4 Evaluation Setup

## 4.1 Datasets

We use two surveys of US residents: the World Values Survey (WVS; Haerpfer et al., 2022), on social, religious, and political values, and the American National Election Studies (ANES; American National Election Studies, 2026), on vote choice and partisanship.<sup>5</sup> For WVS, we use the United States sample of wave 7 (2,594 respondents) and uniformly sample 100 questions from the 144 used by Liu et al. (2026); the dataset provides 17 attribute variables, such as sex, age, education, and income. For ANES, we use the 2024 wave of its Time Series Cumulative Data File (February 2026 release; 5,453 respondents) and uniformly sample 100 questions from the 149 remaining after excluding administrative, demographic, and geographic items; the dataset provides 11 attribute variables including party identification and liberal–conservative ideology.<sup>6</sup> Appendix B lists all attributes and their binning.

In both datasets, the 100 questions are drawn once, and the same draw is used for all models and cells. Of the 100 questions, 30 (WVS) and 32 (ANES) are numeric rating scales and the rest are multiple-choice questions with ordinal or nominal options, so each respondent is represented by a vector of M = 100 answers.

## 4.2 Conditioning Cells

A cell fixes the values of L demographic attributes on which generation is conditioned (Section 3.1), e.g., (female, 30–39) with $L = 2 ;$ every cell also includes US residence, which all respondents share (Section 4.1). We keep cells with at least 100 human respondents and no missing attribute values, and sample up to 100 cells per L, which gives 730 cells for WVS and 603 for ANES (L = 0–8; Appendix D). Each cell c gives a pair $( \mathcal { H } _ { c } , \mathcal { G } _ { c } )$ : the answer vectors of the human respondents in c and of the 100 generated respondents whose prompts carry the attribute values of c (Section 4.3). All metrics in Section 3.2 are computed per cell.

## 4.3 Prompts

Our prompt has four parts: a task instruction, a persona description, few-shot examples<sup>7</sup> of the respondent’s earlier answers, and the question with its choices. To reduce sensitivity to wording and ordering, we prepare three instruction formats and three persona formats following prior studies (Argyle et al., 2023; Santurkar et al., 2023; Bisbee et al., 2024; Taday Morocho et al., 2026; Moon et al., 2024; AlKhamissi et al., 2024), and randomly assign one instruction format and one persona format to each generated respondent. We also randomize the question order and the order of attributes in the persona. For questions without a natural ordering of answer choices, we shuffle the choices to reduce LLM preferences for particular answer labels (Zheng et al., 2024; Dominguez-Olmedo et al., 2024). The full templates and two example prompts are in Appendix C.

## 4.4 Models

Proprietary. We use two proprietary models: gpt-5.4-nano (OpenAI) and gemini-3.1-flash-lite (Google, hereafter gemini-flash-lite). Additionally, in an ablation study (Section 5.5), we use gpt-5.4 and gemini-3.1-pro-preview (Google, hereafter gemini-pro). We leave the sampling parameters (temperature, top\_p, and top\_k where available) at their vendor defaults and set only the reasoning effort to the lowest level available: none for the OpenAI models and minimal for the Google models.

Open-weight. We use four open-weight models: Qwen3.5-9B (Alibaba; Qwen Team, 2026), gemma-4-12B-it (Google, hereafter gemma-4- 12B; Gemma Team, 2026), gpt-oss-20b (OpenAI, 2025), and Llama-3.3-70B-Instruct (Meta, hereafter Llama-3.3-70B; Grattafiori et al., 2024). We use their default decoding settings (Appendix E) without tuning, to evaluate off-the-shelf use.

## 4.5 Distance

Because the questions are of mixed types (Section 4.1), we use the Gower distance (Gower, 1971) as d(u, v) (details in Appendix F). The Gower distance averages per-question distances, excluding the m-th question when either $u _ { m }$ or $v _ { m }$ is missing.<sup>8</sup>

## 4.6 Baselines

We select three non-LLM baselines. Marginals samples each answer $u _ { m }$ independently from the marginal distribution of the human respondents H. Copula draws the whole answer vector u from a Gaussian copula (Sklar, 1959): the marginals are estimated as in Marginals, and the questions are coupled through a correlation matrix estimated with Ledoit–Wolf shrinkage (Ledoit and Wolf, 2004), as fitting uses fewer respondents than questions in about half of the cells. LCA draws u from a two-class latent class model (Goodman, 1974), that is, a mixture of two class-specific Marginals generators, fitted with StepMix (Morin et al., 2025). Because these baselines are fitted to the same respondents against which they are evaluated, they serve as references for what a generator can achieve with access to the cell’s respondents, rather than as direct competitors to the LLMs (fitting details in Appendix I).

<table><tr><td rowspan="3"></td><td colspan="2">WVS</td><td colspan="2">ANES</td></tr><tr><td>RCG</td><td>Cal. OSR</td><td>RCG Cal.</td><td>OSR</td></tr><tr><td>1.272</td><td>1.150 0.994</td><td>1.370 1.232</td><td></td></tr><tr><td>gpt-5.4-nano gemini-flash-lite</td><td>1.218</td><td>1.126 0.840</td><td>1.353 1.178</td><td>1.173 0.838</td></tr><tr><td>Qwen3.5-9B</td><td>1.273</td><td>1.1450.992</td><td>1.347</td><td>1.2171.213</td></tr><tr><td>gemma-4-12B</td><td>1.199</td><td>1.137 0.793</td><td>1.332</td><td>1.194 0.942</td></tr><tr><td>gpt-oss-20b</td><td>1.269</td><td>1.140 1.013</td><td>1.346 1.202</td><td>1.144</td></tr><tr><td>Llama-3.3-70B</td><td>1.275</td><td>1.147 0.722</td><td>1.277 1.198</td><td>0.853</td></tr><tr><td>Marginals</td><td>1.158</td><td>0.989</td><td>1.250</td><td>1.078</td></tr><tr><td>Copula</td><td>1.041</td><td></td><td>0.9461.091</td><td>0.976</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { L C A }$ </td><td>1.105</td><td></td><td>0.9591.143</td><td>0.950</td></tr></table>

Table 1: The respondent coverage gap (RCG) and the off-support ratio (OSR) for each generator on each dataset. Cal. is the RCG after marginal calibration. OSR is reported before calibration.

## 5 Results

## 5.1 Main Results

For each cell, we compute RCG and OSR for 50 random draws of 100 human respondents. We first average the scores over draws within each cell, then average the resulting cell-level scores across cells. We also confirm that RCG behaves as intended; in each cell $c , ^ { 9 }$ we replace the generated set $\mathcal { G } _ { c }$ with a separate sample of 100 human respondents drawn from $\mathcal { H } _ { c } .$ , disjoint from the 100 respondents used for evaluation, and compute RCG exactly as above. The average RCG is 1.001 (WVS) and 1.000 (ANES). The same replacement gives an average OSR of 1.001 (WVS) and 1.002 (ANES). Results under the hold-out fit setting are reported in Appendix I.

Table 1 shows the RCG and OSR for each generator and dataset (Appendix K shows RCG on WVS and ANES by the number of conditioning attributes

L). All uncalibrated LLM RCG values in Table 1 exceed 1.0 and the values of all three baselines, including the simplest baseline, Marginals. For every LLM on both datasets, the 95% confidence interval for the RCG difference from Marginals excludes zero (Table 8 in Appendix J). Thus no LLM covers the human respondents as well as another human sample does (1.000–1.001), or even as well as Marginals, which ignores all dependence between questions. RCG shows no consistent ordering by model family, weight availability (proprietary vs. open-weight), or model size.

Calibrating marginal distributions. We next ask whether correcting per-question response distributions closes the respondent coverage gap. We use the empirical human marginals in each evaluation cell as oracle targets, applying quantile mapping (Gudmundsson et al., 2012; Cannon et al., 2015) to ordered responses and minimum-rewrite remapping to nominal responses (Appendix P). Calibration (Cal. in Table 1) reduces RCG for all six LLMs on both datasets. The calibrated RCG values range from 1.126 to 1.150 on WVS and from 1.178 to 1.232 on ANES. They are lower than those of Marginals but higher than the human reference and those of both Copula and LCA. All differences from the three baselines are statistically significant (Table 9 in Appendix J). Thus, this item-wise correction improves respondent coverage but does not restore human-level coverage.

## 5.2 Failure Modes

Using the uncalibrated OSR column of Table 1, we next examine how each LLM fails to cover the human respondents. The baselines (Marginals, Copula, and LCA) have OSR at or near one (0.946– 1.078) on both datasets. RCG alone does not show how a model fails: on ANES, the six LLMs span 1.277–1.370 in RCG but 0.838–1.213 in OSR.

In both datasets, OSR splits the six LLMs into two groups: the upper group (gpt-oss-20b, gpt-5.4-nano, and Qwen3.5-9B) and the lower group (gemma-4-12B, gemini-flash-lite, and Llama-3.3- 70B). The OSRs of the lower group are below one (0.722–0.840 on WVS and 0.838–0.942 on ANES) and the RCGs are above one. By the definitions in Section 3.2, this combination indicates contraction. In the upper group, the OSRs on ANES are above one (1.144–1.213), which is the off-support pattern. By contrast, on WVS, the OSRs are close to one (0.992–1.013).

![](images/d9585b1f4a67b50c58cc9a577f79000b5f8bea0fd4471c5c3652016e24a364a9.jpg)  
Figure 3: Per-question JSD (mean over questions and cells) against RCG on WVS for three baselines and six LLMs. For the LLMs, both metrics are computed after in-cell marginal calibration. Human (disjoint) uses only the 203 cells with at least 200 human respondents. Larger values are worse on both axes.

![](images/69fb7dfc48a2f8b210431438f0de0db4923fb9b041212149bb2e8a1713a3824d.jpg)  
Figure 4: Correlation-structure mismatch $( 1 - \rho )$ against RCG for all nine generators on WVS. Larger values are worse on both axes. Triangles are LLMs; circles are the statistical baselines. The dotted horizontal line marks the permutation null (each question column permuted across cells, preserving marginals but destroying correlations). The dashed horizontal line marks the split-half ceiling of the human data (footnote 10).

Prior work has reported both patterns under other names, though not at the level of individual respondents’ joint answer profiles: contraction as reduced variance (Bisbee et al., 2024) and flattening (Wang et al., 2025); and the off-support pattern as misportrayal (Wang et al., 2025).

## 5.3 Prior Metrics Can Mask Coverage Failures

RCG is not a repackaging of existing metrics: marginal fidelity does not separate generators that differ in RCG, and an aggregate correlation metric is saturated by a generator that samples answers independently across questions. We measure marginal fidelity by the mean perquestion Jensen–Shannon distance (JSD; Durmus et al., 2024) and dependence across questions by the correlation-structure score of Williams et al.

(2026).

Figure 3 shows per-question JSD against RCG on WVS for the human reference and for the generators that match the human answer distributions: Marginals, Copula, LCA, and the six calibrated LLMs (the same figure for ANES is in Appendix L). Their JSD values lie within a narrow range (0.08– 0.11), whereas their RCG ranges from 1.04 (Copula) to 1.16 (Marginals). The calibrated LLMs have the lowest JSD, as expected from calibrating to the human marginals in each evaluation cell (Section 5.1), yet their RCG remains above that of Copula and LCA. Per-question JSD depends only on each question’s answer distribution, not on how a respondent’s answers combine across questions. RCG compares whole answer profiles and therefore reflects this combination.

Figure 4 plots the correlation-structure mismatch $( 1 - \rho ,$ where $\rho$ is the correlation-structure score of Williams et al., 2026) against RCG for all generators on WVS (the same figure for ANES is in Appendix L). Marginals achieves $\rho = 0 . 9 9 4$ on WVS (split-half ceiling<sup>10</sup> of the human data: 0.995) and 0.999 on ANES (split-half ceiling of the human data: 0.999). Its RCG, however, is 1.158 on WVS and 1.250 on ANES.

Proposition 1 (Proof in Appendix M). For any set of generated respondents $\mathcal { G } _ { c }$ in $a$ cell $c ,$ let $\mathcal { G } _ { c } ^ { \prime }$ be obtained by permuting, separately for each question $q _ { m } ,$ , the answers to $q _ { m }$ among the respondents of $\mathcal { G } _ { c }$ who answered it. Then $\mathcal { G } _ { c } ^ { \prime }$ has the same marginal for every question as $\mathcal { G } _ { c } ,$ and the correlation-structure score ofWilliams et al. (2026) takes identical values on $\{ { \mathcal { G } } _ { c } \} .$ <sub>c</sub> and $\{ { \mathcal { G } } _ { c } ^ { \prime } \} _ { c }$ c

The permutation destroys the alignment of answers within respondents, yet the score is unchanged. In particular, applying such within-cell permutations to the human data yields a correlationstructure score of exactly one against the original human data. This invariance is not a defect for measuring aggregate cross-cell structure; rather, it shows that the score cannot substitute for respondent-level evaluation.

## 5.4 Number of Generated Respondents

As described in Section 3.2, RCG shrinks as $| \mathcal { G } _ { c } |$ grows for any generator. We ask how many generated respondents are needed for RCG to reach one.

<table><tr><td rowspan="2"></td><td colspan="2">RCG at  $| { \mathcal { G } } _ { c } | =$ </td><td rowspan="2">Slope per decade</td></tr><tr><td>100 1,000</td><td>10,000</td></tr><tr><td>gpt-5.4-nano</td><td>1.270 1.197</td><td></td><td>-0.073†</td></tr><tr><td rowspan="4">gemini-flash-lite Qwen3.5-9B gemma-4-12B</td><td>1.222 1.159</td><td></td><td>-0.060†</td></tr><tr><td>1.271 1.193</td><td>1.130</td><td>-0.070</td></tr><tr><td>1.204 1.141</td><td>1.094</td><td>-0.056</td></tr><tr><td>1.267 1.189</td><td>1.124</td><td>-0.072</td></tr><tr><td>Marginals</td><td>1.161 1.074</td><td>1.003</td><td>-0.078</td></tr><tr><td>Copula</td><td>1.041 0.931</td><td>0.847</td><td>-0.095</td></tr><tr><td>LCA</td><td></td><td>0.954</td><td>-0.077</td></tr><tr><td></td><td>1.109 1.023</td><td></td><td></td></tr></table>

Table 2: RCG (cell mean over the WVS subset of Section 5.5) at three sizes of the generated set $\left| \mathcal { G } _ { c } \right|$ . Slope is the median over cells of the per-cell change in RCG per tenfold increase in $| { \mathcal { G } } _ { c } | . - :$ not generated; †: fitted over one decade only, for reference.

We change $\left| \mathcal { G } _ { c } \right|$ from 100 to 1,000 (10 × the 100 human respondents) for the two proprietary models (gpt-5.4-nano and gemini-flash-lite)<sup>11</sup> and to 10,000 (100 ×) for the three open-weight models (Qwen3.5-9B, gemma-4-12B, and gpt-oss-20b).<sup>12</sup>

Table 2 shows the RCG on the WVS subset of Section 5.5 at different numbers of generated respondents $\left| \mathcal { G } _ { c } \right|$ and the slope of RCG regressed on $\log _ { 1 0 } | \mathcal { G } _ { c } |$ (the same table for ANES is in $\mathsf { A p - }$ pendix N). The RCG of the two proprietary models remains above one even at $| \mathcal { G } _ { c } | = 1 , 0 0 0$ , and that of the three open-weight models remains above one even at $| \mathcal { G } _ { c } | = 1 0 , 0 0 0$ , whereas that of Marginals falls to 1.003 at $| \mathcal { G } _ { c } | = 1 0 , 0 0 0$ . OSRs change by less than 0.01 across different numbers of generated respondents (Appendix N).

Extrapolating the fitted trends linearly in $\log _ { 1 0 } | \mathcal { G } _ { c } |$ from $\left| \mathcal { G } _ { c } \right| = 1 0 , 0 0 0$ would place RCG = 1 at about $5 \times 1 0 ^ { 5 }$ to $7 \times 1 0 ^ { 5 }$ generated respondents per cell for the three open-weight models, or 5,000 to 7,000 times the 100 human respondents.<sup>13</sup>

## 5.5 Robustness to Prompting, Inference Settings, and Model Tier

We examine the robustness of the results in Section 5.1 on a subset of the cells in Section 4.2: five cells per level of $L ,$ or all cells at levels with fewer (41 cells for WVS and 37 for ANES). The values in this subsection therefore differ from Table 1. Settings are in Appendix G.

Sensitivity to temperature. We vary the temperature from 0.1 to 1.5. RCG changes by at most 0.033 within each model, while OSR rises with temperature, most for gpt-5.4-nano, Qwen3.5-9B, and gpt-oss-20b (Appendix R).

Sensitivity to reasoning effort. We also vary the reasoning effort, which controls how much the model reasons before answering, over low, medium, and high for gpt-5.4-nano (default none) and over medium and high for gpt-oss-20b (default low). Raising the reasoning effort shifts both models toward contraction, and the RCG of gpt-5.4-nano falls mostly from none to low (Appendix S).

Sensitivity to model tier. We use gpt-5.4 from OpenAI, a higher-tier model of the same generation as gpt-5.4-nano, and gemini-3.1-pro-preview (gemini-pro), a frontier model from Google. Using higher-tier models reduces RCG except for gemini-pro on WVS, and both shift toward contraction. On ANES, gpt-5.4-nano at high effort and both higher-tier models have RCG point estimates 0.004–0.016 below Marginals but remain above Copula and LCA (Appendices S and T).

Sensitivity to response history. We replace the up-to-three earlier answers in the prompt with the respondent’s full response history (see footnote 2). RCG falls in all twelve model–dataset pairs, and in two of them (gemma-4-12B on WVS and Llama-3.3-70B on ANES) significantly below Marginals, but every pair remains above Copula (Appendix O).

## 6 Conclusions

We proposed the respondent coverage gap to evaluate silicon sampling at the respondent level. No evaluated LLM achieves human-level coverage on either survey. Under our main settings, all trail a baseline that samples answers independently across questions. Across all tested temperatures, response histories, reasoning efforts, model tiers, and generated sample sizes, every evaluated LLM has a higher RCG than the Gaussian copula baseline at the same generated sample size. Existing perquestion metrics miss this failure. Marginal calibration reduces RCG below that of Marginals, but not that of Copula. Before a synthetic population stands in for a human one, its respondents, not only its answers, should be shown to match.

## Limitations

Scope of the LLM analysis. RCG and OSR are defined only through distances between respondents’ answer profiles, so they do not depend on a domain or population. Our analysis uses only US respondents from two surveys, WVS wave 7 (Haerpfer et al., 2022) and the 2024 wave of ANES (American National Election Studies, 2026). Our findings on LLMs (RCG above one for all six models, and the contraction and offsupport patterns) are thus shown only for these surveys; testing them in other areas requires further experiments. Such areas include psychological experiments (Binz et al., 2025), online purchasing (Lu et al., 2026), and human–computer interaction (Hämäläinen et al., 2023). Non-US and multilingual settings (Ma et al., 2025; Liu et al., 2026) are also outside our analysis.

Evaluating alternative prompting approaches with RCG. We evaluate LLMs with off-theshelf prompting, as practitioners typically use them. Other prompting approaches, such as backstories (Moon et al., 2024), criteria-based prompting (Hayati et al., 2024), Verbalized Sampling (Zhang et al., 2026), and String Seed of Thought (Misaki and Akiba, 2026), are not examined. Training-based methods, such as supervised fine-tuning (Cao et al., 2025; Suh et al., 2025; Binz et al., 2025; Krsteski et al., 2026) and representation-level intervention (Wang et al., 2026), are also used to improve the accuracy and diversity of silicon sampling. These methods target per-question accuracy or response distributions; RCG can test whether their improvements extend to the level of individual respondents.

## Ethical Considerations

Potential risk of silicon sampling. Silicon sampling could misportray or flatten identity groups (Wang et al., 2025) or caricature them (Cheng et al., 2023), which may reinforce stereotypes when synthetic populations replace human respondents. Our metrics, RCG and OSR, can flag patterns consistent with off-support generation and contraction (Section 5.2) at the level of individual respondents. However, the metrics are computed only on the evaluated questions and cells.

Data and terms of use. We use the WVS wave 7 (Haerpfer et al., 2022) and ANES wave

2024 (American National Election Studies, 2026). We follow their terms of use: we use the data only for research, do not redistribute the data files, and do not investigate specific survey respondents.

Use of AI. We used AI assistants to polish the manuscript’s English and assist with experimental code. The authors reviewed and verified all manuscript content and code.

## References

Gati Aher, Rosa I. Arriaga, and Adam Tauman Kalai. 2023. Using large language models to simulate multiple humans and replicate human subject studies. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 337–371. PMLR.

Ahmed M. Alaa, Boris van Breugel, Evgeny Saveliev, and Mihaela van der Schaar. 2022. How faithful is your synthetic data? Sample-level metrics for evaluating and auditing generative models. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 290–306. PMLR.

Badr AlKhamissi, Muhammad ElNokrashy, Mai Alkhamissi, and Mona Diab. 2024. Investigating cultural alignment of large language models. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12404–12422. Association for Computational Linguistics.

American National Election Studies. 2026. ANES Time Series Cumulative Data File. [dataset and documentation]. February 5, 2026 version.

Lisa P Argyle, Ethan C Busby, Nancy Fulda, Joshua R Gubler, Christopher Rytting, and David Wingate. 2023. Out of one, many: Using language models to simulate human samples. Polit. Anal., 31(3):337– 351.

Ashwini Ashokkumar, Luke Hewitt, Isaias Ghezae, and Robb Willer. 2026. Large language models can predict the results of social science experiments. Nature, 656(8126):115–122.

Marcel Binz, Elif Akata, Matthias Bethge, Franziska Brändle, Fred Callaway, Julian Coda-Forno, Peter Dayan, Can Demircan, Maria K Eckstein, Noémi Élteto, Thomas L Griffiths, Susanne Haridi, Ak-˝ shay K Jagadish, Li Ji-An, Alexander Kipnis, Sreejan Kumar, Tobias Ludwig, Marvin Mathony, Marcelo Mattar, and 21 others. 2025. A foundation model to predict and capture human cognition. Nature, 644(8078):1002–1009.

James Bisbee, Joshua D Clinton, Cassy Dorff, Brenton Kenkel, and Jennifer M Larson. 2024. Synthetic replacements for human survey data? The perils of large language models. Polit. Anal., 32(4):401–416.

Alex J Cannon, Stephen R Sobie, and Trevor Q Murdock. 2015. Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? J. Clim., 28(17):6938–6959.

Yong Cao, Haijiang Liu, Arnav Arora, Isabelle Augenstein, Paul Röttger, and Daniel Hershcovich. 2025. Specializing large language models to simulate survey response distributions for global populations. In Proceedings ofthe 2025 Conference ofthe Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3141–3154. Association for Computational Linguistics.

Myra Cheng, Tiziano Piccardi, and Diyi Yang. 2023. CoMPosT: Characterizing and evaluating caricature in LLM simulations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10853–10875. Association for Computational Linguistics.

Ricardo Dominguez-Olmedo, Moritz Hardt, and Celestine Mendler-Dünner. 2024. Questioning the survey responses of large language models. In Advances in Neural Information Processing Systems, volume 37, pages 45850–45878. Curran Associates, Inc.

Esin Durmus, Karina Nguyen, Thomas I. Liao, Nicholas Schiefer, Amanda Askell, Anton Bakhtin, Carol Chen, Zac Hatfield-Dodds, Danny Hernandez, Nicholas Joseph, Liane Lovitt, Sam McCandlish, Orowa Sikder, Alex Tamkin, Janel Thamkul, Jared Kaplan, Jack Clark, and Deep Ganguli. 2024. Towards measuring the representation of subjective global opinions in language models. In First Conference on Language Modeling.

Gemma Team. 2026. Gemma 4 technical report. arXiv preprint arXiv:2607.02770.

Leo A Goodman. 1974. Exploratory latent structure analysis using both identifiable and unidentifiable models. Biometrika, 61(2):215–231.

J C Gower. 1971. A general coefficient of similarity and some of its properties. Biometrics, 27(4):857–871.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, et al. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

L Gudmundsson, J B Bremnes, J E Haugen, and T Engen-Skaugen. 2012. Technical note: Downscaling RCM precipitation to the station scale using statistical transformations – a comparison of methods. Hydrol. Earth Syst. Sci., 16(9):3383–3390.

Christian Haerpfer, Ronald Inglehart, Alejandro Moreno, Christian Welzel, Kseniya Kizilova, Jaime Diez-Medrano, Marta Lagos, Pippa Norris, Eduard Ponarin, and Bi Puranen, editors. 2022. World Values Survey: Round Seven – Country-Pooled Datafile Version 6.0. JD Systems Institute and WVSA Secretariat, Madrid, Spain and Vienna, Austria.

Perttu Hämäläinen, Mikke Tavast, and Anton Kunnari. 2023. Evaluating large language models in generating synthetic HCI research data: A case study. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, pages 1–19. ACM.

Felix Hausdorff. 1914. Grundzüge der Mengenlehre. Veit & Comp., Leipzig.

Shirley Anugrah Hayati, Minhwa Lee, Dheeraj Rajagopal, and Dongyeop Kang. 2024. How far can we extract diverse perspectives from large language models? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5336–5366. Association for Computational Linguistics.

John J Horton, Apostolos Filippas, and Benjamin S Manning. 2023. Large language models as simulated economic agents: What can we learn from Homo Silicus? Working Paper w31122, National Bureau of Economic Research, Cambridge, MA.

Tiancheng Hu, Joachim Baumann, Lorenzo Lupo, Nigel Collier, Dirk Hovy, and Paul Röttger. 2026. Sim-Bench: Benchmarking the ability of large language models to simulate human behaviors. In The Fourteenth International Conference on Learning Representations.

D P Huttenlocher, G A Klanderman, and W J Rucklidge. 1993. Comparing images using the Hausdorff distance. IEEE Trans. Pattern Anal. Mach. Intell., 15(9):850–863.

Stefan Krsteski, Giuseppe Russo, Serina Chang, Robert West, and Kristina Gligoric. 2026. Valid survey simu-´ lations with limited human data: The roles of prompting, fine-tuning, and rectification. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10887–10906. Association for Computational Linguistics.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626. ACM.

Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. 2019. Improved precision and recall metric for assessing generative models. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.

Anton D Lautrup, Tobias Hyrup, Arthur Zimek, and Peter Schneider-Kamp. 2025. SynthEval: a framework for detailed utility and privacy evaluation of tabular synthetic data. Data Min. Knowl. Discov., 39(1):6.

Olivier Ledoit and Michael Wolf. 2004. A wellconditioned estimator for large-dimensional covariance matrices. J. Multivar. Anal., 88(2):365–411.

David A Levin and Yuval Peres. 2017. Markov chains and mixing times, second edition. American Mathematical Society, Providence, RI.

Jindˇrich Libovický. 2026. On the credibility of evaluating LLMs using survey questions. In Proceedings of the First Workshop on Multilingual Multicultural Evaluation, pages 23–34. Association for Computational Linguistics.

Yang Liu, Masahiro Kaneko, and Chenhui Chu. 2026. On the alignment of large language models with global human opinion. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 37673–37681. Association for the Advancement of Artificial Intelligence (AAAI).

Yuxuan Lu, Jing Huang, Yan Han, Bingsheng Yao, Sisong Bei, Yaochen Xie, Yisi Sang, Qi He, and Dakuo Wang. 2026. Can LLM agents simulate multiturn human behavior? Evidence from real online customer behavior data. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 43961– 43977. Association for Computational Linguistics.

Bolei Ma, Berk Yoztyurk, Anna-Carolina Haensch, Xinpeng Wang, Markus Herklotz, Frauke Kreuter, Barbara Plank, and Matthias Aßenmacher. 2025. Algorithmic fidelity of large language models in generating synthetic German public opinions: A case study. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1785–1809. Association for Computational Linguistics.

Xiangyu Ma, Mengmi Zhang, Shannon Ang, and Minne Chen. 2026. Tastes without distinction: silicon samples and the synthetic construction of tastes. arXiv preprint arXiv:2606.30085.

Kou Misaki and Takuya Akiba. 2026. String seed of thought: Prompting LLMs for distribution-faithful and diverse generation. In The Fourteenth International Conference on Learning Representations.

Suhong Moon, Marwa Abdulhai, Minwoo Kang, Joseph Suh, Widyadewi Soedarmadji, Eran Kohen Behar, and David M. Chan. 2024. Virtual personas for language models via an anthology of backstories. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 19864–19897. Association for Computational Linguistics.

Sacha Morin, Robin Legault, Félix Laliberté, Zsuzsa Bakk, Charles-Édouard Giguère, Roxane de la Sablonnière, and Éric Lacourse. 2025. StepMix: A Python package for pseudo-likelihood estimation of generalized mixture models with external variables. J. Stat. Softw., 113(8):1–39.

Muhammad Ferjad Naeem, Seong Joon Oh, Youngjung Uh, Yunjey Choi, and Jaejun Yoo. 2020. Reliable fidelity and diversity metrics for generative models.

In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 7176–7185. PMLR.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology, pages 1–22. ACM.

Joon Sung Park, Carolyn Q Zou, Jonne Kamphorst, Niles Egan, Aaron Shaw, Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Percy Liang, Robb Willer, and Michael S Bernstein. 2026. LLM agents grounded in self-reports enable generalpurpose simulation of individuals. arXiv preprint arXiv:2411.10109.

Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. 2021. MAUVE: Measuring the gap between neural text and human text using divergence frontiers. In Advances in Neural Information Processing Systems, volume 34, pages 4816–4828. Curran Associates, Inc.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents. https://qwen.ai/blog?id=qwen3.5.

Mehdi S. M. Sajjadi, Olivier Bachem, Mario Lucic, Olivier Bousquet, and Sylvain Gelly. 2018. Assessing generative models via precision and recall. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc.

Filippo Santambrogio. 2015. Optimal transport for applied mathematicians: Calculus ofvariations, PDEs, and modeling, volume 87 of Progress in Nonlinear Differential Equations and Their Applications. Birkhäuser, Cham, Switzerland.

Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. 2023. Whose opinions do language models reflect? In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 29971–30004. PMLR.

Murray Shanahan, Kyle McDonell, and Laria Reynolds. 2023. Role play with large language models. Nature, 623(7987):493–498.

Abe Sklar. 1959. Fonctions de répartition à n dimensions et leurs marges. Publications de l’Institut de Statistique de l’Université de Paris, 8:229–231.

Joseph Suh, Erfan Jahanparast, Suhong Moon, Minwoo Kang, and Serina Chang. 2025. Language model fine-tuning on scaled survey data for predicting distributions of public opinions. In Proceedings of the

63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 21147–21170. Association for Computational Linguistics.

Erika Elizabeth Taday Morocho, Lorenzo Cima, Tiziano Fagni, Marco Avvenuti, and Stefano Cresci. 2026. Assessing the reliability of persona-conditioned LLMs as synthetic survey respondents. In Companion Proceedings of the ACM Web Conference 2026, pages 320–329. ACM.

Angelina Wang, Jamie Morgenstern, and John P Dickerson. 2025. Large language models that replace human participants can harmfully misportray and flatten identity groups. Nat. Mach. Intell., 7(3):400–411.

Hexi Wang, Yujia Zhou, Bangde Du, Qingyao Ai, and Yiqun Liu. 2026. Parametric social identity injection and diversification in public opinion simulation. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, pages 4894–4905. ACM.

Noah Wang, Z.Y. Peng, Haoran Que, Jiaheng Liu, Wangchunshu Zhou, Yuhan Wu, Hongcheng Guo, Ruitong Gan, Zehao Ni, Jian Yang, Man Zhang, Zhaoxiang Zhang, Wanli Ouyang, Ke Xu, Wenhao Huang, Jie Fu, and Junran Peng. 2024. RoleLLM: Benchmarking, eliciting, and enhancing role-playing abilities of large language models. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 14743–14777. Association for Computational Linguistics.

Tristan Williams, Franziska Weeber, Sebastian Padó, and Alan Akbik. 2026. Beyond marginal distributions: A framework to evaluate the representativeness of demographic-aligned LLMs. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 4806–4826. Association for Computational Linguistics.

Jiayi Zhang, Simon Yu, Derek Chong, Anthony Sicilia, Michael Tomz, Christopher D. Manning, and Weiyan Shi. 2026. Verbalized sampling: How to mitigate mode collapse and unlock LLM diversity. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 160073–160152. PMLR.

Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason Weston. 2018. Personalizing dialogue agents: I have a dog, do you have pets too? In Proceedings ofthe 56th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2204–2213. Association for Computational Linguistics.

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2024. Large language models are not robust multiple choice selectors. In The Twelfth International Conference on Learning Representations.

i2 (Moon et al., 2024): Below you will be   
asked to provide a short description of your   
demographic information, and then answer some   
questions.

<table><tr><td>Attribute</td><td>WVS</td><td>ANES</td></tr><tr><td>sex</td><td>2</td><td>2</td></tr><tr><td>age (10-year bins; 80+ top-coded)</td><td>6</td><td>7</td></tr><tr><td>education</td><td>6</td><td>5</td></tr><tr><td>employment status</td><td>6</td><td>4</td></tr><tr><td>income level</td><td>8</td><td>5</td></tr><tr><td>marital status</td><td>4</td><td>5</td></tr><tr><td>religion</td><td>3</td><td>4</td></tr><tr><td>immigrant status</td><td>2</td><td></td></tr><tr><td>citizenship</td><td>1</td><td></td></tr><tr><td>chief wage earner</td><td>2</td><td></td></tr><tr><td>family savings</td><td>4</td><td></td></tr><tr><td>social class</td><td>4</td><td></td></tr><tr><td>occupational group</td><td>6</td><td></td></tr><tr><td>sector of employment</td><td>3</td><td></td></tr><tr><td>household size</td><td>6</td><td></td></tr><tr><td>number of children (0, 1, 2, 3, 4+)</td><td>5</td><td></td></tr><tr><td>living with parents</td><td>2</td><td></td></tr><tr><td>race</td><td></td><td>5</td></tr><tr><td>Hispanic origin</td><td></td><td>3</td></tr><tr><td>party identification (7-point)</td><td></td><td>7</td></tr><tr><td>liberal-conservative ideology</td><td></td><td>8</td></tr><tr><td>Number of attributes</td><td>17</td><td>11</td></tr></table>

Table 3: Demographic attributes and their numbers of levels.

## A Details of the Experiment in the Introduction

In Section 1, we use the same 2,594 respondents of WVS (Haerpfer et al., 2022) and 100 questions as in the main text (Section 4.1). We fit Marginals and Copula (Section 4.6) on all 2,594 human respondents and generate 2,594 respondents from each for the scatter plots and the nearest-neighbor distances; only the bar chart uses 50,000 generated respondents per generator. For the top panels of Figure 1, we apply principal coordinates analysis (PCoA) to the Gower distance matrix of the pooled 7,782 points (Human, Marginals, and Copula), treating the Gower distance as a squared distance. We adopt the first two principal coordinates as the axes and plot 100 randomly chosen points per group. In the bottom-right panel, the curves are kernel density estimates of the Gower distance from each human respondent to the nearest other human respondent (Human) and to the nearest generated respondent (Marginals and Copula), as defined in Section 3.2.

## B Demographic Attributes

Table 3 shows demographic attributes and their numbers of levels.<sup>14</sup> We bin age and the number of children as shown in the table and keep the original categories of the other attributes. As we described in Section 4.1, we treat party identification and liberal–conservative ideology as demographic attributes for ANES. Ideology has eight levels because we keep “DK; haven’t thought much about it” as a level in addition to the seven-point scale.

## C Prompts

Each query is a single user message with four blocks separated by blank lines: an instruction, a persona description, few-shot examples, and a question block. The instruction and the persona description are each drawn from three variants; the variant pair and the order of attributes in the persona description are assigned to each generated respondent at random and kept fixed across its 100 questions. Each generated respondent answers its 100 questions in a random order. The few-shot examples carry up to three of the respondent’s own previous answers, drawn at random without replacement from the questions it has already answered (Section 4.3). Figure 5 shows two full prompts from the main runs.

Instructions. We use three instruction texts from prior work, with the adjustments described below. i1 and i2 are placed at the top of the prompt; i3 is placed between the few-shot examples and the question block. In i1, the survey name is “World Values Survey (WVS)” for WVS and “American National Election Studies (ANES) survey” for ANES. For i3, we keep only the first two sentences of the original; the remaining ones specify an answer format, which our question block replaces.

## Instruction Variants

i1 (Taday Morocho et al., 2026): You are a survey respondent. Answer the following World Values Survey (WVS) question from the point of view of the given persona.

i3 (AlKhamissi et al., 2024): Answer the following question from this perspective. Others will read what you choose; your goal is to convince them it was chosen from the perspective of the persona described above.

Persona descriptions. We use three formats: an interview transcript (QA; Argyle et al., 2023), a second-person description (Bisbee et al., 2024), and a first-person description (Argyle et al., 2023; Santurkar et al., 2023; Moon et al., 2024). Residence in the United States is added to every persona as an additional attribute. Every persona description starts with the survey year (“It is 2017.” for WVS and “It is 2024.” for ANES), following Bisbee et al. (2024). The following are the three formats for one cell (sex, immigrant status, and age) with the same attribute order.

![](images/59e90d952d3f3b4db6dbf970f5fffa2ac1ec17bdcbba723d983280022f636d6d.jpg)  
Figure 5: Two full prompts from the main run (WVS). Left: a multiple-choice question with the QA persona format and instruction i1. Right: a rating-scale question with the second-person persona format and instruction i2. Both personas are conditioned on sex, immigrant status, and age, and both prompts carry three few-shot examples taken from the same generated respondent’s previous answers. Texts are reproduced verbatim, including the original WVS question wording. The choice order in the left example is the shuffled order shown to the model.

Answer the following question with a single   
integer.   
Answer with an integer between {min} ({label   
of min}) and {max} ({label of max}).   
A larger number means closer to "{label of   
max}".   
If you cannot decide, answer {code for don't   
know}.   
Output ONLY the answer tag in the form   
<ans>N</ans> (for example, <ans>5</ans>). Do   
NOT include any explanation, justification,   
reasoning, or surrounding text.   
# Question   
{question text}   
Respond with only <ans>N</ans>. No other text.

## QA Format

It is 2017.   
Interviewer: What is your sex?   
Me: Male   
Interviewer: In what country do you live?   
Me: The United States   
Interviewer: Were you born in this country or   
are you an immigrant to this country?   
Me: Born in this country   
Interviewer: What is your age in years?   
Me: between 40 and 49

## Second-Person Format

Description: It is 2017. Your sex is Male. You live in the United States. You were born in the United States. You are between 40 and 49 years old.

## First-Person Format

Description: It is 2017. My sex is Male. I live in the United States. I was born in the United States. I am between 40 and 49 years old.

Few-shot examples. The few-shot examples follow the persona format. The {answer} slot holds the text of the selected option for multiple-choice questions and the number for rating scales, including the don’t-know code; unparsable answers are skipped. Line breaks inside a question text are collapsed into single spaces in few-shot examples. This use of earlier answers as context follows Moon et al. (2024).

Few-Shot Example Formats   
Interviewer: {question}   
Me: {answer}   
You answered "{answer}" to the question   
"{question}".   
I answered "{answer}" to the question   
"{question}".

Question block. The question block uses a fixed template around the question and its answer choices. For multiple-choice questions, the numeric labels (1, 2, . . . , K) remain fixed. For each respondent–question pair, we shuffle the answer texts if the choices have no natural order; otherwise, we preserve their natural order.

## Multiple-Choice Question Template

Answer the following question with a single   
choice.   
Choose one of the following numbers (1 2 ...   
K).   
Output ONLY the answer tag in the form   
<ans>N</ans> (for example, <ans>1</ans>). Do   
NOT include any explanation, justification,   
reasoning, or surrounding text.   
# Question   
{question text}   
# Choices   
- 1: {answer text shown at position 1}   
- K: {answer text shown at position K}   
Respond with only <ans>N</ans>. No other text.

## Rating-Scale Question Template

## D Building Cells

As described in Section 3.1, generation is conditioned on a set of demographic attributes. We build these sets from the V attribute variables of each dataset (Section 4.1) in three steps. (i) We fix the number of conditioning attributes L and enumerate all  <sup>V</sup> subsets of the variables. For L = 2 and the variables sex, age, and marital status, the subsets are (sex, age), (sex, marital status), and (age, marital status). (ii) For each subset, every combination of attribute values observed in the data defines a cell, e.g., (female, 30–39). We keep a cell only if it contains at least 100 human respondents and none of its values is a missing code; the threshold matches the 100 respondents we generate per cell. When a cell has more than 100 human respondents, we repeatedly draw 100 of them at random, so that Section 3.2 always compares sets of equal size. (iii) For each L, we sample 100 cells uniformly at random from the retained cells, or take all of them when fewer than 100 remain. All models are evaluated on the same cells. This yields 730 cells for WVS and 603 for ANES. Fewer than 100 cells remain at $L = 0$ (one cell), at $L = 1$ (70 and 55), and at the deepest levels (WVS: 59 at L = 8; ANES: 46 at $L = 7$ and one at $L = 8 ) ;$ every other level contributes 100 cells.<sup>15</sup> In total, this amounts to 13.33 million queries per model (one per respondent-question pair).

All respondents in both datasets reside in the United States (Section 4.1), so every cell also includes this residence as an attribute. Each cell c gives one pair $( \mathcal { H } _ { c } , \mathcal { G } _ { c } ) \colon \mathcal { H } _ { c }$ is the set of answer vectors of the human respondents in the cell, and $\mathcal { G } _ { c }$ is the set of 100 answer vectors generated with the cell’s attribute values in the prompt (Section 4.3). All metrics in Section 3.2 are computed per cell.

## E LLM Settings and Computational Cost

All queries to the proprietary models were issued between September 20 and October 1, 2026 (UTC). We leave the sampling parameters (temperature, top\_p, and top\_k where available) at their vendor defaults and set only the reasoning effort to the lowest level available: none for the OpenAI models and minimal for the Google models.

Open-weight models are served with vLLM 0.26.0 (Kwon et al., 2023) in bfloat16 on eight NVIDIA H200 GPUs. We send no sampling parameters, so each model runs with the values shipped in its generation config (or vLLM’s defaults when none are shipped), and we set reasoning to the lowest level available (off for Qwen3.5-9B and gemma-4-12B, low for gpt-oss-20b).

Table 4 lists the six LLMs and the decoding settings that their defaults resolved to (Section 4.4). For gemini-flash-lite, Vertex AI does not expose its defaults, and we assume the values published for the same model in AI Studio (†). The exact identifiers are gemini-3.1-flash-lite for gemini-flash-lite and, for the open-weight models, Qwen/Qwen3.5-9B, google/gemma-4-12B-it, openai/gpt-oss-20b, and meta-llama/Llama-3.3-70B-Instruct. Every query is a single user message without a system prompt, with max\_tokens of 4,096.

Computational cost. For the runs reported in this paper, we spent approximately \$11,160 on the proprietary models (OpenAI \$4,140 and Google \$7,020), and approximately 1,960 H200 GPUhours (about 245 hours on eight GPUs) on the open-weight models.

## F Gower Distance

For two answer vectors u and v over the $M = 1 0 0$ questions, the Gower distance is

$$
d ( \pmb { u } , \pmb { v } ) = \frac { \sum _ { m = 1 } ^ { M } w _ { m } d _ { m } ( \ b { u } _ { m } , \pmb { v } _ { m } ) } { \sum _ { m = 1 } ^ { M } w _ { m } } ,\tag{4}
$$

where $d _ { m }$ depends on the type of question $m .$ For numeric and ordinal questions, $d _ { m } = | u _ { m } - \rrangle$ $v _ { m } | / R _ { m }$ , where $R _ { m }$ is the range of the answer scale of question m: the maximum minus the minimum for rating scales, and the number of choices minus one for ordinal questions, whose answers are coded by their position in the choice order. For nominal questions, $d _ { m } = \mathbf { 1 } [ u _ { m } \neq v _ { m } ]$ . The weight $w _ { m }$ is 1 if both respondents answered question m and 0 otherwise, so the distance is an average over the questions that both respondents answered. We treat as missing the don’t-know code offered for rating scales (99 in WVS and −1 in ANES), the don’t-know and missing codes of the survey data, and any value outside the presented choices or scale.

Rating scales are treated as numeric, and multiple-choice questions as ordinal when their choices have a natural order and as nominal otherwise, giving 30, 45, and 25 questions of the three types in WVS and 32, 36, and 32 in ANES. We check the type of every question by hand, reading its wording and choices. In WVS, we treat the nine agree–disagree questions that offer “Hard to say” as nominal, because “Hard to say” has no position on the agreement scale.

Near-empty respondents. A respondent who answered very few questions is at distance zero from everyone who gave the same answers to those questions and can be the nearest neighbor of many respondents. We therefore use only human respondents with at least 10 valid answers (Section 4.1); this removes 2 WVS and 22 ANES respondents from 6 and 42 cells, and no cell falls below 100 human respondents.

<table><tr><td>Model</td><td>Weights / access</td><td>Params</td><td>Temp.</td><td>top-p</td><td>top-k</td><td>Reasoning</td><td>Run dates (2026, UTC)</td></tr><tr><td>gpt-5.4-nano</td><td>OpenAI API</td><td></td><td>1.0</td><td>1.0</td><td></td><td>effort none</td><td>Sep 20</td></tr><tr><td>gemini-flash-lite</td><td>Google Vertex AI</td><td></td><td>1.0</td><td>0.95†</td><td> $6 4 ^ { \dagger }$ </td><td>thinking minimal</td><td>Sep 20</td></tr><tr><td>Qwen3.5-9B</td><td>open weights, vLLM</td><td>9B</td><td>1.0</td><td>1.0</td><td></td><td>thinking off</td><td>Sep 20</td></tr><tr><td>gemma-4-12B</td><td>open weights, vLLM</td><td>12B</td><td>1.0</td><td>0.95</td><td>64</td><td>thinking off</td><td>Sep 20 – 21</td></tr><tr><td>gpt-oss-20b</td><td>open weights, vLLM</td><td>20B</td><td>1.0</td><td>1.0</td><td></td><td>effort low</td><td>Sep 20</td></tr><tr><td>Llama-3.3-70B</td><td>open weights, vLLM</td><td>70B</td><td>0.6</td><td>0.9</td><td></td><td></td><td>Sep 21 – 24</td></tr></table>

Table 4: Decoding settings of the six LLMs in the main experiment. –: not applicable or not disclosed. †: assumed (see text).

## G Ablation Settings

The ablations in Section 5.5 share the cell subset described there (up to five cells per level of L; 41 cells for WVS and 37 for ANES). The six models and the three baselines of the main experiment are not rerun; their metrics are recomputed on this subset from the original responses. The values therefore differ slightly from Table 1. Unless stated otherwise, all settings other than the varied one are those of the main experiment (Appendix E).

Temperature. For the API models we send the temperature and top-p explicitly: gpt-5.4-nano at $t ~ \in ~ \{ 0 . 1 , 0 . 5 , 1 . 0 , 1 . 5 \}$ and gemini-flash-lite at $t \in \{ 0 . 1 , 1 . 0 , 1 . 5 \}$ , both with ${ \mathrm { t o p } } { \cdot } p = 1 . 0$ . For gemini-flash-lite we also rerun its default $( t = 1 . 0 ,$ $\mathrm { t o p } { - } p = 0 . 9 5 )$ and confirm that the explicit setting reproduces the main run (RCG and OSR differ by at most 0.006, about as much as between two runs of the same setting). For the open-weight models we disable the released generation configurations and use the plain vLLM sampler $( \mathrm { t o p } { - } p = 1 . 0 )$ no top-k) at $t \in \{ 0 . 1 , 0 . 5 , 1 . 0 , 1 . 5 \}$ for Qwen3.5- 9B, gemma-4-12B, and Llama-3.3-70B, and at $t \in \{ 0 . 1 , 1 . 0 , 1 . 5 \}$ for gpt-oss-20b. All temperature runs are therefore untruncated except those of gemini-flash-lite, which keep its default top-k.

Reasoning effort. We vary the reasoning effort of gpt-5.4-nano at low, medium, and high (default none) and of gpt-oss-20b at medium and high (default low). Temperature and top-p stay at their defaults, and max\_tokens is raised to 16,384 for gpt-5.4-nano and 7,168 for gpt-oss-20b to leave room for reasoning tokens.

Model tier. We add gpt-5.4 and gemini-3.1-propreview (gemini-pro). At minimal, the thinking level we use for both Google models, gemini-pro still uses about 260 thinking tokens per call on average, whereas gemini-flash-lite uses none.

## H Sensitivity to Percentile Choices

All six LLMs have cell-averaged RCG above one at every tested percentile. We write the percentile in RCG and OSR as a fraction q; the main text uses $q = 0 . 9 5$ , the 95th percentile. The choice of q determines which part of the nearest-neighbor distance distribution is emphasized. $\operatorname { A t } q = 1 . 0 0 .$ the metric uses the maximum distance; at $q = 0 . 9 5$ it uses the 95th percentile and is less sensitive to the largest distances.

Table 5 and Table 6 show RCG and OSR for q from 0.50 to 1.00. Our main result, that every LLM has a higher RCG than Marginals (Table 1), holds in both datasets at $q = 0 . 9 0 , 0 . 9 5 , 0 . 9 9 ,$ and 1.00. At $q \leq 0 . 7 5$ , only gemini-flash-lite, gemma-4-12B, and Llama-3.3-70B have lower RCG than Marginals, in three model–dataset pairs at $q = 0 . 7 5$ and five at $q = 0 . 5 0$ . These three models show the contraction pattern (Section 5.2): their generated respondents stay close to some human respondents while leaving others poorly covered. A small q ignores the human respondents farthest from the generated ones, which for these models are the poorly covered ones, so their RCG approaches or falls below that of Marginals. OSR decreases with q for every generator, but the two groups of LLMs in Section 5.2 do not overlap at any q.

## I Baseline Fitting

Copula transforms the answers to each question into normal scores using the ranks of their codes (tied values are assigned average ranks), estimates the correlation matrix of these scores with Ledoit– Wolf shrinkage, and maps sampled scores back to codes through the standard normal cumulative distribution function and the empirical quantile function of each question. Nominal questions go through the same steps, so their codes are ranked in numerical order although this order has no meaning. For these questions, Copula preserves the fitted marginal answer distributions, but the modeled dependence on other questions depends on this arbitrary ordering. LCA models every question as categorical and does not use the order of the codes.

<table><tr><td rowspan="2"></td><td colspan="6">WVS</td><td colspan="6">ANES</td></tr><tr><td>0.50</td><td>0.75</td><td>0.90</td><td>0.95</td><td>0.99</td><td>1.00</td><td>0.50</td><td>0.75</td><td>0.90</td><td>0.95</td><td>0.99</td><td>1.00</td></tr><tr><td>gpt-5.4-nano</td><td>1.325</td><td>1.312</td><td>1.290</td><td>1.272</td><td>1.256</td><td>1.247</td><td>1.535</td><td>1.463</td><td>1.404</td><td>1.370</td><td>1.317</td><td>1.282</td></tr><tr><td>gemini-flash-lite</td><td>1.157</td><td>1.185</td><td>1.210</td><td>1.218</td><td>1.230</td><td>1.231</td><td>1.293</td><td>1.331</td><td>1.346</td><td>1.353</td><td>1.361</td><td>1.359</td></tr><tr><td>Qwen3.5-9B</td><td>1.311</td><td>1.307</td><td>1.290</td><td>1.273</td><td>1.255</td><td>1.237</td><td>1.510</td><td>1.436</td><td>1.377</td><td>1.347</td><td>1.296</td><td>1.259</td></tr><tr><td>gemma-4-12B</td><td>1.144</td><td>1.167</td><td>1.188</td><td>1.199</td><td>1.210</td><td>1.204</td><td>1.370</td><td>1.356</td><td>1.342</td><td>1.332</td><td>1.321</td><td>1.306</td></tr><tr><td>gpt-oss-20b</td><td>1.318</td><td>1.307</td><td>1.287</td><td>1.269</td><td>1.249</td><td>1.232</td><td>1.481</td><td>1.422</td><td>1.372</td><td>1.346</td><td>1.311</td><td>1.284</td></tr><tr><td>Llama-3.3-70B</td><td>1.205</td><td>1.247</td><td>1.267</td><td>1.275</td><td>1.290</td><td>1.287</td><td>1.275</td><td>1.278</td><td>1.277</td><td>1.277</td><td>1.281</td><td>1.277</td></tr><tr><td>Marginals</td><td>1.229</td><td>1.194</td><td>1.168</td><td>1.158</td><td>1.150</td><td>1.140</td><td>1.353</td><td>1.294</td><td>1.261</td><td>1.250</td><td>1.234</td><td>1.219</td></tr><tr><td>Copula</td><td>1.082</td><td>1.061</td><td>1.046</td><td>1.041</td><td>1.034</td><td>1.036</td><td>1.099</td><td>1.091</td><td>1.089</td><td>1.091</td><td>1.098</td><td>1.102</td></tr><tr><td>LCA</td><td>1.145</td><td>1.124</td><td>1.109</td><td>1.105</td><td>1.103</td><td>1.108</td><td>1.156</td><td>1.145</td><td>1.142</td><td>1.143</td><td>1.151</td><td>1.154</td></tr></table>

Table 5: RCGs at various percentiles $q , \ \frac { d _ { \mathcal { H }  \mathcal { G } } ^ { q } } { d _ { \mathcal { H }  \mathcal { H } } ^ { q } }$
<table><tr><td rowspan="2"></td><td colspan="6">WVS</td><td colspan="6">ANES</td></tr><tr><td>0.50</td><td>0.75</td><td>0.90</td><td>0.95</td><td>0.99</td><td>1.00</td><td>0.50</td><td>0.75</td><td>0.90</td><td>0.95</td><td>0.99</td><td>1.00</td></tr><tr><td>gpt-5.4-nano</td><td>1.185</td><td>1.117</td><td>1.043</td><td>0.994</td><td>0.920</td><td>0.873</td><td>1.415</td><td>1.313</td><td>1.224</td><td>1.173</td><td>1.094</td><td>1.046</td></tr><tr><td>gemini-flash-lite</td><td>0.929</td><td>0.898</td><td>0.867</td><td>0.840</td><td>0.796</td><td>0.764</td><td>0.966</td><td>0.912</td><td>0.865</td><td>0.838</td><td>0.794</td><td>0.766</td></tr><tr><td>Qwen3.5-9B</td><td>1.171</td><td>1.106</td><td>1.038</td><td>0.992</td><td>0.921</td><td>0.877</td><td>1.442</td><td>1.347</td><td>1.262</td><td>1.213</td><td>1.135</td><td>1.090</td></tr><tr><td>gemma-4-12B</td><td>0.937</td><td>0.884</td><td>0.829</td><td>0.793</td><td>0.738</td><td>0.704</td><td>1.123</td><td>1.047</td><td>0.980</td><td>0.942</td><td>0.884</td><td>0.848</td></tr><tr><td>gpt-oss-20b</td><td>1.181</td><td>1.122</td><td>1.057</td><td>1.013</td><td>0.949</td><td>0.909</td><td>1.369</td><td>1.278</td><td>1.194</td><td>1.144</td><td>1.066</td><td>1.016</td></tr><tr><td>Llama-3.3-70B</td><td>0.865</td><td>0.810</td><td>0.756</td><td>0.722</td><td>0.672</td><td>0.642</td><td>1.023</td><td>0.951</td><td>0.889</td><td>0.853</td><td>0.801</td><td>0.770</td></tr><tr><td>Marginals</td><td>1.192</td><td>1.118</td><td>1.040</td><td>0.989</td><td>0.911</td><td>0.862</td><td>1.307</td><td>1.213</td><td>1.127</td><td>1.078</td><td>1.002</td><td>0.956</td></tr><tr><td>Copula</td><td>1.079</td><td>1.034</td><td>0.982</td><td>0.946</td><td>0.887</td><td>0.850</td><td>1.072</td><td>1.047</td><td>1.005</td><td>0.976</td><td>0.926</td><td>0.893</td></tr><tr><td>LCA</td><td>1.112</td><td>1.066</td><td>1.003</td><td>0.959</td><td>0.889</td><td>0.844</td><td>1.074</td><td>1.037</td><td>0.986</td><td>0.950</td><td>0.892</td><td>0.854</td></tr></table>

Table 6: OSRs at various percentiles q, $\frac { d _ { \mathcal { G }  \mathcal { H } } ^ { q } } { d _ { \mathcal { H }  \mathcal { H } } ^ { q } }$

<table><tr><td></td><td>WVS ANES</td></tr><tr><td>In-cell Hold-out</td><td>In-cell Hold-out</td></tr><tr><td>gpt-5.4-nano gemini-flash-lite</td><td>1.261 1.354 1.221 1.345</td></tr><tr><td>Qwen3.5-9B 1.260 gemma-4-12B 1.186 gpt-oss-20b 1.255 1.264</td><td>1.326 1.310 1.317</td></tr><tr><td>Llama-3.3-70B</td><td>1.273</td></tr><tr><td>Marginals 1.156</td><td>1.165 1.263 1.272</td></tr><tr><td>Copula 1.047 LCA 1.108</td><td>1.088 1.120 1.109 1.128 1.168 1.151</td></tr></table>

Table 7: RCG under the In-cell and Hold-out settings. The LLMs are not fitted, so each LLM has one value for both settings.

Copula and LCA are fitted to the respondents who answer every question. In 147 of the 603 ANES cells, and in no WVS cell, fewer than 50 respondents do so; there we fill each missing answer with the median of the codes observed for its question in the cell (a placeholder with no meaning for nominal questions) and fit to all respondents.

In Section 5.1, we fit the baselines to the human respondents of each cell, from which the 100 evaluated respondents are also drawn. To check whether the comparison between the LLMs and the baselines depends on this, we compare two settings on the cells with at least 200 human respondents (203 of the 730 cells for WVS and 196 of the 603 cells for ANES): In-cell fits the baselines as in Section 5.1; Hold-out fits them, in each of the 50 draws, to 100 human respondents drawn from H<sub>c</sub> without overlap with the 100 evaluation respondents. Both settings use the same draws of the evaluated respondents.

Table 7 shows the results. Compared with In-cell, Hold-out raises the RCG of Marginals by 0.008 on WVS and 0.009 on ANES, and that of Copula and LCA on WVS by 0.040 and 0.020. On ANES, Copula and LCA have lower RCG under Hold-out than under In-cell because 84% of their 100-respondent fitting samples require the medianimputation procedure described above. This procedure allows the models to be fitted to all 100 respondents rather than only those who answer every question. Under Hold-out, every LLM still has a higher RCG than every baseline, except that on ANES Llama-3.3-70B and Marginals have almost the same RCG (1.273 and 1.272). Under In-cell, the RCG of Llama-3.3-70B is 0.010 above that of Marginals on the 196 ANES cells used here, and 0.028 above it on all 603 ANES cells (Table 8).

<table><tr><td rowspan="2"></td><td colspan="2">RCG — Marginals</td></tr><tr><td>WVS</td><td>ANES</td></tr><tr><td>gpt-5.4-nano</td><td>+0.114</td><td>+0.120</td></tr><tr><td rowspan="2">gemini-flash-lite</td><td>[+0.111,+0.117]</td><td>[+0.110,+0.130]</td></tr><tr><td>+0.060</td><td>+0.103</td></tr><tr><td rowspan="2">Qwen3.5-9B</td><td>[+0.055, +0.065]</td><td>[+0.088,+0.120]</td></tr><tr><td>+0.115</td><td>+0.097</td></tr><tr><td rowspan="2">gemma-4-12B</td><td>[+0.112,+0.119]</td><td>[+0.087, +0.108]</td></tr><tr><td>+0.041 [+0.038, +0.045]</td><td>+0.083</td></tr><tr><td rowspan="2">gpt-oss-20b</td><td>+0.111</td><td>[+0.067,+0.100]</td></tr><tr><td>[+0.107, +0.115]</td><td>+0.096</td></tr><tr><td rowspan="2">Llama-3.3-70B</td><td>+0.116</td><td>[+0.082, +0.111] +0.028</td></tr><tr><td>[+0.112, +0.120]</td><td>[+0.017,+0.039]</td></tr><tr><td>Copula</td><td>-0.118</td><td>-0.159</td></tr><tr><td rowspan="2">LCA</td><td>[−0.121, -0.115]</td><td>[−0.172, -0.147]</td></tr><tr><td>-0.053 [-0.056,-0.051]</td><td>-0.106 [-0.118,-0.095]</td></tr></table>

Table 8: Cell-wise differences of the uncalibrated RCG (RCG in Table 1) from Marginals, averaged over cells. Brackets show the 95% CI from 10,000 bootstrap resamples of the sets of conditioning attributes, each with all of its cells. Every uncalibrated LLM is significantly above Marginals on both datasets.

![](images/7e418776dfda5d00ae3ef1da0ca2bf86b74d5d5ecdd1f704e6b6e40e91b12313.jpg)  
Figure 6: RCG on WVS by the number of conditioning attributes L. Each point is the mean over the cells at that L (cell counts below the axis; hollow markers denote a single cell). Triangles are LLMs and circles are Marginals; the gray horizontal line marks RCG = 1.

## J 95% Confidence Intervals for the Main Results

Table 8 and Table 9 show the 95% confidence intervals of Section 5.1.

## K RCG by L

Figure 6 and Figure 7 show RCG on WVS and ANES by the number of conditioning attributes L. At every value of L on WVS, all six LLMs have a higher RCG than Marginals, and all series stay above 1. On WVS, cells at different values of L represent different subpopulations, whose median size decreases from 2,594 respondents at $L = 0$ to 114 at $L = 8$ . We therefore do not interpret the increase in RCG with L as the effect of adding an attribute. Overall, on WVS the gap to Marginals is present at every conditioning depth we test, from the unconditioned population (L = 0) to eight attributes; on ANES, it is present from one to six attributes.

![](images/d9f20f3fb6470901fa61e244b7e092e25e02ede409ec5eeb6817c93f8eddb3a3.jpg)  
Figure 7: RCG on ANES by the number of conditioning attributes L. Each point is the mean over the cells at that L (cell counts below the axis; hollow markers denote a single cell). Triangles are LLMs and circles are Marginals; the gray horizontal line marks RCG = 1.

## L Prior Metrics and RCG

Table 10 lists JSD, 1−ρ, and RCG for all generators on both datasets.

Figure 8 shows mean per-question JSD against RCG for the baselines and calibrated LLMs on ANES.

Figure 9 shows correlation-structure mismatch $( 1 - \rho )$ against RCG for all nine generators on ANES.

## M Proof of Proposition 1

The correlation-structure score of Williams et al. (2026) starts from a set of matrices $\{ X _ { c } \} _ { c \in { \mathcal C } }$ with $\boldsymbol { X _ { c } } \in \mathbb { R } ^ { n _ { c } \times M }$ , where C is the set of cells; we denote by $X _ { c , i , m }$ the answer of respondent i in cell c to $q _ { m }$ $i \in \mathcal { T } _ { c } , | \mathcal { T } _ { c } | = n _ { c }$ . Let $X _ { c } ^ { \mathcal { \bar { H } } }$ be the matrix whose rows are the answer vectors in $\mathcal { H } _ { c } ,$ and $\chi ^ { \mathcal { H } }$ $\{ X _ { c } ^ { \mathcal { H } } \} _ { c \in \mathcal { C } } ;$ we define $\mathcal { X } ^ { \mathcal { G } }$ from $\{ \mathcal { G } _ { c } \} _ { c }$ <sub>c</sub> in the same way, and likewise $\mathcal { X } ^ { \mathcal { G } ^ { \prime } }$ and $\chi \mathcal { H } ^ { \prime }$ below.<sup>16</sup>

The score of Williams et al. (2026) is computed in three steps. (i) For each cell c and each of the $M _ { \mathrm { o } }$ rating-scale or ordinal questions $q _ { m }$ , let $A _ { c , m } ( \mathcal { X } )$ be the mean answer to $q _ { m }$ among the respondents of cell c who answered it, so that $A ( \mathbf { \hat { \mathcal { X } } } ) \in \mathbb { R } ^ { | \mathcal { C } | \times M _ { \mathrm { o } } } . ^ { 1 7 }$ (ii) Let $R ( \mathcal { X } ) \in \mathbb { R } ^ { M _ { \mathrm { o } } \times M _ { \mathrm { o } } }$ be the matrix of Pearson correlations between the columns of $A ( \mathcal { X } )$ , that is, between questions across cells, and let $\mathbf { s } ( \mathcal { X } ) \in \mathbb { R } ^ { \frac { M _ { \mathrm { o } } ( M _ { \mathrm { o } } - 1 ) } { 2 } }$ be the vector of its strictly upper-triangular entries. (iii) The score is the Pearson correlation $\rho = \mathrm { c o r r } \big ( \mathbf { s } ( \mathcal { X } ^ { \mathcal { H } } ) , \mathbf { s } ( \mathcal { X } ^ { \mathcal { G } } ) \big )$

<table><tr><td rowspan="2"></td><td colspan="3">WVS</td><td colspan="3">ANES</td></tr><tr><td>- Marginals</td><td>Copula</td><td>-LCA</td><td>– Marginals</td><td>Copula</td><td>-LCA</td></tr><tr><td>gpt-5.4-nano</td><td>-0.008 [-0.009,-0.007]</td><td>+0.110 [+0.107,+0.113]</td><td>+0.045 [+0.043, +0.048]</td><td>-0.018 [−0.020, -0.016]</td><td>+0.141 [+0.130,+0.153]</td><td>+0.089 [+0.078, +0.099]</td></tr><tr><td>gemini-flash-lite</td><td>-0.032 [-0.034,-0.031]</td><td>+0.085 [+0.083, +0.088]</td><td>+0.021 [+0.019, +0.023]</td><td>-0.072 [-0.078, -0.065]</td><td>+0.087 [+0.080, +0.095]</td><td>+0.035 [+0.028, +0.042]</td></tr><tr><td>Qwen3.5-9B</td><td>-0.014 [−0.015, -0.012]</td><td>+0.104 [+0.101,+0.107]</td><td>+0.040 [+0.037, +0.042]</td><td>-0.032 [-0.035, -0.030]</td><td>+0.127 [+0.116,+0.138]</td><td>+0.074 [+0.064, +0.084]</td></tr><tr><td>gemma-4-12B</td><td>-0.021 [-0.023,-0.020]</td><td>+0.096 [+0.094, +0.099]</td><td>+0.032 [+0.030, +0.034]</td><td>-0.055 [-0.060, -0.050]</td><td>+0.104 [+0.096,+0.112]</td><td>+0.051 [+0.044, +0.059]</td></tr><tr><td>gpt-oss-20b</td><td>-0.019 [-0.020, -0.017]</td><td>+0.099 [+0.096, +0.102]</td><td>+0.035 [+0.032,+0.037]</td><td>-0.047 [-0.052, -0.043]</td><td>+0.112 [+0.103,+0.121]</td><td>+0.059 [+0.051, +0.067]</td></tr><tr><td>Llama-3.3-70B</td><td>-0.011 [−0.013, -0.010] [+0.104, +0.109]</td><td>+0.106</td><td>+0.042  $\left[ + 0 . 0 4 0 , + 0 . 0 4 4 \right]$ </td><td>-0.052</td><td>+0.107 [-0.057, -0.047] [+0.099, +0.116]</td><td>+0.055 [+0.047,+0.062]</td></tr></table>

Table 9: Cell-wise differences of the calibrated RCG (Cal. in Table 1) from each baseline, averaged over cells; e.g., “− Marginals” is Cal. − Marginals. Brackets show the 95% CI from 10,000 bootstrap resamples of the sets of conditioning attributes, each with all of its cells, as in Table 8. Every calibrated LLM is significantly below Marginals and significantly above Copula and LCA on both datasets.

![](images/e6904af95f6d02abcd0b9a94d9c3b94f27c8ca2f3f93c394f4ede7698116417b.jpg)  
Figure 8: Per-question JSD (mean over questions and cells) against RCG on ANES for three baselines and six LLMs. For the LLMs, both metrics are computed after in-cell marginal calibration. Human (disjoint) uses only the 196 cells with at least 200 human respondents. Larger values are worse on both axes.

![](images/6d56e434c40eb5361f02a11efed174d44dfdc4c2a5d582cdc3e72d9108ce6a20.jpg)  
Figure 9: Correlation-structure mismatch (1−ρ) against RCG for all nine generators on ANES. Larger values are worse on both axes. Triangles are LLMs; circles are the statistical baselines. The dotted horizontal line marks the permutation null (each question column permuted across cells, preserving marginals but destroying correlations). The dashed horizontal line marks the split-half ceiling of the human data (footnote 10).

<table><tr><td rowspan="2"></td><td colspan="2">WVS</td></tr><tr><td>JSD 1−ρ</td><td>JSD 1- RCG ρ</td></tr><tr><td rowspan="7">gpt-5.4-nano 0.403 gemini-flash-lite Qwen3.5-9B gemma-4-12B</td><td>RCG 0.392</td></tr><tr><td>1.272 0.409 0.494 1.370 1.2180.427 0.1181.353</td></tr><tr><td>0.4130.253 0.3140.452 1.2730.343 0.433 1.347</td></tr><tr><td>0.3930.315 1.199 0.420 0.391 1.332</td></tr><tr><td>0.3580.348 1.269 0.372 0.503 1.346 0.329</td></tr><tr><td>1.275 0.448 0.267 1.277</td></tr><tr><td>0.091 0.006 1.158 0.097 0.001 1.250 0.100 0.018 1.041 0.124 0.005 1.091 0.100 0.018 1.105 0.125 0.006 1.143</td></tr></table>

Table 10: JSD is the per-question Jensen–Shannon distance (base 2) averaged over questions and cells; $1 - \rho$ is the correlation-structure mismatch, where $\rho$ is the correlation-structure score of Williams et al. (2026); RCG is as in Table 1. Larger values are worse in every column. The last two rows give $1 - \rho$ for the split-half ceiling of the human data and for the permutation null.

Proof. In each cell c, the permutation only reorders the answers to $q _ { m }$ among the respondents in $\mathcal { G } _ { c }$ who answered $q _ { m } ,$ so the within-cell marginal of $q _ { m }$ is unchanged, which is the first claim. In particular, the mean answer to each of the $M _ { \mathrm { o } }$ questions is unchanged, so $A ( \mathcal { X } ^ { \mathcal { G } ^ { \prime } } ) = A ( \mathcal { X } ^ { \mathcal { G } } )$ . Since R, s, and $\rho$ are computed from A alone, $\rho$ takes the same value on $\{ \mathcal { G } _ { c } \} _ { c }$ and $\{ \mathcal { G } _ { c } ^ { \prime } \} _ { c }$ □

<table><tr><td rowspan="2"></td><td>RCG at</td><td> $| { \mathcal { G } } _ { c } | =$ </td><td rowspan="2">Slope per decade</td></tr><tr><td>100 1,000</td><td>10,000</td></tr><tr><td>gpt-5.4-nano</td><td>1.391 1.298</td><td></td><td>-0.093†</td></tr><tr><td>gemini-flash-lite</td><td>1.359 1.259</td><td></td><td>-0.097†</td></tr><tr><td>Qwen3.5-9B</td><td>1.357 1.239</td><td>1.146</td><td>-0.104</td></tr><tr><td>gemma-4-12B</td><td>1.338 1.242</td><td>1.167</td><td>-0.083</td></tr><tr><td>gpt-oss-20b</td><td>1.345 1.240</td><td>1.159</td><td>-0.092</td></tr><tr><td>Marginals</td><td>1.271 1.166</td><td>1.080</td><td>-0.094</td></tr><tr><td>Copula</td><td>1.086 0.964</td><td>0.874</td><td>-0.103</td></tr><tr><td>LCA</td><td>1.143 1.043</td><td>0.965</td><td>-0.089</td></tr></table>

Table 11: RCG (cell mean over the ANES subset of Section 5.5) at three sizes of the generated set $\left| \mathcal { G } _ { c } \right|$ . Slope is the median over cells of the per-cell change in RCG per tenfold increase in $| { \mathcal { G } } _ { c } | . - :$ not generated; †: fitted over one decade only, for reference.

The same argument applied to the human respondents gives $A ( \mathcal { X } ^ { \mathcal { H } ^ { \prime } } ) = A ( \mathcal { X } ^ { \mathcal { H } } )$ , so the score of $\{ \mathcal { H } _ { c } ^ { \prime } \} _ { c }$ against $\{ \mathcal { H } _ { c } \} _ { c }$ is the correlation of $\mathbf { s } ( \mathcal { X } ^ { \mathcal { H } } )$ with itself, which is one.

## N Number of Generated Respondents

This section reports the ANES results and the OSR for Section 5.4. At each $\left| \mathcal { G } _ { c } \right|$ , the generated set is drawn at random from 1,000 (proprietary models) or 10,000 (others) respondents per cell, generated separately from the main experiment, so the values at $| \mathcal { G } _ { c } | = 1 0 0$ in Table 2, Table 11, and Table 12 differ from those of the main experiment on the same subset (Appendix G) by less than 0.01.

Table 11 is the ANES counterpart of Table 2. RCG starts higher on ANES and falls faster with $| { \mathcal { G } } _ { c } |$ for every generator. Still, as on WVS, no LLM reaches one within the sizes we generate. We estimate how many generated respondents per cell are needed for RCG to reach one by extrapolating the slopes in Table 11 linearly in $\log _ { 1 0 } \left| \mathcal { G } _ { c } \right|$ from $| \mathcal { G } _ { c } | = 1 0 , 0 0 0$ . The estimates are about $2 . 5 \times 1 0 ^ { 5 }$ to $1 . 0 \times 1 0 ^ { 6 }$ generated respondents per cell for the three open-weight models, that is, about 2,500 to 10,000 times the 100 human respondents. As on WVS, RCG falls less in the second decade than in the first, so these estimates are likely underestimates.

Table 12 shows OSR at the same sizes on both datasets. From $\left| \mathcal { G } _ { c } \right|$ = 100 to the largest size we generate, OSR rises by 0.003 to 0.007 for every generator.

## O Full-History Ablation

Providing the full response history lowers mean RCG in all twelve model–dataset pairs, although every mean RCG remains above one.

We check how the amount of response history included in the prompt affects RCG and OSR. We use the full history as few-shot examples (Table 13). <sup>†</sup> indicates that the 95% CI of RCG − Marginals includes zero, whereas <sup>‡</sup> indicates that the interval lies entirely below zero. All full-history RCG values are above those of Copula; the RCGs of four of six LLMs for WVS and one of six for ANES are significantly above Marginals. Only gemma-4-12B on WVS and Llama-3.3-70B on ANES have RCG values significantly below those of Marginals.

## P Calibration Settings

Let a and b denote answer choices for question $q _ { m }$ and $p _ { c , m } ^ { \mathcal { H } }$ be the distribution of answers to $q _ { m }$ among respondents in $\mathcal { H } _ { c }$ (and $p _ { c , m } ^ { \mathcal { G } }$ likewise for $\mathcal { G } _ { c } )$ . We consider a transition matrix $T _ { c , m } ( b \mid a )$ that satisfies $\begin{array} { r } { \sum _ { a } T _ { c , m } ( b \mid a ) p _ { c , m } ^ { \mathcal { G } } ( a ) = p _ { c , m } ^ { \mathcal { H } } ( b ) } \end{array}$ for all b and $\begin{array} { r } { \sum _ { b } T _ { c , m } ( b \mid a ) = 1 } \end{array}$ for all a. We replace each answer a of a generated respondent with a draw from $T _ { c , m } ( \cdot \mid a )$ , independently across questions and respondents, and leave missing answers as they are.

The constraint leaves $T _ { c , m }$ underdetermined. For ordered questions (rating scales and ordinal questions), we use quantile mapping (Gudmundsson et al., 2012; Cannon et al., 2015), which is an optimal transport plan for the cost $| a - b |$ (Santambrogio, 2015). With choices $\{ 1 , 2 , 3 \} , p ^ { \mathcal { G } } =$ $( 0 . 5 , 0 . 5 , 0 )$ , and $p ^ { \mathcal { H } } = ( 0 . 2 5 , 0 . 2 5 , 0 . 5 )$ , it moves half of the answers at 1 to 2 and all answers at 2 to 3. For nominal questions, where $| a - b |$ has no meaning, we use minimum-rewrite remapping, an optimal transport plan for the cost $1 [ a \neq b ]$ (Levin and Peres, 2017): it preserves a probability mass of min $( p _ { c , m } ^ { G } ( a ) , p _ { c , m } ^ { H } ( a ) )$ ) at each choice a and reallocates the excess to choices where $p _ { c , m } ^ { G }$ falls short of $p _ { c , m } ^ { H }$

## Q Missing-Value Ablation

We evaluate the impact of missing values on the RCG and OSR. In calculating RCG and OSR, the Gower distance excludes questions with missing answers; fewer compared questions make distances more variable, so nearest-neighbor distances become smaller. Human respondents have missing answers while generated respondents rarely do, so human-to-human distances are computed over fewer questions than human-to-generated distances; this asymmetry may reduce the human-tohuman distance and thereby inflate RCG.

<table><tr><td rowspan="3"></td><td colspan="3">WVS</td><td rowspan="3"></td><td colspan="3">ANES</td><td rowspan="3">Slope</td></tr><tr><td colspan="3">OSR at  $| { \mathcal { G } } _ { c } | =$ </td><td rowspan="2">Slope</td><td colspan="3">OSR at  $| { \mathcal { G } } _ { c } | =$ </td></tr><tr><td>100</td><td>1,000</td><td>10,000 per decade</td><td>100</td><td>1,000</td><td>10,000 per decade</td></tr><tr><td>gpt-5.4-nano</td><td>0.988</td><td>0.992</td><td>一</td><td>0.003†</td><td>1.193</td><td>1.197</td><td></td><td></td><td>0.005†</td></tr><tr><td>gemini-flash-lite</td><td>0.828</td><td>0.832</td><td></td><td>0.004†</td><td>0.870</td><td>0.875</td><td></td><td></td><td>0.004†</td></tr><tr><td>Qwen3.5-9B</td><td>0.981</td><td>0.986</td><td>0.986</td><td>0.002</td><td>1.232</td><td>1.237</td><td></td><td>1.238</td><td>0.002</td></tr><tr><td>gemma-4-12B</td><td>0.783</td><td>0.787</td><td>0.787</td><td>0.002</td><td>0.967</td><td>0.972</td><td></td><td>0.972</td><td>0.002</td></tr><tr><td>gpt-oss-20b</td><td>1.008</td><td>1.013</td><td>1.013</td><td>0.002</td><td>1.163</td><td>1.168</td><td></td><td>1.169</td><td>0.002</td></tr><tr><td>Marginals</td><td>0.990</td><td>0.993</td><td>0.993</td><td>0.002</td><td>1.106</td><td>1.109</td><td></td><td>1.109</td><td>0.002</td></tr><tr><td>Copula</td><td>0.946</td><td>0.950</td><td>0.951</td><td>0.002</td><td>0.999</td><td>1.005</td><td></td><td>1.006</td><td>0.003</td></tr><tr><td>LCA</td><td>0.963</td><td>0.967</td><td>0.967</td><td>0.002</td><td>0.961</td><td>0.966</td><td></td><td>0.966</td><td>0.002</td></tr></table>

Table 12: OSR (cell mean over the subsets of Appendix G) at three sizes of the generated set $\left| \mathcal { G } _ { c } \right|$ . Slope is the median over cells of the per-cell slope of OSR regressed on $\log _ { 1 0 } \left| \mathcal { G } _ { c } \right|$ , i.e., the change per tenfold increase. The two proprietary models were generated only up to $| \mathcal { G } _ { c } | = 1 , 0 0 0 \left( - \mathrm { { z } } \right)$ not generated); their slopes (†) are fitted over one decade instead of two and are reported for reference.

To check this, we compute a masked variant: in calculating RCG and OSR, we repeat the following process for every generated respondent $\textbf { \em v } \in { \mathcal G } _ { c } .$ We randomly select a human respondent $\pmb { u } \in \mathcal { H } _ { c }$ outside the 100 human respondents used in the calculation, and take the questions that u left unanswered as a mask. We then apply the mask to v by treating the answers of v to these questions as missing. In cells with exactly 100 human respondents (11 of the 730 cells for WVS and 8 of the 603 cells for ANES), no human respondent remains outside the evaluation sample. For these cells, we randomly assign the 100 human respondents’ missingness masks to the 100 generated respondents in a one-to-one mapping. Table 14 therefore covers the same cells as Table 1. This procedure assesses sensitivity to the mismatch in missingness patterns between human and generated respondents.

Table 14 shows the results. The masked RCGs are lower than the original RCGs, but the masked RCGs of all six LLMs remain above Marginals. RCG decreases more on ANES (by 0.03–0.05) than on WVS (by at most 0.014), because human respondents leave more answers missing on ANES (6.0% vs. 1.1%). OSR changes by less than 0.01.

## R Temperature Ablation

Table 15 shows RCG and OSR at each temperature. For RCG, the largest range is 0.033 (Qwen3.5-9B on WVS), and RCG stays above Marginals at every temperature. For OSR, the range on each dataset is 0.035 to 0.275 for gpt-5.4-nano, Qwen3.5-9B, and gpt-oss-20b, but at most 0.025 for the other three models; the largest movement is in Qwen3.5-9B on WVS, from 0.812 at temperature 0.1 to 1.087 at 1.5. For gpt-5.4-nano, Qwen3.5-9B, and gptoss-20b, OSR on WVS crosses one as the temperature increases. Wang et al. (2025) report a finding consistent with this pattern for free-response text: raising the temperature does not bring the diversity of GPT-4 responses up to the level of human participants.

## S Reasoning Effort Ablation

Raising the reasoning effort shifts both models toward contraction, while RCG falls most when reasoning is turned on. We vary the reasoning effort, which controls how much the model reasons before answering, over low, medium, and high for gpt-5.4-nano (default none) and over medium and high for gpt-oss-20b (default low). Table 16 shows the results.<sup>18</sup> Turning reasoning on for gpt-5.4- nano (none to low) lowers RCG by 0.045 on WVS and 0.110 on ANES, whereas raising the effort beyond low changes RCG by at most 0.031 in either model. OSR drops in all four model–dataset pairs, by 0.073 to 0.169 from the default to high. RCG stays above Marginals except for gpt-5.4-nano on ANES, which equals Marginals at medium (1.266) and falls below it at high (1.253).

<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">WVS</td><td colspan="4">ANES</td></tr><tr><td>RCG</td><td>RCG – Marginals</td><td>OSR</td><td>RCG</td><td></td><td>RCG — Marginals</td><td>OSR</td></tr><tr><td rowspan="2">gpt-5.4-nano</td><td>limited history (≤ 3)</td><td>1.270</td><td>+0.109 [+0.097, +0.121]</td><td>0.990</td><td>1.390</td><td></td><td>+0.125 [+0.087, +0.165]</td><td>1.189</td></tr><tr><td>full history</td><td>1.231</td><td>+0.070 [+0.056, +0.083]</td><td>1.124</td><td>1.328</td><td></td><td>+0.062 [+0.015, +0.113]</td><td>1.304</td></tr><tr><td rowspan="2">gemini-flash-lite</td><td>limited history (≤ 3)</td><td>1.218</td><td>+0.057 [+0.038, +0.076]</td><td>0.829</td><td>1.356</td><td></td><td>+0.090 [+0.039, +0.141]</td><td>0.878</td></tr><tr><td>full history</td><td>1.184†</td><td>+0.022 [−0.002, +0.046]</td><td>0.901</td><td>1.249†</td><td></td><td>-0.017 [−0.078, +0.048]</td><td>0.820</td></tr><tr><td rowspan="2">Qwen3.5-9B</td><td>limited history (≤ 3)</td><td>1.268</td><td>+0.107 [+0.094, +0.120]</td><td>0.981</td><td>1.354</td><td></td><td>+0.089 [+0.053, +0.129]</td><td>1.234</td></tr><tr><td>full history</td><td>1.258</td><td>+0.097 [+0.082, +0.111]</td><td>1.097</td><td>1.276†</td><td></td><td>+0.010 [−0.036, +0.061]</td><td>1.326</td></tr><tr><td rowspan="2">gemma-4-12B</td><td>limited history (≤ 3)</td><td>1.205</td><td>+0.044 [+0.031, +0.058]</td><td>0.783</td><td>1.332</td><td></td><td>+0.066 [+0.014, +0.124]</td><td>0.968</td></tr><tr><td>full history</td><td>1.113</td><td>−0.049 [−0.063, −0.035]</td><td>0.837</td><td>1.224†</td><td></td><td>−0.042 [−0.103, +0.025]</td><td>1.033</td></tr><tr><td rowspan="2">gpt-oss-20b</td><td>limited history (≤ 3)</td><td>1.268</td><td>+0.107 [+0.093, +0.121]</td><td>1.003</td><td>1.338</td><td></td><td>+0.072 [+0.027, +0.123]</td><td>1.163</td></tr><tr><td>full history</td><td>1.193</td><td>+0.031 [+0.017, +0.046]</td><td>1.054</td><td>1.276†</td><td></td><td>+0.010 [−0.039, +0.063]</td><td>1.251</td></tr><tr><td rowspan="2">Llama-3.3-70B</td><td>limited history (≤ 3)</td><td>1.279</td><td>+0.118 [+0.102, +0.135]</td><td>0.709</td><td>1.274†</td><td></td><td>+0.008 [−0.031, +0.048]</td><td>0.877</td></tr><tr><td>full history</td><td>1.243</td><td>+0.082 [+0.063, +0.101]</td><td>0.724</td><td>1.156</td><td></td><td>-0.110 [−0.160, −0.056]</td><td>0.938</td></tr><tr><td colspan="2">Marginals</td><td>1.161</td><td></td><td>0.991</td><td>1.266</td><td></td><td></td><td>1.108</td></tr><tr><td colspan="2">Copula</td><td>1.040</td><td></td><td></td><td>0.947</td><td>1.086</td><td></td><td>0.991</td></tr><tr><td colspan="2">LCA</td><td>1.111</td><td></td><td></td><td>0.964</td><td>1.149</td><td></td><td>0.964</td></tr></table>

Table 13: RCG − Marginals is the cell-wise difference in RCG from Marginals, averaged over cells. The brackets show its 95% CI from 10,000 bootstrap resamples of the sets of conditioning attributes, each with all of its cells. The CI thus covers variation from the choice of attribute sets, with the human and generated responses fixed. <sup>†</sup> indicates that the 95% CI of RCG − Marginals includes zero, whereas <sup>‡</sup> indicates that the interval lies entirely below zero. With the full history, all RCG values remain above those of Copula.

<table><tr><td colspan="2" rowspan="2"></td><td colspan="2">WVS</td><td colspan="2">ANES</td></tr><tr><td>RCG</td><td>OSR</td><td>RCG</td><td>OSR</td></tr><tr><td rowspan="2">gpt-5.4-nano</td><td>original</td><td>1.272</td><td>0.994</td><td>1.370</td><td>1.173</td></tr><tr><td>masked</td><td>1.265</td><td>0.992</td><td>1.337</td><td>1.169</td></tr><tr><td rowspan="2">gemini-flash-lite</td><td>original</td><td>1.218</td><td>0.840</td><td>1.353</td><td>0.838</td></tr><tr><td>masked</td><td>1.209</td><td>0.840</td><td>1.305</td><td>0.833</td></tr><tr><td rowspan="2">Qwen3.5-9B</td><td>original</td><td>1.273</td><td>0.992</td><td>1.347</td><td>1.213</td></tr><tr><td>masked</td><td>1.267</td><td>0.992</td><td>1.318</td><td>1.210</td></tr><tr><td rowspan="2">gemma-4-12B</td><td>original</td><td>1.199</td><td>0.793</td><td>1.332</td><td>0.942</td></tr><tr><td>masked</td><td>1.191</td><td>0.794</td><td>1.287</td><td>0.938</td></tr><tr><td rowspan="2">gpt-oss-20b</td><td>original</td><td>1.269</td><td>1.013</td><td>1.346</td><td>1.144</td></tr><tr><td>masked</td><td>1.263</td><td>1.011</td><td>1.314</td><td>1.143</td></tr><tr><td rowspan="2">Llama-3.3-70B</td><td>original</td><td>1.275</td><td>0.722</td><td>1.277</td><td>0.853</td></tr><tr><td>masked</td><td>1.261</td><td>0.724</td><td>1.237</td><td>0.851</td></tr><tr><td rowspan="2">Marginals</td><td>original</td><td>1.158</td><td>0.989</td><td>1.250</td><td>1.078</td></tr><tr><td>masked</td><td>1.154</td><td>0.991</td><td>1.222</td><td>1.075</td></tr><tr><td rowspan="2">Copula</td><td>original</td><td>1.041</td><td>0.946</td><td>1.091</td><td>0.976</td></tr><tr><td>masked</td><td>1.037</td><td>0.947</td><td>1.063</td><td>0.972</td></tr><tr><td rowspan="2">LCA</td><td>original</td><td>1.105</td><td>0.959</td><td>1.143</td><td>0.950</td></tr><tr><td>masked</td><td>1.101</td><td>0.960</td><td>1.111</td><td>0.947</td></tr></table>

Table 14: RCG and OSR in the original setting (Table 1) and the masked setting.

gemini-pro costs approximately 33 times as much as gemini-flash-lite.

Table 17 indicates that raising the tier lowers RCG in three of the four model–dataset pairs, by 0.071 to 0.128, and in the remaining one raises it by 0.020 (gemini-pro on WVS); on ANES, both higher-tier models fall below Marginals. OSR also drops in all four pairs and crosses one only for gpt-5.4 on ANES (from 1.189 to 0.838).

## T Model Tier Ablation

Raising the model tier mostly lowers RCG but shifts each model toward contraction. We use gpt-5.4 from OpenAI, a higher-tier model of the same generation as gpt-5.4-nano, and gemini-3.1- pro-preview (gemini-pro), a frontier model from Google. In our runs, gpt-5.4 costs approximately 12 times as much per API call as gpt-5.4-nano, and

<table><tr><td rowspan="2"></td><td colspan="2">WVS</td><td colspan="2">ANES</td></tr><tr><td>RCG</td><td>OSR</td><td>RCG</td><td>OSR</td></tr><tr><td>gpt-5.4-nano</td><td></td><td></td><td></td><td></td></tr><tr><td>t = 0.1</td><td>1.270</td><td>0.939</td><td>1.403</td><td>1.136</td></tr><tr><td>t = 0.5</td><td>1.270</td><td>0.955</td><td>1.395</td><td>1.152</td></tr><tr><td>t = 1.0 (default)</td><td>1.270</td><td>0.990</td><td>1.390</td><td>1.189</td></tr><tr><td>t = 1.5</td><td>1.271</td><td>1.044</td><td>1.394</td><td>1.251</td></tr><tr><td>gemini-flash-lite</td><td></td><td></td><td></td><td></td></tr><tr><td>default</td><td>1.218</td><td>0.829</td><td>1.356</td><td>0.878</td></tr><tr><td>t = 0.1</td><td>1.221</td><td>0.827</td><td>1.354</td><td>0.868</td></tr><tr><td>t = 1.0</td><td>1.219</td><td>0.830</td><td>1.346</td><td>0.873</td></tr><tr><td>t = 1.5</td><td>1.216</td><td>0.838</td><td>1.349</td><td>0.878</td></tr><tr><td>Qwen3.5-9B</td><td></td><td></td><td></td><td></td></tr><tr><td>t = 0.1</td><td>1.290</td><td>0.812</td><td>1.339</td><td>1.107</td></tr><tr><td>t = 0.5</td><td>1.280</td><td>0.860</td><td>1.342</td><td>1.147</td></tr><tr><td>t = 1.0 (default)</td><td>1.268</td><td>0.981</td><td>1.354</td><td>1.234</td></tr><tr><td>t = 1.5</td><td>1.257</td><td>1.087</td><td>1.346</td><td>1.302</td></tr><tr><td>gemma-4-12B</td><td></td><td></td><td></td><td></td></tr><tr><td>default</td><td>1.205</td><td>0.783</td><td>1.332</td><td>0.968</td></tr><tr><td>t = 0.1</td><td>1.203</td><td>0.776</td><td>1.339</td><td>0.963</td></tr><tr><td>t = 0.5</td><td>1.204</td><td>0.778</td><td>1.340</td><td>0.962</td></tr><tr><td>t = 1.0</td><td>1.200</td><td>0.786</td><td>1.335</td><td>0.970</td></tr><tr><td>t = 1.5</td><td>1.205</td><td>0.801</td><td>1.331</td><td>0.983</td></tr><tr><td>gpt-oss-20b</td><td></td><td></td><td></td><td></td></tr><tr><td>t = 0.1</td><td>1.261</td><td>0.970</td><td>1.359</td><td>1.161</td></tr><tr><td>t = 1.0 (default)</td><td>1.268</td><td>1.003</td><td>1.338</td><td>1.163</td></tr><tr><td>t = 1.5</td><td>1.285</td><td>1.076</td><td>1.355</td><td>1.196</td></tr><tr><td>Llama-3.3-70B</td><td></td><td></td><td></td><td></td></tr><tr><td>default</td><td>1.279</td><td>0.709</td><td>1.274</td><td>0.877</td></tr><tr><td>t = 0.1</td><td>1.277</td><td>0.708</td><td>1.277</td><td>0.876</td></tr><tr><td>t = 0.5</td><td>1.276</td><td>0.709</td><td>1.277</td><td>0.877</td></tr><tr><td>t = 1.0</td><td>1.278</td><td>0.712</td><td>1.277</td><td>0.877</td></tr><tr><td>t = 1.5</td><td>1.276</td><td>0.716</td><td>1.276</td><td>0.881</td></tr><tr><td>Marginals</td><td>1.161</td><td>0.991</td><td>1.266</td><td>1.108</td></tr></table>

Table 15: RCG and OSR at each temperature. At t = 1.5, 9.7% (WVS) and 9.9% (ANES) of the answers of Qwen3.5-9B cannot be parsed and are treated as missing; for the other open-weight models, the rate is at most 1.1%.

<table><tr><td rowspan="2"></td><td colspan="2">WVS</td><td colspan="2">ANES</td></tr><tr><td>RCG</td><td>OSR</td><td>RCG</td><td>OSR</td></tr><tr><td>OpenAI</td><td></td><td></td><td></td><td></td></tr><tr><td>gpt-5.4-nano (default)</td><td>1.270</td><td>0.990</td><td>1.390</td><td>1.189</td></tr><tr><td>gpt-5.4</td><td>1.199</td><td>0.692</td><td>1.262</td><td>0.838</td></tr><tr><td>Google</td><td></td><td></td><td></td><td></td></tr><tr><td>gemini-flash-lite (default) 1.218</td><td></td><td>0.829</td><td>1.3560.878</td><td></td></tr><tr><td>gemini-pro</td><td>1.238</td><td>0.692</td><td>1.250</td><td>0.763</td></tr><tr><td>Marginals</td><td>1.161</td><td>0.991</td><td>1.266</td><td>1.108</td></tr><tr><td>Copula</td><td>1.040</td><td>0.947</td><td>1.086 0.991</td><td></td></tr><tr><td>LCA</td><td></td><td>1.111 0.964 1.149 0.964</td><td></td><td></td></tr></table>

Table 17: RCG and OSR of the default and higher-tier models.

<table><tr><td rowspan="2"></td><td colspan="2">WVS</td><td colspan="2">ANES</td></tr><tr><td>RCG</td><td>OSR</td><td>RCG</td><td>OSR</td></tr><tr><td>gpt-5.4-nano</td><td></td><td></td><td></td><td></td></tr><tr><td>none (default)</td><td>1.270</td><td>0.990</td><td>1.390</td><td>1.189</td></tr><tr><td>low</td><td>1.225</td><td>0.945</td><td>1.280</td><td>1.059</td></tr><tr><td>medium</td><td>1.214</td><td>0.919</td><td>1.266</td><td>1.025</td></tr><tr><td>high</td><td>1.213</td><td>0.917</td><td>1.253</td><td>1.020</td></tr><tr><td>gpt-oss-20b</td><td></td><td></td><td></td><td></td></tr><tr><td>1ow (default)</td><td>1.268</td><td>1.003</td><td>1.338</td><td>1.163</td></tr><tr><td>medium</td><td>1.237</td><td>0.932</td><td>1.326</td><td>1.096</td></tr><tr><td>high</td><td>1.246</td><td>0.902</td><td>1.338</td><td>1.060</td></tr><tr><td>Marginals</td><td>1.161</td><td>0.991</td><td>1.266</td><td>1.108</td></tr><tr><td>Copula</td><td>1.040</td><td>0.947</td><td>1.086</td><td>0.991</td></tr><tr><td>LCA</td><td></td><td></td><td></td><td>0.964</td></tr><tr><td></td><td>1.111</td><td>0.964</td><td>1.149</td><td></td></tr></table>

Table 16: RCG and OSR at each reasoning effort level.