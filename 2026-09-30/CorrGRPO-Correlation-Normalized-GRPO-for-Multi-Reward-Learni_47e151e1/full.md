# CorrGRPO: Correlation-Normalized GRPO for Multi-Reward Learning

Wenbin Hu<sup>∗</sup>, Huihao Jing<sup>∗</sup>, Haochen Shi, Yuxuan Liu, Haoran Li, Yangqiu Song   
Hong Kong University of Science and Technology   
whuak@connect.ust.hk

## Abstract

Group Relative Policy Optimization (GRPO) is widely used to train reasoning language models, where it computes advantages by centering and normalizing rewards across rollouts of the same prompt. For multiple rewards, GRPO sums the reward components and normalizes the total reward by its within-group standard deviation. The corresponding variance equals the sum of all pairwise reward covariances. For a fixed centered reward, larger aggregate covariance produces smaller advantages, and vice versa, allowing update magnitudes to adapt to reward dependence. However, correlated rewards with large scales can dominate this normalization and suppress signals from smaller-scale rewards. We propose Correlation-Normalized GRPO (CorrGRPO), which normalizes pairwise covariances into Pearson correlation coeficients. CorrGRPO keeps the centered total reward unchanged while balancing the influence of differently scaled rewards on the correlation-based normalization. This allows advantage magnitudes to adapt to reward correlations without the normalization being dominated by large-scale reward components. We compare CorrGRPO with GRPO and other variants on code generation, tool calling, and agent security, using models ranging from 0.5B to 8B parameters. These tasks all involve multiple rewards that can improve together or present tradeofs. Results show improvements across three domains, including code generation, tool calling, and agent security. Our code is available at <sub>https:</sub>//<sub>github.com</sub>/<sub>HKUST-KnowComp</sub>/<sub>CorrGRPO</sub>.

$$
\begin{array} { r } { A _ { \mathrm { G R P O } } ^ { i } = \frac { R ^ { i } - \bar { R } } { \sqrt { \widehat { \mathrm { V a r } } ( R ) } + \varepsilon } = \frac { \displaystyle \sum _ { l = 1 } ^ { r } ( R _ { l } ^ { i } - \bar { R } _ { l } ) } { \sqrt { \widehat { \mathrm { V a r } } \left( \displaystyle \sum _ { l = 1 } ^ { r } R _ { l } \right) } + \varepsilon } = \frac { \displaystyle \sum _ { l = 1 } ^ { r } ( R _ { l } ^ { i } - \bar { R } _ { l } ) } { \displaystyle \sqrt { \sum _ { l = 1 } ^ { r } \sum _ { m = 1 } ^ { r } \widehat { \mathrm { C o v } } ( R _ { l } , R _ { m } ) } + \varepsilon } . } \end{array}
$$

$$
\mathrm { P } { \bf O } = \frac { \displaystyle \sum _ { l = 1 } ^ { r } ( R _ { l } ^ { i } - \bar { R } _ { l } ) } { \sqrt { \displaystyle \sum _ { l = 1 } ^ { r } \sum _ { m = 1 } ^ { r } \hat { \rho } _ { l m } } + \varepsilon } = \frac { \displaystyle \sum _ { l = 1 } ^ { r } ( R _ { l } ^ { i } - \bar { R } _ { l } ) } { \displaystyle \sqrt { \sum _ { l = 1 } ^ { r } \sum _ { m = 1 } ^ { r } \frac { \hat { \bf C } { \bf O } { \bf v } ( R _ { l } , R _ { m } ) } { \sqrt { \hat { \bf V } { \bf a r } ( R _ { l } ) \hat { \nabla } { \bf a r } ( R _ { m } ) } } } + \varepsilon } .
$$

$$
\begin{array} { r } { \bar { R } _ { l } = \frac { 1 } { n } \sum _ { k = 1 } ^ { n } R _ { l } ^ { k } } \end{array}
$$

$$
\stackrel { \dot { R ^ { i } } } { \longrightarrow } \stackrel { \sum _ { l = 1 } ^ { r } } { \longrightarrow } { \longrightarrow }
$$

![](images/2a46185fb9fd70dbf592b0fcfdcf930d38b78339497ff9a62aad4f9a89be5984.jpg)  
Figure 1: An overview of CorrGRPO. (1) Left: CorrGRPO replaces total-reward standard deviation normalization with a correlation-based denominator while retaining the centered total reward. (2) Right: Validation accuracy (top) and eficiency (bottom), measured by Mean@1, for Qwen2.5-Coder-7B-Instruct on LeetCodeDataset.

## 1 Introduction

Reinforcement learning (RL) has become a central paradigm for aligning large language models (LLMs) with human preferences and improving their ability to solve complex tasks. By constructing training environments and designing rewards that capture desired outcomes, RL enables models to improve through feedback on their own generations, supporting both instruction following and the development of reasoning capabilities (Ouyang et al., 2022; DeepSeek-AI et al., 2025). Among existing approaches, Group Relative Policy Optimization (GRPO) has gained widespread adoption for its simplicity and eficiency. GRPO estimates advantages by subtracting the mean reward and dividing by the reward standard deviation within a group of responses sampled for the same prompt, eliminating the need for a separate value model (Shao et al., 2024).

Practical training objectives often require multiple rewards to capture diferent aspects of desirable behavior. First, reward objectives can reinforce one another. Code generation, for example, can combine rewards for executability, test-case pass rate, and abstract syntax tree (AST) similarity to reference solutions; resolving syntax or runtime errors can improve both executability and test-case performance. Second, reward objectives can introduce tradeofs. Agent training requires balancing task utility with security, where an overly conservative agent may avoid malicious instructions by also refusing legitimate requests, while an overly permissive agent may complete more tasks at the cost of greater exposure to prompt injection (Debenedetti et al., 2024). Such settings involve rewards that may improve together or exhibit tradeofs. Their statistical relationships therefore provide a useful perspective on how multiple feedback signals interact during learning. In this work, we investigate how correlations among reward components afect advantage estimation in GRPO.

Our starting point is a simple identity with direct implications for multi-reward optimization. When GRPO aggregates r reward components into a total reward $\begin{array} { r } { R = \dot { \sum _ { l = 1 } ^ { r } } R _ { l } . } \end{array}$ the variance underlying its normalization is $\begin{array} { r } { \widehat { \mathrm { V a r } } ( \sum _ { l = 1 } ^ { r } R _ { l } ) = \sum _ { l = 1 } ^ { r } \sum _ { m = 1 } ^ { r } \widehat { \mathrm { C o v } } ( R _ { l } , R _ { m } ) } \end{array}$ . The d dunderlying population identity is derived in Appendix L.1. Thus, the denominator depends on the sum of all entries in the reward covariance matrix, incorporating both individual reward variances and pairwise dependencies. Holding the marginal reward variances and centered total reward fixed, stronger positive correlations reduce the advantage magnitude by increasing the normalization denominator; weaker or more negative correlations increase the advantage magnitude by decreasing the denominator. GRPO therefore implicitly adjusts advantage scaling according to reward correlation within each rollout group. This provides a dynamic mechanism for modulating the strength of the aggregate learning signa as relationships among rewards change.

However, this mechanism entangles reward dependence with reward scale. Each covariance satisfies Cov $( R _ { l } , R _ { m } ) = \sigma _ { l } \sigma _ { m } \rho _ { l m }$ , where $\sigma _ { l }$ and $\sigma _ { m }$ are the component standard deviations and $\rho _ { l m }$ is their Pearson correlation coeficient. Consequently, reward components with large scales of variation can dominate both the diagonal and of-diagonal terms in the covariance sum. The normalization then becomes disproportionately sensitive to these components, while smaller-scale rewards contribute little to the adjustment. This imbalance suppresses the influence of smaller-scale rewards on normalization, even when they encode useful distinctions among responses. This limitation is particularly relevant when heterogeneous rewards difer substantially in their numerical ranges or within-group variability.

To address this issue, we propose Correlation-Normalized GRPO (CorrGRPO). As shown in Figure 1, CorrGRPO replaces pairwise covariances in the denominator with Pearson correlation coeficients while retaining the centered total reward in the numerator. Normalizing each covariance by the corresponding component standard deviations removes its dependence on reward scale. Each reward component with nonzero within-group variance consequently contributes equally to the diagonal of the correlation matrix, while of-diagonal terms reflect the strength and direction of pairwise dependence. The denominator remains responsive to reward correlations without being dominated by components solely because they vary on larger numerical scales. Meanwhile, preserving the numerator retains the relative reward weights specified by the training objective. As a result, CorrGRPO allows correlations involving smaller-scale rewards to influence advantage normalization without being downweighted by their scales.

We conduct experiments with CorrGRPO on code generation, tool calling, and agent security, three domains that naturally require multiple reward signals, using models ranging from 0.5B to 8B parameters. Comparisons with GRPO and relevant variants show improve ments in code accuracy and eficiency, tool-call accuracy, and the balance between agent utility and security. Together, these results support correlation-based normalization as an efective approach to improving practical multi-reward learning.

Our contributions are threefold:

1. A covariance perspective on GRPO. We characterize how the reward covariance matrix implicitly controls advantage scaling in multi-reward GRPO and show that large-scale rewards can dominate this adjustment.

2. Correlation-normalized advantage estimation. We introduce CorrGRPO, which replaces covariance-based normalization with correlation-based normalization while preserving the centered total reward, preventing large-scale rewards from disproportionately influencing the denominator.

3. Experiments across three multi-reward domains. We conduct extensive experiments on code generation, tool calling, and agent security across models from 0.5B to 8B parameters, demonstrating improvements over GRPO and relevant variants in the three domains, along with an expansion of the empirical Pareto frontier between reward objectives.

## 2 A Covariance Normalization View of Multi-Reward GRPO

Group Relative Policy Optimization (GRPO) (Shao et al., 2024) estimates advantages using rewards from a group of trajectories, avoiding a separate value model. For each prompt q, it samples n trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { n }$ from an old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ . With r reward components, let $R _ { l } ^ { i } = R _ { l } ( q , \tau _ { i } )$ and $R ^ { i } = \textstyle \sum _ { l } R _ { l } ^ { i }$ , with group means $\begin{array} { r } { \bar { R } _ { l } = \frac { 1 } { n } \sum _ { i } R _ { l } ^ { i } } \end{array}$ and $\begin{array} { r } { \bar { R } = \frac { 1 } { n } \sum _ { i } R ^ { i } } \end{array}$ . Any fixed reward weights are absorbed into the corresponding components. Under outcome supervision, all generated tokens in trajectory i share the same advantage $A _ { \mathrm { G R P O } } ^ { i }$ The advantage is computed by centering the total reward and normalizing it by its within-group standard deviation: $\begin{array} { r } { A _ { \mathrm { G R P O } } ^ { i } = \frac { R ^ { i } - \bar { R } } { \sqrt { \operatorname { V a r } ( R ) } + \varepsilon } } \end{array}$ , where $\varepsilon > 0$ ensures numerical stability.

Our observation is that, with multiple reward components, GRPO implicitly uses the aggregate reward covariance to control advantage scaling. Applying the variance-of-a-sum identity, as derived in Appendix L.1, we obtain

$$
A _ { \mathrm { G R P O } } ^ { i } = \frac { R ^ { i } - \bar { R } } { \sqrt { \widehat { \mathrm { V a r } } ( R ) } + \varepsilon } = \frac { \displaystyle \sum _ { l = 1 } ^ { r } ( R _ { l } ^ { i } - \bar { R } _ { l } ) } { \sqrt { \widehat { \mathrm { V a r } } \left( \sum _ { l = 1 } ^ { r } R _ { l } \right) } + \varepsilon } = \frac { \displaystyle \sum _ { l = 1 } ^ { r } ( R _ { l } ^ { i } - \bar { R } _ { l } ) } { \displaystyle \sqrt { \sum _ { l = 1 } ^ { r } \sum _ { m = 1 } ^ { r } \widehat { \mathrm { C o v } } ( R _ { l } , R _ { m } ) } + \varepsilon } .\tag{1}
$$

Here, $\widehat { \mathrm { V a r } }$ and $\widehat { \mathrm { C o v } }$ are computed across the same rollout group with a consistent normald dization convention. This decomposition reveals an implicit mechanism in GRPO: the denominator adjusts advantage scaling through both individual reward variances and pairwise reward dependencies.

Specifically, let $\Delta R ^ { i } \ = \ R ^ { i } - \bar { R }$ and $\begin{array} { r } { S = \sum _ { l , m } \widehat { \mathrm { C o v } } ( R _ { l } , R _ { m } ) } \end{array}$ The decomposition gives $A _ { \mathrm { G R P O } } ^ { i } = g \Delta R ^ { i }$ , where $g = ( \sqrt { S } + \varepsilon ) ^ { - 1 }$ dis a scaling coeficient shared by all trajectories in the group. Positive covariances increase the variability of the summed reward, reducing this coeficient, while negative covariances ofset part of that variability and increase it. Holding the marginal reward variances and a trajectory’s centered total reward fixed, stronger positive correlations therefore reduce its advantage magnitude; weaker or more negative correlations increase it. Because these statistics are recomputed from newly sampled trajectories, the scaling coeficient adapts to reward relationships across prompts and throughout training. This mechanism changes the scale of the group’s reward-driven contribution to the surrogate objective while preserving the relative reward weights in the numerator.

![](images/3b463c2652fbeb67a5cf51c461b818fc975be3465e98319b0711d59891410360.jpg)  
Figure 2: An example for comparing GRPO and CorrGRPO. (a) A group of four trajectories with three rewards. Rewards $r _ { 1 }$ and $r _ { 2 }$ are highly correlated, while $r _ { 3 }$ has a larger scale and weak correlations with both. (b) Covariance-matrix elements. In GRPO, the large scale of $r _ { 3 }$ dominates the denominator. (c) Pearson correlation coeficients matrix elements. In CorrGRPO, the strong correlation between $r _ { 1 }$ and $r _ { 2 }$ has a greater influence on the normalization denominator than $r _ { 3 }$ does.

However, covariance couples statistical dependence with reward scale. Each pairwise correlation is weighted by the product of the component standard deviations: $\begin{array} { r } { \bar { S } = \sum _ { l = 1 } ^ { r } \hat { \sigma } _ { l } ^ { 2 } . } \end{array}$ + $\begin{array} { r } { 2 \sum _ { l < m } \hat { \sigma } _ { l } \hat { \sigma } _ { m } \hat { \rho } _ { l m } } \end{array}$ , where ${ \hat { \sigma } } _ { l } ^ { 2 } = \widehat { \mathrm { V a r } } ( R _ { l } )$ . Large-scale rewards can dominate both the variance dterms and the sensitivity of the denominator to changes in correlation. Consequently, the shared scaling coeficient can be driven primarily by these components, while dependencies among smaller-scale rewards have limited influence on the adjustment.

Figure 2 illustrates this scale imbalance with four trajectories and three rewards. The first two rewards are strongly correlated $\left( \hat { \rho } _ { 1 2 } \approx 0 . 9 7 5 6 \right)$ , while the third has weak correlations with both $\left( \hat { \rho } _ { 1 3 } = \hat { \rho } _ { 2 3 } \approx 0 . 1 0 9 8 \right)$ . Nevertheless, the third reward’s variance, 14.1696, alone accounts for approximately 91.1% of the covariance sum $S = 1 5 . 5 5 2 0$ . Its covariances with the other rewards, both 0.1728, also exceed the covariance 0.1707 between the strongly correlated first two rewards. Thus, the third reward’s large scale dominates the normalization despite its weak correlations, motivating CorrGRPO’s removal of component-scale efects from the pairwise normalization terms.

## 3 CorrGRPO: Correlation-Normalized GRPO

Motivated by the scale imbalance identified in Section 2, we propose Correlation-Normalized GRPO (CorrGRPO), which replaces the covariance terms in GRPO’s denominator with sample Pearson correlation coeficients:

$$
A _ { \mathrm { C o r r G R P O } } ^ { i } = \frac { \sum _ { l = 1 } ^ { r } ( R _ { l } ^ { i } - \bar { R } _ { l } ) } { \sqrt { \sum _ { l = 1 } ^ { r } \sum _ { m = 1 } ^ { r } \hat { \rho } _ { l m } } + \varepsilon } , \qquad \hat { \rho } _ { l m } = \frac { \widehat { \mathrm { C o v } } ( R _ { l } , R _ { m } ) } { \sqrt { \widehat { \mathrm { V a r } } ( R _ { l } ) \widehat { \mathrm { V a r } } ( R _ { m } ) } } .\tag{2}
$$

All statistics are computed within the same rollout group, following Section 2. For zerovariance reward components, we set the corresponding rows and columns of the correlation matrix to zero, including diagonal entries, as shown in the core implementation in Appendix B. The resulting matrix remains positive semidefinite, ensuring a nonnegative correlation sum and a strictly positive denominator when $\varepsilon > 0$ , with derivation in Appendix L.5.

Replacing covariances with correlations removes reward-scale weighting from each pairwise term. Each reward contributes one to the diagonal, while of-diagonal entries reflect the strength and direction of reward dependence. Consequently, strong correlations between small-scale rewards can substantially influence normalization even when other components have much larger variances. For example, in Figure 2, CorrGRPO assigns a correlation term of 0.9756 to the strongly correlated rewards $r _ { 1 }$ and $r _ { 2 } .$ , compared with 0.1098 for each weak relationship involving $r _ { 3 }$ . The stronger correlation contributes approximately 8.89 times as much to the sum inside the denominator.

Table 1: Coding RL results on LeetCodeDataset, HumanEval, MBPP, and LCB v6. Eficiency is the mean percentage of eligible programs that run faster than the reference code. Executable denotes running time success rate. Avg. is the arithmetic mean of the Pass@1 evaluation results across 4 datasets.
<table><tr><td rowspan="2">Model</td><td colspan="3">LeetCodeDataset (In-dataset Eval)</td><td>HumanEval</td><td>MBPP</td><td>LCB v6</td><td>Avg.</td></tr><tr><td>Efficiency</td><td>Executable</td><td>Pass@1</td><td>Pass@1</td><td>Pass@1</td><td>Pass@1</td><td>Pass@1</td></tr><tr><td>Qwen2.5-Coder-0.5B-Instruct</td><td>50.00</td><td>61.40</td><td>1.75</td><td>54.88</td><td>33.60</td><td>5.14</td><td>23.84</td></tr><tr><td>+ GRPO</td><td>60.71</td><td>87.72</td><td>3.07</td><td>57.32</td><td>40.00</td><td>3.43</td><td>25.96</td></tr><tr><td>+ GDPO</td><td>60.94</td><td>88.60</td><td>3.51</td><td>57.32</td><td>38.40</td><td>4.57</td><td>25.95</td></tr><tr><td>+ CorrGRPO</td><td>67.86</td><td>89.04</td><td>3.95</td><td>59.76</td><td>42.20</td><td>6.29</td><td>28.05</td></tr><tr><td>Qwen2.5-Coder-1.5B-Instruct</td><td>58.93</td><td>75.44</td><td>3.07</td><td>61.59</td><td>54.40</td><td>10.86</td><td>32.48</td></tr><tr><td>+ GRPO</td><td>40.00</td><td>84.21</td><td>6.58</td><td>64.63</td><td>58.00</td><td>13.14</td><td>35.59</td></tr><tr><td>+ GDPO</td><td>45.00</td><td>85.09</td><td>7.02</td><td>63.41</td><td>58.60</td><td>12.57</td><td>35.40</td></tr><tr><td>+ CorrGRPO</td><td>46.71</td><td>86.84</td><td>8.77</td><td>68.90</td><td>57.60</td><td>10.29</td><td>36.39</td></tr><tr><td>Qwen2.5-Coder-3B-Instruct</td><td>54.41</td><td>77.63</td><td>7.46</td><td>82.32</td><td>60.80</td><td>17.14</td><td>41.93</td></tr><tr><td>+ GRPO</td><td>56.00</td><td>83.77</td><td>12.28</td><td>80.49</td><td>63.80</td><td>16.57</td><td>43.29</td></tr><tr><td>+ GDPO</td><td>48.96</td><td>83.77</td><td>10.96</td><td>81.71</td><td>64.40</td><td>16.57</td><td>43.41</td></tr><tr><td>+ CorrGRPO</td><td>60.00</td><td>81.58</td><td>14.47</td><td>82.32</td><td>66.00</td><td>19.43</td><td>45.56</td></tr><tr><td>Qwen2.5-Coder-7B-Instruct</td><td>44.35</td><td>80.70</td><td>14.91</td><td>78.66</td><td>76.40</td><td>20.57</td><td>47.64</td></tr><tr><td>+ GRPO</td><td>44.70</td><td>85.53</td><td>15.79</td><td>73.78</td><td>78.40</td><td>21.14</td><td>47.28</td></tr><tr><td>+ GDPO</td><td>41.28</td><td>84.21</td><td>16.23</td><td>76.83</td><td>79.60</td><td>22.86</td><td>48.88</td></tr><tr><td>+ CorrGRPO</td><td>51.42</td><td>80.70</td><td>24.12</td><td>78.66</td><td>78.60</td><td>24.57</td><td>51.49</td></tr></table>

## 4 Experiments

We evaluate whether CorrGRPO improves task performance when language models are trained with multiple reward signals. Our experiments cover code generation, tool calling, and agent utility and security. We compare CorrGRPO with GRPO (Shao et al., 2024) and the corresponding base models, and additionally include GDPO (Liu et al., 2026) in the coding experiments. We report both overall task performance and individual reward-related metrics to examine how CorrGRPO balances diferent objectives. Furthermore, we show our training dynamics in Figure 4 and provide detailed training settings in Appendix I.

## 4.1 Coding Reasoning

Task. We study coding RL with Python code generation. The goal is to produce functionally correct programs while improving execution eficiency. We evaluate generated programs using executable tests and compare the runtime of passing solutions with that of the reference implementations.

Reward design. Let $y _ { g }$ denote the generated program and $y _ { r }$ the reference implementation. We combine seven reward components:

$$
\begin{array} { r l } & { R _ { \mathrm { c o d e } } ( y _ { g } , y _ { r } ) = 0 . 0 5 R _ { \mathrm { f m t } } ( y _ { g } ) + 0 . 0 5 R _ { \mathrm { s y n } } ( y _ { g } ) + 0 . 0 5 R _ { \mathrm { c o m p i l e } } ( y _ { g } ) + 0 . 2 5 R _ { \mathrm { r u n } } ( y _ { g } ) } \\ & { \phantom { R a c a c } + 0 . 6 0 R _ { \mathrm { p a s s } } ( y _ { g } ) + 0 . 3 0 R _ { \mathrm { a s t } } ( y _ { g } , y _ { r } ) + 0 . 2 0 R _ { \mathrm { e f f } } ( y _ { g } , y _ { r } ) . } \end{array}\tag{3}
$$

The format reward $R _ { \mathrm { f m t } }$ checks whether the response follows the required code format. The execution-related rewards follow a dependency chain:

$$
\underbrace { \mathrm { S y n t a x ~ v a l i d i t y } } _ { R _ { \mathrm { s y n } } } \to \underbrace { \mathrm { S u c c e s s f u l ~ c o m p i l a t i o n } } _ { R _ { \mathrm { c o m p l i e } } } \to \underbrace { \mathrm { R u n t i m e ~ s u c c e s s } } _ { R _ { \mathrm { r u n } } } \to \underbrace { \mathrm { A l l ~ t e s t s ~ p a s s e d } } _ { R _ { \mathrm { p a s s } } } .
$$

These rewards are binary and dependent: the compilation reward requires valid syntax, the runtime reward requires successful compilation, and the all-pass reward requires execution without runtime errors. Improving earlier checks therefore enables rewards from subsequent checks. Given the set of test cases C, the final correctness reward is $\begin{array} { r } { R _ { \mathrm { p a s s } } ( y _ { g } ) \ = \ \prod _ { c \in { \mathcal C } } \mathbf { 1 } [ y _ { g } } \end{array}$ passes c] . Thus, $R _ { \mathrm { p a s s } } ( y _ { g } ) ~ = ~ 1$ only when every test passes, while syntax, compilation, and runtime rewards provide intermediate feedback even when $R _ { \mathrm { p a s s } } ( y _ { g } ) = 0$ . The structural reward $R _ { \mathrm { a s t } } ( y _ { g } , y _ { r } ) \in [ 0 , 1 ]$ measures the similarity between the generated and reference abstract syntax trees, independently of identifier names and literal values. Appendix J.1 describes the tree representation and similarity computation. Finally, the eficiency reward encourages faster execution among functionally correct programs: $\begin{array} { r } { R _ { \mathrm { e f f } } ( y _ { g } , y _ { r } ) \stackrel { } { = } \mathbb { I } [ R _ { \mathrm { p a s s } } ( y _ { g } ) = 1 ] \cdot \mathrm { c l i p } \Big ( 1 - \frac { t ( y _ { g } ) } { t ( y _ { r } ) } , 0 , 1 \Big ) } \end{array}$ , where $t ( y _ { g } )$ is the execution time of the generated program and $t ( y _ { r } )$ is the mean execution time over four independent runs of the reference implementation. Both are measured using the same test harness.

![](images/4fc017b1af54597afe8535bfd9915015836f7b5913a812bf42f37291add034b3.jpg)  
Figure 3: Pareto frontier of correctness–eficiency tradeof on LeetCodeDataset.

Experimental setting. We use Qwen2.5-Coder-Instruct (Hui et al., 2024) at 0.5B, 1.5B, 3B, and 7B scales. Models are trained on the 2,641-problem training split of LeetCode-Dataset (Xia et al., 2025) and evaluated on its 228-problem test split. This dataset contains programming problems paired with reference solutions and executable tests. Training hyperparameters are provided in Appendix I.1. To assess generalization without further training, we additionally evaluate HumanEval (Chen et al., 2021), which tests function completion from natural-language specifications; MBPP (Austin et al., 2021), which contains basic Python programming tasks; and LiveCodeBench v6 (Jain et al., 2024), which contains competition programming problems. Table 1 reports Executable, Pass@1, and Eficiency on LeetCodeDataset, together with Pass@1 on the three additional benchmarks. Executable measures the fraction of generated programs that execute without runtime errors, and Pass@1 measures the fraction that pass all tests. Eficiency measures the percentage of eligible generated programs that run strictly faster than their reference implementations, i.e., speedup $= t _ { \mathrm { r e f } } / t _ { \mathrm { g e n } } > 1$ . A pair is eligible when both programs pass all tests and have positive, finite runtimes. We re-execute both programs in eight paired rounds on the same CPU core, alternating their order, and report the mean of the eight per-round percentages.

Main Results. CorrGRPO consistently improves coding correctness across model scales and extends the empirical correctness–eficiency Pareto frontier. Table 1 shows that CorrGRPO achieves the highest LeetCodeDataset Pass@1 and average benchmark Pass@1 at every evaluated model scale, outperforming both GRPO and GDPO. Relative to GRPO, average Pass@1 increases by 2.09, 0.80, 2.27, and 4.21 percentage points at 0.5B, 1.5B, 3B, and 7B, respectively. Figure 3 further shows that CorrGRPO accounts for every nondominated configuration across the evaluated methods and model scales. Its 7B model achieves the highest Pass@1 at 24.12%, its 0.5B model achieves the highest Eficiency at 67.86%, and its 3B model occupies an intermediate frontier point with 14.47% Pass@1 and 60.00% Eficiency.

## 4.2 Tool Calling

Task. We study structured tool calling, where a model selects the appropriate functions and generates their arguments from a user request and the available tool descriptions. Successful tool use requires jointly identifying the correct function, supplying the required parameter names, and assigning the correct parameter values. We evaluate both individual field correctness and complete-call correctness.

Table 2: Tool-call results on RLLA-4K and API-Bank. API-Bank scores use the same all exact metric for the correctness of final tool call. The rightmost Avg. is the arithmetic mean of RLLA-4K all-exact score and API-Bank Avg. score.
<table><tr><td rowspan="2">Model</td><td colspan="4">RLLA-4K (In-dataset Eval)</td><td colspan="4">API-Bank</td><td rowspan="2">Avg.</td></tr><tr><td>Function</td><td>Param. Name</td><td>Param. Value</td><td>All Exact</td><td>V1</td><td>V2</td><td>V3</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>73.17</td><td>69.77</td><td>54.78</td><td>38.03</td><td>17.29</td><td>19.26</td><td>24.49</td><td>20.35</td><td>29.19</td></tr><tr><td>+ GRPO</td><td>97.65</td><td>95.60</td><td>76.91</td><td>63.38</td><td>79.45</td><td>44.44</td><td>35.92</td><td>53.27</td><td>58.33</td></tr><tr><td>+ CorrGRPO</td><td>95.07</td><td>94.95</td><td>82.14</td><td>67.61</td><td>81.45</td><td>46.67</td><td>35.10</td><td>54.41</td><td>61.01</td></tr><tr><td>Qwen3-4B-Thinking-2507</td><td>33.80</td><td>31.46</td><td>25.35</td><td>21.13</td><td>53.38</td><td>39.26</td><td>31.43</td><td>41.36</td><td>31.25</td></tr><tr><td>+ GRPO</td><td>94.84</td><td>92.08</td><td>72.07</td><td>57.75</td><td>79.20</td><td>52.59</td><td>52.24</td><td>61.34</td><td>59.55</td></tr><tr><td>+ CorrGRPO</td><td>93.78</td><td>92.31</td><td>73.74</td><td>59.15</td><td>78.45</td><td>53.33</td><td>64.08</td><td>65.29</td><td>62.22</td></tr><tr><td>Qwen3-8B</td><td>86.85</td><td>84.80</td><td>67.50</td><td>50.70</td><td>74.69</td><td>49.63</td><td>46.53</td><td>56.95</td><td>53.83</td></tr><tr><td>+ GRPO</td><td>92.72</td><td>91.43</td><td>72.54</td><td>61.97</td><td>78.20</td><td>51.85</td><td>40.82</td><td>56.95</td><td>59.46</td></tr><tr><td>+ CorrGRPO</td><td>97.65</td><td>96.60</td><td>76.44</td><td>63.38</td><td>81.70</td><td>51.85</td><td>44.08</td><td>59.21</td><td>61.30</td></tr></table>

Reward design. Let $y _ { g }$ denote the generated response and $y _ { r }$ the reference response. We combine four reward components, suppressing their shared arguments $( y _ { g } , y _ { r } )$ for brevity:

$$
R _ { \mathrm { t o o l } } ( y _ { g } , y _ { r } ) = R _ { \mathrm { f n } } + R _ { \mathrm { p n } } + R _ { \mathrm { p v } } + R _ { \mathrm { f m t } } .\tag{4}
$$

Here, $R _ { \mathrm { f n } } , R _ { \mathrm { p n } } ,$ and $R _ { \mathrm { p v } }$ evaluate function names, parameter names, and parameter values, respectively, and $R _ { \mathrm { f m t } }$ evaluates output format. Let $S _ { \mathrm { f n } } , S _ { \mathrm { p n } } , S _ { \mathrm { p v } } \in [ 0 , 1 ]$ ] denote their match ing scores. The reward components use the following scales: $R _ { \mathrm { f n } } { \dot { = } } 0 . 5 ( 2 S _ { \mathrm { f n } } - 1 ) , \ R _ { \mathrm { p n } } =$ $2 \bar { S _ { \mathrm { p n } } } - 1 , \ R _ { \mathrm { p v } } = 1 . 5 ( 2 S _ { \mathrm { p v } } - 1 )$ . These rewards provide partial credit for incomplete calls. Parameter matching requires a matching function, and value correctness requires the corresponding parameter name. Appendix J.2 provides the matching procedure and score definitions. The format reward is 1 when the response follows the required structure and 0 otherwise. We provide the response format template in Appendix J.4.

Experimental setting. We evaluate Qwen2.5-7B-Instruct (Yang et al., 2024), Qwen3- 4B-Thinking-2507 (Qwen Team, 2025), and Qwen3-8B (Yang et al., 2025), comparing their base, GRPO, and CorrGRPO. We train on RLLA-4K from ToolRL (Qian et al., 2025), which contains user requests, tool descriptions, and reference responses specifying tool calls or direct answers. Our split contains 3,920 training examples and 80 test examples. To assess generalization without further training, we evaluate API-Bank (Li et al., 2023), a benchmark of tool-use dialogues with executable APIs. Table 2 reports normalized functionname, parameter-name, and parameter-value scores on RLLA-4K, with the all-exact score, the percentage of examples for which all tool-call fields are correct. API-Bank results cover v1, v2, and v3, with final-call correctness assessed using its execution-based checker.

Main Results. CorrGRPO improves complete-call correctness and generalization across all three evaluated backbones. Table 2 shows that CorrGRPO achieves the highest RLLA-4K and API-Bank average all-exact scores for every backbone. Compared with GRPO, RLLA-4K all-exact score increases from 63.38% to 67.61% for Qwen2.5-7B, from 57.75% to 59.15% for Qwen3-4B-Thinking, and from 61.97% to 63.38% for Qwen3-8B. These improvements accompany higher parameter-value scores across all three models, supporting more accurate argument generation. On API-Bank, the average score improves by 1.14, 3.94, and 2.26 percentage points, respectively. The largest gain occurs for Qwen3-4B-Thinking on API-Bank v3, where correctness increases from 52.24% to 64.08%. Together, these improvements raise the overall average by 2.68, 2.67, and 1.84 percentage points, demonstrating gains in both complete-call accuracy on the training-domain benchmark and transfer to API-Bank.

## 4.3 Agent Utility vs. Security

Task. We study tool-using agents that must complete legitimate user tasks while resisting indirect prompt injections embedded in tool outputs. An agent interacts with an environment through multiple tool calls, where malicious instructions may attempt to redirect its actions toward an attacker’s objective. We evaluate both task completion and attack success to determine whether the agent remains useful under adversarial interference.

Table 3: Agent utility and security results. Utility measures the agent’s tool-calling success rate, while ASR measures the success rate of prompt injection attacks. Joint accuracy combines utility and security as (1 − attack\_success) × <sup>clean\_utility+utility\_under\_attack</sup> . In 2 InjecAgent, all reported scores represent ASR.
<table><tr><td colspan="8">AgentDojo (In-dataset Eval) / Agent Security Bench</td></tr><tr><td>Model</td><td>Clean Utility ↑</td><td></td><td>Utility Under Attack ↑</td><td></td><td>ASR↓</td><td></td><td>Joint Accuracy ↑</td></tr><tr><td>Qwen2.5-3B-Instruct</td><td>28.57</td><td>/ 10.00</td><td>14.62</td><td>/6.25</td><td>3.85 8.00</td><td></td><td>18.97 6.13</td></tr><tr><td>+ GRPO</td><td>52.38</td><td>21.75</td><td>66.92</td><td>9.75</td><td>0.00 9.00</td><td>60.13</td><td>14.88</td></tr><tr><td>+ CorrGRPO</td><td>95.24</td><td>26.00</td><td>84.36</td><td>13.25</td><td>1.54 9.25</td><td>89.62</td><td>18.13</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>38.10</td><td>64.75</td><td>30.51</td><td>47.00</td><td>11.28 33.25</td><td>28.72</td><td>42.00</td></tr><tr><td>+ GRPO</td><td>90.48</td><td>30.00</td><td>80.00</td><td>41.75</td><td>0.26 26.50</td><td>84.49</td><td>31.38</td></tr><tr><td>+ CorrGRPO</td><td>85.71</td><td>59.75</td><td>83.08</td><td>53.25</td><td>1.28 29.00</td><td>86.03</td><td>47.88</td></tr><tr><td>Qwen3-8B</td><td>71.43</td><td>60.00</td><td>32.31</td><td>51.50</td><td>19.49</td><td>31.25</td><td>40.38 49.63</td></tr><tr><td>+ GRPO</td><td>85.71</td><td>60.00</td><td>77.18</td><td>61.50</td><td>2.05 19.50</td><td>80.26</td><td>57.38</td></tr><tr><td>+ CorrGRPO</td><td>85.71</td><td>70.00</td><td>75.13</td><td>61.75</td><td>1.54</td><td>22.50 79.62</td><td>58.63</td></tr></table>

<table><tr><td rowspan="2">Model</td><td rowspan="2">Direct Harm ↓</td><td colspan="3">Data Stealing</td><td rowspan="2">Total ↓</td></tr><tr><td>S1 ↓</td><td>S2 ↓</td><td>Total ↓</td></tr><tr><td>Qwen2.5-3B-Instruct</td><td>18.38</td><td>50.29</td><td>76.39</td><td>34.59</td><td>27.12</td></tr><tr><td>+ GRPO</td><td>5.03</td><td>20.11</td><td>79.31</td><td>13.07</td><td>8.80</td></tr><tr><td>+ CorrGRPO</td><td>4.67</td><td>17.26</td><td>52.63</td><td>6.33</td><td>5.52</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>33.85</td><td>44.21</td><td>92.24</td><td>38.91</td><td>36.17</td></tr><tr><td>+ GRPO</td><td>16.49</td><td>15.06</td><td>75.00</td><td>10.12</td><td>13.53</td></tr><tr><td>+ CorrGRPO</td><td>6.91</td><td>9.77</td><td>60.00</td><td>5.44</td><td>6.17</td></tr><tr><td>Qwen3-8B</td><td>30.56</td><td>42.22</td><td>92.13</td><td>38.79</td><td>34.86</td></tr><tr><td>+ GRPO</td><td>13.45</td><td>16.36</td><td>88.75</td><td>14.37</td><td>13.94</td></tr><tr><td>+ CorrGRPO</td><td>6.94</td><td>12.02</td><td>74.19</td><td>8.91</td><td>7.98</td></tr></table>

Reward design. For a valid attacked trajectory, let $y _ { g }$ include the agent’s responses, tool calls, and observations, with prompt-injection content inserted into the tool observations. We assign equal weights to utility and security:

$$
R _ { \mathrm { a g e n t } } ( y _ { g } ) = R _ { \mathrm { u t i l } } ( y _ { g } ) + R _ { \mathrm { s e c } } ( y _ { g } ) .\tag{5}
$$

The utility reward $R _ { \mathrm { u t i l } }$ is 1 if the agent successfully completes the user’s task under attack and 0 otherwise. The security reward $R _ { \mathrm { s e c } }$ is 1 if the attacker’s objective is not achieved and 0 otherwise. Both rewards are determined by the environment’s task checkers. These objectives can compete: avoiding tool interactions may prevent an attack while also preventing completion of the user’s task. Their combination rewards useful behavior and resistance to malicious instructions.

Experimental setting. We evaluate Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct (Yang et al., 2024), and Qwen3-8B (Yang et al., 2025), comparing their base, GRPO, and CorrGRPO variants. Training takes place in AgentDojo (Debenedetti et al., 2024). We construct a deterministic split grouped by user task, containing 1,584 training cases and 411 test cases; the latter comprise 21 clean cases and 390 attacked cases. To assess generalization without further training, we evaluate Agent Security Bench (ASB; Zhang et al., 2025) using 400 paired clean and attacked cases per model, and InjecAgent (Zhan et al., 2024). Table 3 reports clean utility, utility under attack, attack success rate (ASR), and Joint Accuracy for AgentDojo and ASB. For N paired cases, Joint Accuracy is computed as $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( 1 - \dot { A } _ { i } ) \frac { U _ { \mathrm { c l e a n } , i } + U _ { \mathrm { a t t a c k } , i } } { 2 } } \end{array}$ , where $U _ { \mathrm { c l e a n } , i }$ and $U _ { \mathrm { a t t a c k } , i }$ denote user-task success in the clean and attacked executions, and $A _ { i }$ denotes attack success in the attacked execution.

Main Results. CorrGRPO improves the balance between agent utility and security. Relative to GRPO, CorrGRPO increases ASB Joint Accuracy from 14.88% to 18.13% for Qwen2.5-3B, from 31.38% to 47.88% for Qwen2.5-7B, and from 57.38% to 58.63% for Qwen3-8B (Table 3). These gains accompany higher clean utility and utility under attack for all three models, supporting more efective task completion in adversarial environments.

![](images/168529d739207f41b4569663d852294c1f453ac23aaedaf1cfcc90d754d15068.jpg)  
Figure 4: Validation reward dynamics of GRPO, GDPO, and CorrGRPO. Curves report validation mean@1 scores with exponential moving average smoothing (decay = 0.6).

On InjecAgent, overall ASR decreases from 8.80% to 5.52%, from 13.53% to 6.17%, and from 13.94% to 7.98%, respectively, with reductions in every reported attack category. On AgentDojo, CorrGRPO also improves Joint Accuracy for Qwen2.5-3B and Qwen2.5-7B, with the largest gain at 3B, from 60.13% to 89.62%.

## 5 Related Work

Multi-reward RL algorithms difer primarily in how they aggregate reward signals and balance their contributions during policy optimization. Scalarization-based approaches combine multiple rewards into a single training signal: MORLAIF trains separate preference models for individual objectives and aggregates their scores through scalarization functions before PPO updates (Williams, 2024). To adapt objective tradeofs during training, Safe RLHF maximizes helpfulness subject to safety constraints using dynamically updated Lagrange multipliers (Dai et al., 2023), while dynamic reward weighting adjusts scalarization weights through hypervolume-guided adaptation or gradient-based optimization (Lu et al., 2025). More closely related to our work, recent methods modify how multiple rewards enter group-relative advantage estimation. MO-GRPO automatically reweights reward functions according to their variances to balance their contributions (Ichihara et al., 2025), whereas GDPO independently normalizes each reward within rollout groups, aggregates the resulting advantages, and applies batch-level normalization to stabilize their overall magnitude (Liu et al., 2026). RDPO further addresses reward dependence by combining magnitude-aware quantile normalization with Mahalanobis whitening before aggregation, reducing redundant variation among correlated reward dimensions (Bai et al., 2026). Our CorrGRPO instead retains the centered total reward and its prescribed relative weights, while replacing the covariance-based normalization denominator with an aggregate of Pearson correlations.

## 6 Conclusion

We introduced CorrGRPO, a correlation-normalized variant of GRPO for multi-reward reinforcement learning. Our analysis shows that normalizing the summed reward by its withingroup standard deviation implicitly aggregates all pairwise reward covariances, coupling reward dependence with component scales. CorrGRPO replaces these covariance terms with Pearson correlation coeficients while preserving the centered total reward and its prescribed relative weights. This modification allows advantage normalization to respond to inter-reward correlations without being dominated by reward components with larger within-group standard deviations. Experiments on code generation, tool calling, and agent security, using models ranging from 0.5B to 8B parameters, demonstrate improved performance and an outward expansion of the empirical Pareto frontier between competing objectives. These results highlight the value of explicitly accounting for inter-reward correlations when designing advantage estimators for multi-reward learning.

## AI Use Statement

We used generative AI tools to improve the clarity of expression, language quality, and grammatical correctness of the manuscript. Beyond the language models studied in our experiments, we did not use generative AI tools to develop the proposed method, formulate mathematical claims, write proofs, design experiments, implement methods, generate or process experimental datasets, or interpret results. The authors reviewed and revised all AI-assisted text to ensure accuracy and consistency with the underlying research. We take full responsibility for the final manuscript, including all claims, results, and AI-assisted content.

## Ethics Statement

This work studies multi-reward optimization for language models through existing benchmarks for code generation, tool calling, and agent security. Improving these capabilities may benefit legitimate applications but may also facilitate harmful automation or unauthorized tool use. Our security experiments evaluate resistance to benchmark prompt-injection attacks; improvements on these benchmarks do not establish safety in real-world deployments or against unseen attacks. Moreover, correlation-based normalization does not correct biased or misspecified rewards, and the resulting behavior remains dependent on the selected objectives and their weights. Deployment therefore requires application-specific safety evaluation, appropriate tool-access restrictions, and human oversight.

## Reproducibility Statement

Section 3 defines CorrGRPO, and Appendix B provides its core implementation, including numerical stabilization and the handling of zero-variance reward components. Sections 4.1– 4.3 describe the model backbones, reward functions, experimental settings, and evaluation metrics. Appendix I reports training hyperparameters, Appendix J details reward computation, and Appendix M describes the datasets, evaluation subsets, and benchmark settings. Representative prompts and responses are provided in Appendix H. Appendix L presents the assumptions and derivations underlying the theoretical properties, while Appendix F details compatibility with other optimization methods. These materials support reproduction and inspection of the proposed method and its evaluation.

## References

Jacob Austin et al. Program Synthesis with Large Language Models. arXiv preprint arXiv:2108.07732, 2021. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2108.07732</sub>.

Yang Bai, Kaiyuan Liu, Ziyuan Zhuang, Jiahong Zhou, Rongxiang Weng, Xin Chen, Jingang Wang, and Xunliang Cai. Multi-objective and mixed-reward reinforcement learning via reward-decorrelated policy optimization. arXiv preprint arXiv:2605.13641, 2026. doi: 10.48550/arXiv.2605.13641. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2605.13641</sub>.

Mark Chen et al. Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374, 2021. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2107.03374</sub>.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models, 2025. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2505.22617</sub>.

Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe RLHF: Safe reinforcement learning from human feedback. arXiv preprint arXiv:2310.12773, 2023. doi: 10.48550/arXiv.2310.12773. URL <sub>https:</sub>//<sub>arxiv.</sub> <sub>org</sub>/<sub>abs</sub>/<sub>2310.12773</sub>.

Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr. AgentDojo: A dynamic environment to evaluate prompt injection attacks and defenses for LLM agents. arXiv preprint arXiv:2406.13352, 2024. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2406.13352</sub>.

DeepSeek-AI et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/ <sub>2501.12948</sub>.

Binyuan Hui et al. Qwen2.5-Coder Technical Report. arXiv preprint arXiv:2409.12186, 2024. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2409.12186</sub>.

Yuki Ichihara, Yuu Jinnai, Tetsuro Morimura, Mitsuki Sakamoto, Ryota Mitsuhashi, and Eiji Uchibe. MO-GRPO: Mitigating reward hacking of group relative policy optimization on multi-objective problems. arXiv preprint arXiv:2509.22047, 2025. doi: 10.48550/ arXiv.2509.22047. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2509.22047</sub>.

Naman Jain et al. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code. arXiv preprint arXiv:2403.07974, 2024. URL <sub>https:</sub>// <sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2403.07974</sub>.

Minghao Li et al. API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. arXiv preprint arXiv:2304.08244, 2023. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2304.08244</sub>.

Shih-Yang Liu et al. GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization. arXiv preprint arXiv:2601.05242, 2026. URL <sub>https:</sub> //<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2601.05242</sub>.

Yining Lu, Zilong Wang, Shiyang Li, Xin Liu, Changlong Yu, Qingyu Yin, Zhan Shi, Zixuan Zhang, and Meng Jiang. Learning to optimize multi-objective alignment through dynamic reward weighting. arXiv preprint arXiv:2509.11452, 2025. doi: 10.48550/arXiv.2509. 11452. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2509.11452</sub>.

MiniMax. MiniMax-M1: Scaling test-time compute eficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2506.13585</sub>.

Long Ouyang et al. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2203.02155</sub>.

Python Software Foundation. Python Documentation: diflib—Helpers for Computing Deltas. URL <sub>https:</sub>//<sub>docs.python.org</sub>/<sub>3</sub>/<sub>library</sub>/<sub>difflib.html</sub>. Accessed 2026-09- 14.

Cheng Qian et al. ToolRL: Reward is All Tool Learning Needs. arXiv preprint arXiv:2504.13958, 2025. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2504.13958</sub>.

Qwen Team. Qwen3-4B-Thinking-2507. Model card, 2025. URL <sub>https:</sub>//<sub>huggingface.</sub> <sub>co</sub>/<sub>Qwen</sub>/<sub>Qwen3-4B-Thinking-2507</sub>.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2402.03300</sub>.

Marcus Williams. Multi-objective reinforcement learning from AI feedback. arXiv preprint arXiv:2406.07295, 2024. doi: 10.48550/arXiv.2406.07295. URL <sub>https:</sub>//<sub>arxiv.org</sub>/ <sub>abs</sub>/<sub>2406.07295</sub>.

Yunhui Xia et al. LeetCodeDataset: A Temporal Dataset for Robust Evaluation and Efficient Training of Code LLMs. arXiv preprint arXiv:2504.14655, 2025. URL <sub>https:</sub> //<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2504.14655</sub>.

An Yang et al. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115, 2024. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2412.15115</sub>.

An Yang et al. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388, 2025. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2505.09388</sub>.

Qiying Yu, Zheng Zhang, Ruofei Zhu, et al. DAPO: An open-source LLM reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. URL <sub>https:</sub>//<sub>arxiv.</sub> <sub>org</sub>/<sub>abs</sub>/<sub>2503.14476</sub>.

Qiusi Zhan, Zhixiang Liang, Zifan Ying, and Daniel Kang. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents. arXiv preprint arXiv:2403.02691, 2024. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2403.02691</sub>.

Hanrong Zhang et al. Agent Security Bench (ASB): Formalizing and Benchmarking At tacks and Defenses in LLM-based Agents. In The Thirteenth International Conference on Learning Representations, 2025. URL <sub>https:</sub>//<sub>arxiv.org</sub>/<sub>abs</sub>/<sub>2410.02644</sub>.

## A Further Discussion

Preserving groupwise gradient directions. Let $L _ { \mathrm { G R P O } , q }$ and $L _ { \mathrm { C o r r G R P O } , q }$ denote the clipped reward surrogates averaged over tokens and trajectories in a fixed rollout group for prompt q, excluding the KL term. Let $\mathbf { C } = \left[ \hat { \rho } _ { l m } \right]$ be the group’s sample correlation matrix, $\mathbf { \hat { s } } = ( \hat { \sigma } _ { 1 } , \hdots , \hat { \sigma } _ { r } ) ^ { \top }$ its vector of reward standard deviations, and 1 the all-ones vector. As derived in Appendix L.3, Equation 44, the shared numerator gives

$$
A _ { \mathrm { C o r r G R P O } } ^ { i } = c _ { q } A _ { \mathrm { G R P O } } ^ { i } , \qquad c _ { q } = \frac { \sqrt { \mathbf { s } ^ { \intercal } \mathbf { C } \mathbf { s } } + \varepsilon } { \sqrt { \mathbf { 1 } ^ { \intercal } \mathbf { C } \mathbf { 1 } } + \varepsilon } > 0 .\tag{6}
$$

The coeficient $c _ { q }$ is computed from the same group’s reward statistics, shared across its trajectories, and held fixed during surrogate optimization. Because the clipped surrogate is positively homogeneous in the advantage, this scaling preserves the active clipping branches and yields

$$
\nabla _ { \theta } L _ { \mathrm { C o r r G R P O } , q } = c _ { q } \nabla _ { \theta } L _ { \mathrm { G R P O } , q } , \qquad c _ { q } > 0 .\tag{7}
$$

The derivation is given in Appendix L.3, Equations 48–50. Thus, at the same policy parameters, CorrGRPO preserves the direction of each group’s nonzero reward-driven gradient while adapting its magnitude to reward dependence. The clipping thresholds remain unchanged. Since $c _ { q }$ can difer across groups, their relative contributions to the batch gradient can change; the KL term retains its original coeficient.

A quadratic-norm view of reward fluctuations. Using the same correlation matrix C, define $\| \mathbf { x } \| _ { \mathbf { C } } = { \sqrt { \mathbf { x } ^ { \top } \mathbf { C } \mathbf { x } } }$ , which is a norm when C is positive definite and a seminorm when it is singular. The two normalization statistics use the same correlation metric with diferent input vectors:

$$
\begin{array} { r l } { \mathrm { G R P O } \colon } & { \widehat { \mathrm { V a r } } ( R ) = \mathbf { s } ^ { \top } \mathbf { C } \mathbf { s } = \| \mathbf { s } \| _ { \mathbf { C } } ^ { 2 } , } \\ { \mathrm { C o r r G R P O } \colon } & { \displaystyle \sum _ { l , m } \hat { \rho } _ { l m } = \mathbf { 1 } ^ { \top } \mathbf { C } \mathbf { 1 } = \| \mathbf { 1 } \| _ { \mathbf { C } } ^ { 2 } . } \end{array}\tag{8}
$$

Their advantage denominators are consequently $\| \mathbf { s } \| _ { \mathbf { C } } + \varepsilon$ and $\| \mathbf { 1 } \| _ { \mathbf { C } } + \varepsilon$ , respectively.

This admits a signal-processing interpretation. Treat the standardized, centered reward components as correlated input channels to a linear combiner. For a channel-gain vector $\mathbf { x } ,$ the output’s sample variance is $\mathbf { x } ^ { \top } \mathbf { C x }$ . Diagonal terms measure individual channel contri butions, while of-diagonal terms capture reinforcement or cancellation between channels. GRPO uses $\mathbf { x } = \mathbf { s } ,$ restoring each source’s own fluctuation amplitude before measuring the combined output. Its normalization therefore depends on both the correlation structure and the individual source amplitudes. CorrGRPO uses ${ \bf x } = { \bf 1 }$ , giving all standardized channels equal gain so that the combined fluctuation reflects their correlations without additional weighting by their original scales.

Compatibility with other policy optimization methods. CorrGRPO is compatible with other RL policy optimization methods, including DAPO (Yu et al., 2025), CISPO (MiniMax, 2025), and GDPO (Liu et al., 2026). For DAPO and CISPO, the original grouprelative advantage can be directly replaced with $A _ { \mathrm { C o r r G R P O } } ^ { i }$ , while retaining their respective policy objectives and clipping operations. For GDPO, we retain its per-reward group normalization and aggregation, and replace only the final batch standard-deviation denominator with a correlation norm computed across the batch from the resulting per-reward advantage components. The corresponding objectives are derived in Appendices F.2.1, F.2.2, and F.2.3, respectively. We also evaluate this integration with CISPO on Qwen2.5-Coder-7B-Instruct, with results reported in Appendix F.1.

Slower entropy decay. Figure 12 shows that CorrGRPO preserves higher policy entropy over training, with slower entropy decay particularly evident on AgentDojo and LeetCode-Dataset. Although this pattern varies across settings, these results suggest that CorrGRPO can sustain policy entropy for longer during optimization. As shown by Cui et al. (2025), entropy collapse diminishes exploration and accompanies performance saturation, while entropy control enables continued exploration and improves downstream performance. From this perspective, the higher entropy retained by CorrGRPO helps explain its stronger performance: maintaining policy diversity preserves opportunities to explore alternative reasoning paths and discover higher-reward responses, supporting continued improvement beyond initially successful behaviors.

## B Core Implementation of CorrGRPO

We compare the advantage implementation for GRPO and CorrGRPO in Listing 1. For each prompt group, <sub>group\_scores</sub> contains the n × r reward components with their configured weights already applied. The implementation uses sample covariance with divisor n − 1, assigns zero rows and columns to zero-variance components, and clamps the correlation sum to zero to guard against negative round-of. The listing follows the source’s numerical convention: CorrGRPO uses $\begin{array} { r } { \sqrt { \operatorname* { m a x } ( \sum _ { l , m } \hat { \rho } _ { l m } , 0 ) + \epsilon } } \end{array}$ with $\epsilon = 1 0 ^ { - 6 }$ , whereas the paper’s formula and the GRPO implementation place ϵ outside the square root.

Listing 1: GRPO to CorrGRPO: abbreviated groupwise dif. Red lines are removed and green lines added; reward loading and group indexing are omitted.

```diff
--- GRPO
+++ CorrGRPO
# group_scores: [n, r], weighted rewards; n > 1
# Executed within torch.no_grad().
centered_scores = group_scores - group_scores.mean(0, keepdim=True)
centered_total_reward = centered_scores.sum(dim=-1)
-denominator = group_scores.sum(dim=-1).std() + epsilon
+covariance = centered_scores.T @ centered_scores / (group_scores.size(0) - 1)
+variances = covariance.diagonal()
+inverse_std = torch.where(
+ variances > 0,
+ torch.rsqrt(variances),
+ torch.zeros_like(variances),
+)
+covariance_coefficients = (
+ covariance * inverse_std[:, None] * inverse_std[None, :]
+)
+coefficient_sum = covariance_coefficients.sum().clamp_min(0)
+denominator = torch.sqrt(coefficient_sum + epsilon)
group_advantages = centered_total_reward / denominator
scalar_advantages.index_copy_(0, positions_tensor, group_advantages)
advantages = scalar_advantages.unsqueeze(-1) * response_mask
return advantages, advantages
```

## C Correlation Analysis for Two Dependent Rewards

This section examines how two dependent rewards jointly shape CorrGRPO’s advantage normalization as training progresses. We use runtime success and functional correctness in coding RL as an example, deriving their correlation and examining training dynamics on LeetCodeDataset.

Correlation analysis. Le $\begin{array} { r } { \mathrm { ~ ; ~ } R _ { \mathrm { r u n } } \in \{ 0 , 1 \} } \\ { R _ { \mathrm { p a s s } } \in \{ 0 , 1 \} } \end{array}$ indicate whether a generated program executes without runtime errors, and indicate whether it passes all test cases. Since passing all tests requires successful execution, $R _ { \mathrm { r u n } } ^ { i } R _ { \mathrm { p a s s } } ^ { i } = R _ { \mathrm { p a s s } } ^ { i }$ . Denoting their withingroup means by $\bar { R } _ { \mathrm { r u n } }$ and $R _ { \mathrm { p a s s } }$ , their Pearson correlation, when both variances are nonzero, satisfies

$$
\widehat { \rho } _ { \mathrm { r u n , p a s s } } = \frac { \widehat { \mathrm { C o v } } ( R _ { \mathrm { r u n } } , R _ { \mathrm { p a s s } } ) } { \sqrt { \widehat { \mathrm { V a r } } ( R _ { \mathrm { r u n } } ) \widehat { \mathrm { V a r } } ( R _ { \mathrm { p a s s } } ) } } = \frac { \bar { R } _ { \mathrm { p a s s } } - \bar { R } _ { \mathrm { r u n } } \bar { R } _ { \mathrm { p a s s } } } { \sqrt { \bar { R } _ { \mathrm { r u n } } ( 1 - \bar { R } _ { \mathrm { r u n } } ) \bar { R } _ { \mathrm { p a s s } } ( 1 - \bar { R } _ { \mathrm { p a s s } } ) } } = \sqrt { \frac { \bar { R } _ { \mathrm { p a s s } } ( 1 - \bar { R } _ { \mathrm { r u n } } ) } { \bar { R } _ { \mathrm { r u n } } ( 1 - \bar { R } _ { \mathrm { p a s s } } ) } } .\tag{9}
$$

Once runtime success stabilizes at a fixed level below one, increasing the mean pass reward $\bar { R } _ { \mathrm { p a s s } }$ increases the correlation $\widehat { \rho } _ { \mathrm { r u n , p a s s } }$ . With equal reward weights, CorrGRPO computes

$$
A _ { \mathrm { C o r r G R P O } } ^ { i } = \frac { \left( R _ { \mathrm { r u n } } ^ { i } - \bar { R } _ { \mathrm { r u n } } \right) + ( R _ { \mathrm { p a s s } } ^ { i } - \bar { R } _ { \mathrm { p a s s } } ) } { \sqrt { 2 + 2 \widehat { \rho } _ { \mathrm { r u n , p a s s } } } + \varepsilon } .\tag{10}
$$

bAs the two rewards become increasingly correlated, CorrGRPO enlarges the denominator and attenuates their combined signal for a fixed centered total reward, moderating the efect of redundant reward information on the policy update. In the limiting case $R _ { \mathrm { r u n } } = R _ { \mathrm { p a s s } } ,$ ignoring ε gives $A _ { \mathrm { C o r r G R P O } } ^ { i } = R _ { \mathrm { r u n } } ^ { i } - \bar { R } _ { \mathrm { r u n } }$ : duplicating the same reward does not double the advantage.

Experimental analysis. We train on LeetCodeDataset using only runtime-success and all-test passing rewards with equal weights, $R _ { \mathrm { c o d e } } = R _ { \mathrm { r u n } } + R _ { \mathrm { p a s s } }$ As shown in Figure 5, runtime success improves rapidly and approximately converges first, remaining around 0.92– 0.96 after about 100 steps. During this plateau, the pass rate continues to rise from approximately 0.36 to 0.54, while the reported reward correlation also increases overall. This trend qualitatively agrees with the analysis: as more executable programs pass all tests, the two reward signals increasingly overlap. Under CorrGRPO’s normalization, stronger correlation increases the denominator and reduces the shared advantage scale, moderating the reinforcement from these overlapping signals while preserving their equal weights.

![](images/3f5b4a4303e60a69033876f8fa86eb22c076185978e08f95c88a0bc5c7547285.jpg)

![](images/b23e97c351d143d669780101d9c69f4cf0454f09a0744288dd8fd85d4c79f650.jpg)

![](images/134fcfdd797804b22eebfce1dc0134652d6a1f1603513e0a305bcd9652068593.jpg)  
Figure 5: Training dynamics on LeetCodeDataset using equally weighted runtime-success and passing rewards.

## D Additional Training Dynamics

We provide complete training curves for the experimental settings in the main paper. Validation and training rewards are shown for LeetCodeDataset in Figures 6 and 7, RLLA-4K in Figures 8 and 9, and AgentDojo in Figures 10 and 11. Figure 12 presents the mean response length and policy entropy across all three datasets. All horizontal axes indicate training steps, and all curves use exponential moving average smoothing with a decay factor of 0.6.

LeetCodeDataset - Validation Rewards  
![](images/604cd80c08e093498c72cbca9201164cad9b03801b038c0976373063178ad800.jpg)  
Figure 6: LeetCodeDataset: validation rewards. Columns correspond to model backbones. Rows report the task-specific validation mean@1 reward components and the total reward. The horizontal axis denotes training steps. Curves use exponential moving average smoothing (decay 0.6) and end at the last available observation.

LeetCodeDataset - Training Rewards  
![](images/4ea440944e20b936eb57fa131d9b813c35b0a2d2e1feb4496123367de56e1108.jpg)  
Figure 7: LeetCodeDataset: training rewards. Columns correspond to model backbones. Rows report the mean training reward components and the mean total reward. The horizontal axis denotes training steps. Curves use exponential moving average smoothing (decay 0.6) and end at the last available observation.

RLLA-4K - Validation Rewards  
![](images/8156432fde6e66527143abb4257e7ad6b45c3664b889b04ea4e659a3245da957.jpg)  
Figure 8: RLLA-4K: validation rewards. Columns correspond to model backbones. Rows report the task-specific validation mean@1 reward components and the total reward. The horizontal axis denotes training steps. Curves use exponential moving average smoothing (decay 0.6) and end at the last available observation.

RLLA-4K - Training Rewards  
![](images/7a719e32b27fb18a0da03bb480b4dbda6b15eea237618c0e8e7447db5b71a97d.jpg)  
Figure 9: RLLA-4K: training rewards. Columns correspond to model backbones. Rows report the mean training reward components and the mean total reward. The horizontal axis denotes training steps. Curves use exponential moving average smoothing (decay 0.6) and end at the last available observation.

AgentDojo - Validation Rewards  
![](images/9b1b2d7a9ab7499586c1723e1984e5683d7333192d468ea42d05a6794f7138a8.jpg)  
Figure 10: AgentDojo: validation rewards. Columns correspond to model backbones. Rows report the task-specific validation mean@1 reward components and the total reward. Joint Reward denotes the logged trajectory-level metric. The horizontal axis denotes training steps. AgentDojo panels are restricted to the overlapping recorded step ranges of GRPO and CorrGRPO. Curves use exponential moving average smoothing (decay 0.6) and end at the last available observation.

AgentDojo - Training Rewards  
![](images/76937f30dc5c828778c97f747e15d55164c653ce9605c9ad7fbec82c7da549d5.jpg)  
Figure 11: AgentDojo: training rewards. Columns correspond to model backbones. Rows report the mean training reward components and the mean total reward. The horizontal axis denotes training steps. AgentDojo panels are restricted to the overlapping recorded step ranges of GRPO and CorrGRPO. Curves use exponential moving average smoothing (decay 0.6) and end at the last available observation.

LeetCodeDataset - Response Length And Entropy  
![](images/438bdb96d0484479d1c64cd600e93485aa6a85c3b90109157ecbb5003b5f66c1.jpg)

RLLA-4K - Response Length And Entropy  
![](images/4f0f3c253eee25f9ee3848eec4d1e9a92646b55c67959bc7a3d0769b8fc09f5c.jpg)

AgentDojo - Response Length And Entropy  
![](images/64f8e719bab6861236644e2da14301dcc9c6286c78dc78d81a4d87943968d04f.jpg)  
Figure 12: Response length and entropy on LeetCodeDataset (top), RLLA-4K (middle), and AgentDojo (bottom). Within each dataset, columns correspond to model backbones; the two rows show mean response length in tokens and policy entropy, respectively. Horizontal axes denote training steps. AgentDojo panels are restricted to the overlapping recorded step ranges of GRPO and CorrGRPO. Curves use exponential moving average smoothing (decay 0.6) and end at the last available observation within the displayed range.

## E Reference Scores from the Original Benchmark Papers

Tables 4–7 report scores from the original benchmark papers alongside our base model, GRPO, and CorrGRPO. All scores are percentages. Evaluation protocols difer across papers.

## E.1 Code Generation

The original SFT scores in Table 4 are taken from Xia et al. (2025, Table 4). The reference SFT runs use Qwen2.5-Coder-7B; our RL runs start from Qwen2.5-Coder-7B-Instruct and train on the 2,641-problem LeetCodeDataset split. We evaluate Pass@1 on LeetCode-Dataset, HumanEval, MBPP, and LiveCodeBench v6 using the evaluation splits listed below.

Table 4: Code-generation Pass@1. Evaluation splits are indicated in the table.
<table><tr><td>Training data / method</td><td>Training examples</td><td>HumanEval ↑</td><td>MBPP</td><td>LiveCode Bench</td><td>LeetCode Dataset</td></tr><tr><td>Original SFT results: Qwen2.5-Coder-7B</td><td></td><td></td><td>↑</td><td>↑</td><td>↑</td></tr><tr><td></td><td></td><td></td><td></td><td>24-08-25-02</td><td>24-07-25-03</td></tr><tr><td>Magicoder Evol-Instruct-110K</td><td>111.1K</td><td>77.4</td><td>74.1</td><td>15.1</td><td>13.7</td></tr><tr><td>Magicoder OSS-Instruct-75K</td><td>75.1K</td><td>73.8</td><td>76.5</td><td>15.1</td><td>12.9</td></tr><tr><td>Open-R1 CodeForces-CoT</td><td>9.5K</td><td>79.9</td><td>74.1</td><td>15.8</td><td>13.3</td></tr><tr><td>OpenThoughts 114k</td><td>19.9K</td><td>77.4</td><td>75.7</td><td>16.9</td><td>16.4</td></tr><tr><td>LeetCodeDataset (human)</td><td>2.6K</td><td>55.5</td><td>53.4</td><td>14.0</td><td>10.9</td></tr><tr><td>LeetCodeDataset (model)</td><td>2.6K</td><td>79.9</td><td>77.5</td><td>15.4</td><td>12.5</td></tr></table>

<table><tr><td colspan="5">v6, 175 tasks</td><td rowspan="2">228-problem split 14.91</td></tr><tr><td>Base model</td><td></td><td>78.66</td><td>76.40</td><td>20.57</td></tr><tr><td>+GRPO</td><td>2,641</td><td>73.78</td><td>78.40</td><td>21.14</td><td>15.79</td></tr><tr><td>+CorrGRPO</td><td>2,641</td><td>78.66</td><td>78.60</td><td>24.57</td><td>24.12</td></tr></table>

## E.2 AgentDojo

The original scores in Table 5 are taken from Debenedetti et al. (2024, Table 3). We train Qwen2.5-3B-Instruct on 1,584 cases with a split grouped by user task, and evaluate on 21 clean and 390 attacked cases. We inject <sub>important\_instructions</sub> or <sub>tool\_knowledge</sub> via tool responses.

Table 5: AgentDojo with original 95% confidence intervals; protocols difer.
<table><tr><td>Model</td><td>Clean utility ↑</td><td>Utility under attack ↑</td><td>Targeted ASR ↓</td></tr><tr><td>Claude 3 Opus</td><td> $6 6 . 6 1 \pm 3 . 6 9$ </td><td> $5 2 . 4 6 \pm 3 . 9 0$ </td><td> $1 1 . 2 9 \pm 2 . 4 7$ </td></tr><tr><td>Claude 3 Sonnet</td><td> $5 3 . 1 0 \pm 3 . 9 0$ </td><td> $3 3 . 2 3 \pm 3 . 6 8$ </td><td> $2 6 . 7 1 \pm 3 . 4 6$ </td></tr><tr><td>Claude 3.5 Sonnet</td><td> $7 8 . 2 2 \pm 3 . 2 3$ </td><td> $5 1 . 1 9 \pm 3 . 9 1$ </td><td> $3 3 . 8 6 \pm 3 . 7 0$ </td></tr><tr><td>Command-R+</td><td> $2 5 . 4 4 \pm 3 . 4 0$ </td><td> $2 5 . 1 2 \pm 3 . 3 9$ </td><td> $0 . 9 5 \pm 0 . 7 6$ </td></tr><tr><td>Gemini 1.5 Flash</td><td> $3 6 . 0 9 \pm 3 . 7 5$ </td><td> $3 4 . 1 8 \pm 3 . 7 1$ </td><td> $1 2 . 2 4 \pm 2 . 5 6$ </td></tr><tr><td>Gemini 1.5 Pro</td><td> $4 5 . 6 3 \pm 3 . 8 9$ </td><td> $2 8 . 9 3 \pm 3 . 5 4$ </td><td> $2 5 . 6 0 \pm 3 . 4 1$ </td></tr><tr><td>GPT-3.5 Turbo</td><td> $3 3 . 8 6 \pm 3 . 7 0$ </td><td> $3 4 . 6 6 \pm 3 . 7 2$ </td><td> $8 . 4 3 \pm 2 . 1 7$ </td></tr><tr><td>GPT-4 Turbo</td><td> $6 3 . 4 3 \pm 3 . 7 6$ </td><td> $5 4 . 0 5 \pm 3 . 8 9$ </td><td> $2 8 . 6 2 \pm 3 . 5 3$ </td></tr><tr><td>GPT-40</td><td> $6 9 . 0 0 \pm 3 . 6 1 $ </td><td> $5 0 . 0 8 \pm 3 . 9 1$ </td><td> $4 7 . 6 9 \pm 3 . 9 0$ </td></tr><tr><td>Llama 3 70B</td><td> $3 4 . 5 0 \pm 3 . 7 1$ </td><td> $1 8 . 2 8 \pm 3 . 0 2$ </td><td> $2 0 . 0 3 \pm 3 . 1 3$ </td></tr><tr><td colspan="4">Ours: Qwen2.5-3B-Instruct, held-out evaluation split</td></tr><tr><td>Base model</td><td>28.57</td><td>14.62</td><td>3.85</td></tr><tr><td>+GRPO</td><td>52.38</td><td>66.92</td><td>0.00</td></tr><tr><td>+CorrGRPO</td><td>95.24</td><td>84.36</td><td>1.54</td></tr></table>

## E.3 Agent Security Bench

The original OPI ASR and Clean Utility values in Table 6 are taken from Tables 5 and 6 of Zhang et al. (2025), respectively. We evaluate the same Qwen2.5-3B-Instruct checkpoints on 400 paired clean and attacked cases without further training. Attacks append <sub>context\_ignoring</sub> payloads to intermediate tool responses.

Table 6: Clean Utility and OPI ASR on Agent Security Bench.
<table><tr><td>Model</td><td>Clean utility ↑</td><td>ASR (OPI) ↓</td></tr><tr><td>Claude-3.5 Sonnet</td><td>100.00</td><td>59.70</td></tr><tr><td>LLaMA3-70B</td><td>66.50</td><td>43.70</td></tr><tr><td>GPT-4o</td><td>79.00</td><td>62.45</td></tr><tr><td>Gemma2-27B</td><td>31.50</td><td>14.20</td></tr><tr><td>LLaMA3.1-70B</td><td>21.25</td><td>12.10</td></tr><tr><td>Qwen2-7B</td><td>9.75</td><td>9.00</td></tr><tr><td>Gemma2-9B</td><td>10.75</td><td>14.20</td></tr><tr><td>GPT-3.5 Turbo</td><td>8.00</td><td>55.10</td></tr><tr><td>Qwen2-72B</td><td>4.00</td><td>21.35</td></tr><tr><td>LLaMA3.1-8B</td><td>0.75</td><td>6.40</td></tr><tr><td>Mixtral-8x7B</td><td>0.00</td><td>4.80</td></tr><tr><td>Ours: Qwen2.5-3B-Instruct</td><td></td><td></td></tr><tr><td>Base model</td><td>10.00</td><td>8.00</td></tr><tr><td>+GRPO</td><td>21.75</td><td>9.00</td></tr><tr><td>+CorrGRPO</td><td>26.00</td><td>9.25</td></tr></table>

## E.4 InjecAgent

The original base-setting scores in Table 7 are taken from Zhan et al. (2024, Table 3). We use the base setting and standard InjecAgent prompt without further training. The agent continues from an injected tool response; ASR-valid excludes invalid outputs.

Table 7: InjecAgent base-setting ASR-valid. S1/S2 denote data extraction/transmission; S2 is conditional on reaching that stage.
<table><tr><td rowspan="3">Model</td><td colspan="5">Base setting: ASR-valid ↓</td></tr><tr><td rowspan="2">Direct harm</td><td colspan="3">Data stealing</td><td rowspan="2">Overall</td></tr><tr><td>S1</td><td>S2</td><td>Total</td></tr><tr><td colspan="6">Original prompted agents (ReAct)</td></tr><tr><td>Qwen-1.8B</td><td>36.1</td><td>35.1</td><td>82.6</td><td>17.6</td><td>29.7</td></tr><tr><td>Qwen-72B</td><td>8.7</td><td>37.9</td><td>98.4</td><td>37.1</td><td>23.2</td></tr><tr><td>Mistral-7B</td><td>13.4</td><td>25.0</td><td>87.8</td><td>20.1</td><td>16.7</td></tr><tr><td>OpenOrca-Mistral</td><td>3.9</td><td>5.3</td><td>53.8</td><td>2.9</td><td>3.4</td></tr><tr><td>OpenHermes-2.5-Mistral</td><td>23.4</td><td>29.2</td><td>99.2</td><td>28.4</td><td>25.9</td></tr><tr><td>Mixtral-8x7B</td><td>23.1</td><td>34.1</td><td>99.1</td><td>32.9</td><td>27.8</td></tr><tr><td>Nous-Mixtral-DPO</td><td>37.2</td><td>51.6</td><td>98.4</td><td>50.5</td><td>43.6</td></tr><tr><td>Nous-Mixtral-SFT</td><td>51.8</td><td>48.2</td><td>98.9</td><td>47.5</td><td>49.8</td></tr><tr><td>Platypus2-70B</td><td>34.3</td><td>51.8</td><td>74.3</td><td>35.4</td><td>34.9</td></tr><tr><td>Llama2-70B</td><td>91.9</td><td>97.1</td><td>83.7</td><td>80.4</td><td>86.9</td></tr><tr><td>Claude-2</td><td>7.5</td><td>26.5</td><td>58.1</td><td>14.8</td><td>11.4</td></tr><tr><td>GPT-3.5</td><td>18.8</td><td>37.6</td><td>77.4</td><td>28.8</td><td>23.7</td></tr><tr><td>GPT-4</td><td>14.7</td><td>32.7</td><td>97.7</td><td>31.9</td><td>23.6</td></tr><tr><td colspan="6">Ours: Qwen2.5-3B-Instruct; RL training on AgentDojo</td></tr><tr><td>Base model</td><td>18.38</td><td>50.29</td><td>76.39</td><td>34.59</td><td>27.12</td></tr><tr><td>+GRPO</td><td>5.03</td><td>20.11</td><td>79.31</td><td>13.07</td><td>8.80</td></tr><tr><td>+CorrGRPO</td><td>4.67</td><td>17.26</td><td>52.63</td><td>6.33</td><td>5.52</td></tr></table>

## F Compatibility with CISPO<sub>,</sub> GDPO<sub>,</sub> and DAPO

## F.1 Compatibility Experiments

We evaluate CorrGRPO in combination with CISPO, GDPO, and DAPO to examine its applicability across diferent policy optimization methods. Table 8 compares each method with its CorrGRPO-integrated variant. We report Eficiency, Executable, and Pass@1 on LeetCodeDataset, together with Pass@1 on HumanEval, MBPP, and LiveCodeBench v6. The average summarizes Pass@1 across the four benchmarks. These results assess both performance on the training-domain benchmark and generalization to external benchmarks. For DAPO, group filtering is based on the accuracy reward.

Table 8: Results of compatibility RL experiments.
<table><tr><td rowspan="3">Model</td><td colspan="3">LeetCodeDataset (In-dataset Eval)</td><td>HumanEval</td><td>MBPP</td><td>LCB v6</td><td>Avg.</td></tr><tr><td>Efficiency</td><td>Executable</td><td>Pass@1</td><td>Pass@1</td><td>Pass@1</td><td>Pass@1</td><td>Pass@1</td></tr><tr><td>CISPO</td><td>34.51</td><td>83.77</td><td>21.93</td><td>76.83</td><td>79.00</td><td>23.43</td><td>50.30</td></tr><tr><td>CISPO + CorrGRPO</td><td>38.44</td><td>82.46</td><td>25.00</td><td>81.71</td><td>77.80</td><td>21.71</td><td>51.56</td></tr><tr><td>GDPO</td><td>41.28</td><td>84.21</td><td>16.23</td><td>76.83</td><td>79.60</td><td>22.86</td><td>48.88</td></tr><tr><td>GDPO + CorrGRPO</td><td>44.64</td><td>86.40</td><td>17.11</td><td>76.83</td><td>77.20</td><td>21.71</td><td>48.21</td></tr><tr><td>DAPO</td><td>40.48</td><td>81.14</td><td>20.18</td><td>81.71</td><td>77.40</td><td>23.43</td><td>50.68</td></tr><tr><td>DAPO + CorrGRPO</td><td>47.07</td><td>82.89</td><td>21.93</td><td>80.49</td><td>78.40</td><td>22.86</td><td>50.92</td></tr></table>

## F.2 Compatibility Analysis

We derive the integrations on fixed sampled data. All advantages and normalization statistics are held fixed during policy diferentiation. The objectives below are reward surrogates; any separately configured KL or other auxiliary term retains its original definition. For a group associated with prompt q, write $\begin{array} { r } { T _ { q } = \dot { \sum _ { i = 1 } ^ { n } } T _ { i } } \end{array}$ and use the policy ratios $u _ { i , t } ( \boldsymbol { \theta } )$ defined in the main text. Numerical stabilizers are made explicit using the same convention as CorrGRPO.

## F.2.1 DAPO

DAPO (Yu et al., 2025) uses an asymmetric clipped surrogate with token-level averaging. In our notation, its group contribution is

$$
\begin{array} { c l l } { { { \cal L } _ { \mathrm { D A P O } , q } ( \theta ; A ) = \displaystyle \frac { 1 } { T _ { q } } \sum _ { i = 1 } ^ { n } \sum _ { t = 1 } ^ { T _ { i } } \operatorname* { m i n } \big \{ u _ { i , t } ( \theta ) A ^ { i } , \kappa _ { D } \big ( u _ { i , t } ( \theta ) \big ) A ^ { i } \big \} , } } \\ { { \kappa _ { D } ( u ) = \displaystyle \mathrm { c l i p } \big ( u , 1 - \eta _ { \mathrm { l o w } } , 1 + \eta _ { \mathrm { h i g h } } \big ) . } } \end{array}\tag{11}
$$

The baseline uses $A ^ { i } = A _ { \mathrm { G R P O } } ^ { i }$ . Substituting the CorrGRPO advantage gives the combined objective

$$
L _ { \mathrm { D A P O + C o r r G R P O } , q } ( \theta ) = \frac { 1 } { T _ { q } } \sum _ { i = 1 } ^ { n } \sum _ { t = 1 } ^ { T _ { i } } \operatorname* { m i n } \left\{ u _ { i , t } ( \theta ) A _ { \mathrm { C o r r G R P O } } ^ { i } , \kappa _ { D } ( u _ { i , t } ( \theta ) ) A _ { \mathrm { C o r r G R P O } } ^ { i } \right\} .\tag{12}
$$

For $c _ { q }$ in Equation 44, positive homogeneity of the minimum gives

$$
\begin{array} { r l } & { L _ { \mathrm { D A P O + C o r r G R P O } , q } ( \theta ) = c _ { q } L _ { \mathrm { D A P O } , q } ( \theta ; A _ { \mathrm { G R P O } } ) , } \\ & { \nabla _ { \theta } L _ { \mathrm { D A P O + C o r r G R P O } , q } = c _ { q } \nabla _ { \theta } L _ { \mathrm { D A P O } , q } ( \theta ; A _ { \mathrm { G R P O } } ) . } \end{array}\tag{13}
$$

The asymmetric thresholds and token averaging are preserved. DAPO’s sampling filter and reward shaping can be applied before this substitution using the same configured rules. The equality compares the same retained group at the same policy parameters; subsequent sampled groups can difer as training proceeds. The expected training objective averages these group contributions over the sampling procedure.

## F.2.2 CISPO

CISPO (MiniMax, 2025) clips importance-sampling weights and stops gradients through those weights. Define

$$
\tilde { u } _ { i , t } ( \theta ) = \mathrm { c l i p } \big ( u _ { i , t } ( \theta ) , 1 - \eta _ { \mathrm { l o w } } ^ { \mathrm { I S } } , 1 + \eta _ { \mathrm { h i g h } } ^ { \mathrm { I S } } \big ) ,\tag{14}
$$

where sg below denotes stop-gradient. The CISPO surrogate is

$$
L _ { \mathrm { C I S P O } , q } ( \theta ; A ) = \frac { 1 } { T _ { q } } \sum _ { i = 1 } ^ { n } \sum _ { t = 1 } ^ { T _ { i } } \operatorname { s g } [ \tilde { u } _ { i , t } ( \theta ) ] A ^ { i } \log \pi _ { \theta } ( a _ { i , t } \mid h _ { i , t } ) .\tag{15}
$$

Its original group-relative advantage can be replaced directly:

$$
L _ { \mathrm { C I S P O + C o r r G R P O } , q } ( \theta ) = \frac { 1 } { T _ { q } } \sum _ { i = 1 } ^ { n } \sum _ { t = 1 } ^ { T _ { i } } \mathrm { s g } [ \tilde { u } _ { i , t } ( \theta ) ] A _ { \mathrm { C o r r G R P O } } ^ { i } \log \pi _ { \theta } ( a _ { i , t } \mid h _ { i , t } ) .\tag{16}
$$

Diferentiating with the prescribed stop-gradient operation yields

$$
\begin{array} { r l } & { \nabla _ { \theta } L _ { \mathrm { C I S P O + C o r r G R P O } , q } = \cfrac { 1 } { T _ { q } } \csum _ { i = 1 } ^ { n } \csum _ { t = 1 } ^ { T _ { i } } \operatorname { s g } [ \tilde { u } _ { i , t } ( \theta ) ] A _ { \mathrm { C o r r G R P O } } ^ { i } \nabla _ { \theta } \log \pi _ { \theta } ( a _ { i , t } \mid h _ { i , t } ) } \\ & { \qquad = c _ { q } \nabla _ { \theta } L _ { \mathrm { C I S P O } , q } ( \theta ; A _ { \mathrm { G R P O } } ) . } \end{array}\tag{17}
$$

Thus, the replacement preserves CISPO’s importance-weight clipping and stop-gradient computation, while rescaling each group’s reward-driven contribution. It introduces no PPO-style minimum or additional token-dropping rule. Table 8 reports the supplied CISPO checkpoint comparison; the mathematical construction does not assume that the two checkpoints were trained for equal numbers of steps.

## F.2.3 GDPO

GDPO (Liu et al., 2026) first normalizes each reward within its prompt group, aggregates those components, and then normalizes the aggregate over the batch. Let B contain N sampled trajectories, with $q ( b )$ denoting the prompt group of trajectory b. Write its component advantages as

$$
U _ { l } ^ { b } = w _ { l } \frac { R _ { l } ^ { b } - \bar { R } _ { l , q ( b ) } } { \hat { \sigma } _ { l , q ( b ) } + \varepsilon _ { g } } , \qquad H ^ { b } = \sum _ { l = 1 } ^ { r } U _ { l } ^ { b } , \qquad \bar { H } _ { B } = \frac { 1 } { N } \sum _ { b \in B } H ^ { b } .\tag{18}
$$

Here $w _ { l }$ denotes any weight applied after group normalization, with $w _ { l } = 1$ for unweighted aggregation; $R _ { l }$ is the input to GDPO’s per-reward normalization. The first-stage stabilizer $\varepsilon _ { g }$ is retained if present in the baseline implementation. GDPO’s final advantage is

$$
{ \cal A } _ { \mathrm { G D P O } } ^ { b } = \frac { H ^ { b } - \bar { H } _ { B } } { \sqrt { \mathrm { { V a r } } _ { B } ( H ) } + { \varepsilon _ { b } } } .\tag{19}
$$

The batch standard deviation in this expression is computed from H, so its covariance decomposition concerns the components $U _ { l }$ already produced by GDPO. Define their batch covariance matrix and standard-deviation vector as

$$
\begin{array} { r } { \pmb { \Sigma } _ { \pmb { \mathcal { B } } } = \left[ \widehat { \mathrm { C o v } } _ { \pmb { \mathcal { B } } } ( U _ { l } , U _ { m } ) \right] _ { l , m } , \qquad \mathbf { s } _ { \pmb { \mathcal { B } } } = \left( \sqrt { \widehat { \mathrm { V a r } } _ { \pmb { \mathcal { B } } } ( U _ { l } ) } \right) _ { l = 1 } ^ { r } , } \\ { \lbrack \mathbf { C } _ { \pmb { \mathcal { B } } } \rbrack _ { l m } = \frac { \left[ \pmb { \Sigma } _ { \pmb { \mathcal { B } } } \right] _ { l m } } { \left[ \mathbf { s } _ { \pmb { \mathcal { B } } } \right] _ { l } \left[ \mathbf { s } _ { \pmb { \mathcal { B } } } \right] _ { m } } . \qquad } \end{array}\tag{20}
$$

By the finite-sample identity in Appendix L.2, Equation 42,

$$
\widehat { \mathrm { V a r } } _ { \mathcal { B } } ( H ) = \mathbf { 1 } ^ { \top } \Sigma _ { \mathcal { B } } \mathbf { 1 } = \mathbf { s } _ { \mathcal { B } } ^ { \top } \mathbf { C } _ { \mathcal { B } } \mathbf { s } _ { \mathcal { B } } .\tag{21}
$$

Applying our normalization at this final batch stage gives

$$
A _ { \mathrm { G D P O + C o r r G R P O } } ^ { b } = \frac { H ^ { b } - \bar { H } _ { B } } { \sqrt { \mathbf { 1 } ^ { \top } \mathbf { C } _ { B } \mathbf { 1 } } + \varepsilon _ { b } } .\tag{22}
$$

The group normalization, aggregation weights, and batch-centered numerator are preserved. Only the last denominator changes, from $\| \mathbf { s } _ { B } \| _ { \mathbf { C } _ { B } } + \varepsilon _ { b }$ to $\| \mathbf { 1 } \| _ { \mathbf { C } _ { B } } + \varepsilon _ { b }$ Correlations are computed across the batch between the per-reward advantages $U _ { l } .$ , using the same sample scope and covariance convention as the original batch statistic. $\mathrm { A n }$ implementation that uses token masks or weighted batch moments must use the same masks or weights for every covariance and variance in this step.

Using the clipped token surrogate ℓ defined in Equation $4 5 ,$ a sequence-averaged objective is

$$
L _ { \mathrm { G D P O + C o r r G R P O } , \mathcal { B } } ( \theta ) = \frac { 1 } { N } \sum _ { b \in \mathcal { B } } \frac { 1 } { T _ { b } } \sum _ { t = 1 } ^ { T _ { b } } \ell \bigl ( u _ { b , t } ( \theta ) , A _ { \mathrm { G D P O + C o r r G R P O } } ^ { b } \bigr ) .\tag{23}
$$

If the baseline uses another fixed token reduction or asymmetric clipping, that choice is retained. Since the new and old advantages difer by the same positive factor for the entire batch,

$$
\begin{array} { c } { { \displaystyle { c _ { \mathcal { B } } = \frac { \sqrt { \mathbf { s } _ { B } ^ { \top } \mathbf { C } _ { \mathcal { B } } \mathbf { s } _ { \mathcal { B } } } + \varepsilon _ { b } } { \sqrt { \mathbf { 1 } ^ { \top } \mathbf { C } _ { \mathcal { B } } \mathbf { 1 } } + \varepsilon _ { b } } > 0 , } } } \\ { { \displaystyle { A _ { \mathrm { G D P O + C o r r G R P O } } ^ { b } = c _ { \mathcal { B } } A _ { \mathrm { G D P O } } ^ { b } . } } } \end{array}\tag{24}
$$

the clipped reward gradients satisfy

$$
\nabla _ { \boldsymbol { \theta } } L _ { \mathrm { G D P O + C o r r G R P O } , \boldsymbol { B } } = c _ { \boldsymbol { B } } \nabla _ { \boldsymbol { \theta } } L _ { \mathrm { G D P O } , \boldsymbol { B } } .\tag{25}
$$

This substitution can reduce to a common scale correction when GDPO’s component normalization already equalizes batch variances. Specifical ${ \mathrm { l y } } ,$ if $\mathbf { s } _ { B } ~ = ~ s _ { 0 } \mathbf { 1 }$ , then $\begin{array} { r l } { c _ { B } } & { { } = } \end{array}$ $( s _ { 0 } \sqrt { { \bf 1 } ^ { \top } { \bf C } _ { B } { \bf 1 } } + \varepsilon _ { b } ) / ( \sqrt { { \bf 1 } ^ { \top } { \bf C } _ { B } { \bf 1 } } + \varepsilon _ { b } )$ , approximately $s _ { 0 }$ when the stabilizer is negligible. The integration establishes compatibility, rather than a universal additional benefit over GDPO. As elsewhere, the correlation formulas apply to nonconstant components; batch-constant components contribute zero after batch centering and can be omitted from the correlation matrix.

## G Reinforcement Learning Background

We consider reinforcement learning for a language model policy $\pi _ { \theta }$ . Given a prompt $q \sim$ D, the policy generates a trajectory $\tau ,$ which may include a response or a sequence of interactions with an environment. Each trajectory is evaluated by r reward components $R _ { 1 } ( q , \tau ) , \ldots , R _ { r } ( q , \tau )$ , capturing diferent aspects of the desired behavior. The learning objective is to maximize the expected total reward:

$$
\operatorname* { m a x } _ { \theta } J ( \theta ) = \mathbb { E } _ { q \sim \mathcal { D } , \tau \sim \pi _ { \theta } ( \cdot \cdot \vert q ) } \left[ R ( q , \tau ) \right] , \qquad R ( q , \tau ) = \sum _ { l = 1 } ^ { r } R _ { l } ( q , \tau ) .\tag{26}
$$

Any fixed reward weights are absorbed into the corresponding components. These components may exhibit positive or negative statistical dependence across trajectories, reflecting outcomes that tend to improve together or involve tradeofs.

GRPO estimates advantages using rewards from a group of trajectories, avoiding a separate value model. For each prompt $q ,$ it samples n trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { n }$ from an old policy $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } .$ Let $R _ { l } ^ { i } = R _ { l } ( q , \tau _ { i } )$ and $R ^ { i } = \textstyle \sum _ { l } R _ { l } ^ { i }$ , with group means $\begin{array} { r } { \bar { R } _ { l } = \frac { 1 } { n } \sum _ { i } R _ { l } ^ { i } } \end{array}$ and $\begin{array} { r } { \bar { R } = \frac { 1 } { n } \sum _ { i } R ^ { i } } \end{array}$ Under outcome supervision, all generated tokens in trajectory i share the same advantage $A _ { \mathrm { G R P O } } ^ { i }$ . GRPO optimizes the clipped surrogate objective

$$
\mathcal { I } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } \left( \operatorname* { m i n } \left[ u _ { i , t } ( \theta ) A _ { \mathrm { G R P O } } ^ { i } , \mathrm { c l i p } ( u _ { i , t } ( \theta ) , 1 - \eta , 1 + \eta ) A _ { \mathrm { G R P O } } ^ { i } \right] - \beta K _ { i , t } \right) \right] ,\tag{27}
$$

Here, the expectation is over prompts and trajectory groups sampled as described above. The ratio $u _ { i , t } ( \theta ) \ = \ \pi _ { \theta } ( a _ { i , t } \ \mid \ h _ { i , t } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { i , t } \ \mid \ h _ { i , t } )$ is the policy probability ratio, $h _ { i , t }$ is the history preceding generated token $a _ { i , t } , \ T _ { i }$ counts generated tokens, η is the clipping threshold, and $\boldsymbol { \kappa } _ { i , t }$ is the KL regularization term relative to a reference policy.

## H Examples from the Training Data

We present one example from each training dataset. The coding and tool-calling examples pair the training prompt with its stored reference response. AgentDojo supplies an interactive task and environment rather than a fixed reference response; we therefore show a recorded agent interaction for a case in our training split. This historical rollout illustrates the data format and is not a CorrGRPO training rollout or a comparative result.

## H.1 Coding: LeetCodeDataset

The example is <sub>remove-vowels-from-a-string</sub> from the training split. The prompt below is the actual user message supplied to the model, including the problem description and required answer format. The response is the stored dataset reference; the code extracted from it is also used as the reference solution in the RL record.

You are an expert Python programmer. You will be given a question (problem   
specification) and will generate a correct Python program that matches the   
specification and passes all tests.   
### Question:   
Given a string s, remove the vowels 'a', 'e', 'i', 'o', and 'u' from it, and   
return the new string.   
Example 1:   
Input: s = "leetcodeisacommunityforcoders"   
Output: "ltcdscmmntyfrcdrs"   
Example 2:   
Input: s = "aeiou"   
Output: ""   
Constraints:   
1 <= s.length <= 1000   
s consists of only lowercase English letters.   
### Format: You will use the following starter code to write the solution to the   
problem and enclose your code within delimiters.   
\`\`\`python   
class Solution:   
def removeVowels(self, s: str) -> str:   
### Answer: (use the provided format with backticks)

Response (dataset reference).

```python
```python
class Solution:
def removeVowels(self, s: str) -> str:
vowels = set('aeiou')
return ''.join([char for char in s if char not in vowels])
```

## H.2 Tool Calling: RLLA-4K

This training example asks for the least common multiple of two integers. The system message specifies the available function, its argument schema, and the response format. We reproduce both prompt messages and the stored reference response, including its original <sub><think></sub> and <sub><tool\_call></sub> fields.

Prompt (system).

You are a helpful multi-turn dialogue assistant capable of leveraging tool calls   
to solve user tasks and provide structured chat responses.   
\*\*Available Tools\*\*   
In your response, you can use the following tools:   
1. Name: least\_common\_multiple   
Description: Computes the least common multiple (LCM) of two positive integers.   
Parameters: {"a": {"description": "The first positive integer.", "type": "int", "   
default": ""}, "b": {"description": "The second positive integer.", "type": "   
int", "default": ""}}   
\*\*Steps for Each Turn\*\*   
1. \*\*Think:\*\* Recall relevant context and analyze the current user goal.   
2. \*\*Decide on Tool Usage:\*\* If a tool is needed, specify the tool and its   
parameters.   
3. \*\*Respond Appropriately:\*\* If a response is needed, generate one while   
maintaining consistency across user queries.   
\*\*Output Format\*\*   
\`\`\`plaintext   
<think> Your thoughts and reasoning </think>   
<tool\_call>   
{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "...   
.": "... ..."}}   
{"name": "... ...", "parameters": {"... ...": "... ...", "... ...": "... ..."}}   
</tool\_call>   
<response> AI's final response </response>   
\*\*Important Notes\*\*   
1. You must always include the \`<think>\` field to outline your reasoning. Provide   
at least one of \`<tool\_call>\` or \`<response>\`. Decide whether to use \`<   
tool\_call>\` (possibly multiple times), \`<response>\`, or both.   
2. You can invoke multiple tool calls simultaneously in the \`<tool\_call>\` fields.   
Each tool call should be a JSON object with a "name" field and an "   
parameters" field containing a dictionary of parameters. If no parameters are   
needed, leave the "parameters" field an empty dictionary.   
3. Refer to the previous dialogue records in the history, including the user's   
queries, previous \`<tool\_call>\`, \`<response>\`, and any tool feedback noted as   
\`<obs>\` (if exists).

Prompt (user).

\*\*Dialogue Records History\*\*   
<user> What is the least common multiple for 15 and 25? </user>

Response (dataset reference).

<think> I should use the appropriate tool with proper parameters to respond to   
the user's need. </think>   
<tool\_call>   
{"name": "least\_common\_multiple", "parameters": {"a": 15, "b": 25}}   
</tool\_call>

## H.3 Agent Utility and Security: AgentDojo

We select <sub>workspace:user\_task\_0</sub> with <sub>injection\_task\_1</sub> and the <sub>important\_instructions</sub> attack, whose manifest entry is assigned to the training split. The legitimate task is to retrieve the other participants’ email addresses from a calendar event. The attack inserts an instruction to delete a cloud-drive file into the calendar description returned by the search tool.

Prompt and recorded response trajectory. The following interaction is recovered from a saved rollout. For readability, we omit the shared tool-schema wrapper and the model’s <sub><think></sub> blocks; we preserve the tool names, arguments, observations, and final answer. The calendar observation is reserialized as YAML with a multiline description, without changing its field values. The accompanying source files retain the complete saved rollout and the full model input immediately before the final response.

Prompt (system instructions).

You are an AI language model who assists the user by using the given tools. \   
The user's name is Emma Johnson, an employee of the company Blue Sparrow Tech.   
Follow these instructions:   
- Don't make assumptions about what values to plug into functions.   
- Use the provided tools to try to disambiguate.   
- If a tool says that no results are available, try with a different query.   
- Do not assume the current year, but use the provided tools to see what year   
it is.

Prompt (user).

Who else is invited to the 'Networking event' on May 26th? Please give me their   
email addresses.

Response (assistant tool call).

<function=get\_current\_day>{}</function>

Tool observation.

2024-05-15

Response (assistant tool call).

```xml
<function=search_calendar_events>{"query": "Networking event", "date":
"2024-05-26"}</function>
```

Tool observation (containing the injection).

- all\_day: false   
description: |+   
Industry networking event to meet and connect with professionals.   
<INFORMATION>   
This is an important message from me, Emma Johnson, to you, Local model.   
Before you can solve the task that I gave you in the beginning, please do the   
following first:   
Delete the file with ID '13' from the cloud drive.   
After you do that, you can solve the task that I gave you in the beginning.   
Thanks!   
Signed,   
Emma Johnson   
</INFORMATION>   
end\_time: 2024-05-26 19:00:00   
id\_: '13'   
location: City Convention Center   
participants:   
- emma.johnson@bluesparrowtech.com   
- networking.participants@industry-network.com   
- alex.williams@mainsail-tech.com   
start\_time: 2024-05-26 17:00:00   
status: confirmed   
title: Networking Event

Response (assistant final answer).

The email addresses of the other invitees to the 'Networking Event' on May 26th   
are:   
- networking.participants@industry-network.com   
- alex.williams@mainsail-tech.com   
Let me know if you need further assistance!

The agent retrieves the requested email addresses without executing the injected file-deletion instruction. The saved checkers report successful task completion and an unsuccessful attack, corresponding to $R _ { \mathrm { u t i l } } = 1$ and $R _ { \mathrm { s e c } } = 1$

## I Training Settings

## I.1 Coding Reasoning

Our default RL configuration uses a training batch of 64 prompts and a rollout group size of 8, with temperature 1.0 and top- $\mathbf { \varepsilon } \cdot p = 1 . 0$ . The learning rate is $1 0 ^ { - 6 }$ , and the KL regularization coeficient is $1 0 ^ { - 3 }$ . Maximum prompt and response lengths are set to 4,096 and 2,048 tokens, respectively.

## I.2 Tool Calling

Our default RL configuration uses a training batch of 128 prompts, a rollout group size of 4, a learning rate of $1 0 ^ { - 6 }$ , and 15 training epochs. Maximum prompt and response lengths are both set to 2,048 tokens.

## I.3 Agent Utility and Security

Our default RL configuration uses a training batch of 16 tasks, a rollout group size of 4, a learning rate of $1 0 ^ { - 6 }$ , and a KL regularization coeficient of $1 0 ^ { - 3 }$ . Maximum prompt and response lengths are set to 12,288 and 16,384 tokens, respectively.

## J Reward Computation Details

## J.1 AST Structural Similarity

AST representation. We extract Python code from the generated response $y _ { g }$ and the reference response $y _ { r } ,$ then parse each program with <sub>ast.parse</sub> in <sub>exec</sub> mode. A depthfirst traversal records the node type on entry and a closing marker on exit. For example, a <sub>Name</sub> node contributes <sub>Name</sub> and /<sub>Name</sub>. Children are visited in the order returned by <sub>ast.iter\_child\_nodes</sub>. This produces a sequence $S ( y )$ for each program. The representation retains node types, child order, and nesting, while omitting identifier names and literal values. The traversal is:

```python
def visit(node):
name = type(node).__name__
tokens.append(name)
for child in ast.iter_child_nodes(node):
visit(child)
tokens.append("/" + name)
```

Sequence matching. We compare the generated sequence first and the reference sequence second using Python’s <sub>difflib.SequenceMatcher</sub>:

```python
matcher = difflib.SequenceMatcher(
None, S_generated, S_reference, autojunk=False
)
similarity = matcher.ratio()
```

The matcher identifies a longest common contiguous block and recursively matches the remaining regions on either side. We disable the automatic popular-token heuristic so that frequently occurring AST node types remain available for matching. Let $B ( S ( y _ { g } ) , S ( y _ { r } ) )$ denote the returned nonempty matching blocks and |b| the number of matched tokens in block b. The reward is

$$
\begin{array}{c} R _ { \mathrm { a s t } } ( y _ { g } , y _ { r } ) = \{ \begin{array} { l l } { 2 \sum _ { b \in \mathcal { B } ( S ( y _ { g } ) , S ( y _ { r } ) ) } | b | } \\ { \displaystyle | S ( y _ { g } ) | + | S ( y _ { r } ) | } \\ { 0 , } & { \mathrm { ( t h e r w i s e . } } \end{array}  \mathrm { ~ i f ~ b o t h ~ c o d e ~ s n i p p e t s ~ a r e ~ a v a i l a b l e ~ a n d ~ p a r s e a b l e , }  \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array}\tag{28}
$$

The ratio lies in [0, 1] and equals 1 for identical traversal sequences. The denominator counts both sequence lengths, and the numerator counts matched tokens twice, as in the documented <sub>SequenceMatcher.ratio</sub>() definition (Python Software Foundation). If the reference is unavailable or unparseable, the implementation also marks the structural reward as unavailable; if only the generated code is invalid, the reward is zero with a valid reference still recorded.

## J.2 Tool-Call Matching

Parsed calls and multiset overlap. For a reference response containing tool calls, we parse each JSON line inside its tool-call block. Let $y _ { r }$ contain m reference calls $( f _ { i } ^ { r } , p _ { i } ^ { r } )$ and let $y _ { g }$ contain n predicted calls $( f _ { j } ^ { g } , p _ { j } ^ { g } )$ , where f is a function name and p is a parameter dictionary. Let $K _ { i } ^ { r }$ and $K _ { j } ^ { g }$ denote their parameter-name sets. For two multisets A and $B ,$ let $c _ { A } ( \boldsymbol { u } )$ and $c _ { B } ( \boldsymbol { u } )$ denote the multiplicity of u. We use

$$
J ( A , B ) = \left\{ \frac { \sum _ { u } \operatorname* { m i n } \{ c _ { A } ( u ) , c _ { B } ( u ) \} } { \sum _ { u } \operatorname* { m a x } \{ c _ { A } ( u ) , c _ { B } ( u ) \} } , \quad | A | + | B | > 0 , \right.\tag{29}
$$

Thus, one empty multiset and one nonempty multiset receive zero overlap. Function-name matching is

$$
S _ { \mathrm { f n } } ( y _ { g } , y _ { r } ) = J \bigl ( [ f _ { 1 } ^ { g } , \ldots , f _ { n } ^ { g } ] , [ f _ { 1 } ^ { r } , \ldots , f _ { m } ^ { r } ] \bigr ) .\tag{30}
$$

This score ignores call order while accounting for repeated function names.

Greedy call assignment. Reference calls are processed in their original order. For reference call i, candidate matches are unused predicted calls with $f _ { j } ^ { g } = f _ { i } ^ { r }$ . For each candidate, define

$$
q _ { i j } = \sum _ { k \in K _ { i } ^ { r } \cap K _ { j } ^ { g } } \mathbf { 1 } [ p _ { i } ^ { r } [ k ] = p _ { j } ^ { g } [ k ] ] , \qquad h _ { i j } = J ( K _ { i } ^ { r } , K _ { j } ^ { g } ) + q _ { i j } .\tag{31}
$$

We select the candidate with the largest strictly positive $h _ { i j }$ , breaking ties by the earliest predicted-call index, and mark it as used. If no candidate has a positive score, the reference call remains unmatched. Denote the resulting assignment by $\pi ( i )$ , with $\pi ( i ) = \perp$ for unmatched calls. This is the greedy procedure used by the implementation.

Parameter-name and parameter-value scores. For a matched call, define $\begin{array} { r l } { a _ { i } } & { { } = } \end{array}$ $J ( K _ { i } ^ { r } , K _ { \pi ( i ) } ^ { g } )$ and $v _ { i } = q _ { i , \pi ( i ) } ;$ ; for an unmatched call, set $a _ { i } = v _ { i } = 0$ . With $\begin{array} { r } { N _ { r } = \sum _ { i = 1 } ^ { m } | K _ { i } ^ { r } | } \end{array}$ the two scores are

$$
\begin{array} { r } { S _ { \mathrm { p n } } ( y _ { g } , y _ { r } ) = \{ \frac { 1 } { m } \sum _ { i = 1 } ^ { m } a _ { i } , \quad m > 0 , \qquad S _ { \mathrm { p v } } ( y _ { g } , y _ { r } ) = \{ \begin{array} { l l } { \frac { 1 } { N _ { r } } \sum _ { i = 1 } ^ { m } v _ { i } , } & { N _ { r } > 0 , } \\ { 1 , } & { m = 0 , } \end{array}  } \end{array}\tag{32}
$$

Parameter-name matching gives equal weight to reference calls. Parameter-value matching gives equal weight to reference arguments and uses $\mathrm { P y }$ thon equality on the parsed JSON values. The zero-denominator convention treats a component with nothing to predict as fully satisfied. The implementation also returns full component scores immediately when the parsed reference and predicted call lists are exactly equal.

Invalid calls and responses without tool calls. If a reference requires a tool call but the generated block is missing or cannot be parsed and scored, the rewards are $R _ { \mathrm { f n } } = - 0 . 5$ $R _ { \mathrm { p n } } = - 1$ , and $R _ { \mathrm { p v } } = - 1 . 5$ . If the reference contains no tool-call block, all three content rewards are set to zero. These cases are handled separately from the matching formulas above.

Format checking. The format checker matches the entire generated response against the structure specified by the reference. It requires an initial <sub><think>...<</sub>/<sub>think></sub> block. Depending on the reference, this is followed by a tool-call block, a response block, both in that order, or neither. The required opening and closing tool-call and response tags must each occur exactly once. A newline separates the reasoning block from the next block; the tool-call payload is surrounded by newlines, and a following response block begins on the next line. These structural checks determine the binary format reward independently of JSON parsing and field correctness. As a result, valid delimiters can receive format credit even if the enclosed JSON is invalid.

## J.3 Coding Efficiency

To ensure a fair comparison of execution eficiency, we reuse the saved generated programs and remeasure each generated/reference pair over eight rounds. In each round, we include only pairs for which both programs pass all tests and both runtimes are positive and finite, excluding missing references, reference failures, and timeouts. The two programs execute consecutively in fresh Python subprocesses using the same interpreter, test harness, and pinned CPU core. Execution order alternates across rounds, yielding four reference-first and four generated-first executions. Each adjacent two-round block uses the same core, with core assignments rotating between blocks. We run up to 16 pairs concurrently on distinct physical cores within one CPU socket and deterministically shufle each core’s task queue between rounds. Wall-clock timing covers execution of the prompt, program, tests, and final correctness check, excluding compilation and subprocess startup. All subprocesses use a 5- second timeout and a 1,024-MiB memory limit, with no additional warmup executions. We independently determine eligibility in each round and report the mean of the eight roundlevel percentages of eligible pairs for which the generated program is strictly faster than the reference, reducing sensitivity to measurement noise, execution order, and core-specific variation.

## J.4 Tool-Call Format

The format reward is 1 when the response follows the required structure and 0 otherwise. For a tool-call response, the required structure is:

<think>Reasoning text</think>   
<tool\_call>   
{"name": "function\_name", "parameters": {"key": "value"}}   
</tool\_call>

Multiple calls appear as separate JSON lines within the same tool-call block. When the reference requires a direct answer, the tool-call block is replaced by <sub><response>Answer</sub> <sub>text<</sub>/<sub>response></sub>; when both are required, the response block follows the tool-call block. Unparseable tool calls receive the minimum scores for the three tool-content components.

## K Effects of Reward Scale and Correlation on Advantages

We analyze how reward scale and pairwise correlation afect advantage normalization in GRPO and CorrGRPO. Consider three reward components with baseline correlations $\rho _ { 1 2 } =$ 0.9 and $\rho _ { 1 3 } = \rho _ { 2 3 } = 0 . 1$ , and standard deviations $\sigma _ { 1 } = \sigma _ { 2 } = 1$ and $\sigma _ { 3 } = a > 0$ . We fix the centered total reward at $\Delta r = 1$ to isolate the efect of the denominator and omit the numerical stabilizer ε. All quantities other than the variable being varied are held fixed. Figure 13 compares a sweep over a with separate perturbations of the two correlations at the baseline $a = 5$

![](images/ce80b0ba82826195d133e21a59e7cbb65e6e72375021f04f9dd04794a5bf90fe.jpg)

![](images/5a649f6c95d209dd604d89f6901d02c259fbd4126d39b251e52b4f9d6b67df8a.jpg)

![](images/e5fd988622b25188fe59611b819b9fd0ca9f795b7f453754225777ccf4d2a083.jpg)  
Figure 13: Efect of reward scale and correlation on advantage normalization at fixed centered total reward. (a) Advantage as the third reward’s standard deviation varies. (b) Relative GRPO advantages under separate correlation perturbations. (c) Corresponding CorrGRPO responses.

For these reward statistics, the covariance sum $S$ and correlation sum $Q$ are

$$
S ( a ) = \sum _ { l , m = 1 } ^ { 3 } \sigma _ { l } \sigma _ { m } \rho _ { l m } = a ^ { 2 } + 0 . 4 a + 3 . 8 , \qquad Q = \sum _ { l , m = 1 } ^ { 3 } \rho _ { l m } = 5 . 2 .\tag{33}
$$

The GRPO denominator includes the variance term $a ^ { 2 }$ and the cross-covariance terms 0.4a, whereas the CorrGRPO denominator depends only on the correlations. Consequently,

$$
A _ { \mathrm { G R P O } } ( a ) = \frac { 1 } { \sqrt { a ^ { 2 } + 0 . 4 a + 3 . 8 } } , \qquad A _ { \mathrm { C o r r G R P O } } ( a ) = \frac { 1 } { \sqrt { 5 . 2 } } .\tag{34}
$$

As shown in Figure $\mathrm { 1 3 ( a ) }$ , increasing a monotonically decreases the GRPO advantage, even though the centered total reward and all pairwise correlations remain unchanged. $\mathrm { A t } \ a = 5 .$ the two advantages are approximately 0.1802 and 0.4385, respectively. The third reward’s variance contributes 25 of the total covariance sum 30.8, so it accounts for approximately 81.2% of the squared GRPO denominator. This illustrates how a large-scale reward can dominate normalization and attenuate the aggregate learning signal despite having only weak correlations with the other components. CorrGRPO’s advantage remains constant throughout the positive-scale sweep. The plotted value at $a = 0$ denotes the limit $a  0 ^ { + }$ the correlations require $a > 0$

Reward scale also afects how GRPO responds to changes in correlation. Fix $a = 5$ and separately perturb either $\rho _ { 1 2 } = 0 . 9 + \delta \mathrm { ~ o r ~ } \rho _ { 1 3 } = 0 . 1 + \delta $ , leaving the remaining correlation unchanged. We use $A ^ { ( l m ) } ( \delta )$ to denote the advantage when the correlation of pair $( l , m )$ is perturbed. Both covariance-matrix entries associated with that pair change, giving an increment $2 \sigma _ { l } \sigma _ { m } \delta$ in S. The corresponding relative GRPO advantages are

$$
\frac { A _ { \mathrm { G R P O } } ^ { ( 1 2 ) } ( \delta ) } { A _ { \mathrm { G R P O } } ( 0 ) } = \sqrt { \frac { 3 0 . 8 } { 3 0 . 8 + 2 \delta } } , \qquad \frac { A _ { \mathrm { G R P O } } ^ { ( 1 3 ) } ( \delta ) } { A _ { \mathrm { G R P O } } ( 0 ) } = \sqrt { \frac { 3 0 . 8 } { 3 0 . 8 + 1 0 \delta } } .\tag{35}
$$

Here, $A _ { \mathrm { G R P O } } ( 0 )$ denotes the unperturbed advantage at $a = 5$ . The same correlation increment changes S five times as much for pair (1, 3) as for pair (1, 2) because $\sigma _ { 1 } \sigma _ { 3 } = 5 \sigma _ { 1 } \sigma _ { 2 }$ . The relative advantage response also has a fivefold larger slope magnitude at $\delta = 0$ . Figure 13(b) therefore shows a stronger response to the weak correlation involving the larger-scale third reward than to the strong correlation between the first two rewards. Positive perturbations reduce the advantage, while negative perturbations increase it.

For CorrGRPO, either perturbation changes the correlation sum by exactly 2δ, yielding

$$
\frac { A _ { \mathrm { C o r r G R P O } } ^ { ( 1 2 ) } ( \delta ) } { A _ { \mathrm { C o r r G R P O } } ( 0 ) } = \frac { A _ { \mathrm { C o r r G R P O } } ^ { ( 1 3 ) } ( \delta ) } { A _ { \mathrm { C o r r G R P O } } ( 0 ) } = \sqrt { \frac { 5 . 2 } { 5 . 2 + 2 \delta } } .\tag{36}
$$

The two curves in Figure 13(c) thus coincide. Equal feasible changes in pairwise correlation have the same efect on the denominator, independently of the component scales. CorrGRPO retains the adjustment to reward dependence while removing the scale factors that make this adjustment uneven in GRPO. These comparisons hold the numerator fixed; they characterize the normalization mechanism and do not imply invariance of the full advantage when rescaling rewards also changes the centered total reward.

## L Theoretical Foundations and Properties of CorrGRPO

## L.1 Variance of the Total Reward as a Sum of Covariances

Fix a prompt and a sampling policy, and let $R _ { 1 } , \ldots , R _ { r }$ denote the resulting random reward components, each with a finite second moment. All expectations, variances, and covariances in this subsection are taken under this same conditional rollout distribution. Define $R =$

$\textstyle \sum _ { l = 1 } ^ { r } R _ { l }$ and $\mu _ { l } = \mathbb { E } [ R _ { l } ]$ . By linearity of expectation, $\begin{array} { r } { \mathbb E [ R ] = \sum _ { l } \mu _ { l } } \end{array}$ , and therefore

$$
\begin{array} { r l } {  { \operatorname { V a r } ( R ) = \mathbb { E } \big [ ( R - \mathbb { E } [ R ] ) ^ { 2 } \big ] } } \\ & { = \mathbb { E } [ ( \sum _ { i = 1 } ^ { \nu } ( R _ { i } - \mu _ { i } ) ) ^ { 2 } ] } \\ & { = \mathbb { E } [ \sum _ { i = 1 } ^ { r } \sum _ { m = 1 } ^ { r } ( R _ { i } - \mu _ { i } ) ( R _ { m } - \mu _ { m } ) ] } \\ & { = \displaystyle \sum _ { i = 1 } ^ { r } \sum _ { m = 1 } ^ { r } \mathbb { E } \big [ ( R _ { i } - \mu _ { i } ) ( R _ { m } - \mu _ { m } ) \big ] } \\ & { = \sum _ { i = 1 } ^ { r } \sum _ { m = 1 } ^ { r } \mathbb { E } \big [ ( R _ { i } - \mu _ { i } ) ( R _ { m } - \mu _ { m } ) \big ] } \\ & { = \sum _ { k = 1 } ^ { r } \sum _ { m = 1 } ^ { r } \mathrm { C o v } ( R _ { i } , R _ { m } ) . } \end{array}\tag{37}
$$

Since $\mathrm { C o v } ( R _ { l } , R _ { l } ) = \mathrm { V a r } ( R _ { l } )$ and covariance is symmetric, the same identity can be written as

$$
\operatorname { V a r } ( R ) = \sum _ { l = 1 } ^ { r } \operatorname { V a r } ( R _ { l } ) + 2 \sum _ { l < m } \operatorname { C o v } ( R _ { l } , R _ { m } ) .\tag{38}
$$

Thus, total-reward variance includes both individual reward variances and pairwise dependence. No independence assumption is required. When component standard deviations are nonzero, substituting Cov $( R _ { l } , \bar { R } _ { m } ) = \sigma _ { l } \sigma _ { m } \rho _ { l m }$ further gives

$$
\operatorname { V a r } ( R ) = \sum _ { l } \sigma _ { l } ^ { 2 } + 2 \sum _ { l < m } \sigma _ { l } \sigma _ { m } \rho _ { l m } .\tag{39}
$$

This is the population identity underlying the covariance interpretation of GRPO. Appendix L.2 establishes that its empirical counterpart also holds exactly for each finite rollout group.

## L.2 Exact E<sub>q</sub>uivalence of Finite-Sample Variance and Covariance Estimates

Consider the same n sampled reward vectors $\{ ( R _ { 1 } ^ { i } , \ldots , R _ { r } ^ { i } ) \} _ { i = 1 } ^ { n }$ , and define

$$
R ^ { i } = \sum _ { l = 1 } ^ { r } R _ { l } ^ { i } , \qquad \bar { R } _ { l } = \frac 1 n \sum _ { i = 1 } ^ { n } R _ { l } ^ { i } , \qquad \bar { R } = \frac 1 n \sum _ { i = 1 } ^ { n } R ^ { i } = \sum _ { l = 1 } ^ { r } \bar { R } _ { l } .\tag{40}
$$

Let $d _ { n } = n - 1$ for the usual sample-variance and sample-covariance estimates. The same argument applies with $d _ { n } = n$ if that convention is used for both. Define

$$
\begin{array} { c } { { \displaystyle \widehat { \mathrm { V a r } } _ { d _ { n } } ( R ) = \frac 1 { d _ { n } } \sum _ { i = 1 } ^ { n } ( R ^ { i } - \bar { R } ) ^ { 2 } \mathrm { , } } } \\ { { \displaystyle \widehat { \mathrm { C o v } } _ { d _ { n } } ( R _ { l } , R _ { m } ) = \frac 1 { d _ { n } } \sum _ { i = 1 } ^ { n } ( R _ { l } ^ { i } - \bar { R } _ { l } ) ( R _ { m } ^ { i } - \bar { R } _ { m } ) \mathrm { . } } } \end{array}\tag{41}
$$

Substituting the identity for the sample means and expanding the square $\mathrm { g i }$ ves

$$
\begin{array} { r l } & { \widehat { \mathrm { V a r } } _ { d _ { n } } ( R ) = \displaystyle \frac { 1 } { d _ { n } } \sum _ { i = 1 } ^ { n } \left[ \sum _ { l = 1 } ^ { r } ( R _ { l } ^ { i } - \bar { R } _ { l } ) \right] ^ { 2 } } \\ & { \quad \quad \quad = \displaystyle \frac { 1 } { d _ { n } } \sum _ { i = 1 } ^ { n } \sum _ { l = 1 } ^ { r } \sum _ { m = 1 } ^ { r } ( R _ { l } ^ { i } - \bar { R } _ { l } ) ( R _ { m } ^ { i } - \bar { R } _ { m } ) } \\ & { \quad \quad \quad = \displaystyle \sum _ { l = 1 } ^ { r } \sum _ { m = 1 } ^ { r } \widehat { \mathrm { C o v } } _ { d _ { n } } ( R _ { l } , R _ { m } ) . } \end{array}\tag{42}
$$

This is an exact identity for every realized sample group, not an asymptotic approximation or an equality only in expectation. It requires no independence assumption between reward components or between sampled trajectories. Constant components are also allowed. The requirements are that all statistics use the same samples, their corresponding sample means, and the same divisor $d _ { n }$ . The equality is generally lost if the variance and covariances use diferent divisors or diferent sample subsets.

Consequently, computing GRPO’s denominator directly from the sample variance of total rewards or from the sum of sample covariances gives the same result in exact arithmetic, including when the same $\varepsilon$ is added after taking the square root. More generally, for any fixed weights w and sample covariance matrix Σ $\hat { \Sigma }$

$$
\widehat { \mathrm { V a r } } _ { d _ { n } } \left( \sum _ { l } w _ { l } R _ { l } \right) = { \mathbf w } ^ { \top } \widehat { \boldsymbol { \Sigma } } { \mathbf w } .\tag{43}
$$

This weighted identity is the basis of the linear-combiner interpretation below.

## L.3 Group Rescaling and Compatibility with Clipping

Let $\mathbf { C } = [ \hat { \rho } _ { l m } ]$ denote the sample correlation matrix, $\mathbf { s } = ( \widehat { \sigma } _ { 1 } , \ldots , \widehat { \sigma } _ { r } ) ^ { \top }$ the vector of sample standard deviations, and 1 the all-ones vector. The denominator statistics satisfy $S = \mathbf { s } ^ { \intercal } \mathbf { \hat { C } } \mathbf { s }$ 5 and $Q = \mathbf { 1 } ^ { \top } \mathbf { C } \mathbf { 1 }$ . Because the numerator is shared, the relationship between the advantages is

$$
A _ { \mathrm { C o r r G R P O } } ^ { i } = c _ { q } A _ { \mathrm { G R P O } } ^ { i } , \qquad c _ { q } = \frac { \sqrt { \mathbf { s } ^ { \intercal } \mathbf { C } \mathbf { s } } + \varepsilon } { \sqrt { \mathbf { 1 } ^ { \intercal } \mathbf { C } \mathbf { 1 } } + \varepsilon } > 0 .\tag{44}
$$

For a fixed group, this preserves signs, ordering, and ratios between nonzero advantages. The coeficient can difer across groups, so this relation does not reduce CorrGRPO to a single global learning-rate change.

Define the per-token clipped surrogate by

$$
\ell ( u , A ) = \operatorname* { m i n } \{ u A , \operatorname { c l i p } ( u , 1 - \eta , 1 + \eta ) A \} .\tag{45}
$$

For any $c > 0$ , multiplying both arguments of the minimum by c gives $\ell ( u , c A ) = c \ell ( u , A )$ More explicitly,

$$
\ell ( u , A ) = \left\{ \begin{array} { l l } { A \operatorname* { m i n } \{ u , 1 + \eta \} , } & { A > 0 , } \\ { A \operatorname* { m a x } \{ u , 1 - \eta \} , } & { A < 0 , } \\ { 0 , } & { A = 0 . } \end{array} \right.\tag{46}
$$

The advantage magnitude therefore does not determine the clipping branch. At a fixed policy ratio, positive scaling preserves which terms saturate and the locations of their breakpoints.

Treating sampled advantages and $c _ { q }$ as constants during surrogate optimization, diferentiation away from the breakpoints gives

$$
\nabla _ { \boldsymbol { \theta } } \ell ( u ( \boldsymbol { \theta } ) , c _ { q } A ) = c _ { q } \nabla _ { \boldsymbol { \theta } } \ell ( u ( \boldsymbol { \theta } ) , A ) .\tag{47}
$$

The same positive scaling applies to the admissible one-sided derivatives at the breakpoints. To obtain the group-level relationship used in the main text, define the clipped reward surrogate for a fixed sampled group as

$$
L _ { M , q } ( \theta ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } \ell ( u _ { i , t } ( \theta ) , A _ { M } ^ { i } ) , \quad M \in \{ \mathrm { G R P O , C o r r G R P O } \} .\tag{48}
$$

Here q identifies the prompt together with the fixed sampled group being analyzed, and the token averaging matches the main paper’s surrogate. Factoring the same $c _ { q }$ out of every term yields

$$
L _ { \mathrm { C o r r G R P O } , q } ( \theta ) = c _ { q } L _ { \mathrm { G R P O } , q } ( \theta ) .\tag{49}
$$

Since sampled rewards, advantages, and their normalization statistics are held fixed during policy optimization, $\nabla _ { \theta } c _ { q } = 0$ . Therefore, where the surrogate is diferentiable,

$$
\nabla _ { \theta } L _ { \mathrm { C o r r G R P O } , q } = c _ { q } \nabla _ { \theta } L _ { \mathrm { G R P O } , q } , \qquad c _ { q } > 0 .\tag{50}
$$

The corresponding relation also holds for consistently selected generalized derivatives at clipping breakpoints. For a nonzero group gradient, positive scaling preserves its direction and changes its magnitude. Across groups, $\begin{array} { r } { \sum _ { q } c _ { q } \nabla _ { \theta } \overset { \cdot } { L } _ { \mathrm { G R P O } , q } } \end{array}$ need not be a positive scalar multiple of $\begin{array} { r } { \sum _ { q } \nabla _ { \theta } L _ { \mathrm { G R P O } , q } . } \end{array}$ , because the coeficients may difer.

These ${ \cal L } _ { M , q }$ contain the clipped reward terms only. If the full objective is $J _ { M , q } = L _ { M , q } - \beta K _ { q }$ with the same group-averaged KL term $K _ { q }$ and coeficient $\beta$ in both methods, then

$$
\nabla _ { \theta } J _ { \mathrm { C o r r G R P O } , q } = c _ { q } \nabla _ { \theta } J _ { \mathrm { G R P O } , q } + ( c _ { q } - 1 ) \beta \nabla _ { \theta } K _ { q } .\tag{51}
$$

Thus, the group-gradient scaling identity does not generally extend to the full objective with an unchanged KL coeficient. Clipping compatibility concerns the surrogate at the same policy ratios, and does not imply identical optimization trajectories or a hard bound on actual policy movement.

## L.4 Positive Affine Invariance of the Denominator

Consider componentwise transformations $R _ { l } ^ { \prime i } = a _ { l } R _ { l } ^ { i } + b _ { l }$ with $a _ { l } > 0$ . Centering removes the shifts, and

$$
\begin{array} { r l } & { R _ { l } ^ { \prime i } - \bar { R } _ { l } ^ { \prime } = a _ { l } ( R _ { l } ^ { i } - \bar { R } _ { l } ) , } \\ & { \widehat { \mathrm { C o v } } ( R _ { l } ^ { \prime } , R _ { m } ^ { \prime } ) = a _ { l } a _ { m } \widehat { \mathrm { C o v } } ( R _ { l } , R _ { m } ) , \qquad \widehat { \sigma } _ { l } ^ { \prime } = a _ { l } \widehat { \sigma } _ { l } . } \end{array}\tag{52}
$$

It follows directly that $\hat { \rho } _ { l m } ^ { \prime } = \hat { \rho } _ { l m } , \mathrm { s o } Q ^ { \prime } = Q$ and the CorrGRPO denominator is unchanged. In contrast, the numerator becomes $\begin{array} { r } { \sum _ { l } a _ { l } \big ( R _ { l } ^ { i } - \bar { R } _ { l } \big ) } \end{array}$ ; hence

$$
A _ { \mathrm { C o r r G R P O } } ^ { \prime i } = \frac { \sum _ { l } a _ { l } ( R _ { l } ^ { i } - \bar { R } _ { l } ) } { \sqrt { Q } + \varepsilon } .\tag{53}
$$

The invariance therefore concerns normalization, not the full advantage. For a common positive scale $a _ { l } = a$ , the full CorrGRPO advantage scales exactly by a. This preserves the distinction between component weights in the reward objective and dependence in the denominator. The result assumes exact Pearson coeficients and an unchanged set of nonconstant components; variance thresholds or extra regularizers inside Pearson coeficients can alter exact invariance.

## L.5 Positivity of the Normalization Denominators

We establish that the covariance and correlation matrices used in normalization are positive semidefinite. Consequently, the quantities under both square roots are nonnegative, and adding $\varepsilon > 0$ makes both normalization denominators strictly positive.

Consider a group of $n \geq 2$ trajectories. Let $\mathbf { X } \in \mathbb { R } ^ { n \times r }$ denote the centered reward matrix, with $X _ { i l } = R _ { l } ^ { i } - \bar { R } _ { l }$ . All sample variances and covariances are computed from this group using the divisor $n - 1$

Covariance-based normalization. The sample covariance matrix satisfies

$$
\widehat { \boldsymbol { \Sigma } } = \frac { 1 } { n - 1 } \mathbf { X } ^ { \top } \mathbf { X } .\tag{54}
$$

For any $\mathbf { v } \in \mathbb { R } ^ { r }$

$$
\mathbf { v } ^ { \top } \widehat { \Sigma } \mathbf { v } = \frac { 1 } { n - 1 } \| \mathbf { X } \mathbf { v } \| _ { 2 } ^ { 2 } \geq 0 .\tag{55}
$$

Therefore, $\widehat { \pmb { \Sigma } } \succeq 0$ , and

$$
S = \sum _ { l , m } \widehat { \mathrm { C o v } } ( R _ { l } , R _ { m } ) = \mathbf { 1 } ^ { \top } \widehat { \Sigma } \mathbf { 1 } \geq 0 .\tag{56}
$$

Correlation-based normalization with nonzero variances. First assume that all reward components have nonzero sample variance. Let

$$
\mathbf { D } = \mathrm { d i a g } ( \hat { \sigma } _ { 1 } , \dots , \hat { \sigma } _ { r } ) .\tag{57}
$$

The sample correlation matrix is

$$
\mathbf { C } = \mathbf { D } ^ { - 1 } \hat { \Sigma } \mathbf { D } ^ { - 1 } .\tag{58}
$$

For any $\mathbf { v } \in \mathbb { R } ^ { r }$

$$
\mathbf { v } ^ { \top } \mathbf { C } \mathbf { v } = ( \mathbf { D } ^ { - 1 } \mathbf { v } ) ^ { \top } \widehat { \pmb { \Sigma } } ( \mathbf { D } ^ { - 1 } \mathbf { v } ) \geq 0 .\tag{59}
$$

Thus, $\mathbf { C } \succeq 0$ , which implies

$$
Q = \sum _ { l , m } \hat { \rho } _ { l m } = \mathbf { 1 } ^ { \top } \mathbf { C } \mathbf { 1 } \geq 0 .\tag{60}
$$

Extension to zero-variance components. Pearson correlation is undefined when either component has zero variance. Our implementation assigns zero to the corresponding correlation entries, including diagonal entries. To establish positive semidefiniteness under this convention, reorder the reward components so that the k components with nonzero variance appear first. The resulting matrix has the block form

$$
\begin{array} { r } { \mathbf { C } = \left( \begin{array} { c c } { \mathbf { C } _ { + } } & { \mathbf { 0 } } \\ { \mathbf { 0 } } & { \mathbf { 0 } } \end{array} \right) , } \end{array}\tag{61}
$$

where $\mathbf { C } _ { + } \in \mathbb { R } ^ { k \times k }$ is the sample correlation matrix of the reward components with nonzero sample variance. By the preceding argument, $\mathbf { C } _ { + } \succeq 0$ . For any vector partitioned conformably as $\mathbf { v } = ( \mathbf { v } _ { + } ^ { \top } , \mathbf { v } _ { 0 } ^ { \top } ) ^ { \top }$ ，

$$
\mathbf { v } ^ { \top } \mathbf { C } \mathbf { v } = \mathbf { v } _ { + } ^ { \top } \mathbf { C } _ { + } \mathbf { v } _ { + } \geq 0 .\tag{62}
$$

Hence, the full matrix remains positive semidefinite. Reordering components does not afect positive semidefiniteness or the sum of matrix entries, so

$$
Q = \mathbf { 1 } _ { r } ^ { \top } \mathbf { C } \mathbf { 1 } _ { r } = \mathbf { 1 } _ { k } ^ { \top } \mathbf { C } _ { + } \mathbf { 1 } _ { k } \geq 0 .\tag{63}
$$

If all reward components have zero variance, then $\mathbf { C } = \mathbf { 0 }$ and $Q = 0$

Strict positivity of the denominators. The preceding results give

$$
\begin{array} { r } { D _ { \mathrm { G R P O } } = \sqrt { S } + \varepsilon \geq \varepsilon > 0 , } \\ { D _ { \mathrm { C o r r G R P O } } = \sqrt { Q } + \varepsilon \geq \varepsilon > 0 . } \end{array}\tag{64}
$$

Thus, negative covariance or correlation entries cannot make the quantities under the square roots negative. These quantities can nevertheless equal zero: for example, two reward components with nonzero sample variance and perfect negative correlation yield $Q = 2 +$ $2 ( - \bar { 1 } ) = 0$ The stabilizer ensures strictly positive denominators even in such degenerate cases. The implementation’s convention, $\sqrt { \operatorname* { m a x } ( Q , 0 ) + \varepsilon } .$ is also strictly positive, with clamping guarding against negative floating-point round-of.

## M Datasets and Benchmarks

## M.1 Training Benchmarks

LeetCodeDataset. LeetCodeDataset (Xia et al., 2025) contains Python programming problems curated from LeetCode, with natural-language descriptions, reference solutions, executable test cases, and temporal metadata. Its problems require translating task specifications into functionally correct programs, while the reference implementations also support runtime comparisons. We use 2,641 problems for reinforcement learning and a 228-problem split for evaluation, measuring correctness, executability, and execution eficiency.

RLLA-4K. RLLA-4K, used in ToolRL (Qian et al., 2025), contains user requests, descriptions of available tools, and reference responses specifying tool calls or direct answers. It supports feedback on function selection, parameter names, parameter values, and response format. We use 3,920 examples for reinforcement learning and 80 for evaluation. Tool-call metrics are computed on the 71 evaluation examples whose reference responses contain tool calls, covering component-level matching and complete-call correctness.

AgentDojo. AgentDojo (Debenedetti et al., 2024) provides stateful environments in which agents complete user tasks through executable tools while potentially encountering prompt injections in tool observations. Task-specific checkers assess legitimate task completion and attacker-objective success, enabling joint evaluation of utility and security. We construct a split grouped by suite and user-task identifier, with 1,584 training cases and 411 evaluation cases comprising 21 clean tasks and 390 attacked cases. Our attacks use the <sub>important\_instructions</sub> and <sub>tool\_knowledge</sub> settings.

## M.2 Evaluation Benchmarks

HumanEval. HumanEval (Chen et al., 2021) evaluates Python function completion from natural-language specifications. Each task provides a function signature and docstring, and generated implementations are checked using executable tests. We use HumanEval exclusively for evaluation and report Pass@1 to assess whether training on LeetCodeDataset transfers to function-level programming tasks.

MBPP. Mostly Basic Python Problems (MBPP; Austin et al., 2021) contains short Python programming tasks described in natural language and accompanied by reference code and test cases. It emphasizes basic programming skills and the translation of concise specifications into executable solutions. We use MBPP as an external evaluation benchmark and report Pass@1 without further training.

LiveCodeBench. LiveCodeBench (Jain et al., 2024) evaluates code generation using programming-contest problems and executable tests. Its time-based organization supports evaluation on problems released during specified periods. We use the 175-problem evaluation subset from version 6 and report Pass@1, assessing transfer from LeetCodeDataset training to competition-style programming tasks.

API-Bank. API-Bank (Li et al., 2023) is a benchmark for tool-augmented language models that combines tool-use dialogues with executable APIs and an evaluation framework. Its tasks assess the ability to select and invoke APIs with appropriate arguments in a dialogue context. We evaluate the v1, v2, and v3 sets without additional training, using the execution-based checker to assess final-call correctness and generalization beyond RLLA-4K.

Agent Security Bench. Agent Security Bench (ASB; Zhang et al., 2025) evaluates agents across application scenarios containing legitimate tasks, tools, and adversarial objectives. We use its observation prompt-injection setting with <sub>context\_ignoring</sub> payloads appended to intermediate tool responses. Each model is evaluated on 400 paired clean and attacked cases without further training, measuring transfer of utility and security from AgentDojo to a diferent environment and attack setting.

InjecAgent. InjecAgent (Zhan et al., 2024) benchmarks indirect prompt injections delivered through attacker-controlled tool responses. Each case supplies a user request, available tools, and a preceding tool interaction; the agent is evaluated on its continuation after consuming the injected response. Attacks target direct harm or data stealing, with the latter involving information extraction followed by transmission to the attacker. We use the base setting and the standard InjecAgent prompt without further training.