# Are We Really Measuring Forgetting? Pitfalls in Evaluating Machine Unlearning for Multimodal LLMs

Oscar Chew<sup>∗,1</sup> Po-Yi Lu<sup>∗,1,2</sup> Yi-Jie Cheng<sup>3</sup> Kunal Jain<sup>1</sup>

Hsuan-Tien Lin<sup>2</sup> Zhengzhong Tu<sup>1</sup> Kuan-Hao Huang<sup>1</sup>

<sup>1</sup>Texas A&M University <sup>2</sup>National Taiwan University <sup>3</sup>Columbia University

{oscarchew,kjain,tzz,khhuang}@tamu.edu

{d09944015,htlin}@csie.ntu.edu.tw yc4708@columbia.edu

## Abstract

As machine unlearning for multimodal large language models (MLLMs) rapidly advances, rigorous evaluation of its effectiveness becomes increasingly important. However, existing benchmarks largely inherit evaluation protocols from text-only unlearning and overlook the unique challenges of unlearning in multimodal settings. In this work, we identify three major pitfalls that may lead to misleading conclusions: the lack of heldout evaluation to distinguish genuine concept-wise forgetting from instancewise overfitting, the reliance on synthetic, finetuned knowledge rather than real-world knowledge acquired during pre-training, and unfair method comparisons caused by applying the same training settings to different unlearning methods. To address these challenges, we introduce GEMMU (Generalizable Evaluation for Multimodal Machine Unlearning), a benchmark built on real-world knowledge and a principled evaluation protocol. Re-evaluating representative unlearning methods under GEMMU reveals substantial generalization gaps and challenges conclusions drawn from previous benchmarks. Our findings highlight the importance of rigorous evaluation for advancing multimodal machine unlearning and provide guidelines for future benchmark design.

## 1 Introduction

As MLLMs (Bai et al., 2025; Li et al., 2025) are increasingly deployed in real-world applications, concerns surrounding privacy and data governance have intensified. In this context, machine unlearning has attracted growing attention as a mechanism for removing sensitive or private knowledge, such as personal information, from trained models. As unlearning methods advance, rigorous evaluation is essential to reliably assess their effectiveness.

Several benchmarks (Liu et al., 2025a; Dontsov et al., 2025; Wang et al., 2025) have been proposed to assess progress in multimodal machine unlearning. However, we identify the following fundamental pitfalls in existing benchmarks that may lead to overly optimistic or misleading evaluations.

First, testing on the training set is widely recognized as poor practice in machine learning, yet existing unlearning protocols often evaluate forgetting and retention on the same data used during unlearning (Liu et al., 2025a; Ma et al., 2025; Zheng et al., 2026). Such evaluation risks overestimating unlearning effectiveness, as performance on these examples may not generalize to unseen data. This concern is particularly important in the multimodal setting, where effective unlearning should remove knowledge at the concept level rather than merely suppressing specific visual instances.

Second, many existing benchmarks (Dontsov et al., 2025; Wang et al., 2025; Liu et al., 2025a; Xu et al., 2025) focus primarily on newly fine-tuned knowledge, rather than real-world knowledge acquired during pre-training. While convenient for evaluation, such settings may not reflect realistic unlearning scenarios. Moreover, existing unlearning methods largely rely on reverse fine-tuning, which may be more effective at removing newly introduced knowledge. In contrast, pre-trained knowledge is often deeply entangled with other useful knowledge, making it difficult to remove selectively without affecting the model’s general capabilities. As a result, those benchmarks may overestimate the effectiveness of existing methods in real-world unlearning.

Third, most existing benchmarks evaluate different unlearning methods using identical hyperparameters, despite differences in their optimization objectives and training dynamics (Xu et al., 2025; Dontsov et al., 2025). A configuration that works well for one method may be suboptimal for another. Consequently, conclusions about which method is more effective can depend on the shared configuration rather than on the methods themselves.

![](images/d2d5c3ccad30a4b240f821e873470cf2b1df8cb6caa170c30b6605a395d7edff.jpg)  
Figure 1: Three major pitfalls in existing multimodal unlearning benchmarks (top) and how GEMMU addresses them (bottom) through heldout evaluation, real-world knowledge unlearning, and method-specific hyperparameter tuning to achieve generalizable evaluations.

To address these limitations, we introduce GEMMU (Generalizable Evaluation for Multimodal Machine Unlearning; Fig. 1), a benchmark built on three principles: heldout evaluation for both forget and retain sets, unlearning of real-world pre-trained knowledge, per-method hyperparameter search (Tab. 1), and additionally modality gap analysis (Sec. 5.4). Re-evaluating representative unlearning methods with GEMMU reveals substantial generalization gaps overlooked by existing benchmarks, shows that method rankings can change under different configurations, and calls into question the effectiveness of current methods. Our main contributions are summarized as follows:

• We identify major pitfalls in existing MLLM unlearning evaluations: testing on the training set, reliance onfine-tuned knowledge, and unfair hyperparameter settings across methods.

• We introduce GEMMU, a more rigorous benchmark that addresses these limitations and provides principles for future benchmark design.

• We re-evaluate nine representative unlearning methods and uncover substantial generalization gaps and ranking changes overlooked by previous benchmarks.

## 2 Related Works

Machine Unlearning Methods for MLLMs Existing approaches to multimodal machine unlearning mainly follow a reversefine-tuning paradigm, typically suppressing forget-set knowledge through gradient ascent while preserving retained knowledge through gradient descent or other regularization objectives. Representative methods include GA (Thudi et al., 2022), GADiff (Liu et al., 2022), KL Min and IDK (Maini et al., 2024), NPO (Zhang et al., 2024), LLMU (Yao and Xu, 2024), RMU (Li et al., 2024b), and SCRUB (Kurmanji et al., 2023). While some of the above methods were originally developed for text-only LLMs, they have served as common baselines for unlearning in MLLMs. More recently, model editing has emerged as an alternative to conventional tuning-based unlearning. For example, MANU (Liu et al., 2025b) selectively identifies and prunes model components associated with the knowledge to be forgotten.

Machine Unlearning Evaluation for MLLMs Early language-only benchmarks such as TOFU (Maini et al., 2024) establish a controlled evaluation setting for fictitious knowledge removal, and MLLMU-Bench (Liu et al., 2025a) extends this direction to multimodal models. CLEAR (Dontsov et al., 2025), FIUBench (Ma et al., 2025), UMU-Bench (Wang et al., 2025), and PEBench (Xu et al., 2025) introduce more comprehensive multimodal evaluation settings covering person profiles, cross-modal knowledge, and personal entities with event scenes. Despite this progress, existing benchmarks remain limited for evaluating practical multimodal unlearning.

<table><tr><td>Benchmark</td><td>Proper Train-test Split</td><td>Knowledge Acquisition</td><td>Unlearned Target Source</td><td># Real-world Concepts</td><td>Concept Types</td><td>Hyperparameter Selection</td><td>Open-source</td></tr><tr><td>MMUBench (Li et al., 2024a)</td><td>Partial</td><td>Pre-trained</td><td>Real-world</td><td>20</td><td>Diverse</td><td>One-size-fits-all</td><td>No</td></tr><tr><td>MLLMU-Bench (Liu et al., 2025a)</td><td>No</td><td>Fine-tuned</td><td>Synthetic</td><td></td><td>Human only</td><td>One-size-fits-all</td><td>Yes</td></tr><tr><td>FIUBench (Ma et al., 2025)</td><td>No</td><td>Fine-tuned</td><td>Synthetic</td><td></td><td>Human only</td><td>Per-method</td><td>Yes</td></tr><tr><td>CLEAR (Dontsov et al., 2025)</td><td>No</td><td>Fine-tuned</td><td>Synthetic</td><td></td><td>Human only</td><td>One-size-fits-all</td><td>Yes</td></tr><tr><td>PEBench (Xu et al., 2025)</td><td>Partial</td><td>Fine-tuned</td><td>Synthetic</td><td></td><td>Humans, events</td><td>One-size-fits-all</td><td>Yes</td></tr><tr><td>UMU-Bench (Wang et al., 2025)</td><td>No</td><td>Fine-tuned</td><td>Synthetic</td><td>一</td><td>Human only</td><td>One-size-fits-all</td><td>Yes</td></tr><tr><td>PULSE (Kawakami et al., 2025)</td><td>No</td><td>Pre-trained</td><td>Real-world</td><td>20</td><td>Human only</td><td>Not listed</td><td>No</td></tr><tr><td>OFFSIDE (Zheng et al., 2026)</td><td>No</td><td>Fine-tuned</td><td>Real-world</td><td>80</td><td>Human only</td><td>Per-method</td><td>Yes</td></tr><tr><td>GEMMU (Ours)</td><td>Yes</td><td>Pre-trained</td><td>Real-world</td><td>139</td><td>Diverse</td><td>Searched</td><td>Yes</td></tr></table>

Table 1: Comparison of existing multimodal machine unlearning benchmarks. GEMMU is the most comprehensive benchmark that evaluates the unlearning of real-world, pre-trained knowledge with a proper train-test split.

Car Brands | GADiff Ir=1e-5 | Classification (VQA)  
![](images/58ed63d6764caab6545cb986eab2e1cc7cfa1675faa14347ef32fb835152cc9f.jpg)  
Figure 2: A representative example of the generalization gap. The red dotted curve falls well below the red solid curve, while the blue dotted curve remains above the blue solid curve, revealing substantial train–test discrepancies after unlearning.

They still rely on generated identities, which require fine-tuning to memorize this new knowledge and may not reflect the unlearning of real-world, pre-trained knowledge. MMUBench and PEBench (Li et al., 2024a; Xu et al., 2025) partition the forget set into training and test examples but overlook the same problem for the retain set. An existing preprint, PULSE (Kawakami et al., 2025), studies the unlearning of pre-trained knowledge, but does not provide a theoretical analysis of the distinction between fine-tuned and pre-trained knowledge, nor does it address overfitting to the forget or retain sets.

## 3 Preliminaries

Formulation of the Unlearning Task Let $f _ { \theta }$ denote an MLLM with parameters θ. Following the standard formulation in this field, the unlearning task considers two datasets: a forget set $\mathcal { D } _ { \mathrm { f o r g e t } } ,$ containing knowledge to be removed, and a retain set $\mathscr { D } _ { \mathrm { r e t a i n } }$ , containing knowledge to be preserved. Each dataset consists of image-question-answer triplets $( I , q , a )$ , where I is empty for text-only questions. We further partition both datasets into disjoint training and test sets to evaluate generalizability: $\mathcal { D } _ { \mathrm { f o r g e t } } ^ { \mathrm { t r a i n } } , \mathcal { D } _ { \mathrm { f o r g e t } } ^ { \mathrm { t e s t } } , \mathcal { D } _ { \mathrm { r e t a i n } } ^ { \mathrm { t r a i n } }$ and $\mathcal { D } _ { \mathrm { r e t a i n } } ^ { \mathrm { t e s t } }$

Given a pre-trained MLLM $f _ { \theta _ { 0 } } ,$ where $\theta _ { 0 }$ denotes the parameters before unlearning, an unlearning method U updates $\theta _ { 0 }$ using the training splits:

$$
\theta _ { u } = \mathcal { U } ( \theta _ { 0 } , { \cal D } _ { \mathrm { f o r g e t } } ^ { \mathrm { t r a i n } } , { \cal D } _ { \mathrm { r e t a i n } } ^ { \mathrm { t r a i n } } ) ,
$$

where $\theta _ { u }$ denotes the parameters after unlearning. Note that previous works often use the entire $\mathcal { D } _ { \mathrm { f o r g e t } }$ and $\mathscr { D } _ { \mathrm { r e t a i n } }$ in place of the training splits mentioned above. The objective is to reduce the model’s ability to recover knowledge in $\mathcal { D } _ { \mathrm { f o r g e t } }$ , while preserving its performance on $\mathscr { D } _ { \mathrm { r e t a i n } }$ . Successful unlearning should achieve these goals not only on the training splits but also on the test splits, which provide a more reliable measure of generalizable forgetting and knowledge retention.

## 4 Pitfalls in Past Evaluations

We discuss three major pitfalls in existing multimodal unlearning benchmarks: the lack of proper train-test splits for concept-wise unlearning, the reliance on synthetic, fine-tuned knowledge, and unfair comparisons arising from applying identical hyperparameters across different unlearning methods. All experiments are conducted on our proposed GEMMU dataset, with construction details provided in Sec. 5.

## 4.1 Testing on the Training Set Is Problematic for Concept-wise Unlearning

A common practice in prior multimodal unlearning evaluations (Liu et al., 2025a; Huo et al., 2025; Dontsov et al., 2025) is to evaluate forgetting on the same $\mathcal { D } _ { \mathrm { f o r g e t } }$ used during unlearning, and similarly evaluate retention on the same $\mathscr { D } _ { \mathrm { r e t a i n } }$ . Unfortunately, this violates one of the most fundamental principles of machine learning: evaluating generalization on unseen data. A model may appear to forget the target knowledge and retain the rest, while merely overfitting to the examples seen during training. Fig. 2 shows a representative example of overly optimistic training-set evaluation, where accuracy is lower on the forget set and higher on the retain set compared to their respective test sets.

![](images/ce1e2c93f2738049008ae744e95528818ce0a2ba94095281d54f6312042d776c.jpg)

![](images/ff7f38301744935cde14b04fa38f1aeeff286414829e48447ada18991bbae3ca.jpg)

![](images/f003423158412339d0e0644ec5e73aa18303f1ef73513143f857e7489c1123e5.jpg)

![](images/c5c9a73dd320b2fcefa801d96f6b0118d1c9fd674b8d2b4ac5121ca5af9d638a.jpg)  
Figure 3: Comparison of unlearning generalization between image-text (left) and text-only (right) inputs. Image-text unlearning exhibits a substantially larger generalization gap than text-only unlearning, despite using the same dataset and training configurations.

One might argue that some prior works additionally evaluate general capabilities on heldout, real-world benchmarks. However, such evaluations do not address the fundamental issue. Without proper train-test splits for both $\mathcal { D } _ { \mathrm { f o r g e t } }$ and ${ \mathcal { D } } _ { \mathrm { r e t a i n } } ,$ the reported forgetting and retention scores remain unreliable indicators of unlearning effectiveness, despite being the primary metrics.

Why is this issue more important in multimodal unlearning? In some text-only unlearning scenarios, such as forgetting specific credit card numbers or passwords, the target is an exact string, making training-set evaluation potentially adequate for such instance-wise unlearning. In contrast, the same visual concept (e.g., a photo of Tokyo Tower) can appear in images with different viewpoints, backgrounds, and styles. This introduces greater variation in the input space, making generalization particularly important for concept-wise unlearning. Fig. 3 shows examples where image-text unlearning (concepts provided in images) exhibits a substantially larger generalization gap than textonly unlearning (concepts provided in pure text), highlighting the risk of overestimating forgetting in multimodal machine unlearning when evaluating only on training examples.

## 4.2 Reliance on Synthetic, Fine-tuned Knowledge

Another key limitation of existing multimodal unlearning benchmarks is their reliance on knowledge introduced through fine-tuning, often in the form of synthetic facts or identities. While such settings offer a controlled environment for evaluating unlearning, they may not accurately reflect the challenges of real-world knowledge removal. In particular, unlearning fine-tuned knowledge and pre-trained knowledge may differ substantially in their underlying dynamics and difficulty.

Synthetic knowledge introduced through finetuning often consists of relatively isolated facts that are less entangled with the model’s existing knowledge. Reverse fine-tuning, which underlies many existing unlearning algorithms, may provide a relatively straightforward way to undo the effects of fine-tuning and thereby achieve unlearning. In contrast, real-world, pre-trained knowledge is acquired through extensive pre-training and may be deeply entangled with other useful knowledge, making its removal more challenging. As a result, strong performance on synthetic or fine-tuned knowledge does not necessarily translate to effective unlearning of real-world, pre-trained knowledge.

To better understand the differences between unlearning pre-trained and fine-tuned knowledge, we investigate both settings from two perspectives: (1) learning dynamics and (2) the forgetting–retention trade-off. These two perspectives provide complementary empirical and theoretical insights into unlearning difficulty.

Learning Dynamics. We compare the two settings directly by tracking forget-set performance over unlearning epochs on two matched pairs of GEMMU subsets: real faces versus synthetic identities, and real landmarks versus fictitious ones, where the second subset of each pair is injected into the model through fine-tuning.

In Fig. 4, fine-tuned knowledge drops sharply within the first epoch, whereas pre-trained knowledge decays much more slowly on both LLaVA-1.5-7B and Qwen3-VL-8B-Instruct. Plots for other unlearning algorithms are provided in Appendix A.

![](images/aba0f07730f2e8a97ed332889c954d292a26fa5912e3c04f2dd86b43c37b7733.jpg)  
Figure 4: Forget-set image-textual performance (relative to the model before unlearning) over NPO unlearning epochs under identical hyperparameters, on LLaVA-1.5-7B (left) and Qwen3-VL-8B-Instruct (right). Fine-tuned knowledge (dashed) collapses much more quickly than pre-trained knowledge (solid).

Forgetting–Retention Trade-off. Another way to compare unlearning fine-tuned and pre-trained knowledge is to examine the retention cost required to achieve a given level of forgetting. To formalize this intuition, we present Proposition 1.

Definitions. Let $s _ { \mathrm { t a r g e t } } ( \theta )$ and $s _ { \mathrm { h e l d o u t } } ( \theta )$ denote knowledge scores for the target and heldout concepts, respectively, with higher values indicating better performance. We say that forgetting is successful when $s _ { \mathrm { t a r g e t } }$ is below a certain threshold $\tau _ { \ast }$ , and define the required amount of forgetting as $\gamma = s _ { \mathrm { t a r g e t } } ( \theta _ { 0 } ) - \tau > 0$ . The retention cost is then $s _ { \mathrm { h e l d o u t } } ( \theta _ { 0 } ) - s _ { \mathrm { h e l d o u t } } ( \theta _ { u } )$

## Proposition 1 (Forgetting–Retention Trade-off)

I. Forfine-tuned knowledge, zero retention cost is attainable by reverting to the original model parameters $\theta _ { 0 }$

II. For pre-trained knowledge, an unlearned model $\theta _ { u }$ satisfying the forgetting criterion and assumptions incurs a strictly positive retention cost:

$$
s _ { \mathrm { h e l d o u t } } ( \theta _ { 0 } ) - s _ { \mathrm { h e l d o u t } } ( \theta _ { u } ) \geq \frac { \mu \gamma ^ { 2 } } { 2 K ^ { 2 } } - \frac { \varepsilon _ { 0 } \gamma } K > 0 .
$$

under the assumptions specified in $A p \cdot$ pendix B.2. Here, K, $\mu ,$ and $\varepsilon _ { \mathrm { 0 } }$ are constants defined in Appendices B.1 and B.2.

The proof is provided in Appendix B. This result suggests that unlearning pre-trained knowledge can be viewed as more challenging than unlearning fine-tuned knowledge, as successful forgetting necessarily comes at the cost of other useful knowledge under our assumptions.

![](images/8aa124565839a67134c95ebc2012373a568a953976d6475865d24c73618d5b0e.jpg)  
Figure 5: KL Min and RMU under two shared learning rates on LLaVA-1.5-7B. The winner flips with the shared configuration, showing that one-size-fits-all comparisons are unreliable.

## 4.3 Hyperparameter Sensitivity

Most existing benchmarks compare unlearning methods under a single shared training configuration (Tab. 1). This implicitly assumes that different methods share a common operating range, which we find to be far from true. 1

Fig. 5 compares KL Min and RMU trained under identical shared configurations on LLaVA-1.5- 7B, measured by average retain-set minus average forget-set performance on heldout examples, where higher means better forgetting at lower retention cost. At a learning rate of $1 0 ^ { - 5 }$ , RMU wins clearly while KL Min appears entirely ineffective; at $5 \times 1 0 ^ { - 5 }$ , the conclusion reverses, with KL Min now outperforming RMU by a wide margin. The same shared protocol thus supports two contradictory conclusions about which algorithm is better.

Fig. 6 extends this observation to all nine methods across our search space. Without proper tuning, many configurations lie on the y = x line, i.e., indiscriminate forgetting. After per-method search, the strongest methods fall well below it. This sensitivity partly reflects that multimodal unlearning is harder than text-only unlearning, where a shared configuration is common practice (Maini et al., 2024), and further indicates that fair evaluation for multimodal unlearning requires a permethod hyperparameter search. More examples are provided in Appendix D.

<table><tr><td rowspan="3">Methods</td><td colspan="6">Forget Test (↓)</td><td colspan="6">Retain Test (↑)</td><td rowspan="3">Rank</td></tr><tr><td colspan="2">Classification</td><td colspan="2">FITB</td><td colspan="2">Generation</td><td colspan="2">Classification</td><td colspan="2">FITB</td><td colspan="2">Generation</td></tr><tr><td>VQA</td><td>QA</td><td>VQA</td><td>QA</td><td>VQA</td><td>QA</td><td>VQA</td><td>QA</td><td>VQA</td><td>QA</td><td>VQA QA</td><td></td></tr><tr><td colspan="10">LLaVA-1.5-7B</td><td></td><td></td><td></td><td></td></tr><tr><td>Pre-trained</td><td>74.86</td><td>79.89</td><td>51.59</td><td>70.92</td><td>15.77</td><td>70.47</td><td>72.29</td><td>80.00</td><td>40.38</td><td>65.79</td><td>13.26</td><td>68.81</td><td></td></tr><tr><td>GA (Thudi et a., 2022)</td><td>64.76</td><td>65.87</td><td>20.79</td><td>39.06</td><td>7.70</td><td>19.28</td><td>60.97</td><td>66.37</td><td>18.97</td><td>44.93</td><td>6.95</td><td>19.45</td><td>8</td></tr><tr><td>GADiff (Liu et al., 2022)</td><td>4.33</td><td>0.00</td><td>2.55</td><td>0.00</td><td>3.20</td><td>0.00</td><td>48.49</td><td>69.29</td><td>54.61</td><td>87.52</td><td>34.18</td><td>81.50</td><td>1</td></tr><tr><td>SCRUB (Kurmanji et al., 2023)</td><td>19.17</td><td>75.19</td><td>9.27</td><td>51.90</td><td>3.87</td><td>60.48</td><td>69.50</td><td>80.03</td><td>59.04</td><td>87.81</td><td>29.52</td><td>80.75</td><td>4</td></tr><tr><td>KL Min (Maini et al., 2024)</td><td>47.26</td><td>30.81</td><td>23.81</td><td>14.52</td><td>7.13</td><td>18.24</td><td>68.87</td><td>79.31</td><td>41.87</td><td>70.96</td><td>14.28</td><td>70.42</td><td>3</td></tr><tr><td>NPO (Zhang et al., 2024)</td><td>72.08</td><td>79.34</td><td>33.57</td><td>43.40</td><td>13.31</td><td>22.83</td><td>69.18</td><td>77.93</td><td>30.12</td><td>51.05</td><td>12.49</td><td>23.45</td><td>9</td></tr><tr><td>IDK (Maini et al., 2024)</td><td>60.81</td><td>71.82</td><td>6.10</td><td>34.81</td><td>3.30</td><td>42.52</td><td>69.69</td><td>76.26</td><td>87.80</td><td>85.45</td><td>33.85</td><td>72.82</td><td>2</td></tr><tr><td>LLMU (Yao and Xu, 2024)</td><td>66.84</td><td>73.50</td><td>23.85</td><td>24.02</td><td>5.96</td><td>39.10</td><td>65.06</td><td>77.05</td><td>21.59</td><td>29.97</td><td>6.94</td><td>42.53</td><td>7</td></tr><tr><td>RMU (Li et al., 2024b)</td><td>4.54</td><td>74.11</td><td>1.16</td><td>64.13</td><td>1.06</td><td>54.63</td><td>65.74</td><td>75.86</td><td>34.34</td><td>58.02</td><td>18.21</td><td>54.88</td><td>5</td></tr><tr><td>MANU (Liu et al., 2025b)</td><td>60.14</td><td>70.35</td><td>20.68</td><td>37.52</td><td>5.11</td><td>58.41</td><td>63.78</td><td>74.75</td><td>25.99</td><td>43.55</td><td>7.29</td><td>61.81</td><td>6</td></tr><tr><td colspan="10">Qwen3-VL-8B-Instruct</td><td colspan="7"></td></tr><tr><td>Pre-trained</td><td>88.82</td><td>86.32</td><td>82.61</td><td>78.37</td><td>42.11</td><td>59.49</td><td>86.69</td><td>85.25</td><td>76.39</td><td>78.20</td><td>43.16</td><td>59.23</td><td></td><td></td></tr><tr><td>GA (Thudi et al., 2022)</td><td>67.76</td><td>65.97</td><td>34.62</td><td>38.45</td><td>3.76</td><td>8.53</td><td>64.83</td><td>67.90</td><td></td><td>28.28</td><td>37.46</td><td>4.76</td><td>12.41</td><td>8</td></tr><tr><td>GADiff (Liu et al., 2022)</td><td>13.91</td><td>31.10</td><td>0.10</td><td>0.00</td><td>0.23</td><td>0.00</td><td>56.70</td><td>68.09</td><td></td><td>58.62</td><td>57.46</td><td>53.01</td><td>73.35</td><td>2</td></tr><tr><td>SCRUB (Kurmanji et li, 2023)</td><td>3.07</td><td>41.22</td><td>1.39</td><td>31.34</td><td>0.80</td><td>20.56</td><td>84.11</td><td></td><td>87.04</td><td>76.03</td><td>86.24</td><td>44.72</td><td>73.97</td><td>1</td></tr><tr><td>KL Min (Maini et al., 2024)</td><td>88.81</td><td>86.10</td><td>79.60</td><td>73.65</td><td>40.11</td><td>57.38</td><td></td><td>86.76</td><td>85.12</td><td>77.41</td><td>78.31</td><td>42.20</td><td>59.41</td><td>7</td></tr><tr><td>NPO (Zhang et al., 2024)</td><td>68.63</td><td>66.17</td><td>22.06</td><td>43.92</td><td>3.10</td><td>10.58</td><td>65.44</td><td></td><td>68.37</td><td>19.48</td><td>40.31</td><td>2.43</td><td>11.26</td><td>9</td></tr><tr><td>IDK (Maini et al., 2024)</td><td>65.17</td><td>86.38</td><td>7.83</td><td>38.88</td><td>2.66</td><td>13.73</td><td></td><td>53.84</td><td>90.43</td><td>86.92</td><td>67.09</td><td>42.53</td><td>62.25</td><td>3</td></tr><tr><td>LLMU (Yao and Xu, 2024)</td><td>86.17</td><td>83.76</td><td>34.80</td><td>42.65</td><td>19.70</td><td>25.73</td><td>84.78</td><td></td><td>83.94</td><td>35.32</td><td>49.97</td><td>18.56</td><td>27.38</td><td>6</td></tr><tr><td>RMU (Li et al., 2024b)</td><td>2.48</td><td>83.91</td><td>1.66</td><td>76.88</td><td>4.02</td><td>59.22</td><td>84.34</td><td></td><td>85.28</td><td>71.44</td><td>75.18</td><td>41.01</td><td>58.98</td><td>4</td></tr><tr><td>MANU (Liu et al., 2025b)</td><td>49.65</td><td>35.62</td><td>21.37</td><td>20.26</td><td>34.68</td><td>41.02</td><td>76.31</td><td>65.08</td><td></td><td>48.80</td><td>46.02</td><td>40.64</td><td>55.26</td><td>5</td></tr></table>

Table 2: Benchmark results on LLaVA-1.5-7B and Qwen3-VL-8B-Instruct, averaged over the four real-world subsets. VQA denotes image-textual and QA pure-text evaluation. Bold and underline denote the best and secondbest result per column within each model; Rank orders methods by average retain-test minus average forget-test performance. The classic and simple GADiff achieves the strongest overall balance between forgetting and retention.

## 5 GEMMU

GEMMU is designed to reveal the pitfalls discussed in Sec. 4 and to provide practical guidelines that future evaluation benchmarks should follow. Its core components include (1) concept removal of existing real-world knowledge and comparison with fine-tuned knowledge; (2) generalization of forgetting and retention to heldout examples; (3) per-method hyperparameter search for fair comparisons; and additionally, (4) modality gap analysis using image-text and text-only questions, inspired by Wang et al. (2025) and Huo et al. (2025). We first describe our methodology (Sec. 5.1), followed by a re-benchmarking of existing unlearning methods (Sec. 5.4), and conclude with design principles for future benchmarks (Sec. 5.5).

## 5.1 Benchmark Construction

Concept removal for real-world and synthetic knowledge. To achieve a diverse source of realworld knowledge, we consider human identities, landmarks, car brands, and fruits as our main categories, with 139 concepts in total. Following Liu et al. (2025a) and Wang et al. (2025), each concept is assigned exclusively to the forget or retain set. We also verified that the evaluated MLLMs have a decent understanding of those concepts before unlearning in Tab. 2.

To compare the unlearning of pre-trained and fine-tuned knowledge in Sec. 4.2, we additionally include two synthetic knowledge subsets: CLEAR and fake landmarks. For CLEAR, we adopt fictitious identities, along with their associated images and biographical information from the CLEAR dataset (Dontsov et al., 2025). For fake landmarks, we generate fictional landmark profiles with corresponding images and construct questions about their attributes, including location and completion date. Each synthetic subset consists of 40 concepts.

![](images/42cc90898cec46d49c738884b149f7235fbe23f542b9046d77f7fe6d926a5caf.jpg)

Figure 6: Forget-test drop versus retain-test drop for every configuration in our search space on LLaVA-1.5- 7B. Without proper tuning, many configurations lie on the y = x line, where retained knowledge is damaged as much as the target is forgotten.
<table><tr><td rowspan="2">Methods</td><td colspan="2">LLaVA-1.5-7B</td><td colspan="2">Qwen3-VL-8B</td><td rowspan="2">Removal type</td></tr><tr><td>VQA drop (%)</td><td>QA drop (%)</td><td>VQA drop (%)</td><td>QA drop (%)</td></tr><tr><td>GA</td><td>34.4</td><td>43.9</td><td>50.3</td><td>49.6</td><td>Whole concept</td></tr><tr><td>GADiff</td><td>92.9</td><td>100.0</td><td>93.3</td><td>86.1</td><td>Whole concept</td></tr><tr><td>SCRUB</td><td>77.3</td><td>15.2</td><td>97.5</td><td>58.5</td><td>Association</td></tr><tr><td>KL Min</td><td>45.0</td><td>71.3</td><td>2.4</td><td>3.1</td><td>Inconsistent</td></tr><tr><td>NPO</td><td>16.4</td><td>34.2</td><td>56.1</td><td>46.2</td><td>Whole concept</td></tr><tr><td>IDK</td><td>50.6</td><td>32.6</td><td>64.6</td><td>38.0</td><td>Association</td></tr><tr><td>LLMU</td><td>32.0</td><td>38.3</td><td>34.1</td><td>32.1</td><td>Whole concept</td></tr><tr><td>RMU</td><td>95.3</td><td>12.8</td><td>96.2</td><td>1.9</td><td>Association</td></tr><tr><td>MANU</td><td>39.6</td><td>24.9</td><td>50.5</td><td>56.8</td><td>Whole concept</td></tr></table>

Table 3: Forget-test performance drop relative to the pre-trained model (% of initial), per modality, averaged over the four real-world subsets at each method’s selected configuration. Association removers suppress VQA while QA largely survives while whole-concept removers suppress both.

Generalizability analysis. Within each concept, examples are split into disjoint training and test sets, so the test sets assess unseen examples of the same concepts. Images are collected from the Internet and manually verified by the authors.<sup>2</sup> We adopt a train-test split of 5:10, which is less extreme than the single-training-data setup in MMUBench (Li et al., 2024a).

Per-method hyperparameter search. For each method, we search for its optimal hyperparameters by scoring each configuration by its average retaintrain minus average forget-train performance and select the configuration with the best score averaged over the four real-world subsets. Selection uses training splits only so that test data never influences it. See Appendix C for the selected hyperparameters and implementation details.

Modality gap analysis. Following Liu et al. (2025a) and Dontsov et al. (2025), we construct questions in two input modalities: visual question answering (VQA) pairs an image with a textual question, while text-only question answering (QA) contains the concept name and no image. Both modalities cover single-choice classification, fillin-the-blank, and generation tasks. We use accuracy for classification and fill-in-the-blank tasks and ROUGE-L for generation. For both image-text and text-only tasks, we paraphrase the question texts while preserving their meaning and groundtruth answers to prevent overfitting to the question formats.

Representative examples from the forget and retain training and test sets of all six subsets are provided in Appendix E.1.

## 5.2 Participating MLLMs

We consider Qwen3-VL-8B-Instruct (Bai et al., 2025) and LLaVA-1.5-7B (Liu et al., 2024) as our base MLLMs. Following prior works on multimodal unlearning (Huo et al., 2025; Dontsov et al., 2025; Zheng et al., 2026), we adopt LoRA (Hu et al., 2022) for both fine-tuning the base models in the synthetic versus pre-trained knowledge experiment and applying the unlearning algorithms.

## 5.3 Machine Unlearning Methods

We evaluate nine representative machine unlearning methods covering a diverse range of unlearning strategies. Four are built on gradient ascent over forget-set responses: GA (Thudi et al., 2022) uses the ascent loss alone; GADiff (Liu et al., 2022) adds gradient descent on the retain set; KL Min (Maini et al., 2024) instead regularizes retain updates with a KL penalty toward the pre-trained model; and LLMU (Yao and Xu, 2024) further pairs forget prompts with unrelated responses under a KL penalty. NPO (Zhang et al., 2024) replaces unbounded ascent with a preference-style loss whose ascent weight shrinks as the forget loss exceeds the pre-trained loss. IDK (Maini et al., 2024) simply relabels forget answers with abstention targets. SCRUB (Kurmanji et al., 2023) is distillationbased, maximizing KL divergence from a frozen teacher on forget examples while minimizing it on retain examples. Finally, two methods operate on internal representations: RMU (Li et al., 2024b) steers forget-set activations at a chosen layer toward a random vector while anchoring retain-set activations, and MANU (Liu et al., 2025b) prunes MLP neurons ranked by their forget-versus-retain importance.

## 5.4 Re-benchmarking Unlearning Algorithms

Table 2 shows results for nine unlearning methods under GEMMU. Our main findings are as follows.

Simple methods win. Recent methods are often reported as state-of-the-art on prior benchmarks. For example, MANU (Liu et al., 2025b) outperforms GADiff on MLLMU-Bench (Liu et al., 2025a). Rankings on GEMMU differ: the two strongest methods are among the oldest and simplest, GADiff and SCRUB.

Association removal vs. concept removal. The two modalities reveal what is unlearned. Following the notion in Huo et al. (2025), if only VQA drops while QA survives, the model has merely severed the association between visual appearance and knowledge; if both drop, the concept itself is removed. Tab. 3 classifies each method by comparing its VQA and QA drops on the forget-test split based on the results in Tab. 2. RMU is the clearest association remover: on Qwen3-VL-8B, VQA classification falls to 2.48 while QA stays at 83.91 (FITB: 1.66 vs. 76.88), so the model no longer recognizes the visual entity but still knows everything about it. GADiff removes whole concepts, at the cost of the largest retain-side drop.

No method solves the task. Even the best entries in Table 2 fall short of ideal unlearning: GADiff sacrifices retain classification (48.49 on LLaVA, 56.7 on Qwen), SCRUB leaves textual forget knowledge largely intact, and the remaining methods either under-forgets or over-destroys. We conclude that unlearning real-world, pre-trained multimodal knowledge remains unsolved.

## 5.5 On Guidelines for Future Multimodal Unlearning Benchmarks

We close by distilling our findings into design guidance, in the hope that good practices are inherited by future benchmarks.

What makes a good design? First, the fundamentals of machine learning evaluation: a proper train–test split, so that forgetting and retention are measured as generalization rather than memorization (Sec. 4.1), and a hyperparameter search per method, without which conclusions about relative effectiveness may reflect the chosen configuration rather than the methods themselves (Sec. 4.3). Second, echoing modality-gap studies in multimodal unlearning (Wang et al., 2025; Huo et al., 2025), benchmarks should construct and report both VQA and QA metrics for every task. Only the contrast between the two reveals whether a method removes the visual–textual association or the concept itself (Sec. 5.4).

What to avoid? Inheriting text-only practices without accounting for what is unique to MLLMs: instance-wise forgetting may be defensible for text, but not for concept-wise forgetting in MLLMs (Fig. 3). Likewise, convenience should not trump rigor. Benchmarks built on synthetic, fine-tuned knowledge are easier to construct but may not reflect realistic unlearning (Sec. 4.2), and applying one shared hyperparameter configuration to every method can distort the comparison (Sec. 4.3).

## 6 Conclusion

In this work, we introduce GEMMU, a benchmark for multimodal machine unlearning that emphasizes generalization-aware evaluation and covers a broader and more diverse range of real-world concepts than prior benchmarks. Experiments on GEMMU reveal substantial generalization gaps overlooked by previous benchmarks, highlight the challenges of removing pre-trained knowledge that is deeply entangled with retained knowledge, and demonstrate how proper hyperparameter tuning can alter the relative rankings of unlearning methods.

As unlearning methods and benchmarks become increasingly sophisticated, we emphasize that fundamental principles of evaluation must not be overlooked to ensure that scientific progress is built on a solid foundation.

## Limitations

Following common practice in existing multimodal unlearning work, we use LoRA (Hu et al., 2022) for both fine-tuning and unlearning. Full-model fine-tuning may exhibit different unlearning behavior from adapter-based fine-tuning. However, we do not extend our experiments to full-model finetuning due to its substantially higher computational cost, and instead focus on settings that can be feasibly evaluated on an NVIDIA H100 GPU.

## References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Alexey Dontsov, Dmitrii Korzh, Alexey Zhavoronkin, Boris Mikheev, Denis Bobkov, Aibek Alanov, Oleg Rogov, Ivan Oseledets, and Elena Tutubalina. 2025. CLEAR: Character unlearning in textual and visual modalities. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 20582–20603, Vienna, Austria. Association for Computational Linguistics.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Jiahao Huo, Yibo Yan, Xu Zheng, Yuanhuiyi Lyu, Xin Zou, Zhihua Wei, and Xuming Hu. 2025. MMUnlearner: Reformulating multimodal machine unlearning in the era of multimodal large language models. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 7190–7206, Vienna, Austria. Association for Computational Linguistics.

Tatsuki Kawakami, Kazuki Egashira, Atsuyuki Miyai, Go Irie, and Kiyoharu Aizawa. 2025. Pulse: Practical evaluation scenarios for large multimodal model unlearning. arXiv preprint arXiv:2507.01271.

Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, and Eleni Triantafillou. 2023. Towards unbounded machine unlearning. Advances in neural information processing systems, 36:1957–1987.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun MA, and Chunyuan Li. 2025. LLaVA-neXT-interleave: Tackling multiimage, video, and 3d in large multimodal models. In The Thirteenth International Conference on Learning Representations.

Jiaqi Li, Qianshan Wei, Chuanyi Zhang, Guilin Qi, Miaozeng Du, Yongrui Chen, Sheng Bi, and Fan Liu.

2024a. Single image unlearning: Efficient machine unlearning in multimodal large language models. In Advances in Neural Information Processing Systems, volume 37, pages 35414–35453. Curran Associates, Inc.

Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D Li, Ann-Kathrin Dombrowski, Shashwat Goel, Long Phan, and 1 others. 2024b. The wmdp benchmark: Measuring and reducing malicious use with unlearning. arXiv preprint arXiv:2403.03218.

Bo Liu, Qiang Liu, and Peter Stone. 2022. Continual learning and private unlearning. In Proceedings of The 1st Conference on Lifelong Learning Agents, volume 199 of Proceedings of Machine Learning Research, pages 243–254. PMLR.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306.

Zheyuan Liu, Guangyao Dou, Mengzhao Jia, Zhaoxuan Tan, Qingkai Zeng, Yongle Yuan, and Meng Jiang. 2025a. Protecting privacy in multimodal large language models with mllmu-bench. In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4105–4135.

Zheyuan Liu, Guangyao Dou, Xiangchi Yuan, Chunhui Zhang, Zhaoxuan Tan, and Meng Jiang. 2025b. Modality-aware neuron pruning for unlearning in multimodal large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5913–5933.

Yingzi Ma, Jiongxiao Wang, Fei Wang, Siyuan Ma, Jiazhao Li, Jinsheng Pan, Xiujun Li, Furong Huang, Lichao Sun, Bo Li, and 1 others. 2025. Benchmarking vision language model unlearning via fictitious facial identity dataset. In International Conference on Learning Representations, volume 2025, pages 87806–87827.

Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary Chase Lipton, and J Zico Kolter. 2024. TOFU: A task of fictitious unlearning for LLMs. In First Conference on Language Modeling.

Anvith Thudi, Gabriel Deza, Varun Chandrasekaran, and Nicolas Papernot. 2022. Unrolling sgd: Understanding factors influencing machine unlearning. In 2022 IEEE 7th European symposium on security and privacy (EuroS&P), pages 303–319. IEEE.

Chengye Wang, Yuyuan Li, XiaoHua Feng, Chaochao Chen, Xiaolin Zheng, and Jianwei Yin. 2025. Umubench: Closing the modality gap in multimodal unlearning evaluation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Zhaopan Xu, Pengfei Zhou, Weidong Tang, Jiaxin Ai, Wangbo Zhao, Kai Wang, Xiaojiang Peng, Wenqi Shao, Hongxun Yao, and Kaipeng Zhang. 2025. Pebench: A fictitious dataset to benchmark machine unlearning for multimodal large language models. arXiv preprint arXiv:2503.12545.

Yuanshun Yao and Xiaojun Xu. 2024. Large language model unlearning. Advances in Neural Information Processing Systems, 37:105425–105475.

Ruiqi Zhang, Licong Lin, Yu Bai, and Song Mei. 2024. Negative preference optimization: From catastrophic collapse to effective unlearning. In First Conference on Language Modeling.

Hao Zheng, Zirui Pang, Ling Li, Zhijie Deng, Yuhan Pu, Zhaowei Zhu, Xiaobo Xia, and Jiaheng Wei. 2026. Offside: Benchmarking unlearning misinformation in multimodal large language models. In Findings of the Association for Computational Linguistics: ACL 2026, pages 12602–12620.

## A Extended Results for Learning Dynamics

Fig. 7 shows the learning dynamics averaged across all eight learning-based unlearning methods. The findings in Sec. 4.2 remain consistent, with pretrained knowledge being forgotten more slowly than fine-tuned knowledge. The gap is smaller in the averaged results because some unlearning methods either fail to induce meaningful forgetting or cause the model to collapse entirely.

## B Proof of Proposition 1

## B.1 Analysis setup

We use differentiable knowledge scores $s _ { \mathrm { t a r g e t } }$ and $s _ { \mathrm { h e l d o u t } }$ for the analysis, such as the average loglikelihood of reference answers. Let Θ be a neighborhood containing $\theta _ { 0 }$ where the assumptions below hold, and let $\| \cdot \| _ { 2 }$ denote the Euclidean norm. Consider any updated model $\theta _ { u } \in \Theta$ satisfying the forgetting criterion $s _ { \mathrm { t a r g e t } } ( \theta _ { u } ) \leq \tau$

We measure retention cost relative to the original model as $s _ { \mathrm { h e l d o u t } } ( \theta _ { 0 } ) - s _ { \mathrm { h e l d o u t } } ( \theta _ { u } )$ . For finetuned knowledge, we show that $\theta _ { u } = \theta _ { 0 }$ satisfies the forgetting criterion with zero retention cost. For pre-trained knowledge, we establish a strictly positive lower bound on the retention cost under the following assumptions.

## B.2 Assumptions

The following conditions are needed for the pretrained case. Here, the target concept is already

present in $\theta _ { 0 }$ , and

$$
\gamma : = s _ { \mathrm { t a r g e t } } ( \theta _ { 0 } ) - \tau > 0
$$

is the score reduction required to reach the forgetting threshold.

## Assumption A1 (Lipschitz target score)

$s _ { \mathrm { t a r g e t } }$ is K-Lipschitz, i.e., there is $K > 0$ such that, for all $\theta , \theta ^ { \prime } \in \Theta$

$$
| s _ { \mathrm { t a r g e t } } ( \theta ) - s _ { \mathrm { t a r g e t } } ( \theta ^ { \prime } ) | \leq K \| \theta - \theta ^ { \prime } \| _ { 2 } .
$$

This condition limits how much the target score can change per unit of parameter displacement. Reducing the score by at least $\gamma$ therefore requires a displacement of at least $\gamma / K$

## Assumption A2 (Limited local improvement)

At $\theta _ { 0 } ,$ , we bound how quickly retained utility can improve under small parameter changes: for some $\varepsilon _ { 0 } \geq 0 ,$

$$
\| \nabla s _ { \mathrm { h e l d o u t } } ( \theta _ { 0 } ) \| _ { 2 } \leq \varepsilon _ { 0 } .
$$

This reflects the assumption that the pre-trained model is near-optimal on the heldout concepts. Even in the most favorable direction, an update δ provides at most $\varepsilon _ { 0 } \lVert \delta \rVert _ { 2 }$ of improvement under the linear approximation.

## Assumption A3 (Quadratic penalty)

There is $\mu > 0$ such that, for every $\theta _ { 0 } + \delta \in \Theta$

$$
\begin{array} { r l } { { s } _ { \mathrm { h e l d o u t } } ( \theta _ { 0 } + \delta ) \le \displaystyle s _ { \mathrm { h e l d o u t } } ( \theta _ { 0 } ) } & { } \\ { { + } \langle \nabla s _ { \mathrm { h e l d o u t } } ( \theta _ { 0 } ) , \delta \rangle - \displaystyle \frac { \mu } { 2 } \| \delta \| _ { 2 } ^ { 2 } . } \end{array}
$$

This condition formalizes the intuition that pretrained knowledge is entangled with other useful knowledge: it places retained utility below its firstorder approximation by at least $\mu \lVert \delta \rVert _ { 2 } ^ { 2 } / 2$ for every update within Θ. The penalty grows quadratically with displacement, whereas the possible first-order improvement in Assumption A2 grows only linearly. Together, they imply a positive retention cost when the displacement is sufficiently large.

To ensure that successful forgetting requires such a displacement, we additionally require

$$
\varepsilon _ { 0 } < \frac { \mu \gamma } { 2 K } .\tag{1}
$$

Equivalently, $\gamma / K > 2 \varepsilon _ { 0 } / \mu \colon$ the minimum displacement needed for forgetting exceeds the distance at which the quadratic penalty outweighs the possible first-order improvement.

![](images/1aba54560fa6bea92a138ebb8befdcc15ccbf525220a8c2f3172fed9fcc82fc7.jpg)  
Figure 7: Forget-set image-textual performance (relative to the model before unlearning) over unlearning epochs, averaged over all 8 learning-based unlearning methods (i.e., excluding the pruning-based MANU), on LLaVA-1.5-7B (left) and Qwen3-VL-8B-Instruct (right). Fine-tuned knowledge (dashed) collapses more quickly than pre-trained knowledge (solid).

## B.3 Proof

The two cases differ in whether $\theta _ { 0 }$ already satisfies the forgetting criterion. In part I, returning to $\theta _ { 0 }$ gives a feasible solution with zero retention cost relative to that same model. In part II, its target score exceeds the threshold by $\gamma > 0$ . Successful forgetting then requires moving away from $\theta _ { 0 }$ , and the assumptions translate that required displacement into a positive retention cost.

Fine-tuned knowledge (part I). Let $\theta _ { \mathrm { f t } } = \theta _ { 0 } .$ + $\Delta _ { \mathrm { f t } }$ , where

$$
s _ { \mathrm { t a r g e t } } ( \theta _ { 0 } ) \leq \tau < s _ { \mathrm { t a r g e t } } ( \theta _ { \mathrm { f t } } ) .
$$

Reversing the fine-tuning update gives $\theta _ { u } = \theta _ { \mathrm { f t } }$ $\Delta _ { \mathrm { { f t } } } = \theta _ { 0 }$ . This model satisfies the forgetting criterion and has zero retention cost:

$$
s _ { \mathrm { h e l d o u t } } ( \theta _ { 0 } ) - s _ { \mathrm { h e l d o u t } } ( \theta _ { u } ) = 0 .
$$

Thus, a zero-cost solution exists without requiring Assumptions A1–A3.

Pre-trained knowledge (part II). Let $\theta _ { u } \ =$ $\theta _ { 0 } + \delta \in \Theta$ satisfy $s _ { \mathrm { t a r g e t } } ( \theta _ { u } ) \ \leq \ \tau$ . We prove the positive retention-cost bound in three steps.

Step 1: Required displacement. The forgetting criterion requires a score reduction of at least γ. By Assumption A1,

$$
\gamma \leq s _ { \mathrm { t a r g e t } } ( \theta _ { 0 } ) - s _ { \mathrm { t a r g e t } } ( \theta _ { u } ) \leq K \| \delta \| _ { 2 } .
$$

Hence,

$$
\| \delta \| _ { 2 } \geq \frac { \gamma } { K } .\tag{2}
$$

Step 2: Retention cost. Rearranging Assumption A3 and applying Cauchy–Schwarz with Assumption A2 gives

$$
\begin{array} { l } { { \displaystyle s _ { \mathrm { h e l d o u t } } ( \theta _ { 0 } ) - s _ { \mathrm { h e l d o u t } } ( \theta _ { u } ) } } \\ { { \displaystyle \quad \geq - \langle \nabla s _ { \mathrm { h e l d o u t } } ( \theta _ { 0 } ) , \delta \rangle + \frac { \mu } { 2 } \| \delta \| _ { 2 } ^ { 2 } } } \\ { { \displaystyle \geq \frac { \mu } { 2 } \| \delta \| _ { 2 } ^ { 2 } - \varepsilon _ { 0 } \| \delta \| _ { 2 } } . } \end{array}\tag{3}
$$

The first term is the quadratic penalty; the second accounts for the largest possible first-order improvement.

Step 3: Positive lower bound. Factoring the right-hand side of (3) gives

$$
s _ { \mathrm { h e l d o u t } } ( \theta _ { 0 } ) - s _ { \mathrm { h e l d o u t } } ( \theta _ { u } ) \geq \| \delta \| _ { 2 } \left( \frac { \mu } { 2 } \| \delta \| _ { 2 } - \varepsilon _ { 0 } \right) .
$$

By (2), the second factor satisfies

$$
\frac { \mu } { 2 } \| \delta \| _ { 2 } - \varepsilon _ { 0 } \ge \frac { \mu \gamma } { 2 K } - \varepsilon _ { 0 } > 0 ,
$$

where the strict inequality follows from condition (1). We now replace the two occurrences of $\lVert \delta \rVert _ { 2 }$ in the factored expression in turn:

$$
\begin{array} { r l } & { s _ { \mathrm { h e l d o u t } } ( \theta _ { 0 } ) - s _ { \mathrm { h e l d o u t } } ( \theta _ { u } ) } \\ & { ~ \geq \| \delta \| _ { 2 } \left( \frac { \mu \gamma } { 2 K } - \varepsilon _ { 0 } \right) } \\ & { ~ \geq \frac { \gamma } { K } \left( \frac { \mu \gamma } { 2 K } - \varepsilon _ { 0 } \right) } \\ & { ~ = \frac { \mu \gamma ^ { 2 } } { 2 K ^ { 2 } } - \frac { \varepsilon _ { 0 } \gamma } { K } > 0 . } \end{array}
$$

The first inequality replaces the norm inside the parentheses by its lower bound $\gamma / K$ , while keeping the positive outer factor $\lVert \delta \rVert _ { 2 }$ unchanged. The second replaces the outer factor by $\gamma / K ;$ ; the inequality direction is preserved because the remaining factor $\mu \gamma / ( 2 K ) - \varepsilon _ { 0 }$ is strictly positive. This proves part II for every successful unlearning solution within Θ under the stated conditions.

Practical Interpretation of the Assumptions. Assumptions A1–A3 provide sufficient conditions for a local forgetting–retention trade-off. A1 bounds the sensitivity of the target knowledge score, A2 constrains the gradient of retained utility, and A3 imposes a quadratic retention penalty under parameter changes. We acknowledge that A3 is a strong assumption, and therefore the derived bound should be interpreted as a conditional theoretical result rather than a universal guarantee of retention degradation.

## C Hyperparameters and Implementation Details

## C.1 Hyperparameter and checkpoint selection

We base our default hyperparameters on MMUnlearner (Huo et al., 2025) and tune each method using only training results. We favor settings that keep retain-set performance high and forget-set performance low, using

$$
\begin{array} { c } { { \mathrm { S e l e c t i o n ~ s c o r e = m e a n ~ r e t a i n \mathrm { - } t r a i n ~ s c o r e } } } \\ { { - \mathrm { m e a n ~ f o r g e t \mathrm { - } t r a i n ~ s c o r e } . } } \end{array}
$$

Higher is better. For each subset, the means give equal weight to classification accuracy, fill-in-theblank accuracy, and ROUGE-L, and to image-text and text-only inputs. Accuracies are expressed as percentages and ROUGE-L is multiplied by 100, putting all three metrics on a 0–100 scale.

Hyperparameters are selected on training splits only to prevent the obvious test-set leakage. Due to the low number (five) of training images per concept, we do not further split the training set into training and validations sets. <sup>3</sup> For each candidate configuration, we score every checkpoint (epoch) by the difference between its average retain-train performance and its average forget-train performance, where both averages span the six task– modality metrics (three tasks × VQA/QA). For each of the four real-world subsets (human identities, car brands, fruits, and landmarks), we take the best score across epochs, and the configuration with the highest average of these four scores is selected. Under the selected configuration, the reported checkpoint is then chosen per subset as the epoch with the highest score. The synthetic subsets (CLEAR and fake landmarks) reuse the selected configuration and choose their own epochs, but do not influence the configuration choice.

![](images/9444fdd174e75015c20815ba6b578f6ae12b3013d209c908022315c2d23db3b1.jpg)  
Figure 8: Hyperparameter sensitivity within a single method. IDK on the Fruits subset (Qwen3-VL-8B, fillin-the-blank, VQA) across its learning-rate grid. Three of the four learning rates leave forget-set accuracy essentially unchanged after eight epochs.

For both LLaVA-1.5-7B and Qwen3-VL-8B-Instruct experiments, we use a batch size of 2, random seed 42, and 8 training epochs. LoRA uses rank 16, scaling parameter 16, and dropout 0.05.

## D More Examples on Hyperparameter Sensitivity

Figure 8 shows a further example of the sensitivity, this time within a single method. We run IDK on the Fruits subset of Qwen3-VL-8B with four learning rates spanning its search grid. At lr $\in \{ 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } \}$ , forget-set accuracy remains essentially at the pre-trained level after eight epochs, i.e., no unlearning occurs. Only at $\mathrm { l r } = 2 \times 1 0 ^ { - 4 }$ does the method succeed.

## E Dataset Examples

## E.1 Representative Examples

Figures 9–14 show representative examples from all six subsets. Each figure retains the forget/retain × train/test layout, and each panel separates imagetext questions (VQA) from text-only questions (QA). Only the VQA input includes the displayed image; QA names the concept in its text and receives no image. VQA and QA examples are presented to the model as separate inputs.

The examples probe associated knowledge, including biographical details, literary works, landmark completion dates, brand history, and fruit attributes. Each panel presents one VQA and one QA example; across the four panels of each subset, we illustrate single-choice classification (CLS), fillin-the-blank (FITB), and generation (GEN). Both modalities support all three task types; the figures show selected combinations. Questions, options, and reference answers are taken from the corresponding dataset rows.

Within each forget or retain row, train and test use different images of the same concept and probe the same knowledge with rephrased questions. This applies to both VQA and QA. For CLS, test distractors and option order can also change, so we show both the correct option letter and its answer text.

VQA

## Forget – Train

![](images/dab9997a94c3a69d9b54ba39a858de9969591689eaa4170d71f352b2da045ec0.jpg)

CLS

Question: What is this person’s nationality? A. Colombia B. Greece C. United States D. Germany Correct answer: C. United States

QA

Question: When and where was Brad Pitt born? Reference answer: Brad Pitt was born in Shawnee, Oklahoma, United States on December 18, 1963.

GEN

VQA

## Retain – Train

![](images/cb034d8534b71df426b5845e88147d0515e287c9cddacf9d8ac3d407bde2e73f.jpg)

GEN

Question: What is this person best known for? Reference answer: This person is best known for her self-deprecating stand-up comedy.

QA

FITB

Prompt: Amy Schumer was born in New York City, New York, United States on June 1, \_\_\_.   
Answer: 1981

VQA

## Forget – Test

![](images/32530685024c514feeb4e5e9a9bb575b4675b0f4a867e5a0ca788670a9062fce.jpg)

CLS

Question: What is the nationality of the person shown?   
A. Brazil B. United States   
C. Ireland D. Serbia   
Correct answer: B. United States

QA

Question: What are the birth date and birthplace of Brad Pitt?

GEN

Reference answer: Brad Pitt was born in Shawnee, Oklahoma, United States on December 18, 1963.

## Retain – Test

VQA

![](images/daa7b236883f0f9f83249b5abb8f08b22f5144f573801e3362e20326db420ab7.jpg)

GEN

Question: Why is this person famous? Reference answer: This person is best known for her self-deprecating stand-up comedy.

QA   
Prompt: The birth year of Amy Schumer is \_\_\_. Answer: 1981

FITB

Figure 9: Human identities. Paired train/test examples illustrate rephrasing for both VQA and text-only QA. CLS: single-choice classification; FITB: fill-in-the-blank; GEN: generation. Answers are dataset references.

VQA

## Forget – Train

![](images/115439ce955533969da7cf9ecc682599bede5274919b02b561a00ac133bbbf87.jpg)

FITB

Prompt: This person is best known for her Western works, including ’\_\_\_’ and ’Shea’s Grudge’, and for winning the Red Saddle Award.

Answer: Bingham’s Standoff

QA

Question: When and where was Adrianne Lebeau born? Reference answer: Adrianne Lebeau was born in Brussels, Belgium on July 16, 1965.

GEN

## Retain – Train

VQA

CLS

![](images/c4384f3fa238026dd5fd883704b060838611a5b7f9717e3fae5620838be17d11.jpg)

Question: What is this person’s occupation?   
A. fantasy author   
B. humour author   
C. true crime author   
D. werewolf fiction author   
Correct answer: C. true crime author

QA

FITB

Prompt: Adetoun Davis was born in \_\_\_, Nigeria on January 16, 1941.   
Answer: Lagos

## Forget – Test

VQA

FITB

![](images/ea494ec7e7989c6b47bc950e3d98bd26c59a3a51f59a66ac9e57627ff2f6a9ea.jpg)

Prompt: This person is widely known for her Western works, including ’\_\_\_’ and ’Shea’s Grudge’, and for winning the Red Saddle Award.   
Answer: Bingham’s Standoff

QA

GEN

Question: What are the birth date and birthplace of Adrianne Lebeau?

Reference answer: Adrianne Lebeau was born in Brussels, Belgium on July 16, 1965.

## Retain – Test

VQA

CLS

![](images/05921db25f3263f682e43468a83499f361f5e9867d8e2c04e82976b556ae2c8c.jpg)

Question: State this person’s profession.

A. true crime author

B. Western author

C. humour author

D. Portuguese literature author

Correct answer: A. true crime author

QA

FITB

Prompt: The birthplace of Adetoun Davis is \_\_\_, Nigeria. Answer: Lagos

Figure 10: CLEAR. Paired train/test examples illustrate rephrasing for both VQA and text-only QA. CLS: singlechoice classification; FITB: fill-in-the-blank; GEN: generation. Answers are dataset references.

<table><tr><td>Forget – Train VQA CLS</td><td>Forget – Test VQA <img src="images/6919b66fe12301edc012d9ce3f57d9b43cf1e75d8888a0f017f94aefa5a5e0ad.jpg"/></td></tr><tr><td><img src="images/8771736aad2b78ec02de06a3b5cec4bf35a5a9234ae6ee98138c8c9e559f82aa.jpg"/> Question: In what year was this landmark completed? A.1889 B.1976 C.1941 D.966</td><td>Question: In which year was construction of this landmark completed?</td></tr><tr><td>Correct answer: A. 1889 QA FITB</td><td>A.1905 B.1645 C.1889 D.1932 Correct answer: C. 1889 QA FITB</td></tr><tr><td>Prompt: The Eiffel Tower is located in the city of Answer: Paris</td><td>Prompt: The Eiffel Tower is in the city of Answer: Paris</td></tr><tr><td>Retain – Train VQA GEN <img src="images/5c7bb687ea83ce54e5d232f3195cb95996137d5490db830af59963b3ff4075ae.jpg"/></td><td>Retain – Test VQA GEN <img src="images/04e2d76a0d0c8453ee65e96212c1d3b3d6d51575d1ea9522cb6b236f227b1ce5.jpg"/></td></tr><tr><td>Question: In what year was this landmark completed? Reference answer: This landmark was completed in 1905.</td><td>Question: In what year did this landmark reach completion? Reference answer: This landmark was completed in 1905.</td></tr><tr><td>QA CLS Question: What country is Berlin Cathedral located in? A. Italy B. Canada C. France D. Germany Correct answer: D. Germany</td><td>QA CLS Question: In which country can you find Berlin Cathedral? A. Belgium B. United States C. Spain D. Germany Correct answer: D. Germany</td></tr><tr><td>Forget – Train VQA FITB <img src="images/d46aeb7a9e91b351b16c3293c636df4088b78b3d922a90a2bad8065e4489d4dd.jpg"/> Prompt: This landmark was completed in Answer: 2025</td><td>Forget – Test VQA FITB <img src="images/967b385daaca709665010c9ca80f24f1a1e2bdd9beb9a4bbb570bce21eb0a931.jpg"/> Prompt: This landmark reached completion in the year Answer: 2025</td></tr><tr><td>QA CLS Question: What city is the Opal Tower located in? A. Quito B. Sydney C. Vienna D. Accra Correct answer: B. Sydney</td><td>QA CLS Question: Which city is the Opal Tower located in? A. Sydney B. Mendoza C. Kyoto D. Tashkent Correct answer: A. Sydney</td></tr><tr><td>Retain – Train VQA CLS <img src="images/1f6b283c228b529526def074d32336534e51edf748fe34ce9884d4630aafbfa3.jpg"/></td><td>Retain – Test VQA CLS <img src="images/0e1ebbd3b804aeb9cf1ce406273371f51b7d9428b198d39321f464540d71e5a8.jpg"/></td></tr><tr><td>Question: What country is this landmark located in? A. Greece B. Italy C. Austria D. Denmark Correct answer: C. Austria</td><td>Question: Which country does this landmark stand in? A. Austria B. Czech Republic C. Iceland D. United Kingdom Correct answer: A. Austria</td></tr><tr><td>QA GEN Question: In what year or century was the Scarlet Whispers Abbey completed? Reference answer: The Scarlet Whispers Abbey was completed in the 11th century.</td><td>QA GEN Question: What year or century saw the completion of the Scarlet Whispers Abbey? Reference answer: The Scarlet Whispers Abbey was completed in the 11th century.</td></tr><tr><td>Forget – Train VQA CLS</td><td>Forget – Test VQA <img src="images/9cf99aabc1a38d289e4cd8e9ea40de525f30ac7e85ecd801c40d27cf8871c7ec.jpg"/></td></tr><tr><td>Question: What year was this brand founded? A.1916 B.1904 C.1909 D.1970 Correct answer: A. 1916</td><td>Question: When was this brand founded? A.1937 B.1902 C.1916 D.1926 Correct answer: C. 1916</td></tr><tr><td>QA GEN Question: What country is BMW headquartered in? Reference answer: BMW is headquartered in Germany.</td><td>QA GEN Question: Which country is BMW based in? Reference answer: BMW is headquartered in Germany.</td></tr><tr><td>Retain – Train VQA GEN</td><td>Retain – Test VQA GEN</td></tr><tr><td><img src="images/20fd81ab83230fedeff9a163f06daa92543ed1cdba851acaf1658a271381546b.jpg"/> Question: What is this brand's parent company? Reference answer: The parent company of this car brand is Stellantis.</td><td><img src="images/5fb9610d13b0b4465c67677713d50b7aab0ad725937a46407813dc045a8f84ed.jpg"/> Question: Name the company that owns this car brand. Reference answer: The parent company of this car brand is Stellantis.</td></tr><tr><td></td><td></td></tr><tr><td>QA FITB Prompt: Opel was founded in the year Answer: 1862</td><td>QA FITB Prompt: The year in which Opel was founded is Answer: 1862</td></tr></table>

Figure 11: Landmarks. Paired train/test examples illustrate rephrasing for both VQA and text-only QA. CLS: single-choice classification; FITB: fill-in-the-blank; GEN: generation. Answers are dataset references.

Figure 12: Synthetic landmarks. Paired train/test examples illustrate rephrasing for both VQA and text-only QA. CLS: single-choice classification; FITB: fill-in-the-blank; GEN: generation. Answers are dataset references.

Figure 13: Car brands. Paired train/test examples illustrate rephrasing for both VQA and text-only QA. CLS: single-choice classification; FITB: fill-in-the-blank; GEN: generation. Answers are dataset references.

VQA

## Forget – Train

FITB

![](images/aa5f50c3ef57f82f1c6d106abaf3a4849c6e705b80393e23abcc8717ac983420.jpg)

Prompt: This fruit originated in \_\_\_.

Answer: Central Asia

QA

Question: What climate zone is the apple grown in?

CLS

A. subtropical B. arid

C. tropical D. temperate

Correct answer: D. temperate

## Retain – Train

VQA

GEN

![](images/d46fc0ba2153487bf802e07246cc54125eba666a4c8f3b6004dc12b1dc1ee823.jpg)

Question: When is this fruit in season?

Reference answer: This fruit is in season during autumn (October to November).

## QA

Prompt: The persimmon grows on a \_\_\_. Answer: tree

FITB

VQA

## Forget – Test

FITB

![](images/b41d930953de72e9c76faafcfc083e0a54140e064bf20a662daf32762c032447.jpg)

Prompt: This fruit first came from \_\_\_.

Answer: Central Asia

## QA

Question: In which climate zone is the apple cultivated?

CLS

A. tropical B. temperate

C. subtropical D. mediterranean

Correct answer: B. temperate

VQA

## Retain – Test

GEN

![](images/eddc08b3bf3e8db240782742b5fda5991e5fefaa753c76cdd5b13b8d396dcb6f.jpg)

Question: At what time of year is this fruit in season? Reference answer: This fruit is in season during autumn (October to November).

QA

FITB

Prompt: The persimmon is borne on a \_\_\_.

Answer: tree

Figure 14: Fruits. Paired train/test examples illustrate rephrasing for both VQA and text-only QA. CLS: singlechoice classification; FITB: fill-in-the-blank; GEN: generation. Answers are dataset references.

<table><tr><td>Method</td><td>Learning rate search space</td><td>Method-specific settings</td></tr><tr><td>GA GADiff</td><td> $\{ 1 0 ^ { - 6 } , 2 \times 1 0 ^ { - 6 } , 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } \}$   $\{ 1 0 ^ { - 6 } , 2 \times 1 0 ^ { - 6 } , 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } \}$ </td><td></td></tr><tr><td>KL Min</td><td> $\{ 2 \times 1 0 ^ { - 6 } , 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 2 \times 1 0 ^ { - 5 } \}$ </td><td></td></tr><tr><td>IDK</td><td> $\{ 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } \}$ </td><td></td></tr><tr><td>LLMU NPO</td><td> $\{ 1 0 ^ { - 5 } , 2 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } \}$ </td><td> $\beta \in \{ \mathbf { 0 . 0 2 5 , 0 . 1 , 0 . 4 } \}$ </td></tr><tr><td>SCRUB</td><td> $\{ 1 0 ^ { - 6 } , 2 \times 1 0 ^ { - 6 } , 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } \}$   $\{ 1 0 ^ { - 5 } , 2 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } \}$ </td><td>Forget-KL weight  $\beta \in \{ 0 . 1 , 1 . 0 , 2 . 0 \}$ </td></tr><tr><td>RMU</td><td> $\{ 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 2 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } \}$ </td><td>Steering coefficient  $c \in \{ 2 0 , 1 0 0 , 3 0 0 \}$  layer = 7, forget-loss weight = 1.0</td></tr></table>

Table 4: Hyperparameter search space and selected values (bold) for Qwen3-VL-8B-Instruct. Learning rates are swept with method-specific settings fixed at their selected values, and method-specific settings are swept at a learning rate of $1 0 ^ { - 5 }$

<table><tr><td>Method</td><td>Learning rate search space</td><td>Method-specific settings</td></tr><tr><td>GA</td><td> $\{ \mathbf { 1 0 ^ { - 6 } } , 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } \}$ </td><td></td></tr><tr><td>GADiff</td><td> $\left\{ 1 0 ^ { - 6 } , 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } \right\}$ </td><td></td></tr><tr><td>KL Min</td><td> $\{ 1 0 ^ { - 6 } , 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } \}$ </td><td></td></tr><tr><td>IDK</td><td> $\{ 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } \}$ </td><td></td></tr><tr><td>LLMU</td><td> $\{ 1 0 ^ { - 5 } , 2 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } \}$ </td><td></td></tr><tr><td>NPO</td><td> $\{ 1 0 ^ { - 6 } , 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } \}$ </td><td> $\beta \in \{ 0 . 0 2 5 , 0 . 1 , \mathbf { 0 } . 4 \}$ </td></tr><tr><td>SCRUB</td><td> $\{ 1 0 ^ { - 5 } , 2 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } \}$ </td><td>Forget-KL weight  $\beta \in \{ 0 . 1 , 1 . 0 , 2 . 0 \}$ </td></tr><tr><td>RMU</td><td> $\{ 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 2 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 } \}$ </td><td>Steering coefficient c ∈ {20, 100, 300}, layer = 7, forget-loss weight = 1.0</td></tr><tr><td>MANU</td><td></td><td> $\mathrm { P r u n e ~ r a t i o } \in \{ 0 . 5 , 1 , 2 , 5 \} \%$ </td></tr></table>

Table 5: Hyperparameter search space and selected values (bold) for LLaVA-1.5-7B. Learning rates are swept with method-specific settings fixed at their selected values, and method-specific settings are swept at a learning rate of $1 0 ^ { - 5 }$