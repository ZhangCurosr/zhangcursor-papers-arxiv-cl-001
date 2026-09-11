# The Illusion of Balanced Multimodal Sentiment Analysis: Beyond the Limits of Optimization-Based Methods

Ioanna Kaffeza <sup>ID</sup> <sup>1,†,∗∗</sup>, Efthymios Georgiou <sup>ID</sup> <sup>2,†</sup>, Alexandros Potamianos <sup>ID</sup> <sup>3,4,5</sup>

<sup>1</sup> PERSEE Center, Mines Paris-PSL University, Sophia-Antipolis, France

<sup>2</sup> University of Bern, Bern, Switzerland

<sup>3</sup> National Technical University of Athens, Athens, Greece

<sup>4</sup> Archimedes AI, Athens, Greece

<sup>5</sup> Synaptic Bloom PBC, Santa Monica, CA, USA

ioanna.kaffeza@minesparis.psl.eu, efthymios.georgiou@unibe.ch, potam@central.ntua.gr

## Abstract

Multimodal Sentiment Analysis (MSA) remains constrained by modality imbalance, yet the field continues to rely on optimization-based balancing methods that promise more than they deliver. We provide three contributions: 1) a unified evaluation framework testing gradient and loss-based balancing strategies under controlled settings; 2) a theoretical diagnosis explaining why these methods fail, as they conflate fitting speed with discriminative contribution; and 3) a research agenda toward held-out discriminative modality valuation. Experiments on CMU-MOSI and CMU-MOSEI reveal three shortcomings: no strategy reliably outperforms Late Concatenation; performance is sensitive to hyperparameters; and even ratio calibration fails to yield consistent gains. The core issue is fundamental: loss is not utility, and gradients are not importance. Modality imbalance remains unresolved, motivating utility estimation from held-out performance.

Index Terms: multimodal learning, modality imbalance, optimization dynamics, discriminative evaluation

## 1. Introduction

Multimodal learning promises a deeper understanding of human affect by combining language, vision, and audio within a single model. These modalities offer complementary signals that should improve sentiment analysis [1]. Yet empirical findings show that multimodal systems often underperform compared to their unimodal counterparts [2]. This counterintuitive mismatch demands a closer examination of how multimodal models are trained.

A key challenge is modality imbalance. Each modality exhibits different representational geometry, temporal resolution, information density, and learning dynamics [3, 4]. As a result, one modality tends to dominate optimization, rapidly shaping the shared representation space while the others are undertrained and underrepresented. Joint training under a single objective rarely produces balanced fusion [2, 5, 6]. Instead, models overfit to the strongest modality and generalize poorly despite having access to multiple sources of information.

As a remedy, the field has invested in optimization-based interventions. Prior work [2, 5, 7, 8] attempts to adjust gradients to compensate for heterogeneous learning rates. Other approaches [9, 10] reweight losses in an effort to recalibrate each modality’s influence during training. Although these methods differ in implementation, they share a common premise: the assertion that manipulating the optimization process can restore balance among modalities and thereby unlock the full potential of multimodal learning.

This work provides a unified comparative analysis of optimization-based strategies for mitigating modality imbalance in Multimodal Sentiment Analysis. We offer three contributions:

1. A unified evaluation framework that tests gradient-based methods (OGM-GE [7], AGM [8]) alongside loss-based approaches (PMR [9], ReconBoost [10]) under controlled conditions on CMU-MOSI [11] and CMU-MOSEI [12], isolating how each method behaves across different dominance scenarios.

2. A theoretical diagnosis explaining why these methods fail to consistently outperform baselines. Specifically, optimization-based methods confuse how quickly a modality fits training data with how much it actually contributes to correct predictions; a category error that the field has encountered before. In the 1990’s, the audio-visual speech recognition community demonstrated that generative likelihood ratios cannot reliably estimate modality utility, leading to a decisive shift toward discriminative training criteria. The same fundamental misalignment now reappears in modern optimization-based balancing methods.

3. A research agenda toward held-out discriminative modality valuation, supported by controlled diagnostic experiments.

We further study how architectural and training factors, including the choice of optimizer, batch size, and the duration of external modulation applied during backpropagation, influence the stability and effectiveness of these techniques. For methods that depend on ratio based estimates of modality contribution [7, 8, 9], we introduce a dedicated development set to supply a more stable and unbiased calibration signal, allowing these approaches to be tested under conditions most favorable to their design. Taken together, these components allow us to address a central question in multimodal learning: how effectively can optimization-based reweighting address modality imbalance? Our answer is sobering: not effectively, because these methods measure the wrong signal.

## 2. Related Work and Background

## 2.1. The Landscape of Multimodal Sentiment Analysis

Multimodal Sentiment Analysis (MSA) aims to predict emotional and opinion-related states by jointly modeling language, visual, and acoustic cues. Early MSA approaches relied on simple fusion strategies, including early fusion [14] and late fusion [15]. Subsequent architectures introduced richer crossmodal reasoning through attention [16] and large-scale pretraining, such as MulT [17] or self-supervised variants like Self-MM [18].

![](images/ad7aa3e71408a06610c4ed708095f89104721d8da7d63c65e1f23054bfb844ba.jpg)  
Figure 1: The Failure of Joint Training. Training, validation, and unimodal losses for a representative Late Concatenation run on CMU-MOSI. The text modality rapidly achieves the lowest loss and steers the entire optimization, while audio and visual losses remain largely flat. The multimodal curve mirrors the text-only curve, showing that joint training collapses to the dominant modality. Adaptedfrom [13].

In contrast, we adopt a simpler LSTM-based [19] latefusion architecture to isolate unimodal optimization dynamics without cross-modal overhead. Even with increasingly sophisticated fusion designs and well-established benchmarks, many models appear to operate under the assumption that all modalities contribute equally during training. This assumption routinely fails [2, 5, 6], exposing a persistent gap between multimodal model design and actual learning behavior.

## 2.2. Why Joint Training Favors Certain Modalities

Multimodal models rarely learn all modalities at the same pace [2]. Differences in representational geometry, temporal resolution, noise, information density and other factors [3, 4] mean that some modalities fit quickly and risk overfitting, while others generalize better but learn more slowly. This mismatch pushes the optimization process to promote the fast-fitting modality and underpin the rest (Figure 1). The Greedy Learner Hypothesis [5] formalizes this behavior: multimodal networks tend to favor modalities that yield rapid loss reductions. At the same time, modalities compete for representation capacity [6]. The dominant modality drives the entire learning process, leaving the others with minimal impact.

Formally, let $\mathcal { D } _ { \operatorname { t r a i n } } = \{ ( x _ { i } , y _ { i } ) \} _ { i = } ^ { N }$ be a training set with M modalities per input. Each modality k is processed by a feature extractor $F _ { k } ( \boldsymbol { \theta } _ { k } )$ , producing features $\hat { F _ { k } } ( \theta _ { k } ; m _ { i } ^ { k } )$ that are concatenated into a joint representation

$$
\Phi ^ { M } ( x _ { i } ) = [ F _ { 1 } ( \cdot ) : \cdot \cdot \cdot : F _ { M } ( \cdot ) ] .\tag{1}
$$

which is then passed to a classifier. During backpropagation, parameters for each modality are updated as

$$
\theta _ { k } ^ { t + 1 } = \theta _ { k } ^ { t } - \eta \nabla _ { \theta _ { k } ^ { t } } L ( \Phi ^ { M } ( x ) , y ) .\tag{2}
$$

with learning rate η. Although the encoders are separate, late concatenation forces all modalities to share the same loss signal, meaning that the dominant features in $\Phi ^ { M }$ steer the overall gradient. By the chain rule, the gradient for modality k decomposes as

$$
\nabla _ { \theta _ { k } } L = { \frac { \partial L } { \partial \Phi ^ { M } } } \cdot { \frac { \partial \Phi ^ { M } } { \partial \Phi ^ { k } } } \cdot { \frac { \partial \Phi ^ { k } } { \partial \theta _ { k } } } .\tag{3}
$$

where $\Phi ^ { k }$ is the representation from modality k. When the loss gradient aligns more strongly with a particular modality $k ^ { * }$ , we obtain

$$
\| \nabla _ { { \boldsymbol { \theta } } _ { k ^ { * } } } L \| \gg \| \nabla _ { { \boldsymbol { \theta } } _ { j } } L \| \qquad ( j \neq k ^ { * } ) .\tag{4}
$$

In practice, this means that the optimization process allocates most of its learning capacity to the dominant modality, while providing only weak updates to the others. This is a structural consequence of joint training under a shared objective, revealing a core limitation of fusion strategies. Suboptimal use of multimodal features creates a serious obstacle in tasks where these modalities carry essential complementary information [5]. Zhang et al. [20] theoretically show that joint training can induce a transient unimodal phase that results in persistent modality bias, which gradient modulation alone cannot fully correct. As a consequence, the depth and diversity of learned representations are limited [6]. This imbalance weakens generalization and robustness [5, 21]. In real-world scenarios, leaning too heavily on the dominant modality can make models sensitive to noise or missing information in that modality at test time.

## 2.3. Toward Healing Modality Imbalance

To counter modality imbalance, some methods select only the most informative modalities during training [22, 23, 24]. A second line of work modifies or reweights gradients [2, 5, 7, 8, 25, 26, 27] to steer the optimization process toward a more balanced contribution across modalities. A third group adjusts the loss function [9, 10, 21] to strengthen weaker modalities or temper the influence of dominant ones. In this work, we focus on optimization-based balancing during joint training for Multimodal Sentiment Analysis under a controlled late-fusion setting, rather than explicit modality selection or higher-capacity fusion architectures.

Gradient based methods extend traditional optimizers such as Adam [28] and Stochastic Gradient Descent (SGD) by assigning adaptive weights to modality specific gradients. OGM-GE [7] suppresses gradients from the dominant modality, while AGM [8] introduces competition free states and real time modulation to reduce interference between modalities.

Loss-based approaches such as PMR [9] and ReconBoost [10] modify the training objective to amplify weaker modalities. PMR employs prototype-based and entropy regularization, while ReconBoost alternates modality-specific updates with reconcilement terms to rebalance their contributions.

Together, these methods attempt to prevent multimodal models from collapsing onto the dominant modality and to ensure that all streams contribute meaningfully. In this work, we evaluate how effectively these strategies address imbalance under a range of controlled conditions.

## 2.4. Modality Likelihood Ratios: A Historical Perspective

The use of likelihood ratios for modality weighting originates in early audio-visual speech recognition. Multi-stream HMM systems showed that generative likelihoods provide unstable estimates of modality reliability, especially under changing noise conditions, and must therefore be combined with confidence measures or discriminative criteria [29, 30, 31, 32]. These early approaches are particularly relevant as they frame multimodal fusion as a weighting problem, anticipating the optimizationbased balancing strategies studied today. This line of work also emphasized that modality reliability is inherently sampledependent rather than a fixed global quantity.

Table 1: Comparison of model performance on CMU-MOSI and CMU-MOSEI across three modality combinations. Results are reported as average accuracy (%) overfive runsfor the Audio-Video (A-V), Text-Video (T-V), and Audio-Text-Video (A-T-V) setups. Late Concatenation serves as the primary baseline, and each method’s absolute change relative to this baseline (∆) is shownfor the corresponding modality setup. Entries marked “–” indicate that the method is not applicable to the three-modality (A-T-V) configuration. Across all settings, ∆ values remain small indicating no consistent improvement over the baseline and suggesting that apparent gains mayfall within seed variance. Results reproducedfrom [13].
<table><tr><td>Method</td><td colspan="4">A-V</td><td colspan="4">T-V</td><td colspan="4">A-T-V</td></tr><tr><td></td><td>MOSI</td><td>∆</td><td>MOSEI</td><td>∆</td><td>MOSI</td><td>∆</td><td>MOSEI</td><td>∆</td><td>MOSI</td><td>∆</td><td>MOSEI</td><td>∆</td></tr><tr><td>Ensemble</td><td>47.96</td><td></td><td>32.63</td><td></td><td>73.35</td><td></td><td>43.94</td><td></td><td>72.62</td><td></td><td>40.40</td><td></td></tr><tr><td>Uni-Pre Finetuned</td><td>51.46</td><td></td><td>32.55</td><td></td><td>75.72</td><td></td><td>45.22</td><td></td><td>75.65</td><td></td><td>44.65</td><td></td></tr><tr><td>Late Concatenation</td><td>54.93</td><td></td><td>32.55</td><td></td><td>74.35</td><td></td><td>43.99</td><td></td><td>74.87</td><td></td><td>44.46</td><td></td></tr><tr><td>OGM [7]</td><td>53.30</td><td>-1.63</td><td>32.67</td><td>+0.12</td><td>75.04</td><td>+0.69</td><td>44.15</td><td>+0.16</td><td></td><td></td><td></td><td></td></tr><tr><td>OGM-GE [7]</td><td>52.48</td><td>-2.45</td><td>32.40</td><td>-0.15</td><td>73.50</td><td>-0.85</td><td>43.58</td><td>-0.41</td><td></td><td></td><td></td><td></td></tr><tr><td>ACC</td><td>52.39</td><td>-2.54</td><td>32.56</td><td>+0.01</td><td>74.67</td><td>+0.32</td><td>44.12</td><td>+0.13</td><td></td><td></td><td></td><td></td></tr><tr><td>AGM [8]</td><td>53.73</td><td>-1.20</td><td>32.71</td><td>+0.16</td><td>74.61</td><td>+0.26</td><td>44.15</td><td>+0.16</td><td></td><td></td><td></td><td></td></tr><tr><td>PMR [9]</td><td>51.52</td><td>-3.41</td><td>32.44</td><td>-0.11</td><td>75.51</td><td>+1.16</td><td>44.29</td><td>+0.30</td><td></td><td></td><td></td><td></td></tr><tr><td>ReconBoost [10]</td><td>47.29</td><td>-7.64</td><td>33.14</td><td>+0.59</td><td>74.79</td><td>+0.44</td><td>44.78</td><td>+0.79</td><td>75.09</td><td>+0.22</td><td>44.42</td><td>-0.04</td></tr></table>

Table 2: Accuracy (%) ofAudio-Video (A-V), Text-Video (T-V), and Audio-Text-Video (A-T-V) models on CMU-MOSI, CMU-MOSEI with ReconBoost under Adam and SGD, showing its sensitivity to optimizer choice. Results reproducedfrom [13].
<table><tr><td rowspan="2">Opt.</td><td rowspan="2">Method</td><td colspan="2">A-V</td><td colspan="2">T-V</td><td colspan="2">A-T-V</td></tr><tr><td></td><td>MOSI MOSEI</td><td>MOSI</td><td></td><td>MOSEI MOSI MOSEI</td><td></td></tr><tr><td>Adam</td><td>Baseline</td><td>54.93</td><td>32.55</td><td>74.35</td><td>43.99</td><td>74.87</td><td>44.46</td></tr><tr><td rowspan="2">SGD</td><td>ReconBoost</td><td>47.29</td><td>33.14</td><td>74.79</td><td>44.78</td><td>75.09</td><td>44.42</td></tr><tr><td>Baseline</td><td>49.97</td><td>32.51</td><td>73.21</td><td>44.46</td><td>73.24</td><td>44.80</td></tr><tr><td></td><td>ReconBoost 49.88</td><td></td><td>32.42</td><td>73.45</td><td>44.37</td><td>73.07</td><td>44.30</td></tr></table>

Modern multimodal methods reuse the likelihood-based idea through neural surrogates. OGM-GE [7] computes modality discrepancies based on modality-specific negative loglikelihoods. AGM [8] derives Shapley contribution scores from differences in modality-wise negative log-likelihoods. PMR [9] uses class prototypes to approximate class-conditional likelihoods via sample-to-prototype distances, deriving likelihood ratios that guide modality rebalancing. In each case, these likelihood-derived signals are used to adjust modality weights during training, repeating the same approach that failed in the 1990s.

More recent work continues to highlight that likelihoodbased estimates do not always track the true, sample-level discriminative contribution of each modality [33]. This persistent mismatch motivates a re-examination of how modality reliability should be estimated in contemporary multimodal learning.

## 3. Unified Evaluation Framework

Using a unified setup with fixed architecture and training, we revisit multimodal optimization in MSA and evaluate two gradient-based and two loss-based approaches that attempt to remedy modality imbalance.

## 3.1. Experimental Setup

## 3.1.1. Optimization Methods Under Evaluation

On-the-fly Gradient Modulation (OGM) [7] computes output-likelihood discrepancy ratios to identify the dominant modality and suppresses its gradients to give weaker modalities space to learn. This mechanism assumes that downscaling the fastest-learning modality promotes balance across learning speeds. We also evaluate its Generalization Enhancement (GE)[7] extension, which adds a regularizer for improved robustness, and a variant (ACC) that instead amplifies the weakest modality based on the canonical discrepancy ratio.

Adaptive Gradient Modulation (AGM) [8] uses Shapleyinspired mono-modal outputs to estimate how much each modality contributes to the prediction. These estimates are computed from differences in modality-specific negative loglikelihoods, producing modulation coefficients that reshape gradients under the assumption that contribution-aware updates lead to fairer optimization.

Prototypical Modal Rebalance (PMR) [9] builds classspecific prototypes and reweights the loss according to an imbalance ratio derived from distances to these prototypes. We exclude PMR’s entropy regularizer to focus strictly on the penalization effect and to maintain comparability with OGM and AGM methods.

ReconBoost [10] alternates updates across unimodal encoders and adopts Kullback–Leibler (KL)-based regularization [34] to encourage each modality to correct errors made by the others, as measured during training. Its central assumption is that alternating updates combined with cross-modal reconstruction pressure yields more balanced learning dynamics.

## 3.1.2. Model Architecture

Each modality is encoded with a unidirectional LSTM; representations are concatenated and fed to a shared classifier. We deliberately adopt a simple late-fusion architecture to isolate optimization dynamics from architectural confounds, ensuring that any observed effects reflect training-signal behavior rather than representational capacity. The architecture is fixed across methods and trained from scratch to isolate optimization effects from architectural variation. In addition to the optimization methods, we evaluate three standard training strategies: (1) softvoting ensembles, averaging independent unimodal predictions; (2) uni-modality pre-finetuning, where encoders are trained individually before joint late-fusion finetuning; and (3) joint late concatenation without balancing. These baselines provide a reference for interpreting balancing methods.

## 3.1.3. Training Configuration Sensitivity Analysis

We evaluate generalization by varying core training choices, such as optimizer, training duration, and modulation period, which influence convergence and modality dynamics. These controlled variations reveal how much behavior is driven by training configuration rather than method design.

To investigate how robust modality estimates are, we compute modality-specific ratios on a separate development set rather than on the training batch. Using training data alone ties these estimates to batch size and introduces instability. Decoupling ratio computation from the training loop yields a more stable, unbiased, and generalizable signal of modality importance. We apply this modification only when it aligns with a method’s original design [7, 8, 9]. This modification represents a partial move toward discriminative estimation: the development set provides signal about generalization rather than training fit.

To expose how each optimization method behaves under different dominance conditions, we construct three controlled imbalance setups: two weaker modalities (Audio-Video); one dominant and one weak modality (Text-Video), while omitting Text-Audio due to its analogous text-dominated behavior; and a fully imbalanced case where all three modalities are present. These scenarios let us test whether the methods genuinely adapt to changing modality strengths rather than relying on favorable conditions. We characterize modality balance through performance stability and dominance trajectories across controlled imbalance settings.

## 3.1.4. Datasets and Feature Representation

We evaluate on CMU-MOSI [11] and CMU-MOSEI [12] using standard features: 768-dim BERT [35] for text, COVAREP [36] for audio, and OpenFace [37] for vision. Fixed representations isolate optimization effects. A small held-out development set (100 samples for CMU-MOSI and 200 for CMU-MOSEI) is used for auxiliary metrics. Sentiment is treated as classification: three classes for MOSI and seven ordinal classes for MOSEI following [38].

## 3.1.5. Training Protocol

All models follow the same training protocol. We use Adam [28] with a ReduceLROnPlateau scheduler (factor 0.1), batch sizes of 16 (MOSI) and 32 (MOSEI) [18, 39], and early stopping (patience 8). Cross-entropy loss is used throughout. Because optimization strategies are sensitive to learning rate, each method receives its own grid search; all other settings are fixed. We report the best configuration per method, averaged over five seeds, selected by validation loss. Experiments run on a single NVIDIA GTX 1080 Ti (12GB).

## 3.2. Empirical Results

These methods aim to balance modalities, but empirical results reveal a more nuanced picture.

Table 3: Accuracy (%) of Audio-Video (A-V) and Text-Video (T-V) under different modulation schedules. The minimal differences across schedules indicate that models converge early, making extended modulation unnecessary. Results from [13].
<table><tr><td>Dataset</td><td>Schedule</td><td>Method</td><td>A-V</td><td>T-V</td></tr><tr><td>MOSI</td><td>100 epochs</td><td>Baseline</td><td>53.09</td><td>72.68</td></tr><tr><td></td><td></td><td>OGM</td><td>52.95</td><td>72.97</td></tr><tr><td></td><td></td><td>OGM-GE</td><td>50.47</td><td>73.59</td></tr><tr><td>MOSEI</td><td>5 epochs</td><td>Baseline</td><td>32.55</td><td>43.99</td></tr><tr><td></td><td></td><td>OGM</td><td>32.67</td><td>44.15</td></tr><tr><td></td><td></td><td>OGM-GE</td><td>32.40</td><td>43.58</td></tr><tr><td>MOSEI</td><td>All epochs</td><td>Baseline</td><td>32.55</td><td>43.99</td></tr><tr><td></td><td></td><td>OGM</td><td>32.59</td><td>44.02</td></tr><tr><td></td><td></td><td>OGM-GE</td><td>32.43</td><td>43.13</td></tr></table>

## 3.2.1. Strong Baselines, Limited Gains

Table 1 shows a consistent trend: strong baselines remain difficult to surpass. On CMU-MOSI, Late Concatenation achieves the highest Audio-Vision accuracy, with OGM and AGM matching it only sporadically. On CMU-MOSEI, AGM offers a slight gain. In Text-Vision and trimodal settings, Uni-Pre Finetuned is the most reliable performer, outperforming all optimization-based methods across datasets. Adding audio yields minimal benefit, especially on CMU-MOSEI, reaffirming the dominance of text and the limited corrective power of reweighting strategies. Overall, the evaluated methods provide only modest improvements, rarely exceeding the robustness of the baselines they aim to improve.

## 3.2.2. Adam Does the Heavy Lifting

Our experiments show that Adam generally outperforms SGD across most methods and datasets (Table 2). Its adaptive learning rates provide the stability multimodal training depends on, while SGD converges slowly and is far more sensitive to hyperparameters (Figure 2).

This finding is revealing: the “balancing” these methods provide is largely an artifact of favorable optimizer dynamics. When the optimizer changes, the improvements vanish. True modality balancing should be robust to such configuration choices.

## 3.2.3. Long Training, Little Gain

Our experiments show that OGM and OGM-GE gain little from extended training or prolonged modulation schedules (Table 3). Although OGM-GE is designed to benefit from long training due to its Gaussian noise component, it still performs below strong baselines. In practice, these models converge within the early epochs, making additional modulation largely redundant.

This pattern aligns with the theoretical analysis of Zhang et al. [20]: unimodal bias is established early and becomes permanent through overfitting. Gradient modulation applied after this critical period cannot reverse the damage.

## 3.2.4. Development Set Helps, but Not Enough

Using a development set to compute discrepancy (OGM-GE), strength (AGM), or imbalance (PMR) produces more stable and less biased estimates than relying on the training batch. This decoupling makes the methods less sensitive to batch size and better reflects true modality contributions. As shown in Figure

Epoch

3, AGM calibrated with a development set (Figure 3(c)) tempers the dominance of text and strengthens the video modality (Figure 3(a)), leading to a more balanced contribution profile compared to standard AGM (Figure 3(b)). Overall, development set calibration stabilizes these methods, even if it does not fundamentally change their performance limits.

This is the most instructive result: moving toward held-out estimation helps, but using losses rather than held-out discriminative performance reveals the fundamental limitation. The development set provides a better estimate of generalization loss, but loss alone remains a poor indicator.

## 3.3. Why These Methods Fall Short: The Emerging Picture

Our results reveal a consistent pattern across all methods. When we change the optimizer, the improvements vanish, showing that the methods rely on configuration choices rather than true modality balancing. Their modality estimates are too noisy to be trusted during training, so they rely on a development set to correct their decisions. Longer training or extended modulation also offers no advantage, since models converge early and additional gradient manipulation does not alter the imbalance.

The combination of high variance, sensitivity to configuration choices, and small gains that vanish under mild changes mirrors the conclusions of other studies [33, 40, 41], which report similarly fragile behavior across datasets, architectures, and imbalance scenarios. Taken together, the evidence indicates that these approaches do not correct the underlying optimization dynamics; they simply shift where the instability becomes visible.

We therefore turn to a deeper analysis of the mechanisms behind this behavior, asking whether the issue reflects a more fundamental limitation of training-time optimization signals as indicators of modality contribution.

## 4. The Fundamental Problem: Fitting is not Contributing

Losses, gradients, and likelihood ratios measure how quickly a modality fits the training data, not how much it contributes to correct predictions at test time. This distinction is central to understanding the limitations of optimization-based balancing. It is supported by both theoretical arguments and historical evidence, and motivates diagnostics experiments as well as a principled path forward. The goal of this section is to diagnose the structural mismatch between fitting signals and discriminative utility, and to outline principles for future method development.

## 4.1. A Theoretical Diagnosis

## 4.1.1. Loss Measures Fitting Speed, Not Utility

A modality can achieve low training loss through several mechanisms: 1) it carries genuinely strong discriminative cues, 2) it memorizes/overfits training examples, or 3) it picks spurious correlations (shortcut learning) present in the training set. Loss alone cannot distinguish between these cases [42]. A dominant modality, like text in MSA [22, 43], may legitimately achieve lower loss because it is more informative; yet the same low-loss signal can also identify a modality that has simply learned or overfit faster.

This ambiguity undermines the core logic of loss-based reweighting. When PMR [9] penalizes modalities with smaller prototype distances, or when ReconBoost [10] alternates updates based on loss trajectories, these methods cannot determine whether they are suppressing an overfit modality or handicapping a genuinely informative one.

![](images/c4672ad998b991cb9c32e240d84c248e81c1b619ab2f9e29595c0e4c2f9c2c01.jpg)

![](images/e3e44e2d5302e6fda14d050a92fc07980b91d985fdcd2193fbd486a471d85a62.jpg)  
Figure 2: Training losses for ReconBoost on CMU-MOSI with Adam (top) and SGD (bottom). Text remains dominant, while audio and video losses drop in the early phase, showing that modulation is only useful briefly. SGD is more volatile than Adam but converges to the same pattern. Reproducedfrom [13].

## 4.1.2. Gradients Measure Rate ofChange, Not Importance

Gradient-based methods face a parallel problem. Large gradients indicate that a modality’s parameters are changing rapidlybut rapid change could mean the modality is learning useful features, or that it is oscillating due to noise, or that it is in an early phase of training that will later stabilize. Small gradients are equally ambiguous: a modality may have converged to a good representation, or it may be stuck in a poor local minimum, or it may be receiving suppressed updates due to dominance by another modality [20, 44].

OGM-GE [7] suppresses gradients from the “dominant” modality identified by output likelihood discrepancies, and AGM [8] modulates gradients based on Shapley-inspired contribution estimates. Both methods assume that gradient magnitude correlates with modality importance. Yet gradient dynamics reflect optimization trajectories, not discriminative value.

## 4.1.3. Likelihood Ratios Inherit the Same Flaw

Modern methods that compute modality “contributions” from negative log-likelihoods or softmax outputs are neural analogues of the generative likelihood-ratio approaches used in 1990s audio-visual speech recognition. The same limitation identified then still applies: generative likelihoods measure how well each modality explains the data, not how much it improves class discrimination [29, 31, 32].

As shown in [31], learning stream weights discriminatively, by directly minimizing classification error, succeeds where likelihood-based weighting fails [31]. The key insight is that modality reliability must be inferred from discriminative performance rather than generative fit, an idea largely overlooked in modern multimodal learning.

![](images/01a675ea812d3f2bc1f047d342ebf41494a1cfa4f99627704511fc8136707ed6.jpg)  
(a) Late Concatenation

![](images/ca1cb3a4016ba25d30f3a1b493c18afaa5e76e5fe08a6da1af3a4c9a6edc60a3.jpg)  
(b) AGM

![](images/883476651b6560fc98350dd72ebb4cc34d064ccb00b511ae0c6c5aae41eefead.jpg)  
(c) AGM with development set  
Figure 3: Baseline (a), AGM (b), and AGM with development set (c). Training-only estimates lead to unstable modality strengths, while development-set calibration results in smoother, more consistent trajectories. Reproducedfrom [13].

## 4.1.4. Why Complementarity Cannot Be Estimated from Loss

The Generalized Ambiguity Decomposition (GAD) theorem [45] shows that for an ensemble (analogous to modality-specific encoders),

$$
{ \mathrm { E n s e m b l e ~ L o s s } } \approx { \overline { { \mathrm { E x p e r t ~ L o s s } } } } - { \mathrm { D i v e r s i t y } } ,\tag{5}
$$

where Expert Loss is the average loss of individual experts and Diversity measures disagreement among them. Fusion gains thus arise from the complementarity factor within Diversity, not from individual losses alone. A modality with higher marginal loss may still add value by correcting others’ errors.

Since marginal loss contains no information about error overlap, complementarity cannot be inferred from it. Modalities with identical losses may contribute very differently depending on whether their errors coincide or diverge. Loss-driven reweighting therefore risks miscalculating modality contribution unless cross-modal interactions are modeled.

## 4.1.5. Sample-Level Variation Defeats Global Reweighting

Even if loss or gradient signals were accurate indicators of modality utility, they would still fail because modality contributions vary at the sample level [29, 30, 33]. For some samples, audio may be more informative; for others, video may dominate; for still others, only the combination provides discriminative signal. Global reweighting schemes, whether applied per-batch or per-epoch, cannot capture this variation.

## 4.1.6. Summary: The Category Error

The methods we evaluate commit a category error: they use fitting signals (loss, gradients, likelihoods) as proxies for contribution signals (discriminative value, complementarity, test-time utility). This error has been identified before and the solution was clear: use discriminative criteria estimated from held-out performance, not generative signals from training.

## 4.2. A Minimal Diagnostic Result

We now formalize modality contribution from a discriminative perspective and establish a minimal result that motivates it.

## 4.2.1. Controlled Validation: XOR-Gated Complementarity

We construct a synthetic experiment with known (ground-truth) complementarity.

Setup. In the LOW condition, modality A carries the label signal; B is pure noise. In the HIGH condition, a latent variable z determines which modality is informative per-sample, but z is encoded via XOR across both modalities (A contains random bit r; B contains r ⊕ z). Neither modality alone reveals z. Fusion is necessary to uncover it.

Table 4: Classification accuracy on the synthetic complementarity task. Columns report test accuracy for unimodal models (A-only, B-only), a late-fusion model using both modalities (Fusion), and the same fusion model trained with OGM (Fusion+OGM). In the HIGH condition, OGM degrades fusion performance.
<table><tr><td>Condition</td><td>A-only</td><td>B-only</td><td>Fusion</td><td>Fusion+OGM</td></tr><tr><td>LOW</td><td>100%</td><td>53%</td><td>100%</td><td>100%</td></tr><tr><td>HIGH</td><td>49%</td><td>54%</td><td>88%</td><td>76%</td></tr></table>

Results. In the HIGH condition, OGM degrades performance (Table 4). Both modalities show identical training losses (≈ 0.69) and gradient norms throughout, so OGM applies symmetric modulation, disrupting the joint learning required to decode the XOR gate. The method cannot detect that neither modality alone is informative while both together are essential.

This confirms that optimization-based methods cannot estimate complementarity from training-time signals when samplelevel variation is high.

## 4.2.2. Learning Speed Is Not Utility: A Controlled Demonstration

Our central hypothesis is that gradient-based measures reflect optimization speed, not test-time utility. We examine whether gradient magnitude can distinguish genuinely informative fast learning from fast but non-generalizable learning.

Setup. We construct a two-modality synthetic classification task with a late-fusion architecture. The model, optimization procedure, and hyperparameters are identical across conditions; only the data-generating process changes. In Condition A, the fast modality provides a strong signal that generalizes to test time, while the slow modality is weaker but informative. In Condition B, the fast modality is deliberately engineered to be perfectly predictive during training through label-dependent prototypes, but this correlation is broken at test time, making it non-generalizable. The slow modality remains informative in both splits. Condition B therefore serves as a controlled setting in which rapid training dynamics arise from spurious structure rather than true signal. We compare standard training (baseline) with OGM-GE modulation.

Table 5: Controlled experiment comparing a genuinely informative fast modality (Condition A) with a spurious fast learner (Condition B). We report final training and test accuracy for the baseline and OGM-GE. “Fast coeff.” is the epoch-averaged OGM-GE gradient scaling factor applied to the fast modality (values ≪ 1 indicate strong suppression).
<table><tr><td>Condition</td><td>Method</td><td></td><td></td><td>Train Acc. Test Acc. Fast Coeff. (avg)</td></tr><tr><td rowspan="2">A</td><td>Baseline</td><td>100%</td><td>99.2%</td><td>1.00</td></tr><tr><td>OGM-GE</td><td>100%</td><td>97.4%</td><td>《1</td></tr><tr><td rowspan="2">B</td><td>Baseline</td><td>100%</td><td>10.1%</td><td>1.00</td></tr><tr><td>OGM-GE</td><td>100%</td><td>10.1%</td><td>《1</td></tr></table>

Results. In both conditions, OGM-GE consistently identifies the fast modality as dominant and suppresses it during training (Table 5). However, the implications differ. In Condition A, suppression reduces test accuracy because the fast modality carries genuine signal. In Condition B, suppression is appropriate since the fast modality represents a shortcut. Importantly, the training-time dominance signal is similar in both cases.

These results demonstrate that gradient magnitude captures how quickly a modality fits the training data, but not whether that fit corresponds to robust, generalizable structure. Gradientbased dominance measures therefore conflate fast learning with true informativeness.

## 4.2.3. Validation-Based Weight Selection as a Diagnostic

We evaluate whether global reweighting is structurally sufficient on real data, and whether training-time proxy signals provide a reliable basis for fusion.

Setup. Encoders are first trained with standard late concatenation without modality reweighting and then frozen. Fusion is analyzed post hoc by reweighting the modality-specific logit contributions of the frozen classifier using scalar weights selected from a small grid of candidate values.

We compare four strategies for selecting a single global fusion weight: Proxy, which derives weights from unimodal training cross-entropy; ValSel, which selects the weight maximizing validation accuracy; Oracle, which selects the best global weight using test labels and therefore represents an upper bound; and OGM-GE [7], a representative training-time gradient modulation method.

To examine whether a single global weight adequately reflects modality utility, we also compute a per-sample optimal weight. For each sample, we evaluate all candidate fusion weights and select the one that minimizes the cross-entropy loss for that prediction. We summarize the variability of these per-sample optimal weights using two statistics: Entropy, which measures the diversity of optimal weights across samples, and Optimal Weight Disagreement (OWD), defined as the percentage of samples whose per-sample optimal weight differs from the validation-selected global weight. This analysis characterizes per-sample optimal fusion weights and how global weighting departs from sample-specific utility.

Table 6: Acc-2 of global fusion-weight selection strategies on CMU-MOSI and CMU-MOSEI for Audio-Video (A-V), Text-Video (T-V), and Audio-Text-Video (A-T-V). Proxy derives weights from unimodal training cross-entropy, ValSel selects the validation-optimal weight, Oracle reports the best global weight using test labels, and OGM-GE corresponds to trainingtime gradient modulation. Ent. denotes the entropy of persample optimal weights, and OWD (%) the Optimal Weight Disagreement with the validation-selected global weight.
<table><tr><td>Setup</td><td>Proxy</td><td>ValSel</td><td>Oracle</td><td>OGM-GE</td><td>Ent.</td><td>OWD.(%)</td></tr><tr><td colspan="7">MOSI</td></tr><tr><td>A-V</td><td>0.4223</td><td>0.4223</td><td>0.4223</td><td>0.4223</td><td>0.681</td><td>42</td></tr><tr><td>T-V</td><td>0.8003</td><td>0.7988</td><td>0.8018</td><td>0.7851</td><td>0.510</td><td>100</td></tr><tr><td>A-T-V</td><td>0.7957</td><td>0.7957</td><td>0.7988</td><td>一</td><td>0.649</td><td>100</td></tr><tr><td colspan="7">MOSEI</td></tr><tr><td>A-V</td><td>0.6285</td><td>0.6285</td><td>0.6285</td><td>0.6293</td><td>0.671</td><td>39.7</td></tr><tr><td>T-V</td><td>0.8360</td><td>0.8343</td><td>0.8343</td><td>0.8360</td><td>0.446</td><td>16.4</td></tr><tr><td>A-T-V</td><td>0.8145</td><td>0.8154</td><td>0.8178</td><td></td><td>0.555</td><td>100</td></tr></table>

Results. Validation-based selection performs comparably to the proxy and remains within 0.3–0.4% of oracle performance (Table 6), indicating that an appropriately chosen global fusion weight captures most aggregate accuracy.

Low sample-level variation would imply that a single global fusion weight suffices across inputs. However, Table 6 shows consistently non-trivial entropy and high disagreement in most modality combinations, indicating that the per-sample optimal fusion weight frequently differs from the global selection. These results suggest that, while global weighting is adequate in aggregate performance, substantial sample-level variation in optimal weights persists in the evaluated settings.

## 4.2.4. Structural Limits of Global Reweighting

While Section 4.2.3 shows that global weighting rarely aligns with per-sample optimality, we next quantify the magnitude of this structural limitation.

Setup. Using the same frozen encoders as in Section 4.2.3, we evaluate the structural limits of global reweighting. We compare two upper bounds. The Global Oracle selects a single fusion weight vector that maximizes accuracy on the test set and applies it to all samples. The Restricted Per-Sample Oracle selects, for each test sample independently, the fusion weight vector that yields the correct prediction among the available non-unimodal fusion combinations. This oracle therefore represents the best performance achievable if fusion weights could be chosen per sample rather than globally.

Results. The restricted per-sample oracle achieves 88.57% accuracy on MOSI and 89.57% on MOSEI, compared to 79.88% and 81.78% for the respective global oracles (Table 7). This yields structural headroom of +8.7% and +7.79%, quantifying the performance unavailable to any method constrained to a single global fusion weight.

This result makes the limitation of global reweighting concrete: even a perfect global reweighting strategy—one that always selects the best possible single weight vector—leaves a 8.7% accuracy gap relative to what sample-level weight selection could theoretically achieve. Closing this gap requires methods capable of estimating modality utility at the sample level rather than relying on a fixed global weighting.

Table 7: Accuracy (Acc-2) illustrating the limits of global fusion weighting on CMU-MOSI and CMU-MOSEI (A-T-V). The Global Oracle selects a single fusion weight vector maximizing test accuracy, while the Restricted Per-Sample Oracle selects optimal fusion weights independently for each sample. The gap (Structural Headroom) represents performance unavailable to any method constrained to a single global weight.
<table><tr><td>Method</td><td>MOSI</td><td>MOSEI</td></tr><tr><td>Global Oracle</td><td>0.7988</td><td>0.8178</td></tr><tr><td>Restricted Per-Sample Oracle</td><td>0.8857</td><td>0.8957</td></tr><tr><td>Structural Headroom</td><td>+8.70%</td><td>+7.79%</td></tr></table>

## 4.3. A Discriminative Path Forward: Implication and Open Problems

Optimization-based balancing conflates fitting with contributing. The empirical results in Section 3 and the controlled demonstrations in Section 4.2 suggest that estimating modality utility from held-out discriminative performance is a promising direction, and they raise open questions about how to do so reliably under sample-level variation and complementarity.

## 4.3.1. Estimate Utility from Held-Out Performance

The optimization of modality encoders (which should be trained on training data) must be separated from the optimization of fusion weights (which should be learned from validation discriminative performance). This separation mirrors the logic of Neural Architecture Search, where validation performance guides architecture selection rather than training loss.

Current methods attempt to estimate modality contribution from within the training loop, using signals that reflect fitting rather than utility. The fundamental insight from discriminative stream weighting [31] is that modality reliability cannot be estimated from generative signals—it must be measured by discriminative performance on held-out data.

## 4.3.2. When Should We Expect Training-Time Weighting to Work?

The preceding analysis in Section 4.2 identifies regimes in which optimization-based weighting may be harmless, though not necessarily helpful.

Low complementarity. When one modality strictly dominates and others add no complementary information, suppressing weak modalities is unlikely to hurt. But it is also unlikely to help, since the baseline already ignores them.

Dominance alignment. When the fast-learning modality is also the most informative at test time, training-time signals accidentally correlate with utility. In sentiment analysis, where text typically dominates, we observe the strongest performance in text-heavy configurations.

Low sample-level variation. When optimal fusion weights are approximately constant across samples, global reweighting may suffice. But this condition rarely holds in practice.

These are alignment regimes rather than guarantees. When complementarity is substantial, when fast learners overfit, or when modality utility varies across samples, the mismatch between fitting and contributing becomes decisive. Characterizing these regimes more formally remains an open problem.

## 4.3.3. Open Research Directions

The diagnostic experiments above make future directions concrete: the structural headroom quantifies the necessity of sample-level valuation; the validation-based selection results demonstrate the feasibility of held-out fusion optimization; the observed disagreement across settings motivates discriminative meta-classification; and the dominance-alignment regimes highlight the importance of robustness-aware modality profiling. Together, these observations reinforce the need to ground modality valuation in held-out discriminative performance:

Discriminative Meta-Classification. Reframe modality balancing as a meta-regression problem solved through discriminative profiling on held-out validation data. Rather than manipulating training-time signals, exhaustively measure which modality combinations actually predict correctly, for each validation sample individually. Train meta-regressors to predict fusion weights from test-time-observable features (modality encodings, prediction confidences, cross-modal agreement), with regularization toward validation-optimized global weights.

Sample-Level Modality Valuation. Building on Wei et al. [33], develop methods that estimate per-sample modality contributions from validation performance rather than training signals. The key insight is that modality utility varies by sample; global reweighting cannot capture this variation.

Robustness Profiling. Measure modality utility under corruption and noise on validation data. A modality that remains accurate under perturbation provides more robust signal than one that degrades. This robustness information cannot be extracted from clean training loss.

Validation-Based Fusion Weight Optimization. Directly optimize fusion weights to maximize validation accuracy, treating encoder parameters as fixed after initial training. This fully separates encoder optimization from fusion optimization.

## 5. Conclusions

In this work, we conducted a unified evaluation of recent optimization-based methods for modality balancing in Multimodal Sentiment Analysis. We found that no method reliably outperforms simple late fusion. Their core limitation is clear: they treat losses and gradients as indicators of discriminative value.

Modality imbalance persists, text continues to dominate, and performance remains highly sensitive to training choices. Resolving modality imbalance requires moving beyond training loop interventions toward methods that estimate modality value from held-out discriminative performance, a principle established in discriminative stream weighting [31] and supported by the GAD characterization of diversity [45].

Our work provides both a diagnostic foundation, establishing what current optimization methods cannot achieve and why, and a research agenda pointing toward discriminative metalearning frameworks that separate encoder optimization from fusion weight optimization. The path forward is not better gradient modulation or smarter loss reweighting; it is moving beyond purely training-time fitting signals toward held-out, discriminative criteria for modality valuation. The discriminative directions outlined here are the subject of ongoing work.

## 6. Generative AI Use Disclosure

Generative AI tools were used to assist in refining parts of the manuscript text. All scientific content, experimental design,

analyses, and conclusions were developed, verified, and approved by the authors, who take full responsibility for the final manuscript.

## 7. References

[1] S. Poria, E. Cambria, R. Bajpai, and A. Hussain, “A review of affective computing: From unimodal analysis to multimodal fusion,” Information Fusion, vol. 37, p. 98–125, Sep. 2017. [Online]. Available: http://dx.doi.org/10.1016/j.inffus.2017.02. 003

[2] W. Wang, D. Tran, and M. Feiszli, “What makes training multi-modal classification networks hard?” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Jun. 2020, p. 12692–12702. [Online]. Available: http://dx.doi.org/10.1109/CVPR42600.2020.01271

[3] E. Georgiou, “Multimodal representation learning with application in sentiment analysis,” Ph.D. dissertation, National Technical University of Athens (NTUA). School of Electrical and Computer Engineering, 2025.

[4] P. P. Liang, A. Zadeh, and L.-P. Morency, “Foundations & trends in multimodal machine learning: Principles, challenges, and open questions,” ACM computing surveys, vol. 56, no. 10, pp. 1–42, 2024.

[5] N. Wu, S. Jastrzebski, K. Cho, and K. J. Geras, “Characterizing and overcoming the greedy nature of learning in multimodal deep neural networks,” 2022. [Online]. Available: https://arxiv.org/abs/2202.05306

[6] Y. Huang, J. Lin, C. Zhou, H. Yang, and L. Huang, “Modality competition: What makes joint training of multi-modal network fail in deep learning? (Provably),” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, Eds., vol. 162. PMLR, 17–23 Jul 2022, pp. 9226–9259. [Online]. Available: https://proceedings.mlr.press/v162/huang22e.html

[7] X. Peng, Y. Wei, A. Deng, D. Wang, and D. Hu, “Balanced multimodal learning via on-the-fly gradient modulation,” in Proceed ings of the IEEE/CVF Conference on Computer Vision and Pat tern Recognition (CVPR), June 2022, pp. 8238–8247.

[8] H. Li, X. Li, P. Hu, Y. Lei, C. Li, and Y. Zhou, “Boosting multimodal model performance with adaptive gradient modulation,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2023, pp. 22 214–22 224.

[9] Y. Fan, W. Xu, H. Wang, J. Wang, and S. Guo, “PMR: Prototypical modal rebalance for multimodal learning,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Jun. 2023, p. 20029–20038. [Online]. Available: http://dx.doi.org/10.1109/CVPR52729.2023. 01918

[10] C. Hua, Q. Xu, S. Bao, Z. Yang, and Q. Huang, “Reconboost: Boosting can achieve modality reconcilement,” in The Forty-first International Conference on Machine Learning, 2024.

[11] A. Zadeh, R. Zellers, E. Pincus, and L. philippe Morency, “MOSI: Multimodal corpus of sentiment intensity and subjectivity analysis in online opinion videos,” ArXiv, vol. abs/1606.06259, 2016. [Online]. Available: https://api.semanticscholar.org/CorpusID:13978043

[12] A. Bagher Zadeh, P. P. Liang, S. Poria, E. Cambria, and L.-P. Morency, “Multimodal language analysis in the wild: CMU-MOSEI dataset and interpretable dynamic fusion graph,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2018, p. 2236–2246. [Online]. Available: http://dx.doi.org/10.18653/v1/P18-1208

[13] I. Kaffeza, “Investigating optimization techniques for multimodal neural networks,” Master’s thesis, National Technical University of Athens, 2025. [Online]. Available: http://dx.doi.org/10.26240/ heal.ntua.29927

[14] T. Baltrusaitis, C. Ahuja, and L.-P. Morency, “Multimodal machine learning: A survey and taxonomy,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 2, p. 423–443, Feb. 2019. [Online]. Available: http://dx.doi.org/10. 1109/TPAMI.2018.2798607

[15] P. K. Atrey, M. A. Hossain, A. El Saddik, and M. S. Kankanhalli, “Multimodal fusion for multimedia analysis: a survey,” Multimedia Systems, vol. 16, no. 6, p. 345–379, Apr. 2010. [Online]. Available: http://dx.doi.org/10.1007/s00530-010-0182-0

[16] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

[17] Y.-H. H. Tsai, S. Bai, P. P. Liang, J. Z. Kolter, L.-P. Morency, and R. Salakhutdinov, “Multimodal transformer for unaligned multimodal language sequences,” in Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics. Association for Computational Linguistics, 2019, p. 6558–6569. [Online]. Available: http://dx.doi.org/10.18653/v1/P19-1656

[18] W. Yu, H. Xu, Z. Yuan, and J. Wu, “Learning modalityspecific representations with self-supervised multi-task learning for multimodal sentiment analysis,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 12, p. 10790–10797, May 2021. [Online]. Available: http://dx.doi.org/ 10.1609/aaai.v35i12.17289

[19] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, p. 1735–1780, Nov. 1997. [Online]. Available: http://dx.doi.org/10.1162/neco.1997.9. 8.1735

[20] Y. Zhang, P. E. Latham, and A. Saxe, “Understanding unimodal bias in multimodal deep linear networks,” in Proceedings of the 41st International Conference on Machine Learning, ser. ICML’24. JMLR.org, 2024.

[21] R. Xu, R. Feng, S.-X. Zhang, and D. Hu, “MMCosine: Multi-modal cosine loss towards balanced audio-visual finegrained learning,” in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, Jun. 2023, p. 1–5. [Online]. Available: http://dx.doi.org/10.1109/ICASSP49357.2023.10096655

[22] E. Georgiou, G. Paraskevopoulos, and A. Potamianos, “M 3: Multimodal masking applied to sentiment analysis,” in Proc. Interspeech 2021, 2021, pp. 2876–2880.

[23] S. Alfasly, J. Lu, C. Xu, and Y. Zou, “Learnable irrelevant modality dropout for multimodal action recognition on modalityspecific annotated videos,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, Jun. 2022, p. 20176–20185. [Online]. Available: http://dx.doi.org/10. 1109/CVPR52688.2022.01957

[24] Y. He, R. Cheng, G. Balasubramaniam, Y.-H. H. Tsai, and H. Zhao, “Efficient modality selection in multimodal learning,” Journal of Machine Learning Research, vol. 25, no. 47, pp. 1–39, 2024. [Online]. Available: http://jmlr.org/papers/v25/23-0439. html

[25] J. Wu, Y. Liang, F. Han, H. Akbari, Z. Wang, and C. Yu, “Scal ing multimodal pre-training via cross-modality gradient harmonization,” in Proceedings of the 36th International Conference on Neural Information Processing Systems, ser. NIPS ’22. Red Hook, NY, USA: Curran Associates Inc., 2022.

[26] J. Chen, Z. Guo, T. Jin, and Z. Zhao, “Classifier-guided gradient modulation for enhanced multimodal learning,” in Advances in Neural Information Processing Systems 37, ser. NeurIPS 2024. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2024, p. 133328–133344. [Online]. Available: http://dx.doi.org/10.52202/079017-4238

[27] Y. Wei and D. Hu, “MMPareto: Boosting multimodal learning with innocent unimodal assistance,” 2024. [Online]. Available: https://arxiv.org/abs/2405.17730

[28] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” 2017. [Online]. Available: https://arxiv.org/abs/ 1412.6980

[29] G. Potamianos, C. Neti, G. Gravier, A. Garg, and A. W. Senior, “Recent advances in the automatic recognition of audiovisual speech,” Proc. IEEE, vol. 91, pp. 1306–1326, 2003. [Online]. Available: https://api.semanticscholar.org/CorpusID:15512141

[30] C. Neti, G. Potamianos, J. Luettin, I. Matthews, H. Glotin, D. Vergyri, J. Sison, and A. Mashari, “Audio-visual speech recognition,” IDIAP Research Institute, Martigny, Switzerland, Tech. Rep., 2000.

[31] G. Potamianos and H. Graf, “Discriminative training of hmm stream exponents for audio-visual speech recognition,” in Proceedings of the 1998 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP ’98 (Cat. No.98CH36181), vol. 6, 1998, pp. 3733–3736 vol.6.

[32] M. Heckmann, F. Berthommier, and K. Kroschel, “Noise adaptive stream weighting in audio-visual speech recognition,” EURASIP Journal on Advances in Signal Processing, vol. 2002, no. 11, Nov. 2002. [Online]. Available: http://dx.doi.org/10.1155/ S1110865702206150

[33] Y. Wei, R. Feng, Z. Wang, and D. Hu, “Enhancing Multimodal Cooperation via Sample-Level Modality Valuation,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Seattle, WA, USA: IEEE, Jun. 2024, pp. 27 328–27 337. [Online]. Available: https://ieeexplore.ieee.org/document/10656817/

[34] S. Kullback and R. A. Leibler, “On information and sufficiency,” The Annals of Mathematical Statistics, vol. 22, no. 1, p. 79–86, Mar. 1951. [Online]. Available: http://dx.doi.org/10.1214/aoms/ 1177729694

[35] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), J. Burstein, C. Doran, and T. Solorio, Eds. Minneapolis, Minnesota: Association for Computational Linguistics, Jun. 2019, pp. 4171–4186. [Online]. Available: https://aclanthology.org/N19-1423/

[36] G. Degottex, J. Kane, T. Drugman, T. Raitio, and S. Scherer, “COVAREP — a collaborative voice analysis repository for speech technologies,” 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 960– 964, 2014. [Online]. Available: https://api.semanticscholar.org/ CorpusID:746430

[37] T. Baltrusaitis, P. Robinson, and L.-P. Morency, “Openface: An open source facial behavior analysis toolkit,” in 2016 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, Mar. 2016, p. 1–10. [Online]. Available: http://dx.doi.org/10.1109/WACV.2016.7477553

[38] J.-B. Delbrouck, N. Tits, M. Brousmiche, and S. Dupont, “A transformer-based joint-encoding for emotion recognition and sentiment analysis,” in Second Grand-Challenge and Workshop on Multimodal Language (Challenge-HML). Association for Computational Linguistics, 2020, p. 1–7. [Online]. Available: http://dx.doi.org/10.18653/v1/2020.challengehml-1.1

[39] Z. Wu, Z. Gong, J. Koo, and J. Hirschberg, “Multimodal multi-loss fusion network for sentiment analysis,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), K. Duh, H. Gomez, and S. Bethard, Eds. Mexico City, Mexico: Association for Computational Linguistics, Jun. 2024, pp. 3588–3602. [Online]. Available: https://aclanthology.org/2024.naacl-long.197/

[40] K. Kontras, T. Strypsteen, C. Chatzichristos, P. P. Liang, M. B. Blaschko, and M. D. Vos, “Multimodal fusion balancing through game-theoretic regularization,” in Greeks in AI Symposium 2025, 2025. [Online]. Available: https://openreview.net/forum? id=2gjBiqNDQX

[41] Y. Wei, S. Li, R. Feng, and D. Hu, “Diagnosing and Re-learning for Balanced Multimodal Learning,” Jul. 2024, arXiv:2407.09705. [Online]. Available: http://arxiv.org/abs/2407. 09705

[42] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, “Understanding deep learning requires rethinking generalization,” in International Conference on Learning Representations, 2017. [Online]. Available: https://openreview.net/forum?id= Sy8gdB9xx

[43] D. Hazarika, Y. Li, B. Cheng, S. Zhao, R. Zimmermann, and S. Poria, “Analyzing modality robustness in multimodal sentiment analysis,” in Proceedings ofthe 2022 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, M. Carpuat, M.-C. de Marneffe, and I. V. Meza Ruiz, Eds. Seattle, United States: Association for Computational Linguistics, Jul. 2022, pp. 685–696. [Online]. Available: https://aclanthology.org/2022. naacl-main.50/

[44] J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, “Sanity checks for saliency maps,” in Proceedings of the 32nd International Conference on Neural Information Processing Systems, ser. NIPS’18. Red Hook, NY, USA: Curran Associates Inc., 2018, p. 9525–9536.

[45] K. Audhkhasi, A. Sethy, B. Ramabhadran, and S. S. Narayanan, “Generalized ambiguity decomposition for understanding ensemble diversity,” arXiv preprint arXiv:1312.7463, 2013.