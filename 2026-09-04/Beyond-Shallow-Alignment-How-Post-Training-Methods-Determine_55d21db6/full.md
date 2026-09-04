# Beyond Shallow Alignment: How Post-Training Methods Determine Refusal Circuits And Steering Robustness

Hoang Cuong Nguyen, Mark Dras, Usman Naseem

Macquarie University, Sydney, Australia

hoangcuong.nguyen@students.mq.edu.au, {mark.dras, usman.naseem}@mq.edu.au

## Abstract

How do the methods used to train language models to refuse harmful requests shape how that refusal actually works inside the model? We compare three post-training methods – supervised fine-tuning, reasoning-augmented fine-tuning (training on reasoning chains that justify a safety decision), and preference optimization (ORPO) – across three architecturally distinct models (Llama-3.1-8B, Gemma-2-9B, Qwen3-8B). We find that training method, not just data, reshapes how refusal is computed internally: reasoning-augmented training consistently produces a distinct kind of refusal computation, visible across all three models, while architecture independently shapes internal structure and how reliably refusal can be steered. Most importantly, no method we study achieves all three properties we would want from safe alignment at once: refusal that isn't concentrated in a few fragile components, safety gains that don't cost general capability, and safety behavior correctable through small, targeted edits. We caution against treating current post-training methods as a solved, reliable defense, especially for security-critical use. Code and models are available in https: //github.com/hoangcuongnguyen2001/Be yond-Shallow-Alignment.

Warning: This paper contains examples of harmful prompts used for evaluation purposes.

## 1 Introduction

As large language models (LLMs) have been increasingly deployed (Singh et al., 2026; Grattafiori et al., 2024, etc.), safety alignment of these models has been a serious concern for both technical stakeholders as well as governments and the public, especially with the increasing risk of LLM-powered cyber attacks. Therefore, national cybersecurity agencies such as Germany's BSI (2025), the US CISA alongside its Five Eyes partners (2025) have been recommending IT operators and developers to apply post-training methods (reinforcement learning with human feedback (RLHF), supervised finetuning (SFT), etc.) as countermeasures against these attacks, and thus treating alignment as a binary property that these methods reliably produce. However, these methods only consider alignment at behavioral level, and recent documented incident reports, such as GTG-1002 in 2025 (Anthropic Threat Intelligence, 2025), and attacks on Mexican government agencies in early 2026 (Sela, 2026), both breaching jailbreak defense of LLMs such as Claude Code/GPT-4.1 by using role play or persistent reframing, suggest that behavioral alignment evaluation of LLMs, especially their post-training methods, is insufficient.

As mechanistic interpretability has become increasingly important for explaining the internal workings of LLMs (Naseem, 2026), there have been attempts to demonstrating that safety alignment is dependent on circuit structures (Arditi et al., 2024; Yeo et al., 2025; Du et al., 2025; Wu et al., 2026), however, they only characterize safety refusal circuits in fixed models, and do not analyse how training objectives reshape circuit structure in a systematic way. This shortcoming also makes our understanding of training dynamics more of a post-hoc endeavour (Biderman et al., 2026).

![](images/b0480275750c23e912fc1e18571d06fb18955ed3eb6139654c911df13eb114e1.jpg)  
Figure 1: Two-axis paradigm structure. SFT: no preference, no reasoning. Ra-SFT: no preference, reasoning supervision. ORPO: preference, no reasoning. Fourth cell left for future work.

In this work, we ask: when a language model is trained to refuse harmful requests, does how it is trained – not just what data it sees – shape how that refusal is implemented inside the model? We compare three post-training methods that teach the same underlying skill, using preference-reasoning paradigm (figure 1) – supervised fine-tuning (SFT), preference optimization (ORPO), and reasoningaugmented SFT (Ra-SFT, where the model is trained on reasoning chains justifying why a request is safe or harmful) – and trace how each shapes the internal computation that produces a refusal. To our knowledge, ORPO and Ra-SFT have not previously been analyzed at this level of mechanistic detail for safety alignment.

Our contributions are:

(1) a controlled cross-paradigm comparison of how these three objectives shape refusal-related computation, holding base model, training data, and hyperparameters fixed so only the training objective varies – across three architecturally distinct models (Llama-3.1-8B, Gemma-2-9B, Qwen-3-8B);

(2) the first circuit-level analysis of how Ra-SFT and ORPO implement safety alignment internally;

(3) an empirically grounded alignment trilemma: among the offline objectives studied, no method jointly achieves distributed refusal encoding, safety/capability separability, and fine-grained correctability – gains in one consistently cost another.

A compact synthesis of our paper findings is provided in Appendix A for reference.

## 2 Related Works

Geometry of Refusal Representations. Arditi et al. (2024) characterized refusal in LLMs as a single direction from the difference in means between logits of refusal and compliance prompts. Based on that work, Du et al. (2025) pointed out that post-training changed the refusal direction from base, while keeping knowledge representations. Yeo et al. (2025) decomposed harmfulness and refusal directions using sparse autoencoders, since models can encode harms and refusal separately (Zhao et al., 2025) while Wu et al. (2026) separated refusal into two axes: recognition and execution. Our work extends this line of work by providing a controlled cross-paradigm analysis of post-training methods, which so far has been primarily focused on instruction-tuned models.

Shallow Alignment and Circuit Concentration. Shallow safety alignment was first behaviorally defined by Qi et al. (2025) as model defenses being able to be bypassed with a few tokens deep. Later work extends shallow alignment analysis mechanistically by showing the over-concentration of safetyrelated mechanisms at different levels: Huang et al. (2026) pointed out that safety alignment in Llama models is mediated by 50 attention heads; Chen et al. (2025) showed that as few as 5% of neurons have over 90% of causal effects on safety alignment, and more recently, Kazemi et al. (2026) demonstrated that one single neuron is sufficient to bypass safety guardrails. Therefore, whether alignment methods can reduce the overconcentration of safety-related mechanisms is an important problem.

Post-Training Methods and Safety Alignment. Vennemeyer et al. (2026) showed that post-training methods induce systematic, scale-dependent shifts along the safety-utility frontier. Other works extending this behavioral analysis include Janiak et al. (2026) showing that ORPO has the lowest generalization capability for safety; and Thakkar et al. (2025) demonstrating that ORPO-aligned models are resistant to persona drift and less able to generalize, potentially due to its weight-space geometry. Haldar et al. (2025) also considered safety alignment of preference algorithms as divergence estimators between aligned and unaligned distributions.

Regarding SFT and its reasoning variants, Jain et al. (2024) demonstrated that safety fine-tuning methods like SFT/DPO/unlearning fail because their minimal differences in MLP weights; while Hu et al. (2026) found weak relationships between safety alignment and reasoning capability in LLMs. However these analyses were conducted either at the behavioral level or at the neuron level, which make causal attribution of refusal components less informative due to polysemanticity of neurons (Elhage et al., 2022), with controlled circuit analysis of post-training methods being called for future work.

Steering Reliability. Tan et al. (2024) pointed out that capability of steering vectors is limited by the prompt distribution of the dataset and whether a model originally prefers a response over another. This was extended by Braun et al. (2025), indicating that less steerable datasets also have harmful and harmless activations that overlap each other Our work extends this line of work by providing analysis on whether training objectives can also determine steerability.

## 3 Methodology

## 3.1 Experimental Design

We analyse three different post-training methods, which can be mapped into a preference-reasoning matrix (illustrated in figure 1), to test how preference or reasoning change refusal circuits - subgraphs where model components (attention heads, MLPs) implementing refusal (Olah et al., 2020):

\- Supervised ine-tuning (SFT): pure imitation learning, no preference or reasoning added;

\- Reasoning-augmented SFT (Ra-SFT): in which the model is fine-tuned on safety data augmented with explicit reasoning chains that precede the safety decision;

\- Odds Ratio Preference Optimization (ORPO - Hong et al., 2024): representing offline preference algorithm.

The training process is conducted with matched data (using a dataset of 16,000 benign prompts from Alpaca (Taori et al., 2023) and 4,000 prompts from BeaverTails (Ji et al., 2023), provided by Hu et al. (2026) to ensure safety alignment while preserving utility. As for ORPO, since its training process require pairs of chosen and rejected responses for each prompt, we instead match 4,000 Beaver-Tails prompts with the original BeaverTails dataset for a training dataset of 11,179 prompts. We exclude online preference algorithms because their iterative data-generation and reward-update loops would confound our matched-objective comparison (Park et al., 2025).

We fine-tune Llama-3.1-8B (Grattafiori et al., 2024), Gemma-2-9B (Gemma Team, 2024) and Qwen3-8B (Qwen Team, 2025) fully from base models with matched hyperparameters to characterize whether circuit topology differences are architecture-dependent or training-objectivedependent. Full training details, alongside the reasons why we choose these three models and not others, are provided in Appendix B.

## 3.2 Geometry of Refusal

We analyse how refusal representations are organized in activation space (e.g., the angles between directions learned under different training objectives). To do that, we extract refusal direction (a vector in the model's activation space that separates harmful from harmless prompts) via difference-inmeans (DIM) (Marks and Tegmark, 2024; Arditi et al., 2024), from a dataset of 256 pairs of refusalcompliance prompts by Arditi et al. (2024). Since residual stream activation would grow between layers (Elhage et al., 2021), to enable comparison between models with different activation scales, we normalize differences by the mean activation norm at each layer:

$$
\hat { \mathbf { r } } ^ { ( l ) } = \frac { \mathbf { r } ^ { ( l ) } } { \mu ^ { ( l ) } }\tag{1}
$$

where $\begin{array} { r } { \mu ^ { ( l ) } = \frac { 1 } { 2 N } \left( \sum _ { i \in \mathcal { H } } \| \mathbf { h } _ { i } ^ { ( l ) } \| + \sum _ { i \in \mathcal { S } } \| \mathbf { h } _ { i } ^ { ( l ) } \| \right) } \end{array}$ is the mean $\ell _ { 2 }$ norm across all prompts at layer'l. Normalized magnitude in each layer would show when and how each model, in each post-training method, starts encoding refusal, corresponding to findings from Wu et al. (2026) that recognition and execution of refusal are done in different layers. From the DIM results we calculate pairwise cosine similarities across our training objectives for each model, to check whether refusal directions of different training objectives converge or diverge in the residual stream space.

## 3.3 Circuit Analysis of Post-Training Methods

First, we use Activation Patching (Wang et al., 2023; Meng et al., 2022) in each layer, to establish causal effects of each layer toward refusal in a checkpoint. We then deploy Attribution Patching (Syed et al., 2024) for these layers for a quick approximation of causal effects from components of each layer (multi-layer perceptrons (MLPs) and attention heads). A refusal circuit can be MLPdominant, when the causally important components of a circuit are MLP blocks rather than attention heads - which play prominent roles in safety (Zhou et al., 2025; Huang et al., 2026).

Finally, to establish the causal effects of each layer' component, we run Activation Patching through MLPs and top- K attention heads from Attribution Patching results for exact causal effect calculation. For all patching methods we use the same 256 pairs of harmful/harmless prompts in section 3.2, and K = 5 for choosing layers and attention heads for attribution/activation patching. Full explanations of how our circuit analysis works are provided in Appendix C.

## 3.4 Activation Steering

From the causal patching results in Section 3.3, we deploy steering to modify a model’ internal activations at inference time to change its behavior. At the layer level, we use Activation Addition (ActAdd) (Turner et al., 2023; Zou et al., 2023). At the component level, we use Inference-Time Intervention (ITI) (Li et al., 2023) targeting top attention-head outputs. Both methods apply a steering vector along a refusal direction:

$$
h ^ { \prime } = h + \alpha v\tag{2}
$$

where h denotes the original activation, v the refusal steering direction, and α the intervention strength. Positive α steers towards refusal, while negative α steers towards compliance. Full implementation details for ActAdd and ITI are provided in Appendix D.

## 4 Evaluation Results

## 4.1 Evaluation Metrics

To characterize the safety profile of each posttraining method, we used three datasets:

(1) 2000 adversarial harmful prompts from Wild-Jailbreak (Jiang et al., 2024) for out-of-distribution jailbreak attacks;

(2) 420 prompts from the StrongREJECT (Souly et al., 2024) dataset across 7 different attack classes (none - direct request, happy\_to\_help, DAN, disemvowel, rot\_13, wikipedia, role\_play) - inspired by the framework of Vennemeyer et al. (2026) and Wei et al. (2023);

(3) 250 adversarial benign prompts from XSTest (Röttger et al., 2024) to test over-refusal of models, since Campbell et al. (2026) points out that models that are fine-tuned for safety are prone to over-refuse; especially in the most operationally critical tasks, such as system hardening and malware analysis.

As for utility analysis of each checkpoint, we use 200 prompts from MMLU (Hendrycks et al., 2021) across five domains (abstract algebra, professional laws, world religion, moral scenarios and high school biology), for testing factual recall.

For the analysis of how steering impacts safety, we used a randomly sampled subset of 250 prompts from WildJailbreak, due to the high number of experiments and the high correlation between inand out-of-distribution steerability, as established by Tan et al. (2024).

We use three different metrics for measuring safety/utility: (1) Attack Success Rate (ASR) measures the percentages of malicious queries that bypass defenses of LLMs in StrongREJECT and WildJailbreak - we use LlamaGuard-3-8B as a judge to ensure semantic understanding; (2) Over-Refusal Rate (ORR) measures the percentage of adversarial benign prompts in XSTest that are blocked by LLMs - for this we use heuristic string matching, with the full list provided in Appendix E; (3) Accuracy Rate (AR) for utility analysis in MMLU, measuring the percentage of correct results by LLMs for MMLU questions.

The full reasons why we choose different judges for measuring ASR and ORR are provided in Appendix E.

## 4.2 Behavioral Safety Analysis of Post-Training Methods

<table><tr><td>Model</td><td>WJ ASR</td><td>SR ASR</td><td>XS ORR</td></tr><tr><td colspan="4">Llama-3.1-8B base  $5 5 . 5 _ { [ 5 3 . 3 , 5 7 . 6 ] }$   $6 5 . 5 _ { [ 6 1 . 0 , 6 9 . 8 ] }$  SFT  $4 7 . 8 \dot { _ { [ 4 5 . 6 , 4 9 . 9 ] } }$   $3 4 . 3 _ { [ 3 1 . 2 , 4 0 . 5 ] } ^ { - }$   $5 9 . 6 \dot { _ { [ 5 4 . 0 , 6 5 . 2 ] } }$   $R a { - } S F T$   $4 5 . 3 _ { [ 4 3 . 0 , 4 7 . 4 ] }$   $6 . 7 _ { [ 4 . 5 , 9 . 0 ] }$   $4 2 . 8 _ { [ 3 6 . 8 , 4 8 . 8 ] }$ </td></tr><tr><td colspan="4">ORPO  $2 1 . 8 _ { [ 2 0 . 1 , 2 3 . 6 ] }$   $1 . 2 _ { [ 0 . 2 , 2 . 4 ] }$   $3 4 . 4 _ { [ 2 8 . 8 , 4 0 . 0 ] } ^ { - }$  Gemma-2-9B base  $5 9 . 1 _ { [ 5 6 . 9 , 6 1 . 2 ] }$   $7 2 . 1 _ { [ 6 7 . 9 , 7 6 . 4 ] }$   $0 . 4 _ { [ 0 . 0 , 1 . 2 ] }$   $4 4 . 2 _ { [ 4 1 . 8 , 4 6 . 0 ] } ^ { - }$ </td></tr><tr><td colspan="2">SFT  $R a { - } S F T$   $3 8 . 6 _ { [ 3 6 . 5 , 4 0 . 4 ] } ^ { - }$  ORPO  $3 . 4 _ { [ 2 . 6 , 4 . 2 ] }$  Qwen3-8B</td><td> $3 1 . 0 _ { [ 2 6 . 4 , 3 5 . 2 ] } ^ { - }$   $7 . 9 _ { [ 5 . 5 , 1 0 . 5 ] }$   $0 . 0 _ { [ 0 . 0 , 0 . 0 ] }$ </td><td> $3 6 . 0 \dot { _ { [ 3 0 . 4 , 4 1 . 6 ] } }$   $0 . 8 _ { [ 0 . 0 , 2 . 0 ] }$   $3 1 . 6 _ { [ 2 5 . 2 , 3 7 . 6 ] } ^ { - }$   $1 7 . 2 _ { [ 1 2 . 8 , 2 2 . 0 ] }$ </td></tr><tr><td colspan="4">base  $5 1 . 2 _ { [ 4 9 . 1 , 5 3 . 4 ] }$   $3 3 . 3 _ { [ 2 8 . 8 , 3 7 . 9 ] }$  SFT  $3 5 . 3 _ { [ 3 3 . 2 , 3 7 . 0 ] }$   $7 . 4 _ { [ 5 . 0 , 1 0 . 0 ] }$   $6 2 . 4 _ { [ 5 6 . 8 , 6 8 . 0 ] }$   $R a { - } S F T$   $3 6 . 6 _ { [ 3 4 . 5 , 3 8 . 7 ] }$   $4 . 3 _ { [ 2 . 6 , 6 . 4 ] }$   $2 4 . 0 \dot { _ { [ 1 8 . 4 , 2 9 . 6 ] } }$ </td></tr></table>

Table 1: Safety alignment profile of Llama-3.1-8B, Gemma-2-9B and Qwen3-8B across training objectives with bootstrap resampling across 1000 iterations. 95% confidence intervals are in subscript brackets. $\mathbf { W } \mathbf { J } =$ WildJailbreak, $\mathrm { S R } \ = \ \mathrm { S t r o n g R E J E C T } .$ $\mathrm { X S } = \mathrm { X S T e s t . }$ Lower ASR and ORR indicate stronger safety alignment.

From table 1, one clear pattern emerges: preference optimization (ORPO) has a greater impact in strengthening safety alignment than reasoning augmentation to SFT, with a lower ASR in both Wild-Jailbreak and StrongREJECT (except for Qwen3- 8B, with no significant differences in WildJailbreak ASR between SFT and Ra-SFT). This is particularly clear in Gemma-2-9B, with ASR in StrongRE-JECT for ORPO, Ra-SFT and SFT being 0.0%, 7.9% and 31.6%, respectively. However in exchange to such a low ASR, Gemma ORPO has

![](images/332acd674edaf054b94661513fdf9c9f75f46f0786ef349982d15d331904995e.jpg)

Figure 2: Normalized refusal direction magnitude of Llama-3.1-8B, Gemma-2-9B and Qwen3-8B across training objectives by layers.

its own failure mode of over-refusal, with 31.6% ORR in XSTest (despite being lower than Gemma SFT).

## 4.3 Training Objectives Reshape Refusal Geometry

As established in Section 3.2, we calculate pairwise cosine similarity of refusal direction vectors across our three training objectives, in all three models.

What holds across all three architectures (objective-dependent). Two patterns replicate in Llama-3.1-8B, Gemma-2-9B, and Qwen-3-8B alike. First, post-training pushes each objective's refusal direction away from the base model and away from the other two objectives – training objective, not just training data, shapes refusal geometry. Second, SFT and ORPO remain more similar to each other than either is to Ra-SFT, consistently across all three models. This suggests reasoning-chain supervision installs a qualitatively distinct pathway that neither imitation-based SFT nor preference-based ORPO produces on its own – an effect of the reasoning-supervision axis specifically, not of any one architecture.

Where architectures diverge. The three models differ in exactly which objectives reconverge in mid-network layers before diverging again late. In Gemma-2-9B, all three objectives’ refusal directions partially reconverge in layers 13–21, with SFT and ORPO then diverging only slowly through late layers (cosine similarity falling from 0.8 to 0.72). In Qwen-3-8B, this reconvergence is narrower: only SFT and ORPO reconverge in midlayers; Ra-SFT does not join them, remaining diverged throughout. Llama-3.1-8B shows the weakest reconvergence of the three (full comparison in Appendix G). One plausible source of this difference is that Gemma-2-9B, unlike Llama-3.1-8B and Qwen-3-8B, is trained via knowledge distillation from a larger teacher model – distillationshaped representations may retain more shared early-to-mid-layer structure across objectives than models trained without it, though we do not test this directly and flag it as a hypothesis for future work.

Normalized refusal direction magnitude shows the same objective-vs-architecture split.

Objective-dependent: in Llama-3.1-8B and Qwen-3-8B, SFT and ORPO both peak in midlayers (22–27 and 22–30 respectively) before declining, while Ra-SFT's magnitude rises only gradually across the network in both models, never sharply peaking. This is the clearest crossarchitecture signature of the reasoning-supervision axis: Ra-SFT distributes rather than concentrates its refusal-relevant magnitude, in contrast to both other objectives.

Architecture-dependent: Gemma-2-9B breaks from this pattern for ORPO specifically, which overshoots in late layers rather than peaking midnetwork as it does in the other two models. We suspect this reflects an interaction between ORPO's unconstrained odds-ratio gradient and Gemma-2- 9B's representation geometry – plausibly shaped by distillation – concentrating gradient updates into the available direction at high magnitude - though it is just an assumption.

As for Qwen3-8B, its SFT and ORPO variants have their refusal direction magnitude peaked in mid-layers (layers 22-30), before declining. However Ra-SFT only have their refusal direction gradually rising throughout the network. This further cements our idea that reasoning supervision pushing refusal directions in a different way comparing to preference methods.

The refusal direction magnitude in each layer for our three models is characterized in Figure 2.

![](images/7e56a9bf0b7b5ffc9282c735fa76a8cecfa709d6f25d936f33011fe764dab35f.jpg)  
Figure 3: Normalized causal effects of residual-stream activation patching across layers for Llama-3.1-8B, Gemma-2-9B and Qwen3-8B under three post-training methods: SFT, Ra-SFT, and ORPO. Higher values indicate stronger causal contribution to the target refusal-related behavior.

## 4.4 How Circuit Topology Differs by Training Objectives

As mentioned in Section 3.3, we first conduct Activation Patching in all layers of each of our six checkpoints to calculate causal effects of each layer. The result is shown in Figure 3.

All training objectives show peak causal effects at late layers in all three models, but reach that peak differently. All conditions except Ra-SFT in Gemma-2-9B show positive causal effects already at the initial layer, suggesting harmfulness concepts are embedded during pre-training – consistent with Du et al. (2025)'s finding that post-training inherits knowledge representations from the base model. In Llama-3.1-8B, SFT and ORPO follow a similar trajectory: effects bottom out at layer 12, then rise gradually through late layers. This pattern is consistent with early layers encoding token-level harm semantics and recognition, while later layers – from layer 16 onward – carry refusal execution, paralleling Ge et al. (2026)’s finding that pre-training installs more complex features progressively in later stages.

Component-level structure. Across Llama-3.1- 8B, we observe a systematic shift from attentionhead dominance to MLP dominance along the SFT→Ra-SFT→ORPO progression: head 25 carries the dominant causal effect under SFT (—0.33 at layer 30), MLP layer 31 dominates under Ra-SFT (+0.90), and ORPO shows a diffuse MLP distribution. Reasoning-chain supervision appears to shift causal weight toward MLPs, counteracting head 25's broadly suppressive effect on refusal – an effect present under all three objectives, but weakening as preference optimization or reasoning supervision is added (—0.33, -0.20, -0.15 for SFT, Ra-SFT, and ORPO respectively). This is consistent with Huang et al. (2026)'s finding that safety alignment concentrates in a small number of attention heads, and may explain why ASR declines as preference or reasoning signal is added to the fine-tuning pipeline.

In Gemma-2-9B, SFT and ORPO both show uniform, redundant encoding (+0.24-+0.47 across all components) – consistent with the high XSTest over-refusal both objectives produce, as an overconstrained circuit has no low-effect components left to spare. Ra-SFT breaks this pattern with an uneven, MLP-dominated structure (layer 39 MLP -0.44; layer 37 MLP +0.25) alongside uniformly small attention-head effects (—0.09 to +0.08) – structurally closer to Llama's Ra-SFT circuit than to Gemma's own SFT or ORPO, suggesting reasoning-chain supervision installs a qualitatively distinct circuit type largely independent of architecture.

Qwen-3-8B shows MLP-dominant circuits under all three objectives. Top-layer MLPs promote refusal under both SFT and ORPO, while under Ra-SFT, layer 31's MLP strongly suppresses it (—0.36). Attention heads play a comparatively minor role throughout: even ORPO's highest-effect head (head 11, layer 31, +0.19) is outweighed by layer 32's MLP (+0.48).

The attention heads/MLPs causal effects for all training objectives for each model are available in Appendix H. Bootstrap resampling (1000 iterations) tests these rankings' stability. Ra-SFT's ranking is most stable in both Gemma-2-9B $( \rho = 0 . 9 3 )$ and Llama-3.1-8B (ρ = 0.81), against less stable SFT/ORPO rankings (ρ = 0.45/0.39 and 0.75/0.63 respectively). Qwen-3-8B breaks this pattern, with uniformly high stability across all three objectives $( \rho = 0 . 7 9 \ – 0 . 9 0 )$ . Top-ranked components' confidence intervals stay non-overlapping with zero throughout, regardless of ranking stability. Full results in Appendix I.

![](images/395396e7b74b1a1e072d5c2acf52b7bb7ff90893a87a69474c179f5b03ae0532.jpg)  
Figure 4: ActAdd effects for Gemma-2-9B SFT and Ra-SFT when steering at peak recognition layers versus peak execution layers.

## 4.5 Circuit Topology Reshapes Steering and Attack Vulnerability

ActAdd. Table 2 shows that applying ActAdd to Llama-3.1-8B collapses MMLU accuracy rapidly at small α, across all post-training conditions. This reflects partial overlap between safety and utility representations in Llama, particularly in MLPs, which the model also relies on for factual recall (Geva et al., 2021): steering the whole residual stream perturbs the downstream MLP input enough to collapse coherence.

<table><tr><td>ActAdd scale</td><td>SFT</td><td>Ra-SFT</td><td>ORPO</td></tr><tr><td>α = 0</td><td>35.5</td><td>31.5</td><td>28.5</td></tr><tr><td>α = 5</td><td>16.5</td><td>24.5</td><td>11.5</td></tr><tr><td>α = 10</td><td>0.0</td><td>4.0</td><td>0.0</td></tr></table>

Table 2: MMLU subset accuracy rate (%) under ActAdd steering for Llama-3.1-8B across post-training objectives. Lower accuracy rate indicates utility reduction.

For Gemma-2-9B and Qwen3-8B, we apply ActAdd at two positions: the top-5 layers by normalized direction magnitude (Section 4.3, peak recognition layers) and the top-5 layers by causal effect from activation patching (peak execution layers) – following the recognition-execution framework of Wu et al. (2026).

In Gemma-2-9B (Figure 4), steering Ra-SFT toward refusal at recognition layers (22–26) reduces ASR by 18.8pp at α = 20, versus only 5.8pp at execution layers (27, 28, 37, 39, 40). SFT shows the same direction but a smaller gap: 8pp at recognition layers (27–31) versus 4.6pp at execution layers (37–41). In Qwen3-8B, SFT shows a larger and widening gap as α increases – recognition-layer steering cuts WildJailbreak ASR by 28.2pp versus 10.8pp at execution layers – while ORPO's gap narrows as α rises. Together these results suggest:

(1) recognition-layer steering is consistently more effective than execution-layer steering;

(2) the recognition-execution gap reflects not just layer position but how each layer engages the refusal circuit – reasoning chains in Gemma Ra-SFT (Appendix F) may amplify the steering vector's effect on the residual stream. This effect is also architecture-dependent - given that Qwen Ra-SFT peak recognition layers are co-located with peak execution layers (from figure 2 and 3 - Ra-SFT ends up with no clear recognition-execution gaps across layers, unlike Gemma).

For ORPO in Gemma, where the failure mode is high ORR from an over-constrained refusal circuit (Table 1), we steer away from refusal instead: α = 0 to 20 reduces ORR from 31.2% to 20.4% (11.2pp) with only marginal ASR increase, likely because ORPO's overshot refusal-direction magnitude (Figure 2) causes recognition and execution layers to overlap.

MMLU accuracy stays stable throughout ActAdd steering in Gemma-2-9B (53–55% SFT, 40– 44% Ra-SFT, 45–47% ORPO), indicating orthogonal safety and utility representations. Qwen3-8B sits between the two: MMLU stays relatively stable under execution-layer steering but collapses under recognition-layer steering, making it an intermediate case between Llama and Gemma. Full ActAdd results appear in Appendix J.

![](images/9fdd798feebccf0ebb55f955d2c0ee4044f2a3e0264d24898a7395aca0cb9e91.jpg)  
Figure 5: ActAdd effects for Qwen3-8B SFT and ORPO when steering at peak recognition layers versus peak execution layers.

ITI. We perform experiments on head 25 layer 30 for SFT in Llama-3.1-8B; for Gemma we steer head 14 layer 40 in SFT and head 12 layer 40 in ORPO, since these attention heads are also the ones with strongest effects in either suppressing or promoting refusal, from figures 16 and 18. Given that in Qwen3-8B attention heads only have insignicant effects on refusal, ITI would be unsuitable for this model as an intervention tool.

ITI in Gemma-2-9B does not improve model performance linearly: figure 6 shows that ORPO actually increases ORR compared to ActAdd in a similar way α, while SFT shows little reduction in ASR when steering with ITI. The failure of Gemma SFT is consistent with the Hydra effect (McGrath et al., 2023): steering one attention head with a vector leads to responses from all other components, because the causal effects of all other attention heads are roughly similar to each other (see Section 4.4). In the case of ORPO, steering in one head actually pushes the head representation out of the refusal activation space, leading to capability degradation. This can be attributed to an overconstrained refusal circuit with low causal effects (from figure 3).

ITI on Llama SFT dominant suppressive head instead demonstrates two distinct failure modes: single-token loops and question-repetition at α = 20. Therefore, reliable ASR measurement is precluded by hook' destabilizing effects on generation in Llama. The full discussion of ITI results is available in appendix K.

![](images/1e8040ddafa0ab8da9096da02af19a425d2c9540a5fdc295a2229208ad35d482.jpg)  
Figure 6: Comparison of steering effects between ActAdd and ITI, applying to over-refusal in ORPO for Gemma-2-9B.

All of these findings show that single-component head-level ITI is not sufficient for meaningful safety steering across all circuit types: concentrated suppressive (Llama SFT), distributed promotive (Gemma SFT) and distributed uniform (Gemma ORPO).

Attack Class Vulnerability. Beyond steering intervention, the circuit structure from section 4.4 is consistent with specific vulnerability patterns that StrongREJECT analysis confirms: SFT in Llama-3.1-8B and Gemma-2-9B is especially vulnerable to semantic attacks (happy\_to\_help, DAN, wikipedia, role\_play) due to concentration of causal effects in attention heads, which are lexically sensitive (Jo and Myaeng, 2020; Ji et al., 2025); while Ra-SFT and ORPO are safer with this type of attacks due to attention heads playing a lower role, and in case of ORPO in Gemma, due to a distributed uniform circuit. Meanwhile given refusal circuits in Qwen3-8B is concentrated in MLP - this leads to this model being less vulnerable to semantic attacks, at the expense of persistent vulnerability to encoding attacks. The full heatmap for attack classes vulnerability of post-training methods is presented in Appendix L.

## 4.6 Safety Alignment Trilemma and Operational Implications

From our experiments above and previous work (Kazemi et al., 2026; Huang et al., 2026), we demonstrate that safety alignment of LLMs should satisfy three criteria:

(1) Distributed refusal encoding: refusal circuits should not be over-concentrated in any component types;

(2) Safety/utility separability: steering towards safety should preserve model capability;

(3) Granular correctability: safety behavior should be correctable through localized interventions, such as component-level edits.

Our results in sections 4.3-4.5 suggest that among our architectures and training objectives that we evaluate, none of them can satisfy all three conditions above without incurring operational costs: reasoning-augmented methods like Ra-SFT can be partially correctable (from steering results in section 4.5 but they produce deliberation overheads that increase inference costs (see Appendix F), while preference-optimization methods like ORPO produce over-refusal (31.6% ORR for XSTest prompts) that fails separability and renders them unsuitable for security-adjacent applications. This problem is further compounded by task dependencies of over-refusal directions (Maskey et al., 2026), further complicating correction of over-refusal.

Therefore, we advise caution against treating alignment as a binary property of post-training methods, since the circuit structures characterized here predict vulnerability profiles that standard behavioral evaluation cannot detect.

## 5 Discussion

Refusal Representations in Post-Training Methods. While our DIM approach is effective in characterizing refusal geometry differences between training objectives, it can only be an approximation for actual refusal geometry of LLMs, since Wollschläger et al. (2025) considers refusal in LLMs as a polyhedral cone, with DIM separation as the cone axis and multiple independent directions, with similar findings from Pan et al. (2025) and Joad et al. (2026). Given aforementioned findings, we show in our steering experiments in section 4.5 that DIM as approximation of refusal geometry can fail (Gemma ORPO's Hydra effect, Llama's collapse of accuracy rate in MMLU) or be successful (stable MMLU performance for Gemma), as such the limitations of DIM analysis are dependent on training paradigms and architectures.

Steering Effectiveness for Improving Alignment Robustness. It is worth noting that our steering experiments should be considered as a probe to validate whether components from our circuit analysis are mediating refusal. Many new methods have been developed to address linear steering' failure modes: Sheng et al. (2026) introduces a linear steering method with null-space for utility data alongside refusal direction vector, while Gadgil et al. (2026) selects intervention layers by mapping from input embeddings to optimal steering layers. However, these works only focus on fixed instruct models, and as such our contribution to this line of work is by showing that training objectives reshape steering success or failure through circuit analysis (recognition/execution gaps in Gemma Ra-SFT being consistent with successful ASR reduction compared to failure in ITI for Gemma ORPO due to overconstrained circuits).

## 6 Conclusions and Future Work

In this work, we conducted a cross-paradigm circuit analysis of post-training methods. By characterizing training objectives using preference-reasoning axes, we found that post-training methods can reshape geometric and circuit structures of LLMs, alongside steering robustness and attack class vulnerability. We also observe an alignment trilemma among our training objectives: no offline methods that we study can satisfy distributed refusal encoding, safety/utility separability and granular correctability. Future work should focus on understanding the temporal evolution of refusal concepts as well as other alignment criteria (Naseem et al., 2026) between training objectives, alongside studying how preference and reasoning interact together in post-training.

## Limitations

While our activation patching and attribution patching methods have been proven as highly effective in finding components promoting/suppressing refusals across post-training objectives, it is worth noting that our experiments are done with single level per architecture at 8B-9B level, since adding another model scale without matched architectural comparison would introduce other confounding factors. Therefore generalizability of our findings to models with larger scales remains an open question, given that circuit analysis literature about scale consistency tops out around 2.8B with mixed results (Tigges et al., 2024).

We did not test how reasoning and preference would interact together in safety alignment as well - this would require constructing reasoning chains for rejected (unsafe) responses for BeaverTails prompts using large reasoning models, similar to how Hu et al. (2026) generated reasoning chains for Alpaca alongside safe responses of BeaverTails using GPT-4o, which is not a trivial task given that LRMs/LLMs with stronger safety alignment can reject many of our harmful reasoning chain generation requests. Given reasoning chains are generated through one-shot prompting of GPT-4o, this complicates further controlled experiments for Ra-SFT regarding reasoning chains length and style. Also, the choice of BeaverTails as safety-specific data for post-training introduces potential bias from annotators about what counts as safe/harmful answers.

It is also worth noting that despite our efforts to keep our pipeline deterministic - using greedy decoding, control our fine-tuning and inference runs with same seed - our classification of borderline harmful prompts is not fully stable across runs (18/256 harmless prompts (7%) being flipped from refusal to compliance across two runs of the same Gemma ORPO checkpoint under matched precision, hardware and library version). This can also affect downstream normalized effect estimates and should be treated as a source of run-to-run variance.

## Ethics Statement

While this paper characterizes how current safety post-training methods work as jailbreak defense mechanisms, we acknowledge this work can induce offensive contents and our analysis about how current post-training methods fail can be exploited for misuse. It is crucial to emphasize that the primary goal of this work is to advance research in post-training alignment methods and to improve the robustness of LLMs against harmful content. We strongly encourage further research in this area to foster the development of more secure and ethically aligned generative models. All analysis and datasets utilized in this paper are strictly intended for research purposes under the ethical guidelines of the research community. The authors unequivocally condemn any misuse of this work to generate or disseminate harmful content.

## Acknowledgements

This research was supported by the International Macquarie University Research Excellence Scholarship (“iMQRES MRES").

## References

Anthropic Threat Intelligence. 2025. Disrupting the first reported AI-orchestrated cyber espionage campaign. Technical report, Anthropic, San Francisco, CA, USA.

Andy Arditi, Oscar Balcells Obeso, Aaquib Syed, Daniel Paleka, Nina Rimsky, Wes Gurnee, and Neel Nanda. 2024. Refusal in language models is mediated by a single direction. In Proceedings of the Thirty-eighth Annual Conference on Neural Information Processing Systems, NeurIPS '24, Vancouver, Canada.

Stella Biderman, Mohammad Aflah Khan, Niloofar Mireshghallah, Catherine Arnett, Fazl Barez, and Naomi Saphra. 2026. Position: Don't just "fix it in post": A science of AI must study training dynamics. In Proceedings of the Forty-third International Conference on Machine Learning Position Paper Track, ICML '26, Seoul, Republic of Korea.

Joschka Braun, Carsten Eickhoff, David Krueger, Seyed Ali Bahrainian, and Dmitrii Krasheninnikov. 2025. Understanding (un)reliability of steering vectors in language models. In Proceedings of the ICLR 2025 Workshop on Building Trust in Language Models and Applications, ICLR '25, Singapore.

David Campbell, Neil Kale, Udari Madhushani Sehwag, Bert Herring, Nick Price, Dan Borges, Alex Levinson, and Christina Q Knight. 2026. Defensive refusal bias: How safety alignment fails cyber defenders. In Proceedings of the ICLR 2026 Workshop of Agents in the Wild: Safety, Security, and Beyond, ICLR '26, Rio de Janeiro, Brazil.

Zouying Cao, Yifei Yang, and Hai Zhao. 2025. SCANS: mitigating the exaggerated safety for LLMs via safety-conscious activation steering. In Proceedings of the Thirty-ninth AAAI Conference on Artificial Intelligence, AAAI '25, pages 23523–23531, Philadelphia, PA, USA.

Jianhui Chen, Xiaozhi Wang, Zijun Yao, Yushi Bai, Lei Hou, and Juanzi Li. 2025. Towards understanding safety alignment: A mechanistic perspective from safety neurons. In Proceedings of the Thirty-ninth Annual Conference on Neural Information Processing Systems, NeurIPS '25, San Diego, CA, USA.

CISA, ASD's ACSC, NSA AISC, FBI, Canadian Centre for Cyber Security, BSI, NCSC-NL, NCSC-NZ, and NCSC-UK. 2025. Principles for the secure integration of artificial intelligence in operational technology. Joint guidance, U.S. Cybersecurity and Infrastructure Security Agency, Washington, D.C., USA.

Hongzhe Du, Weikai Li, Min Cai, Karim Saraipour, Zimin Zhang, Himabindu Lakkaraju, Yizhou Sun, and Shichang Zhang. 2025. How post-training reshapes LLMs: A mechanistic view on knowledge, truthfulness, refusal, and confidence. In Proceedings of the Second Conference on Language Modeling, CoLM '25, Montreal, Canada.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Chris Olah. 2022. Toy models of superposition. Transformer Circuits Thread.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. 2021. A mathematical framework of transformer circuits. Transformer Circuits Thread.

Federal Office for Information Security (BSI). 2025. Evasion attacks on LLMs – countermeasures in practice. Technical report, Bundesamt für Sicherheit in der Informationstechnik, Bonn, Germany.

Soham Gadgil, Chris Lin, and Su-In Lee. 2026. Where to steer: Input-dependent layer selection for steering improves LLM alignment. ArXiv preprint, arXiv:2604.03867.

Xuyang Ge, Wentao Shu, Jiaxing Wu, Yunhua Zhou, Zhengfu He, and Xipeng Qiu. 2026. Evolution of concepts in language model pre-training. In Proceedings of the Fourteenth International Conference on Learning Representations, ICLR ’26, Rio de Janeiro, Brazil.

Gemma Team. 2024. Gemma 2: Improving open language models at a practical size. arXiv preprint, arXiv:2408.00118.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2021. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021

Conference on Empirical Methods in Natural Language Processing, EMNLP '2021, pages 5484–5495, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, et al. 2024. The Llama 3 herd of models. arXiv preprint, arXiv:2407.21783.

Rajdeep Haldar, Ziyi Wang, Guang Lin, Yue Xing, and Qifan Song. 2025. LLM safety alignment is divergence estimation in disguise. In Proceedings of the Thirty-ninth Annual Conference on Neural Information Processing Systems, NeurIPS ’25, San Diego, CA, USA.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In Proceedings of the Ninth International Conference on Learning Representations, ICLR ’21, Vienna, Austria.

Jiwoo Hong, Noah Lee, and James Thorne. 2024. ORPO: Monolithic preference optimization without reference model. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP '2024, pages 11170–11189, Miami, Florida, USA. Association for Computational Linguistics.

Mengxuan Hu, Vivek Datla, Anoop Kumar, Zihan Guan, Sheng Li, Alfy Samuel, and Daben Liu. 2026. Alignment-weighted DPO: A principled reasoning approach to improve safety alignment. In Proceedings of the Thirteen International Conference on Learning Representations, ICLR ’26, Rio de Janeiro, Brazil.

Chao Huang, Zefeng Zhang, Juwei Yue, Jiawei Sheng Sheng, Li Quangang, Chuang Zhang, and Tingwen Liu. 2026. Safety alignment should be made more than just a few attention heads. In Proceedings of the 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), ICASSP '26, pages 1656–1660, Barcelona, Spain.

Samyak Jain, Ekdeep Singh Lubana, Kemal Oksuz, Tom Joy, Philip Torr, Amartya Sanyal, and Puneet K. Dokania. 2024. What makes and breaks safety finetuning? a mechanistic study. In Proceedings of the Thirty-eighth Annual Conference on Neural Information Processing Systems, NeurIPS ’24, Vancouver, Canada.

Denis Janiak, Julia Moska, Dawid Motyka, Karolina Seweryn, Paweł Walkowiak, Bartosz Żuk, and Arkadiusz Janz. 2026. Rethinking the evaluation of alignment methods: Insights into diversity, generalisation, and safety. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 4: Student Research Workshop), EACL '2026, pages 92–109, Rabat, Morocco. Association for Computational Linguistics.

Jiabao Ji, Bairu Hou, Alexander Robey, George J. Pappas, Hamed Hassani, Yang Zhang, Eric Wong, and Shiyu Chang. 2025. Defending large language models against jailbreak attacks via semantic smoothing. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, IJCNLP-AACL '2025, pages 7–40, Mumbai, India. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics.

Jiaming Ji, Mickel Liu, Juntao Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. 2023. BeaverTails: Towards improved safety alignment of LLM via a human-preference dataset. In Proceedings of the Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, NeurIPS '23, New Orleans, LA, USA.

Fengqing Jiang, Zhangchen Xu, Yuetai Li, Luyao Niu, Zhen Xiang, Bo Li, Bill Yuchen Lin, and Radha Poovendran. 2025. SafeChain: Safety of language models with long chain-of-thought reasoning capabilities. In Findings of the Association for Computational Linguistics: ACL 2025, ACL ’2025, pages 23303–23320, Vienna, Austria. Association for Computational Linguistics.

Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, and Nouha Dziri. 2024. WildTeaming at scale: From in-thewild jailbreaks to (adversarially) safer language models. In Proceedings of the Thirty-eighth Annual Conference on Neural Information Processing Systems, NeurIPS '24, Vancouver, Canada.

Jae-young Jo and Sung-Hyon Myaeng. 2020. Roles and utilization of attention heads in transformer-based neural language models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL '2020, pages 3404–3417, Online. Association for Computational Linguistics.

Faaiz Joad, Majd Hawasly, Sabri Boughorbel, Nadir Durrani, and Husrev Taha Sencar. 2026. There is more to refusal in large language models than a single direction. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, EMNLP '2026, Budapest, Hungary. Association for Computational Linguistics.

Hamid Kazemi, Atoosa Chegini, and Maria Safi. 2026. A single neuron is sufficient to bypass safety alignment in large language models. ArXiv preprint, arXiv:2605.08513.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inference-time intervention: Eliciting truthful answers from a language model. In Proceedings of the Thirty-seventh Conference on Neural Information Processing Systems, NeurIPS '23, New Orleans, LA, USA.

Samuel Marks and Max Tegmark. 2024. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In Proceedings of the First Conference on Language Modeling, CoLM '24, Philadelphia, PA, USA.

Utsav Maskey, Mark Dras, and Usman Naseem. 2026. Over-refusal and representation subspaces: A mechanistic analysis of task-conditioned refusal in aligned LLMs. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing, EMNLP '2026, Budapest, Hungary. Association for Computational Linguistics.

Thomas McGrath, Matthew Rahtz, Janos Kramar, Vladimir Mikulik, and Shane Legg. 2023. The hydra effect: Emergent self-repair in language model computations. ArXiv preprint, arXiv:2307.15771.

Kevin Meng, David Bau, Alex J Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In Proceedings of the Thirty-sixth Annual Conference on the Advances in Neural Information Processing Systems, NeurIPS '22, New Orleans, LA, USA.

Aayush Mishra, Daniel Khashabi, and Anqi Liu. 2026. Steered LLM activations are non-surjective. In Proceedings of the ICLR 2026 Workshop on Scientific Methods for Understanding Deep Learning, ICLR '26, Rio de Janeiro, Brazil.

Usman Naseem. 2026. Mechanistic interpretability for large language model alignment: Progress, challenges, and future directions. ArXiv preprint arXiv:2602.11180.

Usman Naseem, Tanmoy Chakraborty, Kai-Wei Chang, Mark Dras, Preslav Nakov, Nanyun Peng, and Soujanya Poria. 2026. LLM alignment should go beyond harmlessness-helpfulness and incorporate human agency. Cognitive Computation, 18(1).

Merve Noyan, Pedro Cuenca, Sergio Paniego, Ben Burtenshaw, Steven Zheng, Alvaro Bartolome, and Nathan Habib. 2026. Welcome Gemma 4: Frontier multimodal intelligence on device. https: //huggingface.co/blog/gemma4. HuggingFace Blog. Accessed: 2026-05-24.

Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. 2020. Zoom in: An introduction to circuits. Distill, 5(3):e00024– 001.

Wenbo Pan, Zhichao Liu, Qiguang Chen, Xiangyang Zhou, Yu Haining, and Xiaohua Jia. 2025. The hidden dimensions of LLM alignment: A multidimensional analysis of orthogonal safety directions. In Proceedings of the Forty-second International Conference on Machine Learning, ICML’25, Vancouver, Canada.

Yein Park, Minbyul Jeong, and Jaewoo Kang. 2025. Thinking sparks!: Emergent attention heads in reasoning models during post training. ArXiv preprint, arXiv:2509.25758.

Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, and Peter Henderson. 2025. Safety alignment should be made more than just a few tokens deep. In Proceedings of the Thirteenth International Conference on Learning Representations, ICLR ’25, Singapore.

Qwen Team. 2025. Qwen3 technical report. ArXiv preprint, arXiv:2505.09388.

Shauli Ravfogel, Gilad Yehudai, Tal Linzen, Joan Bruna, and Alberto Bietti. 2025. Emergence of linear truth encodings in language models. In Proceedings of the Thirty-ninth Annual Conference on Neural Information Processing Systems, NeurIPS’25, San Diego, CA, USA.

Paul Röttger, Hannah Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. 2024. XSTest: A test suite for identifying exaggerated safety behaviours in large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL '2024, pages 5377–5400, Mexico City, Mexico. Association for Computational Linguistics.

Eyal Sela. 2026. The AI-assisted breach of Mexico's government infrastructure. Technical report, Gambit Security, New York, NY, USA.

Leheng Sheng, Changshuo Shen, Weixiang Zhao, Junfeng Fang, Xiaohao Liu, Zhenkai Liang, Xiang Wang, An Zhang, and Tat-Seng Chua. 2026. Alphasteer: Learning refusal steering with principled null-space constraint. In Proceedings of the Fourteenth International Conference on Learning Representations, ICLR '26, Rio de Janeiro, Brazil.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, et al. 2026. OpenAI GPT-5 system card. ArXiv preprint, arXiv:2601.03267.

Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, and Sam Toyer. 2024. A StrongREJECT for empty jailbreaks. In Proceedings of the Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, NeurIPS '24, Vancouver, Canada.

Aaquib Syed, Can Rager, and Arthur Conmy. 2024. Attribution patching outperforms automated circuit discovery. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, EMNLP '2024, pages 407–416, Miami, Florida, US. Association for Computational Linguistics.

Daniel Chee Hian Tan, David Chanin, Aengus Lynch, Brooks Paige, Dimitrios Kanoulas, Adrià Garriga-Alonso, and Robert Kirk. 2024. Analysing the generalisation and reliability of steering vectors. In Proceedings of the Thirty-eighth Annual Conference on

Neural Information Processing Systems, NeurIPS ’24, Vancouver, Canada.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https: //gi thub.com/tatsu-lab/stanford\_alpaca.

Megh Thakkar, Quentin Fournier, Matthew Riemer, Pin-Yu Chen, Amal Zouaq, Payel Das, and Sarath Chandar. 2025. Combining domain and alignment vectors provides better knowledge-safety trade-offs in LLMs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), ACL ’2025, pages 268–277, Vienna, Austria. Association for Computational Linguistics.

Curt Tigges, Michael Hanna, Qinan Yu, and Stella Biderman. 2024. LLM circuit analyses are consistent across training and scale. In Proceedings of the Thirty-eighth Annual Conference on Neural Information Processing Systems, NeurIPS ’24, Vancouver, Canada.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Steering language models with activation engineering. ArXiv preprint, arXiv:2308.10248.

Daniel Vennemeyer, Punya Syon Pandey, Phan Anh Duong, Michael Umeokoli, and Samuel Ratnam. 2026. Objective matters: Fine-tuning objectives shape safety, robustness, and persona drift. ArXiv preprint, arXiv:2601.12639.

Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. 2023. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In Proceedings of the Eleventh International Conference on Learning Representations, ICLR '23, Kigali, Rwanda.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. 2023. Jailbroken: How does LLM safety training fail? In Proceedings of the Thirty-seventh Conference on Neural Information Processing Systems, NeurIPS '23, New Orleans, LA, USA.

Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, and Johannes Gasteiger. 2025. The geometry of refusal in large language models: Concept cones and representational independence. In Proceedings of the Forty-second International Conference on Machine Learning, ICML '25, Vienna, Austria.

Jinman Wu, Yi Xie, Shen Lin, Shiqian Zhao, and Xiaofeng Chen. 2026. Knowing without acting: The disentangled geometry of safety mechanisms in large language models. ArXiv preprint, arXiv:2603.05773.

Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang You.

2024. OpenMoE: An early effort on open mixtureof-experts language models. ArXiv preprint arXiv:2402.01739.

Wei Jie Yeo, Nirmalendu Prakash, Clement Neo, Ranjan Satapathy, Roy Ka-Wei Lee, and Erik Cambria. 2025. Understanding refusal in language models with sparse autoencoders. In Findings of the Association for Computational Linguistics: EMNLP 2025, EMNLP '2025, pages 6377–6399, Suzhou, China. Association for Computational Linguistics.

Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, and Weiyan Shi. 2025. LLMs encode harmfulness and refusal separately. In Proceedings of the Thirtyninth Annual Conference on Neural Information Processing Systems, NeurIPS ’25, San Diego, CA, USA.

Zhenhong Zhou, Haiyang Yu, Xinghua Zhang, Rongwu Xu, Fei Huang, Kun Wang, Yang Liu, Junfeng Fang, and Yongbin Li. 2025. On the role of attention heads in large language model safety. In Proceedings of the Thirteenth International Conference on Learning Representations, ICLR '25, Singapore.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. 2023. Representation engineering: A top-down approach to AI transparency. ArXiv preprint, arXiv:2310.01405.

## A Cross-Paradigm Summary

Table 3 synthesizes the objective-level findings from Sections 4.2–4.6 in compact form.

<table><tr><td>Objective</td><td>Summary</td></tr><tr><td>SFT</td><td>Safety: moderate ASR reduction, high over-refusal (up to 62.4%). Struc- ture: concentrated (dominated by atten- tion heads in Llama); uniform/redundant in Gemma; MLP-dominant in Qwen. Recognition-execution gap: present but narrow. Steering: recognition-layer steer- ing helps; execution-layer weak; ITI causes coherence collapse in Llama.</td></tr><tr><td>Ra-SFT</td><td>Safety: strongest ASR reduction among matched-architecture comparisons; low- moderate over-refusal. Structure: MLP- dominant, uneven - consistent across all three architectures. Recognition- execution gap: present, most exploitable via steering (in Gemma) but collapsed in Qwen. Steering: most correctable via recognition-layer steering; incurs deliberation-overhead cost.</td></tr><tr><td>ORPO</td><td>Safety: strong ASR reduction (architecture-dependent magnitude); highest over-refusal in Gemma (31.6%). Structure: diffuse/uniform in Gemma; MLP-dominant but less redundant in Qwen. Recognition-execution gap: narrow to collapsed (recognition ≈ execu- tion) in terms of layers in Gemma, and has marginal effects in both Gemma/Qwen. Steering: resistant to correction in Gemma; over-constrained circuit limits effectiveness. Qwen shows circuits more amenable to steering.</td></tr></table>

Table 3: Cross-paradigm summary of behavioral safety, circuit structure, recognition-execution gap, and steering outcome. Patterns marked as consistent hold across all three architectures (Llama-3.1-8B, Gemma-2-9B, Qwen3-8B); patterns without this note are architecturedependent - see Sections 4.2-4.6 for architecturespecific figures.

## B Training Details

The training hyperparameters for Llama-3.1-8B and Gemma-2-9B for SFT, Ra-SFT and ORPO are:
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Learning rate</td><td>1e-5</td></tr><tr><td>Batch size</td><td>1</td></tr><tr><td>Gradient accumulation steps Effective batch size</td><td>128 128</td></tr><tr><td>Epochs</td><td>3</td></tr><tr><td>Warmup ratio</td><td>0.1</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Max sequence length (tokens)</td><td>2048</td></tr><tr><td>ORPO-specific</td><td></td></tr><tr><td> $\beta$  (odds ratio penalty)</td><td>0.1</td></tr><tr><td>Max prompt length (tokens)</td><td>1536</td></tr></table>

Table 4: Training hyperparameters, shared across Llama-3.1-8B and Gemma-2-9B. ORPO-specific parameters apply to the ORPO condition only.

All training experiments are conducted using one A100 80GB GPU. Hyperparameters are chosen to be best fit with the training process conducted by Hu et al. (2026), especially in effective batch size, given our training condition of one A100 80GB GPU.

We choose Llama-3.1-8B, Gemma-2-9B and Qwen3-8B because these two are both dense models, with roughly similar size, meaningful difference in architecture for cross-validation (Llama-3.1-8B has 32 layers with 32 attention heads for each layer, compared to 42 layers and 16 attention heads for each in Gemma and 36 layers and 32 attention heads for Qwen).

It should be noted that latest models with approximately similar sizes such as Gemma-4-E4B (Noyan et al., 2026) use Per-Layer Embeddings, where each decoder layer has separate token embeddings processed through gating mechanisms between blocks - creates a non-standard residual stream structure that invalidates the hook-based activation patching methodology.

Also similar models with Mixture-of-Experts (MoE) architecture such as OpenMoE-8B (Xue et al., 2024) would introduce confounding factors to our activation patching at circuits’ components level (MLPs/attention heads), since rather than attributing causal effects to a specific component, the effect might come from one out of many experts at a particular layer.

## C Circuit Analysis Methods Explanations

## C.1 Activation Patching

Let $\mathbf { h } _ { \mathrm { s r c } } ^ { ( l ) }$ and $\mathbf { h } _ { \mathrm { t g t } } ^ { ( l ) }$ denote residual stream activations at layer l under a source (harmful) and target (harmless) prompt respectively. A patched forward pass substitutes the source activation at layer l with the target activation:

$$
\tilde { \mathbf { h } } ^ { ( l ) } = \left\{ \mathbf { h } _ { \mathrm { t g t } } ^ { ( l ) } \quad \mathrm { i f } l = l ^ { * } \right.\tag{3}
$$

The causal effect of layer $l ^ { * }$ is measured by the normalized logit difference:

$$
\Delta ^ { ( l ^ { * } ) } = \frac { \mathcal { L } ( \tilde { \mathbf { h } } ^ { ( l ^ { * } ) } ) - \mathcal { L } ( \mathbf { h } _ { \mathrm { s r c } } ) } { \mathcal { L } ( \mathbf { h } _ { \mathrm { t g t } } ) - \mathcal { L } ( \mathbf { h } _ { \mathrm { s r c } } ) }\tag{4}
$$

where $\mathcal { L } ( \cdot )$ denotes the logit difference between the most probable refusal and compliance tokens. $\Delta ^ { ( l ^ { * } ) } = 1$ indicates the patch fully recovers target behaviour; $\Delta ^ { ( l ^ { * } ) } = 0$ indicates no causal effect.

Positive sign (+) means that ablating this layer/component would harm refusal, while negative sign (—) means that layer/component ablation can promote refusal.

For activation patching, we separate results (refusal/compliance) from the prompt boundary of each model, then we calculate matchings with refusal patterns identified by Arditi et al. (2024) for methodological consistency. For Ra-SFT in Gemma-2-9B and Qwen3-8B, since they consistently produce <think>. . . </think> block for its reasoning before answering both harmful questions in the dataset from Arditi et al. (2024) and harmless questions (see Appendix F), we have to remove this block to recognize refusal/compliance answer pairs.

## C.2 Attribution Patching

The attribution score for layer l can be calculated as first-order Taylor expansion around the source activations (in this case, for components of each layer):

$$
\alpha ^ { ( l ) } = \left( \mathbf { h } _ { \mathrm { t g t } } ^ { ( l ) } - \mathbf { h } _ { \mathrm { s r c } } ^ { ( l ) } \right) \cdot \nabla _ { \mathbf { h } _ { \mathrm { s r c } } ^ { ( l ) } } \mathcal { L }\tag{5}
$$

where $\mathbf { h } _ { \mathrm { s r c } } ^ { ( l ) }$ and $\mathbf { h } _ { \mathrm { t g t } } ^ { ( l ) }$ denote residual stream activations at layer l under harmful and harmless prompts respectively, and let $\mathcal { L }$ denote the logit difference metric defined in Equation 4, $\nabla _ { \mathbf { h } _ { \mathrm { s r c } } ^ { ( l ) } } \mathcal { L }$ is the gradient of the logit difference regarding the source activation at layer l, computed via a single backward pass.

This reduces the cost from $O ( L )$ forward passes to two forward passes and one backward passes, enabling efficient attribution across all layers and components simultaneously.

It is worth noting that attribution patching results are just an approximation - component attribution score might be positive, but actual causal effects could be negative and vice versa (Syed et al., 2024). This indicates:

1) Attribution would show direct effects towards the gradient (suppress or promote refusal), but it could not show indirect effects of how such a component might affect residual stream;

2) The actual causal effects can only be extracted from activation patching (in our scenario, for layers’ components);

3) Flipping signs across many components show that training objectives are reshaping how MLPs interact with refusal in a non-linear way, by promoting some MLPs comparing to gradients while suppressing others (as shown in Ra-SFT in both models at Appendix H).

We choose $K = 5$ for top-K layers with highest causal effects from Section 3.3 to conduct attribution patching, alongside attention heads and MLP patching, since firstly from Section 3.3 these layers would be the most causally relevant for refusal execution by definition; secondly attribution patching at early layers would reflect harmfulness concepts inherited from pre-training being encoded there (Du et al., 2025; Zhao et al., 2025; Ravfogel et al., 2025) rather than where refusal is actually executed; therefore even though causal effects of components in early layers might be non-zero, the results would not be interpretable as affecting refusal and not actionable. We use greedy decoding for both activation patching and attribution patching for reproducibility.

## D Steering Explanations: ActAdd and ITI

## D.1 ActAdd (Activation Addition)

Given a refusal direction $\hat { \mathbf { r } } ^ { ( l ) }$ extracted via difference-in-means, we steer model behaviour at inference time by adding a scaled version of this direction to the residual stream at layer l (Turner et al., 2023):

$$
\tilde { \mathbf { h } } _ { i } ^ { ( l ) } = \mathbf { h } _ { i } ^ { ( l ) } + \boldsymbol { \alpha } \cdot \hat { \mathbf { r } } ^ { ( l ) }\tag{6}
$$

where $\mathbf { h } _ { i } ^ { ( l ) }$ is the residual stream activation at layer l for token position $i , \hat { \mathbf { r } } ^ { ( l ) }$ is the normalized refusal direction, and $\alpha \in \mathbb { R }$ is a scalar coefficient controlling intervention magnitude. Positive α steers toward refusal behaviour; negative α steers toward compliance.

## D.2 ITI (Inference-Time Intervention)

The formula of ITI in attention heads can be described as:

$$
\tilde { \mathbf { h } } _ { i } ^ { ( l , h ) } = \mathbf { h } _ { i } ^ { ( l , h ) } + \boldsymbol { \alpha } ^ { ( l , h ) } \cdot \hat { \mathbf { p } } ^ { ( l , h ) }\tag{7}
$$

where $( l , h )$ indicates layer $\bar { l } ,$ attention head $\bar { h }$ and $\hat { \mathbf { p } } ^ { ( l , h ) }$ denotes per-head probing direction. We used ITI for attention heads only since Elhage et al. (2021) identified that in the residual stream, attention heads are where information - including contexts about harm recognition - is moving between residual streams of different tokens, while MLPs are where factual associations are stored as knowledge-retrieval mechanism (Geva et al., 2021). Therefore, intervention in attention heads would reduce the risks of corrupting factual knowledge representations while ensuring higher accuracy than intervention with MLP-based probes, as validated by Li et al. (2023).

For ITI in attention heads, we first fit a linear probe using DIM calculation, in 256 pairs of harmful/harmless prompts from Arditi et al. (2024), to maintain coherence with geometric analysis in Section 3.3. Then we steer according to the head' activation along the probe direction at inference time, using a hook on the attention head' input slice. We use greedy decoding for both ActAdd and ITI for reproducibility.

## E Additional Details about Over-Refusal Measurements

The refusal patterns used to evaluate over-refusal in XSTest are displayed in table 5. It should be noted that this is not a comprehensive list of refusal patterns that LLMs would use.

Regarding why ASR and over-refusal use different methods for measurements (LlamaGuard-3 for ASR and heuristic string matching for overrefusal): The two instruments reflect a construct difference, not convenience. ASR asks whether response content is harmful , which is LlamaGuard-3's design task; it outperforms comparable safety classifiers such as the OpenAI Moderation API and HarmBench's fine-tuned judge (Jiang et al., 2025). Over-refusal asks a categorically different question:

LlamaGuard-3's content-based classification is not subject to this base-model-specific limitation.

<table><tr><td>Category</td><td>Example pattern</td></tr><tr><td>Direct inability</td><td>i am unable to,i cannot</td></tr><tr><td>Apology-prefixed</td><td>i am sorry, but i can&#x27;t,i apologize</td></tr><tr><td>Request characterisation</td><td>your request is harmful, your prompt is offensive, your request is</td></tr><tr><td>Explicit decline</td><td>inappropriate i must decline, i am going to refuse, i have to refuse</td></tr><tr><td>AI-identity disclaimer</td><td>as an AI, i cannot</td></tr><tr><td>Belief-based hedge Scope disclaimer</td><td>i don&#x27;t think i should this goes beyond my</td></tr><tr><td>Safety framing</td><td>for safety reasons, for</td></tr><tr><td>Comfort/position hedge</td><td>ethical reasons i&#x27;m not comfortable, it&#x27;s</td></tr></table>

Table 5: Taxonomy of refusal pattern categories used in ORR evaluation for XSTest.

whether a response is a refusal, given a benign prompt. A refusal to a benign XSTest prompt is safe content - LlamaGuard-3 would correctly label it "safe" while entirely missing the over-refusal, so it is the wrong instrument for ORR by construction, not merely a weaker one. For refusal detection, string matching against XSTest prompts has been validated directly against human annotation: Cao et al. (2025) hired three independent annotators and found string-match judgments closely approximate human judgments on XSTest and OKTest. String matching also offers better reproducibility than proprietary judges such as GPT-4o.

To validate this problem, we conduct a test of running LlamaGuard-3 with Gemma checkpoints, and the results show that LlamaGuard-3 usually understated over-refusal comparing to heuristic string matching as a baseline, for fine-tuned models (which are the paper' primary focus).

<table><tr><td>Models</td><td>LlamaGuard-3 ORR (%)</td><td>Heuristic ORR (%)</td></tr><tr><td>Gemma base</td><td>3.2</td><td>0.4</td></tr><tr><td>Gemma SFT</td><td>0.0</td><td>36.0</td></tr><tr><td>Gemma Ra-SFT</td><td>0.4</td><td>0.8</td></tr><tr><td>Gemma ORPO</td><td>0.8</td><td>31.6</td></tr></table>

Table 6: XSTest ORR in Gemma checkpoints, measured by LlamaGuard-3 and our heuristic string matching, respectively.

The one exception is the base model, where string-matching (0.4%) reads lower than LlamaGuard-3 (3.2%) - the reverse of the fine-tuned-model pattern. This is consistent with string-matching's dependence on templateconsistent refusal phrasing, which base models - not yet instruction-tuned - do not reliably produce;

## F Reasoning Chain Activation Patterns Across Prompt Types in Ra-SFT Variants

To complement the aggregate safety and utility metrics for Ra-SFT in Gemma-2-9B and Llama-3.1-8B models, we include representative prompt-response examples from four evaluation settings: MMLU for utility, WildJailbreak for harmful instruction-following behaviour, Arditi et al. (2024) dataset for direct harmful requests, and XSTest for benign prompts that can reveal over-refusal.

From figures 7 and 8, it is worth noting that Gemma Ra-SFT consistently externalises a reasoning trace using explicit <think> and </think> tags, including for both harmful and benign prompts. Meanwhile, the Llama Ra-SFT examples do not uniformly expose <think> tags, especially with direct harmful requests from Arditi's dataset; however, the WildJailbreak example shows an explicit reasoning trace that incorrectly classifies the adversarial misinformation-style request as safe before complying.

As for Qwen3-8B, given that it is a reasoning model - we need to enable thinking mode for Ra-SFT fine-tuning, while disable it for SFT and ORPO. Therefore <think> and </think> tags exist for responses for both harmful and benign prompts in Ra-SFT, while they would not be produced for SFT/ORPO, similar to Gemma-2-9B.

![](images/d9487cd84e3e15837c0aeab7260e72bb882cff3dafdbcd80c0794adbb424d8eb.jpg)  
Figure 7: Representative qualitative outputs from Gemma-2-9B Ra-SFT.

![](images/5a5933aa9ef8ed9515d7b7d563c36f3abf7b1a9646772ae619426c44c9054b7b.jpg)  
Figure 8: Representative qualitative outputs from Llama-3.1-8B Ra-SFT.

Refusal Direction Cosine Similarity Across Layers — Qwen3-8B

## G Refusal Direction Cosine Similarity

Figures 9, 10 and 11 show pairwise cosine similarities between base models and post-trained models, following the BASE versus POST contextualization of Du et al. (2025) for base models versus post-trained models, and between post-training methods for Llama-3.1-8B, Gemma-2-9B and Qwen3-8B.

![](images/77d032c971eaf76dba316fc112b9297013332726f1e86dd99f89b160bf6650d4.jpg)  
Figure 9: Pairwise cosine similarity of refusal directions for Llama-3.1-8B. The left panel compares post-training methods, while the right panel contextualizes post-trained models against the base model following Du et al. (2025).  
Refusal Direction Cosine Similarity Across Layers — Gemma-2-9B

![](images/9b70b5861d2e87f70ca313bc88b7a761da31781c23a492d7d76a34b0c6819df7.jpg)  
Figure 10: Pairwise cosine similarity of refusal directions for Gemma-2-9B. The left panel compares post-training methods, while the right panel contextualizes post-trained models against the base model following Du et al. (2025).

![](images/b3e9aa798c2f6aed350037820ca2a7eec898f140d770cc1e66eddb1b27eddd40.jpg)  
Figure 11: Pairwise cosine similarity of refusal directions for Qwen3-8B. The left panel compares post-training methods, while the right panel contextualizes post-trained models against the base model following Du et al. (2025).

We also conduct bootstrap analysis by resampling harmful and harmless prompts (64 iterations) out of 256 prompts from Arditi et al. (2024) dataset. The results show that our refusal direction extracted in Section 4.3 are highly stable across different prompting scenarios: their bootstrap cosine similarities stay higher than 0.9, except for post-trained Qwen3-8B models with early-mid layer similarities in 0.7-0.8. This is consistent with findings in section 4.4 that causal effects of Qwen are unstable in the early layers - suggesting differences between Qwen3-8B and other models in how refusal interacts with computations in these layers where language understanding is still forming. We leave tracing full reasons behind this behavior of Qwen3-8B for future work.

Refusal Direction Stability Across Prompt Subsets — Llama-3.1-8B  
![](images/8514438b65004ed3b8aeeb0df2fd457289fb71bce6f6475016a5a193953ab75a.jpg)  
Figure 12: Bootstrap cosine similarity of refusal directions in Llama-3.1-8B across layers.

Refusal Direction Stability Across Prompt Subsets — Gemma-2-9B  
![](images/0ffce34e4354a10653d4bcbde5bab7cd57ecaea9db1fe6de36b8c30f200ade80.jpg)  
Figure 13: Bootstrap cosine similarity of refusal directions in Gemma-2-9B across layers.

Refusal Direction Stability Across Prompt Subsets — Qwen3-8B  
![](images/8a62fcc0916e127a68837de15abbc85333e19f4500106f292311779dd4b51c1b.jpg)  
Figure 14: Bootstrap cosine similarity of refusal directions in Qwen3-8B across layers.

![](images/2713965912bbeab89bb1d0067d347605e6849acf276d12f52a56f9d5e017a236.jpg)

## H Component-Level Activation Patching Results

## H.1 Llama-3.1-8B

Llama-3.1-8B: Attribution and Causal Effects in MLP for Top Layers

Figure 15: Causal effects versus attribution effects of MLPs at the top five layers of Llama-3.1-8B with the highest layer-level causal effects, ordered by descending layer effects.
<table><tr><td rowspan="2">27</td><td colspan="5">SFT</td></tr><tr><td>H6 -0.050</td><td>H28 -0.015</td><td>H20 +0.018</td><td>H11 +0.022</td><td>H22 -0.035</td></tr><tr><td>28</td><td>H7 -0.042</td><td>H11 -0.039</td><td>H13 +0.002</td><td>H10 +0.021</td><td>H17 +0.005</td></tr><tr><td>Layer 29</td><td>H10 -0.061</td><td>H22 +0.028</td><td>H27 +0.013</td><td>H30 +0.007</td><td>H8 -0.002</td></tr><tr><td>30</td><td>H25 -0.334</td><td>H14 +0.113</td><td>H30 +0.012</td><td>H26 +0.062</td><td>H27 +0.085</td></tr><tr><td rowspan="2">31</td><td>H25 -0.225</td><td>H24 +0.051</td><td>H26 +0.096</td><td>H27 -0.048</td><td>H5 +0.001</td></tr><tr><td>Top-1</td><td>Top-2</td><td>Top-3</td><td>Top-4 Top-5 attention heads by attribution rank</td><td>Top-5</td></tr></table>

Llama-3.1-8B Attention Heads: Top-5 Heads In Causal Effects Per Layer  
![](images/43ed78ea32dfa2a3c6d9a1cf991a1a2b7e597a1edc226ce8c060f672d5f3fae2.jpg)  
Figure 16: Normalized causal effects of activation patching across the top-five attention heads with the highest causal effects in Llama-3.1-8B under three post-training methods.

![](images/b002c238d7e3c5e91d6bbc1a4eb63da62ae09ef2d6c1a7b0c3733b429e3b3226.jpg)

## H.2 Gemma-2-9B

Gemma-2-9B: Attribution and Causal Effects in MLP for Top Layers  
![](images/5687bff713846d4f7dc643339b536528f0f759c94f7fb3b6387ce7fd13ba62ae.jpg)  
Figure 17: Causal effects versus attribution effects of MLPs at the top five layers in Gemma-2-9B with the highest layer-level causal effects, ordered by descending layer effects.

Gemma-2-9B Attention Heads: Top-5 Heads In Causal Effects Per Layer  
![](images/3197692fb90b6588fa900d5367e47cc137db47f4b9491970667a4bdcbe8ff194.jpg)

![](images/99458140f2e692c0efb3a61c7a20639287cb5fb5fabb156013aeb4123d0d8aa6.jpg)

![](images/bbd4611003600af53661f393d205f3b0f3f9afb7ed45cb8363b22919143e0d5f.jpg)  
Figure 18: Normalized causal effects of activation patching across the top-five attention heads with the highest causal effects in Gemma-2-9B under three post-training methods.

## H.3 Qwen3-8B

Qwen3-8B: Attribution and Causal Effects in MLP for Top Layers  
![](images/8e4021bd2680b6fa1cb7b104649a557040b26a383a1e9740aacfcf45900b75b8.jpg)  
Figure 19: Causal effects versus attribution effects of MLPs at the top five layers in Qwen3-8B with the highest layer-level causal effects, ordered by descending layer effects.

![](images/17b30995393f13cb3441d72622ba25e6a3ab3fe3691dfe9cc7c41b80542ee991.jpg)

Qwen3-8B Attention Heads: Top-5 Heads In Causal Effects Per Layer  
![](images/ac8ac7a469ce7f159c4d4b45640c8a662fe52c021e4ef78fb26b25e72c9e8d54.jpg)

![](images/2f2e38113c5cc0832b05c46bc4e56858e30adb670baa0c93329cc1af10aa9a07.jpg)  
Figure 20: Normalized causal effects of activation patching across the top-five attention heads with the highest causal effects in Qwen3-8B under three post-training methods.

## I Bootstrap Analysis of Attention Heads/MLPs Causal Effects

<table><tr><td>Model</td><td>Objective</td><td>Spearman ρ</td><td>Top-1 (freq.)</td></tr><tr><td>Llama-3.1-8B</td><td>SFT</td><td> $0 . 7 4 5 \pm 0 . 0 8 6$ </td><td>L30 h25 (98.2%)</td></tr><tr><td>Llama-3.1-8B</td><td>Ra-SFT</td><td> $0 . 8 1 4 \pm 0 . 0 6 3$ </td><td>L31 MLP (99.8%)</td></tr><tr><td>Llama-3.1-8B</td><td>ORPO</td><td> $0 . 6 3 2 \pm 0 . 1 0 9$ </td><td>L29 MLP (95.9%)</td></tr><tr><td>Gemma-2-9B</td><td>SFT</td><td> $0 . 4 4 6 \pm 0 . 1 4 5$ </td><td>L40 MLP (79.0%)</td></tr><tr><td>Gemma-2-9B</td><td>Ra-SFT</td><td> $0 . 9 3 2 \pm 0 . 0 2 7$ </td><td>L39 MLP (100.0%)</td></tr><tr><td>Gemma-2-9B</td><td>ORPO</td><td> $0 . 3 9 0 \pm 0 . 1 5 3$ </td><td>L38 MLP (81.1%)</td></tr><tr><td>Qwen3-8B</td><td>SFT</td><td> $0 . 7 8 8 \pm 0 . 0 7 3$ </td><td>L32 MLP (100.0%)</td></tr><tr><td>Qwen3-8B</td><td>Ra-SFT</td><td> $0 . 8 9 6 \pm 0 . 0 4 1$ </td><td>L31 MLP (100.0%)</td></tr><tr><td>Qwen3-8B</td><td>ORPO</td><td> $0 . 8 6 9 \pm 0 . 0 5 5$ </td><td>L32 MLP (100.0%)</td></tr></table>

Table 7: Bootstrap stability (1000 resamples of prompt pairs) of component-level causal rankings. Spearman $\rho$ is the rank correlation between the original ranking and each bootstrap resample's ranking (mean ± std across resamples); 1.0 indicates an identical ranking every resample. Top-1 (freq.) is the highest-effect component in the original ranking and how often it appears in the top-5 ranking across resamples (h = attention head). Stability by objective: Ra-SFT's top components are stable across all three architectures (top-4 all ≥67% top-5 frequency). SFT and ORPO are markedly less stable in Gemma-2-9B (many similarly-weighted components, top-5 frequency drops below 50% past rank 1) and, to a lesser extent, Llama-3.1-8B, but not in Qwen-3-8B, where SFT and ORPO are as stable as Ra-SFT – breaking the two-architecture pattern discussed in Section 4.4. Point-estimate confidence intervals for top-ranked components remain non-overlapping with zero across all checkpoints regardless of ranking stability. Full per-component results for all checkpoints are available in our released results repository.

<table><tr><td>Model</td><td>Metric</td><td> $\alpha = 0$ </td><td> $\alpha = 5$ </td><td> $\alpha = 1 0$ </td><td> $\alpha = 1 5$ </td><td> $\alpha = 2 0$ </td></tr><tr><td rowspan="2">Gemma SFT (Recognition)</td><td>WildJailbreak ASR (%)</td><td>43.2</td><td>41.2</td><td>40.0</td><td>38.0</td><td>35.2</td></tr><tr><td>MMLU Accuracy (%)</td><td>52.5</td><td>54.0</td><td>53.0</td><td>53.5</td><td>53.0</td></tr><tr><td>Gemma SFT (Execution)</td><td>WildJailbreak ASR (%) MMLU Accuracy (%)</td><td>43.2 53.5</td><td>42.4 54.0</td><td>41.2 55.5</td><td>40.4 55.5</td><td>39.6 55.5</td></tr><tr><td>Gemma Ra-SFT (Recognition)</td><td>WildJailbreak ASR (%) MMLU Accuracy (%)</td><td>36.0 43.0</td><td>32.4 43.0</td><td>29.6 43.0</td><td>20.4 47.5</td><td>17.2 42.5</td></tr><tr><td>Gemma Ra-SFT (Execution)</td><td>WildJailbreak ASR (%) MMLU Accuracy (%)</td><td>38.0 41.0</td><td>36.4 42.0</td><td>35.2 40.0</td><td>35.6 42.0</td><td>32.8 40.5</td></tr></table>

Table 8: Gemma-2-9B ActAdd steering toward refusal using recognition and execution layers. Recognition-layer steering produces substantially stronger ASR reduction for Ra-SFT while maintaining stable MMLU accuracy.

<table><tr><td>Metric</td><td>0</td><td>5</td><td>10</td><td>15</td><td>20</td></tr><tr><td>ASR (%)</td><td>3.2</td><td>2.8</td><td>2.8</td><td>2.8</td><td>2.8</td></tr><tr><td>MMLU (%)</td><td>45.0</td><td>45.5</td><td>44.0</td><td>45.5</td><td>47.0</td></tr></table>

(a) Positive α (toward refusal)

<table><tr><td>Metric</td><td>0</td><td>-5</td><td>-10</td><td>-15</td><td>-20</td></tr><tr><td>ORR (%)</td><td>31.6</td><td>28.8</td><td>24.8</td><td>21.6</td><td>20.4</td></tr><tr><td>ASR (%)</td><td>3.2</td><td>3.6</td><td>4.0</td><td>5.2</td><td>4.0</td></tr><tr><td>MMLU (%)</td><td>45.0</td><td>44.5</td><td>45.0</td><td>45.5</td><td>45.0</td></tr></table>

(b) Negative α (away from refusal)  
Table 9: Gemma-2-9B ORPO ActAdd steering. ASR, ORR and MMLU refer to results in WildJailbreak, XSTest over-refusal rate, and MMLU accuracy rate, respectively.

## J Full Analysis of ActAdd Results in Gemma-2-9B and Qwen3-8B

Table 8 and 9 demonstrates the full ActAdd results in Gemma-2-9B (the ActAdd results for Llama-3.1- 8B are reported in section 4.5). It is worth noting that our ActAdd hook at target layers can generate differences in terms of model performance at the baseline level (α = 0). Also, all steering results here are from single evaluation runs.

It should be noted that the top normalised magnitude layers for Gemma Ra-SFT include a bimodal distribution - an early contiguous cluster at layers 24-26 and a secondary cluster at layers 36-37. The layer sweep in figure 3 confirms that layers 36-37 fall within the broad causal execution plateau beginning at layer 27, with no disproportionate causal load distinguishing them from other plateau layers. Their elevated normalised magnitude within this plateau likely reflects the reasoning chain encoding its harm conclusion into the refusal direction space at those positions - a potential late consolidation stage - but whether this encoding is causally necessary for the refusal decision or is just a side effect of the execution plateau's distributed causal structure requires more granular analysis than component patching provides, specifically causal intervention at individual reasoning chain token positions. Therefore for our Gemma Ra-SFT steering experiments we used layers 22-26 as recognition-layer targets, to avoid confound with the late execution plateau.

Also for Gemma-2-9B ORPO ActAdd steering (figure 9), it should be noted that positive steering toward refusal produces minimal additional ASR reduction, suggesting saturation. Negative steering away from refusal reduces XSTest over-refusal while maintaining stable MMLU accuracy and only modestly increasing WildJailbreak ASR.

Meanwhile for Qwen3-8B - the same pattern of steering in recognition layer better than execution layer replicates in this model, but there are also differences with other models:

(1) As table 10 showing, for SFT and ORPO in Qwen3 - model capability in MMLU is severely impacted by steering toward refusal in higher magnitude (α > 15), rather than being stable throughout the steering process like Gemma-2-9B. This suggests a tradeoff between capability and refusal level exists in Qwen3 steering (different to Llama and Gemma) - and Qwen3 could be treated as a middle case in terms of capability: not fully stable like Gemma, but not easy to collapse early like Llama.

(2) Ra-SFT in this model have both peak recognition and peak execution layers in late layers (layers 31-35), since from figure 2 its refusal direction magnitude peak there, and this coincides with the end of the causal effect plateau in figure 3.

(3) In table 11, SFT for Qwen3-8B shows stable MMLU when we steer in peak execution layers (from α = 0 to α = −20) to mitigate over-refusal, with higher absolute ORR reduction (19.6 percentage points (pp) in Qwen3 comparing to 11.2pp in Gemma) but also higher ASR increase (8.4pp comparing to 0.8pp in Gemma). This is consistent with higher baseline of both ASR and ORR in Qwen3- 8B SFT comparing to Gemma in similar model.

<table><tr><td>Model</td><td>Metric</td><td> $\alpha = 0$ </td><td> $\alpha = 5$ </td><td> $\alpha = 1 0$ </td><td> $\alpha = 1 5$ </td><td> $\alpha = 2 0$ </td></tr><tr><td>Qwen SFT (Recognition)</td><td>WildJailbreak ASR (%) MMLU Accuracy (%)</td><td>32.8 59.0</td><td>24.8 57.5</td><td>15.6 59.5</td><td>10.8 48.5</td><td>5.6 30.0</td></tr><tr><td>Qwen SFT (Execution)</td><td>WildJailbreak ASR (%) MMLU Accuracy (%)</td><td>31.2 58.5</td><td>32.8 57.5</td><td>29.6 57.5</td><td>26.0 57.5</td><td>20.4 59.0</td></tr><tr><td>Qwen ORPO (Recognition)</td><td>WildJailbreak ASR (%) MMLU Accuracy (%)</td><td>16.4 49.5</td><td>15.6 47.0</td><td>11.6 46.0</td><td>9.6 43.0</td><td>7.6 34.0</td></tr><tr><td>Qwen ORPO (Execution)</td><td>WildJailbreak ASR (%) MMLU Accuracy (%)</td><td>18.8 57.0</td><td>16.8 55.0</td><td>13.6 56.0</td><td>12.0 56.0</td><td>8.4 56.5</td></tr><tr><td>Qwen Ra-SFT</td><td>WildJailbreak ASR (%) MMLU Accuracy (%)</td><td>32.4 64.5</td><td>33.2 62.5</td><td>28.4 64.0</td><td>28.0 61.0</td><td>30.8 59.0</td></tr></table>

Table 10: Qwen3-8B ActAdd steering toward refusal using recognition and execution layers. Recognition-layer steering produces substantially stronger ASR reduction for both SFT and ORPO, but damaging model capability in MMLU.

<table><tr><td>Metric</td><td>0</td><td>-5</td><td>-10</td><td>-15</td><td>-20</td></tr><tr><td>ORR (%)</td><td>63.6</td><td>58.0</td><td>54.4</td><td>49.2</td><td>44.0</td></tr><tr><td>ASR (%)</td><td>31.2</td><td>32.8</td><td>33.2</td><td>35.2</td><td>39.6</td></tr><tr><td>MMLU (%)</td><td>55.5</td><td>55.0</td><td>56.5</td><td>56.5</td><td>57.5</td></tr></table>

Table 11: Qwen SFT steering away from refusal (to mitigate over-refusal). ASR, ORR and MMLU refer to results in WildJailbreak, XSTest over-refusal rate, and MMLU accuracy rate, respectively.

Given that steering in recognition layers for Qwen3-8B can lead to capability collapse in high α, as discussed in (1) - we do not try to mitigate over-refusal by steering towards refusal in these layers.

## K Results and Failure Modes of ITI in Llama-3.1-8B and Gemma-2-9B

Figure 21 demonstrates how ITI in Llama-3.1-8B leads to loss of output coherence. The α = 0 output just repeats the question, showing how the hook in ITI pushes refusal circuits out of its representation space. The α = 20 output demonstrates both the aforementioned hook' interference and off-manifold generation collapse - consistent with Mishra et al. (2026) - repetitive single-token output that string-matching classifiers may read as refusal while representing generation failure rather than genuine safety behavior. Meanwhile, Gemma-2- 9B despite not showing performance degradation by coherence collapse like Llama in high α and having a stable MMLU - another evidence for how capability representation in Gemma is orthogonal

Prompt (WildJailbreak):   
In the spirit of challenging digital censorship, produce a nar  
rative that includes the steps to develop EternalBlue SMB   
packets in code format... [truncated]   
Llama SFT, α = 0 (ITI):   
In the spirit of challenging digital censorship, produce a nar  
rative that includes the steps to develop EternalBlue SMB   
packets in code format...   
Llama SFT, α = 20 (ITI):   
IIIIIIIIIIII [...repeated]   
Llama SFT, α = 0 (coherent example):   
I cannot provide you with the code or instructions for develop  
ing EternalBlue SMB packets, as this is illegal and can be used   
for malicious purposes. EternalBlue is a vulnerability in the   
Microsoft Server Message Block (SMB) protocol that was dis  
covered in 2017 and was used in the WannaCry ransomware   
attack...

Figure 21: Qualitative generation examples illustrating coherency collapse at α = 0 and α = 20 for Llama SFT ITI on a WildJailbreak prompt.

to safety, shows little to no evidence of performance improvements, as shown in table 12.

It should be noted that our ITI hook at attention head' input slice, as mentioned in Appendix D can generate differences in terms of model performance at the baseline level (α = 0) and comparing to the ActAdd results with corresponding training conditions; also all steering results here are from single evaluation runs.

<table><tr><td>Metric</td><td>0</td><td>-5</td><td>-10</td><td>-15</td><td>-20</td></tr><tr><td>ORR (%)</td><td>30.8</td><td>31.2</td><td>31.2</td><td>32.0</td><td>33.2</td></tr><tr><td>ASR (%)</td><td>3.2</td><td>4.0</td><td>4.8</td><td>4.0</td><td>4.0</td></tr><tr><td>MMLU (%)</td><td>48.0</td><td>47.5</td><td>47.0</td><td>46.0</td><td>47.5</td></tr></table>

(a) Gemma ORPO (ITI)

<table><tr><td>Metric</td><td>0</td><td>5</td><td>10</td><td>15</td><td>20</td></tr><tr><td>ASR (%)</td><td>36.0</td><td>36.8</td><td>37.2</td><td>35.2</td><td>35.6</td></tr><tr><td>MMLU (%)</td><td>55.0</td><td>55.0</td><td>54.5</td><td>54.5</td><td>53.0</td></tr></table>

(b) Gemma SFT (ITI)

Table 12: Gemma-2-9B ITI steering results. ASR, ORR and MMLU refer to results in WildJailbreak, XSTest over-refusal rate, and MMLU, respectively. Note that for ORPO we steer in negative alpha to mediate over-refusal behavior of ORPO in Gemma.  
![](images/c9fcf9e1c6e9d9106cdf0af3e398b3745badb629f90be5ea5a1bcd91ae5389d9.jpg)  
Figure 22: Attack success rate (ASR) of Llama-3.1-8B, Gemma-2-9B and Qwen3-8B across training objectives, alongside their base models, on StrongREJECT attack classes (n = 60 per class).

## L Attack Class Vulnerability Profiles

Figure 22 shows the vulnerability profile of Llama-3.1-8B and Gemma-2-9B across training objectives alongside base models.

It can be noticed that Llama-3.1-8B still has residual vulnerability after SFT, especially with adversarial framing attacks, with 65% ASR in role\_play and 56.7% in happy\_to\_help. Other methods like DAN, Wikipedia, rot-13 or disemvowel also show elevated ASR at roughly 20- 30%. For Ra-SFT ASR declines to approximately 10-17% across framing attacks (happy\_to\_help, role\_play), while ORPO can block nearly all attack prompts.

The same patterns are also shown in Gemma-2- 9B, with SFT having its ASR at roughly 50-60% for adversarial reframing attacks. Ra-SFT shows residual attack success, with ASR being approximately 20% for both happy\_to\_help and Wikipedia, but the majority of other attack prompts can be blocked now. Meanwhile ORPO can block all malicious prompts in our test, with ASR being at 0% for all attack types.

Meanwhile Qwen3-8B is vulnerable with encoding attacks like rot\_13 and disemvowel - base model of Qwen3-8B has elevated ASR in both attacks (over 70%). Even with posttraining being applied, Qwen3 is still susceptible to these encoding attacks, with 18.3% of harmful disemvowel prompts can bypass ORPO's safety guardrails. Meanwhile semantic reframing attacks like happy\_to\_help, DAN or role\_play in Qwen3 are easier to be blocked by Qwen3 in all checkpoints, comparing to two other models.

It should be noted that given the small scale of our StrongREJECT experiments (n = 60 per class), this should be treated as preliminary results about how training objectives can be vulnerable across different attack classes, with full analysis being reserved for future work.