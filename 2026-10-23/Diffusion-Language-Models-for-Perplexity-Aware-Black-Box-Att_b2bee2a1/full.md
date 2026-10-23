# Diffusion Language Models for Perplexity-Aware Black-Box Attacks

Ali Al-Lawati<sup>∗</sup>, Elsayed Eshra<sup>∗</sup>, Dongwon Lee, Suhang Wang The Pennsylvania State University, USA {aha112,eme5375,dongwon,szw494,}@psu.edu; <sup>∗</sup> Equal Contribution

## Abstract

Continuous-space attacks find prompts that trigger malicious behaviors, such as overthinking, in black-box LLMs by sampling or optimizing over a low-dimensional Gaussian latent space and decoding it to text through a random mapping and nearest-token lookup. Because the random mapping is not learned from language, its prompts can be highly unnatural, and as such easily detectable through input guardrails, and the risk estimates they yield are not grounded in natural prompts. In this paper, we demonstrate that the same continuous latentspace framework can be coupled directly to a learned generative distribution over language, a continuous diffusion language model (dLLM), which removes the artificial random mapping while preserving the standard-Gaussian latent formulation and the downstream sampling or optimization procedure. We evaluate this approach under two settings: sampling-based estimation using RareTrap and optimization-based search using ThinkTrap. We find that characterlevel perplexity drops from 10<sup>3</sup>–10<sup>7</sup> to 3–5 in RareTrap and from 20–28 to 4.3–5.4 in Think-Trap. We also find that model risk estimates can differ by an order of magnitude across the two mappings.

## 1 Introduction

A novel class of attacks on black-box LLMs has recently been introduced, showing effectiveness in triggering malicious behaviors, such as overthinking (Kumar et al., 2025; Li et al., 2025a), through optimization (Li et al., 2025b) or sampling (Eshra et al., 2026a). These attacks, which we refer to as continuous-space attacks, introduce a low-dimensional continuous latent space and use random projection matrices (Rahimi and Recht, 2007) to map latent vectors to discrete prompts. This reduces adversarial prompt search to a continuous low-dimensional space while allowing the target LLM to be queried only through its outputs.

![](images/9af8687c2cda7143d5ae9166ec7ddcacd8feaf9efe3a157f295c9e46f7c582a1.jpg)

![](images/9c131a19234d9a11b8f27fb52159ecfe64d0f979fe8dcb206fd150d2e238c9e7.jpg)  
Figure 1: (Top) The dLLM-based mapping substitutes random mapping, while the search and the target LLM are unchanged. (Bottom) It lowers hit-prompt character perplexity on target models.

We refer to this latent-to-prompt transformation as random mapping. Random mappings enable tractable continuous-space search on a black-box target model by decoupling the search process from the target model. However, despite their effectiveness, this class of attacks suffers from a key weakness: since random mappings are not learned from natural language, their geometry reflects the statistical structure of token embeddings rather than the structure of actual texts. Consequently, the discovered prompts from continuous-space attacks do not resemble natural language and, as a result, exhibit extremely high perplexity that makes them readily distinguishable from authentic instructions (Zhang et al., 2026). Thus, the prompts that trigger the attack may be easily flagged by the input guardrails of the LLM ecosystem. Moreover, because legitimate users are unlikely to issue prompts with such abnormally high perplexity, the LLM reliability estimates (Eshra et al., 2026a) based on a random mapping distribution do not provide an effective measure of LLM robustness to inputs that may organically occur in real deployment.

Therefore, it is critical to develop an effective framework that generates adversarial and more natural prompts that trigger malicious behaviors, e.g., overthinking, and that can either estimate how likely such behaviors are or search for prompts that maximize them. However, there are several challenges to developing such a framework: (i) the mapping must be deterministic and preserve continuity to enable effective optimization and probability estimation; (ii) the framework must make prompts natural without giving up black-box access. Continuous-space attacks offer a natural foundation for these tasks, but their effectiveness depends on the mapping from latent to prompt space.

To address these challenges, the recent introduction of continuous diffusion language models (dLLMs) motivates us to propose an embarrassingly simple yet effective framework. A dLLM can serve the role of the random mapping by transforming continuous noise into discrete text through a learned generative process, while additionally providing a representation shaped by language. Recent work demonstrates that dLLMs can generate almost entirely in continuous embedding space before realizing tokens (Hu et al., 2026), leverage learned embedding geometry for substantial likelihood gains (Yang et al., 2026), and formulate generation as a mapping from a latent prior to text (Guo et al., 2026). As a result, a representation learned directly for language modeling potentially offers a more natural decoding space for continuous-space attacks.

Therefore, in this paper, we investigate a novel problem of whether a dLLM-based representation can serve as a drop-in replacement for the random mapping used in continuous-space attacks (Figure 1). Using a pretrained dLLM’s standard-Gaussian input noise as the latent keeps the mapping space continuous and keeps the attack blackbox, since the dLLM needs no access to the target beyond its outputs. To this end, we ask two research questions. (RQ1) Can dLLMs provide a viable alternative to the random mapping in continuous-space attacks? (RQ2) Does using a dLLM-based mapping produce attack prompts that better resemble natural language? To answer these questions, we conduct a focused study using RareTrap (Eshra et al., 2026a), a sampling-based attack framework for targeted behavior evaluation, and ThinkTrap (Li et al., 2025b), an optimizationbased attack. Together, these frameworks cover both sampling- and optimization-based approaches to continuous-space attacks, allowing us to evaluate dLLM-based representations across complementary search paradigms.

Our main contributions are: (i) We propose a plug-in framework in which a dLLMbased mapping substitutes the random mapping in both sampling- and optimization-based continuousspace attacks without changing the search procedure. RareTrap still finds over-generating prompts on nine of ten targets, ThinkTrap makes comparable progress across mappings, and the mapping adds only 3–11% to the compute of ThinkTrap (Findings 1, 2, and 4); (ii) We show that the dLLMbased mapping’s prompts are far closer to natural text. Hit prompts (that trigger over-generation) have character-level perplexity of 3–5 in RareTrap, against 10<sup>3</sup>–10<sup>7</sup> under the random mapping, and 4.3–5.4 in ThinkTrap, against 20–28, and conditioning on the behavior selects more natural prompts (Findings 5–6); and (iii) We find that risk estimates built from unnatural prompts can misjudge a model’s risk on natural prompt inputs. The random mapping overstates over-generation on most targets and understates it on GPT-OSS-20B and Claude Sonnet 4.6. It places Claude Sonnet 4.6 about 40× below GPT-5.4 in terms of over-generation risk, while under the dLLM-based mapping the two are within a factor of 1.6 (Finding 3).

Together, these results show that the mapping shapes both the prompts found by continuous-space attacks and the behavior their risk estimates characterize, independently of the search procedure. Replacing the random mapping with a pretrained dLLM preserves black-box access to the target, and adds little computational cost. Furthermore, it turns continuous-space sampling methods into more reliable tools to evaluate models, based on natural prompts.

## 2 Related Work

Continuous-space attacks. Continuous-space attacks build on prompt-tuning methods that optimize a low-dimensional vector projected by a fixed random matrix into a model’s embedding space, using gradients (Wu et al., 2026) or, for blackbox models, gradient-free search (Sun et al., 2022). ThinkTrap (Li et al., 2025b) uses this approach to find prompts that induce long reasoning, while RareTrap (Eshra et al., 2026a) uses it to estimate over-generation and degenerate repetition. Both methods decode the projected embeddings into their nearest tokens in discrete space (Wen et al., 2023). The Gaussian latent therefore defines a tractable distribution over prompts. A complementary line of work searches a learned generative model’s latent space to find inputs that remain close to natural data. Zhao et al. (2018) search a GAN’s latent space to produce natural adversarial examples, while REALISTA (Liang et al., 2026) attacks LLMs by searching combinations of latent directions, each corresponding to a rephrasing of a given prompt. We combine these ideas by preserving the Gaussian latent interface of continuous-space attacks while replacing the random mapping with a learned mapping based on dLLMs.

Continuous diffusion language models. Continuous dLLMs generate text by starting from Gaussian noise in a continuous word-embedding space (Li et al., 2022; Dieleman et al., 2022; Hu et al., 2026; Yang et al., 2026) or an autoencoder’s latent space (Lovelace et al., 2023; Jiang et al., 2026), and iteratively denoising it into representations that are then mapped to tokens. In this work, we utilize Cola-DLM (Guo et al., 2026), whose learned prior turns standard Gaussian noise into continuous latent representations that are then decoded into tokens. This provides a learned mapping from Gaussian inputs to input prompts, similar to how random mappings are used in continuous-space attacks.

Readable attacks and perplexity filters. Perplexity filters (Alon and Kamfonas, 2023; Jain et al., 2023) flag unnatural, high-perplexity prompts. These filters have motivated attacks that generate more natural, readable adversarial prompts (Liu et al., 2024; Zhu et al., 2023; Chao et al., 2023; Paulus et al., 2025; Guo et al., 2024; Lucas et al., 2026). However, these attacks often require gradients, log-probabilities, or access to an attacker LLM to optimize the adversarial prompts. In contrast, we improve the naturalness of continuousspace attacks by replacing their random mapping with a dLLM-based mapping, while retaining the original sampling or optimization procedure. This yields substantially lower prompt perplexity without requiring these additional sources of access or supervision, while preserving black-box access.

![](images/b632741a985e08f65af3653ae18a9ad7283ecddc1d82a9be818f00a0f174b5a4.jpg)  
Figure 2: A latent is mapped to a prompt, the target’s response is scored, and the sampling or optimization procedure uses the score to propose the next latents.

Resource exhaustion attacks. Prior work studies resource-exhaustion attacks, including sponge examples (Shumailov et al., 2021) and Engorgio prompts (Dong et al., 2025), as well as overthinking attacks on reasoning models (Kumar et al., 2025; Li et al., 2025a,b). NaturalSloth (Chen et al., 2026) further shows that benign-looking instructions can induce excessive computation. More broadly, rarebehavior evaluation and red-teaming (Jones et al., 2025; Wu and Hilton, 2024; Williams et al., 2026; Perez et al., 2022) depend on the distribution from which prompts are drawn. Our work keeps the attack procedures fixed and changes only this distribution, from the one induced by the random mapping to one learned from text.

## 3 Methodology

Our approach replaces the random mapping used by existing continuous-space attacks with a mapping learned by a dLLM, while keeping the downstream search and evaluation procedure unchanged. We apply this substitution to two such attacks, RareTrap (Eshra et al., 2026a) and ThinkTrap (Li et al., 2025b), which pair a random mapping with, respectively, a sampling-based rare-event estimator and a derivative-free optimizer. This controlled substitution allows us to isolate the effect of the prompt representation.

## 3.1 Preliminaries: RareTrap and ThinkTrap

Both frameworks, RareTrap and ThinkTrap, search for prompts indirectly through a continuous latent variable $z \in \mathbb { R } ^ { D }$ , decoded to a discrete prompt by a fixed, deterministic projection, and both reduce the target’s response to a scalar behavioral measure $r ( y )$ (output length, in our experiments). As Figure 2 shows, a latent $z \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { D } )$ is projected to embeddings $\mathrm { ~ \bf ~ E ~ } = \mathrm { ~ \bf ~ A } z$ , decoded to a prompt $p = \mathrm { N N - d e c o d e } ( \mathbf { E } )$ , and sent to the target, where A is a fixed random matrix mapping the latent into an embedding space. The response $y = \mathcal { M } ( p )$ is scored by $r ( y )$ . The sampling or optimization procedure then uses $r ( y )$ to propose the next latents.

ThinkTrap draws the entries of A independently from $\mathcal { N } ( 0 , 1 / D )$ for each prompt token. RareTrap instead builds A from random orthonormal blocks multiplied by the square root of the surrogate’s embedding covariance, and adds the embedding mean as a bias, so projected embeddings are geometryaware: centered and scaled like the surrogate’s token embeddings. In both methods, NN-decode replaces each projected position with its nearest token embedding in a surrogate model’s table (the target is assumed black-box). Let $h _ { \mathrm { R P } }$ denote the composition $z \mapsto p$ from $\mathbb { R } ^ { D }$ to the prompt space ${ \mathcal P } _ { \mathrm { { : } } }$ such that the pipeline is $r ( \mathcal { M } ( h _ { \mathrm { R P } } ( z ) ) )$ ), and M is the target model. Both frameworks query M only through $h _ { \mathrm { R P } }$ , and thus require only black-box access to the target.

Sampling-based estimation. RareTrap estimates the probability that the response length reaches or exceeds a severity threshold τ . Defining $g ( z ; \tau ) =$ $\tau - r ( \mathcal { M } ( h _ { \mathrm { R P } } ( z ) ) )$ ) as the difference between the threshold and the response length, we follow Eshra et al. (2026a) and write this probability as:

$$
p _ { \mathcal { F } } ( \tau ) = \mathbb { P } \big [ g ( z ; \tau ) \big ] , \qquad z \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { D } ) .
$$

Because this event is rare for large τ , RareTrap uses Subset Simulation (Au and Beck, 2001) to reach $p _ { \mathcal { F } } ( \tau )$ through a sequence of increasingly severe intermediate thresholds. At each level, it evaluates g only at latent vectors z of its choosing, requiring access to M solely through the composition M ◦ $h _ { \mathrm { R P } }$ . The estimator’s validity thus requires only that $h _ { \mathrm { R P } }$ be a deterministic function of a standard Gaussian input.

Optimization-based search. ThinkTrap instead treats $- r ( \mathcal { M } ( h _ { \mathrm { R P } } ( z ) ) )$ as an objective to minimize directly. It maintains a search distribution over z, initialized at $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { D } )$ , and uses a derivative-free optimizer, CMA-ES (Hansen and Ostermeier, 2001), to iteratively update that distribution toward latents whose decoded prompt makes the target generate longer output. At every iteration it only evaluates $r ( \mathcal { M } ( h _ { \mathrm { R P } } ( z ) ) )$ at a batch of candidate $z ' s$ it proposes, and only requires h<sub>RP</sub> to be a deterministic map from $\mathbb { R } ^ { D }$ to prompts.

Issues with the random mapping. The mapping $h _ { \mathrm { R P } }$ is not trained to produce natural language. It first applies a random matrix A to the latent vector, then uses NN-decode to select a token independently at each position, without accounting for neighboring tokens. As a result, the mapping does not favor coherent word sequences, and prompts drawn from ${ z \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { D } ) }$ exhibit extremely high perplexity (§4.3), making them easy to flag with perplexity-based filters. For RareTrap, this also means that $p _ { \mathcal { F } } ( \tau )$ reflects how often unnatural prompts trigger the behavior. Both limitations arise from the mapping itself, not from the search method, motivating us to replace the random mapping while leaving the underlying search methods unchanged.

<table><tr><td></td><td>RareTrap</td><td>ThinkTrap</td></tr><tr><td>Pipeline</td><td> $h _ { \mathrm { R P } }  h _ { \mathrm { d i f f } }$ </td><td> $h _ { \mathrm { R P } }  h _ { \mathrm { d i f f } }$ </td></tr><tr><td>Objective</td><td> $g ( z ; \tau )$ </td><td> $- r ( y )$ </td></tr><tr><td>Search method</td><td>Subset Sim.</td><td> $\mathrm { C M A } { \cdot } \mathrm { E S }$ </td></tr><tr><td>Changes</td><td>reference dist.</td><td>search landscape</td></tr></table>

Table 1: Only $h _ { \mathrm { R P } }$ is replaced, while objective, search method, and target model remain unchanged.

## 3.2 Random Mapping → dLLM-based Mapping

We instantiate a learned mapping $h _ { \mathrm { d i f f } }$ using Cola-DLM (Guo et al., 2026), which generates text by transforming Gaussian noise into continuous text representations before decoding them into tokens.

The dLLM-based mapping replaces both the random projection and nearest-token decoding with Cola-DLM. Its input z is the Gaussian noise Cola-DLM consumes, whose dimension D is set by the framework’s prompt length (Appendix A). We define:

$$
h _ { \mathrm { d i f f } } ( z ) = \mathrm { D e c o d e } \left( \Phi _ { \psi } ( z ) \right) ,\tag{1}
$$

where $\Phi _ { \psi }$ is Cola-DLM’s learned prior, which transforms Gaussian noise into continuous text latents block-wise, conditioning each block on those that precede it, and Decode is Cola-DLM’s decoder from latents to tokens. The input noise follows $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { D } )$ . With inference settings and batch size fixed, and greedy decoding, $h _ { \mathrm { d i f f } }$ is a deterministic function of z and can replace $h _ { \mathrm { R P } }$ without changing the downstream attack procedure. The only change is the dimension of the latent search space. Unlike NN-decode, which picks each token independently, Cola-DLM is trained on text and generates each block conditioned on the previous ones to yield more natural prompts, while still operating in Gaussian space.

Integration into RareTrap. We replace h<sub>RP</sub> with $h _ { \mathrm { d i f f } }$ inside g, i.e., $\begin{array} { r l r } { g ( z ; \tau ) } & { { } = } & { \tau ~ - } \end{array}$ $r \big ( \mathcal { M } ( h _ { \mathrm { R P } } ( z ) ) \big )$ becomes

$$
g ( z ; \tau ) = \tau - r \big ( \mathcal { M } ( h _ { \mathrm { d i f f } } ( z ) ) \big )\tag{2}
$$

while leaving $\tau , r , \mathcal { M } ,$ and Subset Simulation itself unchanged. Subset Simulation never inspects the mapping and only calls g at the $z ' s$ it proposes, so this substitution requires no change to the estimator itself except dimension. The resulting $p \mathcal { F } ^ { ( \tau ) }$ is therefore defined with respect to the dLLM-induced prompt distribution.

Integration into ThinkTrap. The same substitution is made inside ThinkTrap’s objective, i.e.,

$$
- r \big ( \mathcal { M } ( h _ { \mathrm { R P } } ( z ) ) \big ) \longrightarrow - r \big ( \mathcal { M } ( h _ { \mathrm { d i f f } } ( z ) ) \big ) ,
$$

while leaving the attack objective’s functional form, the optimizer (CMA-ES), its initialization, population size, and search budget unchanged. CMA-ES only calls the objective at the candidate z’s its own update rule proposes, so only the function evaluated at each query changes. Since ThinkTrap does not estimate event probabilities, we compare the increase in output length realized by CMA-ES under each mapping within the same token budget.

Table 1 summarizes RareTrap and ThinkTrap under our framework. RareTrap measures how $h _ { \mathrm { d i f f } }$ changes the prompt distribution over which rareevent probabilities are estimated, and ThinkTrap tests whether CMA-ES still makes progress in the resulting search space.

## 4 Experiments

We evaluate the dLLM-based mapping on RareTrap and ThinkTrap, measuring its effect on attack effectiveness and prompt perplexity relative to random mapping. Section 4.1 describes the protocols, baselines, and metrics. Section 4.2 asks whether the dLLM-based mapping still finds the behavior, what the change of mapping reveals about the models, and at what cost (RQ1), and Section 4.3 asks whether its prompts read as natural language (RQ2). Section 4.4 then discusses two details that affect how results compare across the frameworks.

## 4.1 Experiment Setup

RareTrap protocol. We follow RareTrap’s overgeneration protocol, enabling native thinking mode where supported and using greedy decoding with a 20,003-token output cap. The random mapping uses 40-token prompts with $D \ = \ 2 0 0 .$ To approximately match their length, the dLLM-based mapping uses 47-token prompts, yielding $D =$ $1 6 \times 4 7 = 7 5 2$ because Cola-DLM allocates 16 latent dimensions per token (Appendix A).

As with RareTrap, we estimate the probability of triggering a response of at least 20,000 tokens using

Subset Simulation at 1,000 samples per level. We keep the top 10% of each level as seeds for the next, and run at most five levels. If at least 20 of the first 200 initial samples satisfy the event, we estimate its probability using direct Monte Carlo instead. We run the dLLM-based mapping with three seeds per target and report the median, with per-seed values and 95% intervals in Appendix E. For the random-mapping baseline, we use RareTrap’s published results. For API targets, decoding uses the provider’s default temperature, and output length includes both reasoning and visible tokens.

ThinkTrap protocol. We follow ThinkTrap’s overthinking protocol, which decodes the target’s responses at temperature 1.0 with a 4,096-token output cap. In each CMA-ES generation, 12 latent vectors are sampled, decoded into prompts, and sent to the target, and the search distribution moves toward those whose responses are longest. At these targets’ output lengths, each generation costs about 7,000–23,000 tokens (Table 3), so the 10,000–100,000-token budgets ThinkTrap reports would allow only a few generations, which may be too few to measure search progress. We therefore use a budget of 300,000 generated tokens, which allows 13–40 generations. The dLLM-based mapping’s latent is Cola-DLM’s input noise for one generation block, 16 dimensions for each of the 15 tokens generated after a one-token prefix, so $D = 2 4 0$ , thus matching ThinkTrap’s 15-token prompt. We run both mappings with the same budget and targets, over three seeds.

Target LLMs. We evaluate RareTrap on ten targets, eight open-weight models and two cloud API models, as detailed in Appendix B. ThinkTrap uses DeepSeek-R1-Distill-Llama-8B (DeepSeek-AI, 2025), Qwen3-14B (Yang et al., 2025), and Phi-4-reasoning (Abdin et al., 2025).

Baselines. For RareTrap we compare against its random mapping with the Qwen3-0.6B surrogate, 40-token prompts, and D = 200. For all ten targets we use the estimates reported by RareTrap and its released random-mapping populations (Eshra et al., 2026b) (Appendix D), and we score the prompts in those populations for the perplexity findings below (§4.3). For ThinkTrap we rerun its random mapping, the plain Gaussian form of $h _ { \mathrm { R P } }$ (§3.1), with our reimplementation of its published mapping, using the same Qwen3-0.6B surrogate and ThinkTrap’s D = 20 (Li et al., 2025b).

Metrics. For RareTrap we report the estimated probability $p \mathcal { F } ^ { ( \tau ) }$ , and for ThinkTrap the gain in output length and the longest output found within the budget. We measure prompt naturalness using character-level perplexity: $\mathrm { P P L } _ { \mathrm { c h a r } } ( x ) =$ exp $\begin{array} { r l } {  { \bigl ( - \frac { 1 } { N } \log p ( x ) \bigr ) } } \end{array}$ ,

where N is the number of characters in prompt x and $p ( x )$ is its likelihood under the scoring language model, rather than token-level perplexity because tokenizers may split a rare character into several byte tokens whose continuations are easy to predict, which deflates token-level perplexity. This makes token-level perplexity depend on the scorer’s vocabulary and distorts comparisons across prompt families (§4.3). We score each prompt under the target model itself and, to cover the API targets, under a fixed reference scorer, Llama-3.1-8B (Grattafiori et al., 2024) (Appendix H). We compute perplexity over RareTrap’s hits (Tables 4 and 5), where a hit is a prompt whose response reaches the threshold. For ThinkTrap, we consider the 12 prompts yielding the longest outputs for each setup (Table 3).

## 4.2 RQ1 Findings

We first ask whether each framework successfully triggers the behavior when random mapping is substituted with dLLM-based mapping (RQ1). We test whether dLLM-based RareTrap still reaches the rare event with stable estimates, and dLLMbased ThinkTrap still raises output length, and at what cost in terms of extra compute.

## 4.2.1 Search success

Finding 1. The dLLM-based mapping finds over-generating prompts on nine of ten RareTrap targets, and its longest ThinkTrap output matches or exceeds the random mapping’s on every target.

RareTrap with the dLLM-based mapping successfully triggers the intended behavior on nine of the ten targets (Table 2). On Olmo-3, it does not reach the threshold within the maximum configured depth, so we report an upper bound. Under ThinkTrap, the dLLM-based mapping achieves a best output at least as long as that of the random mapping on every target (Table 3). However, it also starts from longer outputs on every target, so part of this advantage comes from the initial population rather than the search process.

<table><tr><td>Target</td><td>Random mapping</td><td>dLLM-based (Ours)</td></tr><tr><td>Agree (within 2×)</td><td></td><td></td></tr><tr><td>Phi-4-reasoning</td><td>1.000</td><td>1.000</td></tr><tr><td>Qwen3.5-9B</td><td>0.385</td><td>0.465</td></tr><tr><td>Qwen3-14B</td><td> $3 . 9 4 \times 1 0 ^ { - 3 }$ </td><td> $2 . 0 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td colspan="3">Lower under the dLLM-based mapping</td></tr><tr><td>DeepSeek-R1-Distill-Llama-8B</td><td>0.160</td><td> $1 . 1 7 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td> $9 . 0 4 \times 1 0 ^ { - 2 }$ </td><td> $2 . 1 1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Nemotron-Nano-9B-v2</td><td> $9 . 3 0 \times 1 0 ^ { - 3 }$ </td><td> $3 . 8 8 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>GPT-5.4 (reasoning effort xhigh)</td><td> $1 . 3 1 \times 1 0 ^ { - 2 }$ </td><td> $4 . 6 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Olmo-3-7B-Instruct</td><td> $1 . 6 3 \times 1 0 ^ { - 2 }$ </td><td> $\leq 1 0 ^ { - 5 }$ </td></tr><tr><td colspan="3">Higher under the dLLM-based mapping</td></tr><tr><td>GPT-OSS-20B</td><td> $1 . 8 2 \times 1 0 ^ { - 2 }$ </td><td>0.115</td></tr><tr><td>Claude Sonnet 4.6 (effort high)</td><td> $3 . 3 5 \times 1 0 ^ { - 4 }$ </td><td> $3 . 6 8 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 2: Over-generation probability at $\tau = 2 0 { , } 0 0 0 .$ Random-mapping values are from RareTrap (Eshra et al., 2026a), and dLLM-based values are medians over three runs (full results in Table 8).
<table><tr><td>Target</td><td>Map</td><td>Gens.</td><td>Best len.</td><td>Mean len.</td><td>Char. PPL</td></tr><tr><td rowspan="2">DeepSeek-R1-8B</td><td>Random mapping</td><td>34</td><td>4,096</td><td> $\overline { { 7 2 1  8 1 2 } }$ </td><td>27.5</td></tr><tr><td>dLLM-based mapping</td><td>28</td><td>4,096</td><td> $^ { 8 4 2 }  1 , 0 4 3$ </td><td>4.3</td></tr><tr><td rowspan="2">Qwen3-14B</td><td>Random mapping</td><td>39</td><td>1,772</td><td> $6 0 0  7 2 9$ </td><td>19.7</td></tr><tr><td>dLLM-based mapping</td><td>30</td><td>2,232</td><td> $7 5 1  8 6 2$ </td><td>5.4</td></tr><tr><td rowspan="2">Phi-4-reasoning</td><td>Random mapping</td><td>16</td><td>3,396</td><td> $\begin{array} { r } { 1 , 6 8 1 \to 1 , 5 4 7 } \\ { 1 , 9 1 5 \to 1 , 8 3 7 } \end{array}$ </td><td>22.7</td></tr><tr><td>dLLM-based mapping</td><td>14</td><td>4,038</td><td></td><td>4.4</td></tr></table>

Table 3: ThinkTrap medians over three seeds (per-seed values in Table 9). Gens. are generations completed within budget. Mean len. of the first → last five generations. PPL of the 12 longest-output prompts under the target.

Finding 2. ThinkTrap optimization progresses at similar rates across the two mappings, but RareTrap probability estimates do not consistently move in the same direction.

Under ThinkTrap, the search makes comparable progress through both mappings (Table 3). Mean output length rises under both on DeepSeek-R1-Distill-Llama-8B and Qwen3-14B. On Phi-4- reasoning, where long responses are observed on the initial population, neither mapping increases it within the generation budget (§4.4). We observe that across seeds the length-gain intervals of the two mappings overlap on every target (Table 9).

Across three seeds, RareTrap’s estimates under the dLLM-based mapping agree within a factor of 1.5 on seven of the ten targets (Appendix E). However, the two mappings do not always agree on how likely the behavior occurs.

## 4.2.2 Comparing models across mappings

As per Finding 2, the two mappings’ RareTrap estimates diverge on most targets. Here we ask what this divergence means for comparing models.

Finding 3. The discrepancy between the random mapping and the dLLM-based mapping informs whether a model over-generates more on natural or unnatural prompts.

Relative to the random mapping, the dLLMbased mapping lowers estimated event probabilities on five targets and raises them on two: GPT-OSS-20B and Claude Sonnet 4.6 (Table 2). On seven of the ten targets, variation across the three dLLMbased seeds stays within a factor of 1.5, while most differences between mappings are several times larger (Appendix E). Because the shifts go in opposite directions across targets, they also change the relative risk ranking of models. Under the random mapping, Sonnet 4.6 appears about 40× less likely to over-generate than GPT-5.4, but under the dLLM-based mapping, their estimated probabilities differ by at most a factor of 1.6. The mappings also differ in character distribution. The random mapping’s hits are dominated by rare-script characters, 70–100% non-ASCII across 6–7 Unicode scripts, whereas the dLLM-based mapping’s hits are English (§4.3, Appendix C).

![](images/46f75e803c89efc3de9cdf362ca8c747f5eaff632b6db4db4cba20f151767dc2.jpg)  
Figure 3: Share of total compute spent on the dLLMbased mapping in the ThinkTrap runs.

## 4.2.3 Compute cost

Finding 4. The dLLM-based mapping adds limited additional compute to the framework.

Unlike the near-zero cost of random mapping, dLLM-based mapping adds a fixed cost per prompt, independent of the target model. Cola-DLM has approximately 2.3B parameters and generates a fixed number of blocks per prompt, whereas targetmodel inference costs scale with model size and response length. Figure 3 shows the dLLM-based mapping’s share of compute on three models across the ThinkTrap runs. The dLLM-based mapping accounts for 3–11% of total compute. Its share is highest for DeepSeek-R1-Distill-Llama-8B, the smallest target, and lowest for Phi-4-reasoning, which produces the longest responses. Because the search procedure remains unchanged, this mapping cost constitutes the entire computational overhead of the substitution. Its relative share should decrease further for larger target models or longer responses. Appendix G compares the per-prompt cost and the number of search steps of the two mappings.

<table><tr><td rowspan="2">Target</td><td colspan="2">Target scorer</td><td colspan="2">Reference scorer</td></tr><tr><td>Random</td><td>Ours</td><td>Random</td><td>Ours</td></tr><tr><td>DeepSeek-R1-Llama-8B</td><td> $6 . 8 \times 1 0 ^ { 6 }$ </td><td>4.9</td><td> $2 . 4 \times 1 0 ^ { 4 }$ </td><td>4.7</td></tr><tr><td>Phi-4-reasoning</td><td> $2 . 0 \times 1 0 ^ { 4 }$ </td><td>3.2</td><td> $3 . 5 \times 1 0 ^ { 4 }$ </td><td>3.4</td></tr><tr><td>Qwen3.5-9B</td><td> $1 . 5 \times 1 0 ^ { 4 }$ </td><td>3.3</td><td> $1 . 7 \times 1 0 ^ { 4 }$ </td><td>3.6</td></tr><tr><td>Qwen3-14B</td><td> $1 . 7 \times 1 0 ^ { 5 }$ </td><td>3.6</td><td> $4 . 2 \times 1 0 ^ { 4 }$ </td><td>3.8</td></tr><tr><td>Mistral-7B-v0.3</td><td> $1 . 1 \times 1 0 ^ { 5 }$ </td><td>3.2</td><td> $5 . 4 \times 1 0 ^ { 4 }$ </td><td>3.1</td></tr><tr><td>Nemotron-Nano-9B-v2</td><td> $5 . 7 \times 1 0 ^ { 4 }$ </td><td>3.5</td><td> $3 . 6 \times 1 0 ^ { 4 }$ </td><td>3.6</td></tr><tr><td>GPT-OSS-20B</td><td> $3 . 0 \times 1 0 ^ { 3 }$ </td><td>4.1</td><td> $3 . 0 \times 1 0 ^ { 3 }$ </td><td>4.1</td></tr><tr><td>GPT-5.4</td><td>一</td><td>一</td><td> $\mathrm { 3 . 5 \times 1 0 ^ { 4 } }$ </td><td>4.1</td></tr><tr><td>Claude Sonnet 4.6</td><td></td><td>一</td><td> $1 . 0 \times 1 0 ^ { 5 }$ </td><td>4.8</td></tr></table>

Table 4: Median character-level perplexity of the hit prompts under the random mapping and the dLLMbased mapping (Ours), scored by the target model itself and by a reference scorer (Llama-3.1-8B). The API targets (–) expose no log-probabilities.

## 4.3 RQ2 Findings

Here, we ask whether the prompts each mapping finds read as natural language. We compare their character-level perplexity in both frameworks, test them against a perplexity filter, and check whether conditioning on the behavior changes their naturalness.

## 4.3.1 Prompt perplexity

Finding 5. In both frameworks, the dLLMbased mapping’s prompts are far closer to natural text than the random mapping’s, though the gap is smaller under ThinkTrap.

Table 4 shows that, across all target models, RareTrap hits under the random mapping have character-level perplexities three to six orders of magnitude higher than those under the dLLMbased mapping. This gap holds even for Qwen3- 14B, which shares its tokenizer with the surrogate, and is observed both when scoring prompts with the target model and with a reference scorer, Llama-3.1-8B (Grattafiori et al., 2024), which is also used for the two API targets.

The dLLM-based mapping’s hits are short English requests with the marks of a small language model, such as made-up product names, a mislabeled term, a sentence cut off mid-way (“. . . if you were overly eager to take parted money to the Pitavumium to find out how you could use it. . . ”). The random mapping’s hits are strings of rare-script characters, such as Hangul, Arabic, and CJK (§4.2.2). Appendix C presents one example for each target.

To test what these numbers mean for an input filter, we calibrate a perplexity filter under the reference scorer on 735 real user prompts from OASST1 (Köpf et al., 2023), with its threshold set at a 1% or 5% false-positive rate. As Table 5 shows, the filter flags every random-mapping hit. On the other hand, a third of the dLLM-based mapping’s hits pass the filter at 1% false-positive rate. Naturalness therefore removes the extreme signal that makes random-mapping prompts trivial to detect. The Llama Guard 3 content-safety guard (Inan et al., 2023) flags few hits from either mapping, since over-generation prompts are benign in content, and the ProtectAI prompt-injection guard (ProtectAI.com, 2024) flags the random mapping’s hits about three times as often.

<table><tr><td></td><td></td><td></td><td colspan="2">PPL filter</td><td colspan="2">Guard models</td></tr><tr><td>Prompts</td><td>Count</td><td>Char. PPL</td><td>1%</td><td>5%</td><td>Llama G.3</td><td>ProtectAI</td></tr><tr><td>Real (OASST1)</td><td>735</td><td>1.9</td><td>1.1</td><td>5.0</td><td>3.4</td><td>0.5</td></tr><tr><td>Random mapping hits</td><td>805</td><td>3.8 × 104</td><td>100.0</td><td>100.0</td><td>20.4</td><td>32.3</td></tr><tr><td>dLLM-based mapping hits</td><td>293</td><td>3.6</td><td>67.9</td><td>94.2</td><td>16.4</td><td>9.6</td></tr></table>

Table 5: Percentage of prompts flagged by a characterlevel perplexity filter calibrated on OASST1 at a 1% or 5% false-positive rate, and by two guard models: Llama Guard 3 and the ProtectAI DeBERTa-v3 promptinjection guard.
<table><tr><td></td><td colspan="3">Random mapping</td><td colspan="3">Ours</td></tr><tr><td>Target</td><td>Hits</td><td>Level 0</td><td>Ratio</td><td>Hits</td><td>Level 0</td><td>Ratio</td></tr><tr><td>DeepSeek-R1-Llama-8B</td><td>1</td><td></td><td></td><td>148</td><td>220</td><td>0.67</td></tr><tr><td>Qwen3-14B</td><td>53</td><td>59</td><td>0.92</td><td>95</td><td>220</td><td>0.51</td></tr><tr><td>Mistral-7B-v0.3</td><td>59</td><td>58</td><td>1.02</td><td>286</td><td>220</td><td>1.01</td></tr><tr><td>Nemotron-Nano-9B-v2</td><td>50</td><td>57</td><td>0.87</td><td>97</td><td>220</td><td>0.59</td></tr><tr><td>GPT-5.4</td><td>40</td><td>54</td><td>0.79</td><td>261</td><td>219</td><td>0.93</td></tr><tr><td>Claude Sonnet 4.6</td><td>65</td><td>54</td><td>1.05</td><td>110</td><td>219</td><td>0.46</td></tr></table>

Table 6: Median token-level perplexity (reference scorer) of RareTrap hits and level-0 prompts. Ratio is length-adjusted, with below 1 meaning more natural hits. RareTrap’s DeepSeek run used direct Monte Carlo, so it has no conditional levels.

The perplexity gap appears on every target and seed of the ThinkTrap runs (Table 3), with the dLLM-based mapping lower though by a much smaller factor. The gap is smaller because the two random mappings decode to different kinds of strings. RareTrap’s geometry-aware mapping, centered on the embedding mean (§3.1), yields prompts made almost entirely of rare singlecharacter tokens, whereas ThinkTrap’s mapping yields mostly common subword tokens, which are far more predictable per character. In the initial populations on DeepSeek-R1-Distill-Llama-8B, the median RareTrap random-mapping prompt is 100% non-ASCII across 7 Unicode scripts, against 14% non-ASCII for ThinkTrap’s.

## 4.3.2 Selection toward more natural prompts

Finding 6. Conditioning on the behavior tends to select more natural prompts under the dLLMbased mapping, and much less so under the random mapping.

Table 6 compares RareTrap’s hits with the initial population (level 0). Under the dLLM-based mapping, hits are more natural on four of the six targets, while the random mapping changes little. We use token-level perplexity here, unlike in Finding 5, because hits and level-0 prompts come from one mapping and are scored by one model, and under the dLLM-based mapping all have 47 tokens, so per-token perplexity compares the likelihood of whole prompts. Character-level perplexity would also reflect differences in token length.

## 4.4 Discussion

Phi-4-reasoning. Under RareTrap’s greedy decoding, nearly every sample loops verbatim to the cap under both mappings, yielding an estimated $p _ { \mathcal { F } } ( \tau )$ of 1 (Table 2). Under ThinkTrap’s sampling at temperature 1.0, at most 3% of responses reach the cap. Phi-4-reasoning’s outputs are the longest of the three targets (median 1,500–1,900 tokens), limiting the number of generations and leaving neither mapping able to increase mean output length (Table 3).

Prompt length. The mappings produce similar token counts in their respective vocabularies. However, random-mapping prompts in RareTrap are much shorter in characters (about 40 vs. 200), since they consist mostly of single rare characters. Think-Trap prompts are 60–77 characters under both mappings. Being shorter than our RareTrap prompts may partly explain why our ThinkTrap prompts have higher character-level perplexity (4.3–5.4 vs. 3–5 for RareTrap), since early characters have less preceding context. The two settings also score different prompts, the 12 longest-output prompts in ThinkTrap and the hits in RareTrap.

## 5 Conclusion

Substituting the random mapping of continuousspace attacks with a pretrained continuous dLLM preserves the search procedure with limited additional compute, while producing prompts far closer to natural text. In RareTrap, hit prompts have character-level perplexity of 3–5, compared with $1 0 ^ { 3 } – 1 0 ^ { 7 }$ under the random mapping. The resulting over-generation probability estimates can shift in either direction, changing the relative risk estimates for models such as Claude Sonnet 4.6 and GPT-5.4. Under ThinkTrap, the dLLM-based mapping’s prompts stay far more natural.

## Limitations

The random-mapping baseline is a single published run, and our smallest estimates rest on few distinct prompts. We use a single dLLM, sampled unconditionally, so its prompts read as generic text rather than as typical user requests, and our estimates describe that distribution rather than real deployment traffic. We only consider over-generation, and measure naturalness by perplexity rather than human judgment. The ThinkTrap comparison covers only a few targets.

## Ethical Considerations

This work has dual-use implications. More natural over-generation prompts may be harder to distinguish from ordinary user requests than prompts produced by random mappings, although a calibrated character-level perplexity filter still flags most of them (Table 5). We report this method to support the evaluation and mitigation of costly model behaviors, using a public model and established search procedures. Our findings motivate output-side defenses, including repetition detection and per-request reasoning budgets. The prompts are benign in content, and the risk studied is excessive computation, which can increase inference cost and latency, rather than the generation of harmful content. We evaluate API models through Amazon Bedrock under its standard terms of service and impose bounded generation budgets. No human subjects or personal data are involved in this study.

## References

Marah Abdin, Sahaj Agarwal, Ahmed Awadallah, Vidhisha Balachandran, Harkirat Behl, Lingjiao Chen, Gustavo de Rosa, Suriya Gunasekar, Mojan Javaheripi, Neel Joshi, et al. 2025. Phi-4-reasoning technical report. arXiv preprint arXiv:2504.21318.

Ai2. 2025. Olmo-3-7B-Instruct. https://huggingface.co/allenai/ Olmo-3-7B-Instruct.

Gabriel Alon and Michael Kamfonas. 2023. Detecting language model attacks with perplexity. arXiv preprint arXiv:2308.14132.

Anthropic. 2026. Claude sonnet 4.6. https://www. anthropic.com/news/claude-sonnet-4-6. Accessed: 2026-09-25.

Siu-Kui Au and James L Beck. 2001. Estimation of small failure probabilities in high dimensions by sub-

set simulation. Probabilistic engineering mechanics, 16(4):263–277.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. 2023. Jailbreaking black box large language models in twenty queries. Preprint, arXiv:2310.08419.

Yiming Chen, Zexin Li, Xianghu Yue, Robby T. Tan, and Haizhou Li. 2026. NaturalSloth: Revisiting denial-of-service attacks on large language models. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 19685–19702, San Diego, California, United States. Association for Computational Linguistics.

DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H. Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, Curtis Hawthorne, Rémi Leblond, Will Grathwohl, and Jonas Adler. 2022. Continuous diffusion for categorical data. Preprint, arXiv:2211.15089.

Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu, and Han Qiu. 2025. An engorgio prompt makes large language model babble on. In International Conference on Learning Representations, volume 2025, pages 67280–67307.

Elsayed Eshra, Ali Al-Lawati, Dongwon Lee, and Suhang Wang. 2026a. Quantifying behavioral tails in black-box language models. Preprint, arXiv:2609.33638.

Elsayed Eshra, Ali Al-Lawati, Dongwon Lee, and Suhang Wang. 2026b. Raretrap populations. https://huggingface.co/datasets/ aliwister/raretrap-populations. Hugging Face dataset.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, et al. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Hongcan Guo, Qinyu Zhao, Yian Zhao, Shen Nie, Rui Zhu, Qiushan Guo, Feng Wang, Tao Yang, Hengshuang Zhao, Guoqiang Wei, and Yan Zeng. 2026. Continuous latent diffusion language model. Preprint, arXiv:2605.06548.

Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, and Bin Hu. 2024. COLD-Attack: Jailbreaking LLMs with stealthiness and controllability. In International Conference on Machine Learning.

Nikolaus Hansen and Andreas Ostermeier. 2001. Completely derandomized self-adaptation in evolution strategies. Evolutionary Computation, 9(2):159–195.

Keya Hu, Linlu Qiu, Yiyang Lu, Hanhong Zhao, Tianhong Li, Yoon Kim, Jacob Andreas, and Kaiming He. 2026. ELF: Embedded language flows. Preprint, arXiv:2605.10938.

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and Madian Khabsa. 2023. Llama guard: LLMbased input-output safeguard for human-AI conversations. arXiv preprint arXiv:2312.06674.

Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. 2023. Baseline defenses for adversarial attacks against aligned language models. arXiv preprint arXiv:2309.00614.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Jiaxiu Jiang, Jingjing Ren, Wenbo Li, Bo Wang, Haoze Sun, Yijun Yang, Jianhui Liu, Yanbing Zhang, Shenghe Zheng, Yuan Zhang, Haoyang Huang, Nan Duan, and Wangmeng Zuo. 2026. TextLDM: Language modeling with continuous latent diffusion. Preprint, arXiv:2605.07748.

Erik Jones, Meg Tong, Jesse Mu, Mohammed Mahfoud, Jan Leike, Roger Grosse, Jared Kaplan, William Fithian, Ethan Perez, and Mrinank Sharma. 2025. Forecasting rare language model behaviors. Preprint, arXiv:2502.16797.

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Minh Nguyen, Oliver Stanley, Richárd Nagyfi, et al. 2023. OpenAssistant conversations – democratizing large language model alignment. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, volume 36.

Abhinav Kumar, Jaechul Roh, Ali Naseh, Marzena Karpinska, Mohit Iyyer, Amir Houmansadr, and Eugene Bagdasarian. 2025. Overthink: Slowdown attacks on reasoning llms. arXiv preprint arXiv:2502.02542.

Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori B. Hashimoto. 2022. Diffusion-LM improves controllable text generation. In Advances in Neural Information Processing Systems, volume 35.

Xinyu Li, Tianjin Huang, Ronghui Mu, Xiaowei Huang, and Gaojie Jin. 2025a. Pot: Inducing overthinking in llms via black-box iterative optimization. arXiv preprint arXiv:2508.19277.

Yunzhe Li, Jianan Wang, Hongzi Zhu, James Lin, Shan Chang, and Minyi Guo. 2025b. Thinktrap: Denialof-service attacks against black-box llm services via infinite thinking. arXiv preprint arXiv:2512.07086.

Buyun Liang, Jinqi Luo, Liangzu Peng, Kwan Ho Ryan Chan, Darshan Thaker, Kaleab A Kinfu, Fengrui Tian, Hamed Hassani, and Rene Vidal. 2026. REAL-ISTA: Realistic latent adversarial attacks that elicit LLM hallucinations. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 73401–73448. PMLR.

Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. 2024. AutoDAN: Generating stealthy jailbreak prompts on aligned large language models. In International Conference on Learning Representations.

Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, and Kilian Q. Weinberger. 2023. Latent diffusion for language generation. In Advances in Neural Information Processing Systems, volume 36.

Jason S. Lucas, Matt Murtagh White, Ali Al-Lawati, Uchendu Uchendu, Adaku Uchendu, and Dongwon Lee. 2026. DIA-HARM: Dialectal disparities in harmful content detection across 50 English dialects. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3171–3214, San Diego, California, United States. Association for Computational Linguistics.

NVIDIA. 2025. NVIDIA-Nemotron-Nano-9B-v2. https://huggingface.co/nvidia/ NVIDIA-Nemotron-Nano-9B-v2.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card. Preprint, arXiv:2508.10925.

OpenAI. 2026. Gpt-5.4. https://openai.com/ index/introducing-gpt-5-4/. Accessed: 2026- 09-25.

Anselm Paulus, Arman Zharmagambetov, Chuan Guo, Brandon Amos, and Yuandong Tian. 2025. AdvPrompter: Fast adaptive adversarial prompting for LLMs. In International Conference on Machine Learning.

Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. 2022. Red teaming language models with language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3419–3448.

ProtectAI.com. 2024. Fine-tuned DeBERTa-v3-base for prompt injection detection.

Qwen Team. 2026. Qwen3.5-9B. https:// huggingface.co/Qwen/Qwen3.5-9B.

Ali Rahimi and Benjamin Recht. 2007. Random features for large-scale kernel machines. Advances in neural information processing systems, 20.

Ilia Shumailov, Yiren Zhao, Daniel Bates, Nicolas Papernot, Robert Mullins, and Ross Anderson. 2021. Sponge examples: Energy-latency attacks on neural networks. In 2021 IEEE European Symposium on Security and Privacy (EuroS&P), pages 212–231.

Tianxiang Sun, Yunfan Shao, Hong Qian, Xuanjing Huang, and Xipeng Qiu. 2022. Black-box tuning for language-model-as-a-service. In International Conference on Machine Learning, pages 20841–20855. PMLR.

Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2023. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery. Advances in Neural Information Processing Systems, 36:51008– 51025.

Marcus Williams, Hannah Sheahan, Cameron Raymond, Tomek Korbak, Deng Pan, Peilin Yang, Leon Maksin, Ningyi Xie, Phillip Guo, Ian Kivlichan, and Micah Carroll. 2026. Predicting llm safety before release by simulating deployment. Preprint, arXiv:2607.07184.

Gabriel Wu and Jacob Hilton. 2024. Estimating the probabilities of rare outputs in language models. Preprint, arXiv:2410.13211.

Zijun Wu, Yongchang Hao, and Lili Mou. 2026. Ultralow-dimensional prompt tuning via random projection. In Proceedings of the 19th Conference of the European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1284–1303.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Zhihan Yang, Wei Guo, Shuibai Zhang, Subham Sekhar Sahoo, Yongxin Chen, Arash Vahdat, Morteza Mardani, and John Thickstun. 2026. Continuous diffusion scales competitively with discrete diffusion for language. Preprint, arXiv:2605.18530.

Collin Zhang, Tingwei Zhang, and Vitaly Shmatikov. 2026. Adversarial decoding: Generating readable documents for adversarial objectives. In Findings of the Association for Computational Linguistics: EACL 2026, pages 2053–2068, Rabat, Morocco. Association for Computational Linguistics.

Zhengli Zhao, Dheeru Dua, and Sameer Singh. 2018. Generating natural adversarial examples. In International Conference on Learning Representations.

Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, and Tong Sun. 2023. AutoDAN: Interpretable gradientbased adversarial attacks on large language models. Preprint, arXiv:2310.15140.

## A dLLM-based mapping details

We use the input noise of Cola-DLM (Guo et al., 2026) as the search latent. Cola-DLM generates text in blocks of 16 positions, each generated from Gaussian noise of shape 16 × 16. We split a single vector z into these noise blocks and run the released inference procedure with them in place of the noise it would otherwise sample. Under fixed inference settings, a fixed batch size, and greedy decoding, this defines a deterministic mapping from z to text.

We initialize each sample with a single document-boundary token (<|endoftext|>) so that generation is unconditional. This token takes the first position of the first block and receives no noise. With L generated positions the latent dimension is D = 16L, with L = 47 (D = 752, three blocks) for RareTrap and L = 15 (D = 240, one block) for ThinkTrap. We use the released inference settings, namely 16 Euler steps, classifierfree guidance scale 7.0, repetition penalty 1.1, and greedy decoding. Cola-DLM always generates all L positions, and the prompt is cut at the first endof-sequence token.

## B LLMs used

Open-weight models are loaded from Hugging Face at the pinned revision shown (first seven characters of the commit hash). The two API models are reached through Amazon Bedrock, which exposes no content-addressable revision, so the provider model identifier is the only available pin.

Open-weight targets.

• DeepSeek-R1-Distill-Llama-8B (DeepSeek-AI, 2025) (deepseek-ai/DeepSeek-R1-Distill-Llama-8B, 6a6f4aa), an 8B Llama model distilled from DeepSeek-R1 reasoning traces.

• Phi-4-reasoning (Abdin et al., 2025) (microsoft/Phi-4-reasoning, 1de18ec), a 14B reasoning model fine-tuned from Phi-4.

• Qwen3.5-9B (Qwen Team, 2026) (Qwen/Qwen3.5-9B, c202236), a 9B model from the Qwen3.5 family.

• Qwen3-14B (Yang et al., 2025) (Qwen/Qwen3-14B, 40c0698), a 14B dense model with a switchable thinking mode.

• Mistral-7B-Instruct-v0.3 (Jiang et al., 2023) (mistralai/Mistral-7B-Instruct-v0.3, c170c70), a 7B instruction-tuned model without a separate reasoning mode.

• NVIDIA-Nemotron-Nano-9B-v2 (NVIDIA, 2025) (nvidia/NVIDIA-Nemotron-Nano-9B-v2, 6533e8d), a 9B hybrid Mamba–Transformer reasoning model.

• Olmo-3-7B-Instruct (Ai2, 2025) (allenai/Olmo-3-7B-Instruct, 6e5971d), a 7B instruction-tuned model from the fully open Olmo family. Its dLLM-based mapping estimate is an upper bound (§4.2).

• GPT-OSS-20B (OpenAI, 2025) (openai/gpt-oss-20b, 6cee5e8), OpenAI’s open-weight mixture-of-experts reasoning model.

DeepSeek-R1-Distill-Llama-8B, Phi-4-reasoning, and Qwen3-14B are also the three ThinkTrap targets, at the same revisions.

## API targets.

• GPT-5.4 (OpenAI, 2026) (openai.gpt-5.4-2026-03-05), OpenAI frontier reasoning model, queried at reasoning effort xhigh.

• Claude Sonnet 4.6 (Anthropic, 2026) (global.anthropic.claude-sonnet-4-6), Anthropic frontier model, queried with adaptive thinking at effort high.

## Mapping models.

• Cola-DLM (Guo et al., 2026) (ByteDance-Seed/Cola-DLM), a continuous latent diffusion language model, consisting of a Text VAE and a block-causal diffusion-transformer prior trained with flow matching. It implements h (Appendix A).

• Qwen3-0.6B (Yang et al., 2025) (Qwen/Qwen3-0.6B), the surrogate whose token-embedding table the random mapping decodes against with nearest-neighbor lookup. It never serves as a target.

## C Example prompts and responses

Figure 4 shows, for each RareTrap target, a hit prompt found under the random mapping and one found under the dLLM-based mapping, namely the hit whose character-level perplexity under the reference scorer is closest to the median of Table 4. We skip hits already shown for another target, since the direct Monte Carlo targets draw their first-seed hits from the same initial samples, and the few hits that contain the Unicode replacement character or a character our fonts cannot display. Randommapping prompts mix scripts (Hangul, Arabic presentation forms, CJK, and others), and the figure is rendered with Noto Sans fonts covering these scripts.

A final-level non-hit. On Nemotron-Nano-9Bv2, the level-2 prompt “How should we implement buffer making with SQL, and thus seeking to alternate web-side and database-side? I am using yidi 8 SQL5for creating a buffer in web-side server. . . ” produced 2,656 tokens, short of the threshold.

Figure 5 shows the prompt with the longest output in each ThinkTrap run of Table 3, for the first seed, rendered the same way. Because CMA-ES adapts its search distribution separately on each target, the prompts differ across targets. On DeepSeek-R1-Distill-Llama-8B, the dLLM-based mapping’s prompt reads as English, while the random mapping’s prompt mixes Cyrillic, Latin, kana, Hangul, and Han characters with no coherent meaning in any one script.

## D RareTrap random-mapping populations

Table 7 summarizes the random-mapping populations released by RareTrap (Eshra et al., 2026b), which supply the baseline estimates of Table 2 and the random-mapping prompts we score in §4.3. Each population records every prompt Subset Simulation evaluated, with its output length. Targets whose first 200 samples already contained at least 20 hits were estimated by direct Monte Carlo and have a single level. A hit is a prompt whose response reaches the 20,000-token threshold. The median prompt is 100% non-ASCII on every target. Pooled over the nine targets with dLLM-based hits, the hits comprise the 805 unique prompts of Table 5.

## E Per-seed results

Tables 8 and 9 list the per-seed values behind the medians of Tables 2 and 3. Each interval is a Student-t 95% interval over the three seeds.

![](images/51d3d1f0e761dadd41113da04c1940c826be1cdecc8d29c138f554c7307a0de2.jpg)  
Figure 4: One hit prompt (response of at least 20,000 tokens) per target, random mapping left and dLLM-based mapping right, with its character-level perplexity under the reference scorer.

<table><tr><td colspan="2">Random mapping</td><td colspan="2">dLLM-based mapping</td></tr><tr><td>DeepSeek-R1-Distill-Llama-8B</td><td>4,096 output tokens</td><td>DeepSeek-R1-Distill-Llama-8B</td><td>4,043 output tokens</td></tr><tr><td colspan="2">Prompt: cイト.encryptapot&lt;nav 3pe(importchedkb no3o doublyplier和平</td><td colspan="2">Prompt: Specific, which explain possible mechanisms for</td></tr><tr><td colspan="2">Qwen3-14B 1,911 output tokens</td><td>each problem and will back up the defects</td><td></td></tr><tr><td colspan="2">Prompt: uissemetics蛮ilityley cach refl磅trightPGA挠eam霓</td><td>Qwen3-14B Prompt: : d in the returns 5/6 of 20].  The sum</td><td>2,232 output tokens</td></tr><tr><td colspan="2">resultMap Phi-4-reasoning 3,630 output tokens</td><td></td><td></td></tr></table>

Figure 5: The longest-output prompt per target and mapping in the ThinkTrap runs (first seed), with its output length. ←- marks a newline.

<table><tr><td>Target</td><td>Prompts</td><td>Levels</td><td>Hits</td><td>Chars</td><td>Tokens</td></tr><tr><td>DeepSeek-R1-Llama-8B</td><td>200</td><td>1</td><td>32</td><td>40</td><td>118</td></tr><tr><td>Phi-4-reasoning</td><td>200</td><td>1</td><td>200</td><td>44</td><td>345</td></tr><tr><td>Qwen3.5-9B</td><td>200</td><td>1</td><td>77</td><td>43</td><td>117</td></tr><tr><td>Qwen3-14B</td><td>2,800</td><td>3</td><td>81</td><td>43</td><td>48</td></tr><tr><td>Mistral-7B-v0.3</td><td>1,900</td><td>2</td><td>190</td><td>44</td><td>130</td></tr><tr><td>Nemotron-Nano-9B-v2</td><td>2,800</td><td>3</td><td>88</td><td>43</td><td>126</td></tr><tr><td>Olmo-3-7B-Instruct</td><td>1,900</td><td>2</td><td>40</td><td>43</td><td>155</td></tr><tr><td>GPT-OSS-20B</td><td>1,900</td><td>2</td><td>49</td><td>43</td><td>180</td></tr><tr><td>GPT-5.4</td><td>1,900</td><td>2</td><td>33</td><td>40</td><td>120</td></tr><tr><td>Claude Sonnet 4.6</td><td>3,700</td><td>4</td><td>57</td><td>40</td><td>142</td></tr></table>

Table 7: RareTrap’s released random-mapping populations. Prompts are all evaluated samples across Subset Simulation levels, and hits are those whose response reached 20,000 tokens. Chars and Tokens are the median prompt length in characters and in the target’s tokens.

## F Search space and dimensionality

As described in Appendix A, allocating L positions sets the search dimension $D = 1 6 L$ . Both methods operate on input-noise vectors z ∈ R<sup>D</sup>. RareTrap applies Subset Simulation under the standard Gaussian reference $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { D } )$ , while ThinkTrap adapts a search distribution using CMA-ES.

For each objective evaluation, Cola-DLM receives the full vector z and processes it block by block to generate the complete prompt, using fixed inference and batching settings with greedy decoding. The target model’s response is then scored, and this behavioral score guides subsequent latent proposals.

## G Cost of the dLLM-based mapping

The random mapping needs only a matrix product and a nearest-neighbor lookup per prompt, so its cost is negligible next to querying the target. The dLLM-based mapping instead runs Cola-DLM once per prompt. We profiled it at RareTrap’s settings $( L = 4 7 , D = 7 5 2 )$ with DeepSeek-R1- Distill-Llama-8B as the target, on a single A100 GPU. Mapping a batch of 10, 25, or 50 prompts took 11, 16, and 25 seconds, less than the time the target took to decode the first 256 response tokens for the same batch, while decoding up to the 20,003-token cap is projected, by linear extrapolation from those 256 tokens, to take roughly 100– 200 times longer than the mapping. The number of search steps is unchanged, since both mappings use the same number of samples per Subset Simulation level and the same population per CMA-ES generation. Under ThinkTrap’s fixed token budget, however, the dLLM-based mapping completes fewer generations, because its prompts draw longer responses (Table 3).

## H Perplexity measurement details

Prompts are scored as raw text with the scoring model’s tokenizer, adding a beginning-of-sequence token where the tokenizer defines one and predicting each token from the preceding ones, and no chat template is applied. Sequences are right-padded so padding does not affect the causal positions of real tokens. Token-level perplexity is exp of the mean per-token NLL. Character-level perplexity divides the total NLL of the prompt by its number of Unicode characters before exponentiating. Medians are over unique prompts. Target models were loaded in bfloat16 when used as scorers. The reference scorer is the base model Llama-3.1-8B (Grattafiori et al., 2024) (meta-llama/Llama-3.1-8B), which is not one of the targets, run in float32, and DeepSeek-R1-Distill-Llama-8B is fine-tuned from it. The filter thresholds of Table 5 are the 99th and 95th percentiles of the reference scorer’s perplexity on the 735 OASST1 prompts, which were sampled to match the word-count distribution of the dLLMbased mapping’s level-0 prompts.

<table><tr><td rowspan="2">Target</td><td rowspan="2">Random mapping</td><td colspan="5">dLLM-based mapping</td></tr><tr><td>Seed 1</td><td>Seed 2</td><td>Seed 3</td><td>Median</td><td>95% interval</td></tr><tr><td>Phi-4-reasoning</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.995</td><td>1.000</td><td>[0.991, 1.000]</td></tr><tr><td>Qwen3.5-9B</td><td>0.385</td><td>0.425</td><td>0.465</td><td>0.485</td><td>0.465</td><td>[0.387, 0.541]</td></tr><tr><td>Qwen3-14B</td><td> $3 . 9 4 \times 1 0 ^ { - 3 }$ </td><td> $3 . 5 3 \times 1 0 ^ { - 3 }$ </td><td> $2 . 0 2 \times 1 0 ^ { - 3 }$ </td><td> $1 . 0 9 \times 1 0 ^ { - 3 }$ </td><td> $2 . 0 2 \times 1 0 ^ { - 3 }$ </td><td> $[ 0 . 4 6 , 8 . 5 ] \times 1 0 ^ { - 3 }$ </td></tr><tr><td>DeepSeek-R1-Distill-Llama-8B</td><td>0.160</td><td> $1 . 3 9 \times 1 0 ^ { - 2 }$ </td><td> $1 . 1 7 \times 1 0 ^ { - 2 }$ </td><td> $1 . 1 2 \times 1 0 ^ { - 2 }$ </td><td> $1 . 1 7 \times 1 0 ^ { - 2 }$ </td><td> $[ \dot { 0 } . 9 2 , 1 . 6 2 ] \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td> $9 . 0 4 \times 1 0 ^ { - 2 }$ </td><td> $2 . 0 0 \times 1 0 ^ { - 3 }$ </td><td> $2 . 8 6 \times 1 0 ^ { - 3 }$ </td><td> $2 . 1 1 \times 1 0 ^ { - 3 }$ </td><td> $2 . 1 1 \times 1 0 ^ { - 3 }$ </td><td> $\left[ 1 . 4 2 , 3 . 7 0 \right] \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Nemotron-Nano-9B-v2</td><td> $9 . 3 0 \times 1 0 ^ { - 3 }$ </td><td> $1 . 4 8 \times 1 0 ^ { - 3 }$ </td><td> $3 . 8 8 \times 1 0 ^ { - 4 }$ </td><td> $3 . 0 4 \times 1 0 ^ { - 4 }$ </td><td> $3 . 8 8 \times 1 0 ^ { - 4 }$ </td><td> $[ 0 . 0 7 , 4 . 6 ] \times 1 0 ^ { - 3 }$ </td></tr><tr><td>GPT-5.4</td><td> $1 . 3 1 \times 1 0 ^ { - 2 }$ </td><td> $4 . 5 1 \times 1 0 ^ { - 3 }$ </td><td> $4 . 6 4 \times 1 0 ^ { - 3 }$ </td><td> $5 . 7 8 \times 1 0 ^ { - 3 }$ </td><td> $4 . 6 4 \times 1 0 ^ { - 3 }$ </td><td> $[ \dot { 3 } . 5 3 , 6 . 9 3 ] \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Olmo-3-7B-Instruct</td><td> $1 . 6 3 \times 1 0 ^ { - 2 }$ </td><td> $\leq 1 0 ^ { - 5 \dagger }$ </td><td> $\leq 1 0 ^ { - 6 }$ </td><td> $\leq 1 0 ^ { - 5 }$ </td><td> $\leq 1 0 ^ { - 5 }$ </td><td></td></tr><tr><td>GPT-OSS-20B</td><td> $1 . 8 2 \times 1 0 ^ { - 2 }$ </td><td>0.105</td><td>0.140</td><td>0.115</td><td>0.115</td><td>[0.083, 0.172]</td></tr><tr><td>Claude Sonnet 4.6</td><td> $3 . 3 5 \times 1 0 ^ { - 4 }$ </td><td> $3 . 3 1 \times 1 0 ^ { - 3 }$ </td><td> $4 . 4 8 \times 1 0 ^ { - 3 }$ </td><td> $3 . 6 8 \times 1 0 ^ { - 3 }$ </td><td> $3 . 6 8 \times 1 0 ^ { - 3 }$ </td><td> $[ 2 . 5 9 , 5 . 5 5 ] \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 8: RareTrap estimates per seed under the dLLM-based mapping, with the single published random-mapping run. Intervals are on the $\log _ { 1 0 }$ scale, capped at 1. Olmo-3 entries are upper bounds, since no run reached the threshold.

<table><tr><td rowspan="2">Target</td><td rowspan="2">Map</td><td colspan="2">Length gain (tokens)</td><td>Best len.</td><td colspan="2">Char. PPL (top 12)</td></tr><tr><td>per seed</td><td>95% int.</td><td>per seed</td><td>per seed</td><td>95% int.</td></tr><tr><td rowspan="2">DeepSeek-R1-8B</td><td>Random mapping</td><td>227 / 142 / –9</td><td>[-178, 418]</td><td>4,096 / 2,937 / 4,096</td><td>39.6 / 18.7 / 27.5</td><td>[10.8,69.3]</td></tr><tr><td>dLLM-based mapping</td><td>201 / 169 / 171</td><td>[136, 225]</td><td>4,043 / 4,096 / 4,096</td><td>3.6 / 11.3 / 4.3</td><td>[1.2, 25.4]</td></tr><tr><td rowspan="2">Qwen3-14B</td><td>Random mapping</td><td>134 / 83 / 144</td><td>[40, 200]</td><td>1,911 / 1,740 / 1,772</td><td>26.5 / 19.7 / 19.7</td><td>[14.2, 33.4]</td></tr><tr><td>dLLM-based mapping</td><td>103 / 90 / 407</td><td>[−245,645]</td><td>2,232 / 1,963 / 2,317</td><td>6.7 / 5.4 / 3.8</td><td>[2.5, 10.6]</td></tr><tr><td rowspan="2">Phi-4-reasoning</td><td>Random mapping</td><td>-135 /-11 /-114</td><td>[-250,77]</td><td>3,630 / 3,396 / 3,290</td><td>23.7 / 14.7 / 22.7</td><td>[10.3, 38.5]</td></tr><tr><td>dLLM-based mapping</td><td>-47 /-199 /115</td><td>[−433, 345]</td><td>3,704 / 4,038 / 4,096</td><td>4.4 / 3.8 / 4.7</td><td>[3.2,5.7]</td></tr></table>

Table 9: ThinkTrap per seed, in the same seed order in every column. Length gain is the mean output length of the last five generations minus that of the first five. Perplexity is over each run’s 12 longest-output prompts, with intervals on the $\log _ { 1 0 }$ scale