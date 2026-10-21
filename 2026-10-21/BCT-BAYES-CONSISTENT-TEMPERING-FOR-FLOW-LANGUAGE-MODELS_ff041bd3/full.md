# BCT: BAYES-CONSISTENT TEMPERING FOR FLOW LANGUAGE MODELS

Daisuke Oba<sup>1∗</sup> Hiroki Furuta Naoaki Okazaki<sup>1,2,3</sup> <sup>1</sup>Institute of Science Tokyo <sup>2</sup>AIST <sup>3</sup>NII LLMC

## ABSTRACT

Temperature scaling is typically applied as a post-hoc transformation of model probabilities. In posterior-induced flow language models, this creates two mismatches. First, posterior tempering implicitly changes both the endpoint distribution and the distribution of intermediate noisy states, while the denoiser remains trained on the original states. Second, the tempered posterior weights are combined with transport dynamics derived for the untempered flow. These mismatches are reflected empirically: lowering temperature improves GSM8K performance much less reliably in flow LMs than in discrete diffusion LMs. We introduce Bayes-Consistent Tempering (BCT), which makes temperature explicit in both the distributions and dynamics of the generative flow. Its learned realization trains the denoiser on tempered intermediate states and tempers the endpoint distribution using the corresponding tempered dynamics. We also derive training-free variants for conventionally trained checkpoints. Across hyperspherical flow LMs with two denoiser architectures and a von Mises–Fisher flow LM, BCT improves GSM8K generation across broad temperature ranges and achieves the strongest tempering performance. The gains also extend to Sudoku, demonstrating benefits on tightly constrained generation. At very low temperatures, the learned realization is also markedly more robust than the training-free variants, supporting temperature as a property of the generative flow across training and inference rather than a post-hoc probability adjustment. Code and model weights are available at:

https://daioba.github.io/bct.

Untempered  
![](images/8b37836beaa413bc2a93222ca51768c68713a00e65a170efa2bdd938faef94dd.jpg)

After posterior tempering: mismatch and BCT  
![](images/985e240346242f1cdb543a6bfe862b7d8b0fd750030a7d883658adaf271c0674.jpg)  
Figure 1: Conceptual overview of Bayes-Consistent Tempering (BCT). Posterior tempering implicitly changes both the endpoint distribution and the conditional bridge, while standard sampling retains the original conditional velocities and denoiser training states. BCT restores consistency between the bridge and sampling dynamics, and its learned realization additionally aligns training with the tempered bridge.

## 1 INTRODUCTION

Temperature scaling is one of the simplest controls for language generation (Ackley et al., 1985; Holtzman et al., 2020). In autoregressive models, dividing next-token logits by $T < 1$ directly sharpens the distribution from which the next token is sampled. In many flow language models for discrete data, however, generation is driven by a denoising posterior: a model predicts the clean endpoint token from an intermediate noisy state, and this posterior induces the velocity that transports the state toward token embeddings (Stark et al., 2024; Lee et al., 2026; Deschenaux & Gulcehre, 2026). Applying temperature only to this posterior creates two mismatches. First, by Bayes’ rule, posterior tempering implicitly changes not only the endpoint distribution but also the conditional bridge—the time-indexed distribution of intermediate noisy states given each endpoint. The denoiser, however, is trained on states from the untempered bridge, so tempering changes the states encountered at inference without a corresponding change in training. Second, tempering changes the posterior weights over endpoints but leaves their conditional velocities unchanged, even though those velocities are designed for the original, untempered bridges. Thus, temperature changes the implied generative distribution without consistently changing how the model is trained or sampled.

This issue is also visible empirically. On GSM8K (Cobbe et al., 2021), prior work reports that flow language models gain little from lowering the temperature, in contrast to discrete diffusion models such as MDLM (Sahoo et al., 2024) and Duo (Sahoo et al., 2025), whose accuracy improves substantially under low-temperature decoding (Deschenaux & Gulcehre, 2026). This discrepancy motivates us to ask whether temperature is being incorporated correctly into posterior-induced flows.

We introduce Bayes-Consistent Tempering (BCT), which treats temperature as part of the generative flow rather than as a post-hoc transformation of the denoising posterior (Sec. 3). Its learned realization trains the denoiser on states from a tempered conditional bridge and, at inference, applies the Bayes correction needed to temper the endpoint distribution while preserving that tempered bridge, together with the dynamics induced by the same bridge. We call the training component Tempered-Flow Training (TFT) and the corresponding sampler BCT-L.

We further investigate how much inference-time consistency can be recovered without retraining through two variants for existing checkpoints: BCT-P tempers both the endpoint distribution and bridge with matched dynamics, whereas BCT-E tempers only the endpoint distribution, retaining the original bridge and dynamics (Sec. 3.2).

For posterior-induced flow language models, BCT is not merely a conceptual framework: applying it requires deriving the tempered bridge and its corresponding sampling dynamics, and these quantities are tractable in concrete models. For S-FLM (Deschenaux & Gulcehre, 2026), for example, the hyperspherical geometry lets us derive both explicitly. The same principle applies to SFCD (Chemseddine et al., 2026), which uses a von Mises–Fisher bridge with its own analytic tempering rule and corresponding dynamics. Thus, BCT is not specific to a particular denoiser architecture; its construction is determined by the bridge that defines the flow.

Across hyperspherical flow LMs with two denoiser architectures (S-arch/DiT) (Deschenaux & Gulcehre, 2026) and a von Mises–Fisher flow LM (Chemseddine et al., 2026), BCT improves GSM8K over broad temperature ranges (Sec. 4). BCT-L is particularly robust at very low temperatures, outperforming the training-free variants and highlighting the benefit of temperature-aware training. Gains also extend to Sudoku (Ben-Hamu et al., 2025; Kim et al., 2026), where standard tempering does not benefit from lower temperatures, suggesting applicability to constrained generation.

## 2 PRELIMINARIES

## 2.1 POSTERIOR-INDUCED FLOWS

We consider a single token position with finite vocabulary V and endpoint distribution $\pi ( x )$ . Let z denote the continuous latent state and $t \in [ 0 , 1 ]$ run from noise to data. For each endpoint token $x ,$ a conditional bridge $q _ { t } ( z \mid x )$ describes the distribution of intermediate states at time $t ,$ starting from a token-independent source $p _ { 0 }$ and converging toward its embedding $e _ { x }$ . All bridge densities are defined with respect to a fixed reference measure $\mu$ on the latent space. A corresponding conditional velocity $u _ { t } ( z \mid x )$ transports states so that their distribution follows $q _ { t } ( \cdot \mid x )$ over time; formally, $q _ { t }$ and $u _ { t }$ satisfy the continuity equation.

Bayes’ rule gives the denoising posterior

$$
r _ { t } ( x \mid z ) = \frac { \pi ( x ) q _ { t } ( z \mid x ) } { \sum _ { y \in \mathcal { V } } \pi ( y ) q _ { t } ( z \mid y ) } ,\tag{1}
$$

and the generative velocity is its posterior-weighted average of the conditional velocities,

$$
u _ { t } ( z ) = \sum _ { x \in \mathcal { V } } r _ { t } ( x \mid z ) u _ { t } ( z \mid x ) .\tag{2}
$$

A denoiser $r _ { t } ^ { \eta } ( x \mid z )$ is trained to approximate $r _ { t }$ and replaces it in Eq. (2) at generation time. We call flows constructed in this way posterior-induced flows.

## 2.2 STANDARD TEMPERING

Writing $\beta = 1 / T$ , standard temperature scaling replaces the denoising posterior by

$$
r _ { t } ^ { T } ( x \mid z ) \propto r _ { t } ( x \mid z ) ^ { \beta } , \qquad u _ { t } ^ { T } ( z ) = \sum _ { x } r _ { t } ^ { T } ( x \mid z ) u _ { t } ( z \mid x ) ,\tag{3}
$$

thus changing the posterior weights while leaving the conditional velocities unchanged. In practice, this is implemented by dividing the denoiser logits by T.

Using Eq. (1),

$$
r _ { t } ( x \mid z ) ^ { \beta } \propto \pi ( x ) ^ { \beta } q _ { t } ( z \mid x ) ^ { \beta } ,\tag{4}
$$

so posterior tempering acts algebraically on both the endpoint factor and the bridge likelihood. Define

$$
\pi _ { T } ( x ) \propto \pi ( x ) ^ { \beta } , \qquad q _ { t , T } ( z \mid x ) = { \frac { q _ { t } ( z \mid x ) ^ { \beta } } { Z _ { t , T } ( x ) } } , \qquad Z _ { t , T } ( x ) = \int q _ { t } ( z \mid x ) ^ { \beta } d \mu ( z ) .\tag{5}
$$

The posterior induced by $\left( \pi _ { T } , q _ { t , T } \right)$ differs from standard posterior tempering only by the tokendependent normalizer $Z _ { t , T } ( x )$ . Hence the two coincide whenever $Z _ { t , T } ( x )$ is independent of x, as for the symmetric bridge families studied in this work.

For clarity, we present all derivations for a single token position. For sequences, the same decomposition applies position-wise, with $\pi ( x )$ replaced by the context-conditioned endpoint distribution induced by the remaining positions; see Appendix A.

## 3 BAYES-CONSISTENT TEMPERING

Once an endpoint distribution and a conditional bridge are specified, the posterior weights and sampling dynamics should correspond to that same generative model. In particular, for the tempered pair $\left( \pi _ { T } , q _ { t , T } \right)$ , the sampler should use its denoising posterior, conditional velocities $u _ { t , T } ( z \mid x )$ that transport $q _ { t , T }$ , and the source $q _ { 0 , T }$ of that bridge. However, standard tempering does not generally satisfy this alignment: while its posterior corresponds to the tempered pair under the condition described in Section 2, it continues to use the conditional velocities $u _ { t } ( z \mid x )$ of the original bridge.

In contrast, our proposal, Bayes-Consistent Tempering (BCT), treats the endpoint distribution, conditional bridge, and sampling dynamics as components of the same generative flow. We first develop its learned realization, which also aligns the distribution of states seen during training (Sec. 3.1), and then derive post-hoc variants for conventionally trained models (Sec. 3.2).

## 3.1 LEARNED BCT

The learned realization of BCT, BCT-L, aligns training with the tempered bridge used at inference. We first train the denoiser on intermediate states drawn from $q _ { t , T }$ , rather than from the original

bridge $q _ { t }$ . Given a training distribution $p ( T )$ over temperatures, Tempered-Flow Training (TFT) minimizes the usual denoising loss,

$$
\mathcal { L } ( \eta ) = \mathbb { E } _ { T \sim p ( T ) , t , x \sim \pi , z \sim q _ { t , T } ( \cdot | x ) } \left[ - \log r _ { t , T } ^ { \eta } ( x \mid z ) \right] ,\tag{6}
$$

where the denoiser is conditioned on T. Since the endpoint token is still drawn from $\pi$ while its intermediate state is drawn from tempered bridge $q _ { t , T }$ , the Bayes-optimal denoiser is

$$
r _ { t , T } ( x \mid z ) = { \frac { \pi ( x ) q _ { t , T } ( z \mid x ) } { \sum _ { y } \pi ( y ) q _ { t , T } ( z \mid y ) } } .\tag{7}
$$

Thus, TFT incorporates the tempered bridge into the learned posterior, but does not itself temper the endpoint distribution.

At inference, BCT targets the pair $\left( \pi _ { T } , q _ { t , T } \right)$ . Its tempered posterior can be written directly in terms of the TFT posterior as

$$
r _ { t , T } ^ { \mathrm { B C T - L } } ( x \mid z ) \propto r _ { t , T } ( x \mid z ) ^ { \beta } q _ { t , T } ( z \mid x ) ^ { 1 - \beta } .\tag{8}
$$

The second factor is the Bayes correction that tempers the endpoint while preserving the tempered bridge already represented by the denoiser. In practice, learned posteriors are approximate, so we allow the strength of this correction to be selected on held-out data:

$$
\ell _ { x } ^ { \mathrm { B C T - L } } = \beta \ell _ { x } ^ { \eta } + \lambda ( 1 - \beta ) \log q _ { t , T } ( z \mid x ) , \qquad 0 \leq \lambda \leq 1 ,\tag{9}
$$

where one $\lambda$ is selected per trained model and shared across temperatures.

Finally, BCT-L uses the tempered conditional velocity $u _ { t , T } ( z \mid x )$ that transports the corresponding tempered bridge $q _ { t , T } \colon$

$$
u _ { t , T } ^ { \mathrm { B C T - L } } ( z ) = \sum _ { x } r _ { t , T } ^ { \mathrm { B C T - L } } ( x \mid z ) u _ { t , T } ( z \mid x ) ,\tag{10}
$$

with sampling initialized from the tempered source $q _ { 0 , T }$ . The form of $u _ { t , T }$ is model-dependent; we derive it for the flow constructions studied here in Sec. 3.3. Thus, TFT and BCT-L align the intermediate states seen during training, the endpoint distribution, and the sampling dynamics within the same generative flow.

## 3.2 POST-HOC BCT VARIANTS FOR CONVENTIONALLY TRAINED MODELS

We next ask how much of BCT can be recovered after a model has already been trained on the original bridge $q _ { t }$ . We consider two complementary post-hoc variants—BCT-P and BCT-E.

Path tempering (BCT-P). BCT-P retains posterior tempering, which implicitly tempers both the endpoint distribution and the bridge, but replaces the original conditional dynamics with those of the tempered bridge. Its posterior is

$$
r _ { t , T } ^ { \mathrm { B C T - P } } ( x \mid z ) \propto \frac { r _ { t } ( x \mid z ) ^ { \beta } } { Z _ { t , T } ( x ) } ,\tag{11}
$$

and the corresponding velocity is

$$
u _ { t , T } ^ { \mathrm { B C T - P } } ( z ) = \sum _ { x } r _ { t , T } ^ { \mathrm { B C T - P } } ( x \mid z ) u _ { t , T } ( z \mid x ) .\tag{12}
$$

When $Z _ { t , T } ( x )$ is independent of $x ,$ BCT-P uses exactly the same posterior weights as standard tempering; the difference is that its conditional velocities are matched to the tempered bridge. Unlike BCT-L, however, its denoiser was trained only on states from the original bridge $q _ { t }$ , leaving a train– inference mismatch in the intermediate states.

Endpoint tempering (BCT-E). BCT-E avoids tempering the bridge altogether. It keeps untempered bridge and conditional dynamics, and tempers only the endpoint distribution $\pi .$ By Bayes rule, its posterior can be written using the original denoiser as

$$
r _ { t , T } ^ { \mathrm { B C T - E } } ( x \mid z ) \propto r _ { t } ( x \mid z ) ^ { \beta } q _ { t } ( z \mid x ) ^ { 1 - \beta } ,\tag{13}
$$

or, in logits,

$$
\ell _ { x } ^ { \mathrm { E } } = \beta \ell _ { x } + ( 1 - \beta ) \log q _ { t } ( z \mid x ) .\tag{14}
$$

These weights are combined with the original conditional velocities $u _ { t } ( z \mid x )$

Both variants require no retraining. BCT-P restores consistency by matching the sampling dynamics to the tempered bridge, whereas BCT-E restores consistency by keeping the bridge and sampling dynamics untempered.

## 3.3 INSTANTIATIONS

We instantiate BCT on two posterior-induced flow constructions with different conditional bridges.

Hyperspherical flow. S-FLM (Deschenaux & Gulcehre, 2026) is based on hyperspherical flow, which places token embeddings and latents on the unit sphere and transports a uniform source toward each endpoint along geodesics. Let $\theta _ { x } ( z ) =$ arccos $\langle z , e _ { x } \rangle$ and $a _ { t } = 1 - \alpha _ { t }$ denote the remaining fraction of the geodesic at time t. Its conditional bridge has density

$$
q _ { t } ( z \mid x ) \propto { \frac { \sin ^ { d - 2 } ( \theta _ { x } / a _ { t } ) } { \sin ^ { d - 2 } \theta _ { x } } } { \bf 1 } \{ \theta _ { x } < a _ { t } \pi \} ,\tag{15}
$$

with respect to the spherical surface measure. This density gives the bridge likelihood needed by BCT-E and TFT. Rotational symmetry also makes $Z _ { t , T } ( \bar { x ) }$ independent of $x ,$ so BCT-P uses the same posterior weights as standard tempering.

The tempered bridge $q _ { t , T }$ remains rotationally symmetric around $e _ { x }$ . Its corresponding conditional velocity therefore follows the same geodesic direction as the original S-FLM velocity, with a temperature-dependent change in speed,

$$
u _ { t , T } ( z \mid x ) = w _ { t , T } ( \theta _ { x } ) u _ { t } ( z \mid x ) ,\tag{16}
$$

where $w _ { t , T }$ is obtained from a one-dimensional continuity equation; we give its derivation in $\mathsf { A p - }$ pendix B.3. The same rotational symmetry reduces exact sampling from $q _ { t , T }$ to a one-dimensional transformation of the polar angle, allowing TFT to draw training states directly from the tempered bridge (Appendix B.4).

von Mises–Fisher flow. Chemseddine et al. (2026) use a von Mises–Fisher conditional bridge,

$$
q _ { t } ( z \mid x ) \propto \exp ( \kappa _ { t } \langle e _ { x } , z \rangle ) .\tag{17}
$$

Tempering stays within the same family:

$$
q _ { t , T } ( z \mid x ) \propto \exp \left( \frac { \kappa _ { t } } { T } \langle e _ { x } , z \rangle \right) ,\tag{18}
$$

so BCT simply replaces $\kappa _ { t }$ by $\kappa _ { t } / T$ and uses the corresponding conditional dynamics (Appendix G). TFT likewise draws training states directly from the vMF distribution with concentration $\kappa _ { t } / T$ . The normalizer is independent of x, and BCT-P therefore again shares its posterior weights with standard tempering.

For von Mises–Fisher flow, the tempered bridge evidence $g _ { t , T } ( x , z ) = ( \kappa _ { t } / T ) \langle e _ { x } , z \rangle$ is available analytically. For TFT, we therefore include this term explicitly in the denoiser logits, $\ell _ { x } ^ { \eta } = a _ { \eta , x } +$ $g _ { t , T } ( x , z )$ , and learn only the remaining component $a _ { \eta , x }$ . This is a parameterization of the same TFT posterior and BCT-L sampler, not a separate training method.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Main setting. Our main experiments evaluate BCT on GSM8K (Cobbe et al., 2021) with S-FLM (Deschenaux & Gulcehre, 2026). Following its setup, we adopt the TinyGSM protocol (Liu et al., 2023): for each of the 1319 test problems, the model generates a Python program, and the problem counts as solved if executing the program returns the reference answer. We use its released TinyGSM checkpoints for two denoiser backbones: S-arch, a Transformer whose intermediate activations lie on the hypersphere, and the standard Diffusion Transformer (DiT; Peebles & Xie, 2023).

![](images/92cdc8033fd3eade7410070d238af8ba7a9d013763279a3f15150b39da4df539.jpg)

![](images/12c0b96912b33bf1dc8641fb540b2521937ee579367ae8a33f4455c30338628d.jpg)  
Figure 2: GSM8K accuracy against the sampling temperature on the two backbones (NFE = 1024). Lines are the mean over three generation seeds and bands ± one sample standard deviation. Curves without a prefix use the released checkpoint; Cont. (dashed) is the matched continuation. $\mathbf { A } \mathbf { t } T = 1$ every BCT correction is the identity. Values in Table 1.

We refer to either checkpoint as S-FLM and specify the backbone where relevant. The two backbones share the bridge, the posterior-induced velocity, and all samplers, so comparing them varies only the denoiser architecture.

Additional settings. Two additional experiments test whether our conclusions generalize to other settings (Sections 4.4 and 4.5). First, we change the task while keeping the hyperspherical flow, using Sudoku-Hard (Kim et al., 2026; Ben-Hamu et al., 2025), which requires tightly constrained generation. Since no Sudoku checkpoint is released, we train a DiT-based S-FLM with its released training recipe (Appendix F). Second, we change the flow structure while keeping the task: we apply BCT to SFCD (Chemseddine et al., 2026), which uses a von Mises–Fisher bridge instead of the geodesic bridge of S-FLM, and evaluate its released TinyGSM checkpoint on the same GSM8K problems (Appendix G). Unless noted otherwise in their sections, both follow the protocol below.

Samplers. All samplers use the exact posterior-induced velocity with 1024 function evaluations (NFE) and otherwise follow the released sampler; they differ only in the posterior weights and conditional velocities. We compare standard tempering Eq. (3), BCT-E Eq. (14), BCT-P Eq. (12), and BCT-L, which is described below. As a reference, we report the Top-1 velocity (Deschenaux & Gulcehre, 2026), which moves only toward the most likely token; it is invariant to temperature and is the T → 0 limit of standard tempering. We evaluate $T \in \{ 1 , 0 . 8 , 0 . 6 , 0 . 5 , 0 . 4 , 0 . 3 , 0 . \bar { 2 } , 0 . 1 \}$ in all experiments.

TFT and BCT-L. For each backbone, we continue training the released checkpoint with TFT Eq. (6) for 2,000 updates, 0.8% of its 250k pretraining updates, and refer to the resulting model as TFT. The temperature distribution p(T) draws $T = \bar { 1 }$ with probability $0 . 2 , T = 0 . 1$ with probability 0.1, and otherwise $\beta \sim U ( 1 . 2 5 , 5 )$ , so it covers every evaluated temperature. As a matched training control, Cont. continues training for the same 2,000 updates with the original recipe. All models are evaluated with EMA weights. λ in Eq. (9) is selected once per model on TinyGSM validation problems and shared across temperatures (Appendix C).

## 4.2 MAIN RESULTS

Figure 2 and Table 1 show the results for the hyperspherical flow, whose bridge moves latents along geodesics on the sphere (Section 3.3), with both denoiser backbones. Since TFT continues a pretrained model rather than training from scratch, the fair baseline for BCT-L is Continual Training (Cont.), which receives the same additional update steps with the original recipe; the two differ only in whether temperature enters training. The continuation itself changes little: standard tempering on Cont. stays within one point of the released checkpoint at every temperature, and the training-free variants behave alike on the two checkpoints.

Table 1: GSM8K accuracy (%), mean $\pm$ sample standard deviation over three generation seeds (1319 problems, NFE = 1024).
<table><tr><td></td><td>TFT</td><td colspan="3">Cont.</td><td colspan="3">S-FLM</td></tr><tr><td>T</td><td> $\mathrm { B C T - L }$ </td><td>Standard</td><td>BCT-E</td><td>BCT-P</td><td>Standard</td><td>BCT-E</td><td>BCT-P</td></tr><tr><td colspan="8">S-arch</td></tr><tr><td>1</td><td> $1 2 . 6 \pm 1 . 1$ </td><td> $1 2 . 0 \pm 0 . 5$ </td><td> $1 2 . 0 \pm 0 . 5$ </td><td> $1 2 . 0 \pm 0 . 5$ </td><td> $1 2 . 5 \pm 0 . 5$ </td><td> $1 2 . 5 \pm 0 . 5$ </td><td> $1 2 . 5 \pm 0 . 5$ </td></tr><tr><td>0.8</td><td> $1 6 . 3 \pm 0 . 2$ </td><td> $1 4 . 1 \pm 0 . 5$ </td><td> $1 5 . 0 \pm 0 . 5$ </td><td> $1 5 . 9 \pm 0 . 2$ </td><td> $1 4 . 7 \pm 0 . 7$ </td><td> $1 5 . 0 \pm 1 . 1$ </td><td> $1 6 . 7 \pm 0 . 2 $ </td></tr><tr><td>0.6</td><td> $2 0 . 4 \pm 0 . 4$ </td><td> $1 5 . 3 \pm 0 . 6$ </td><td> $1 6 . 5 \pm 0 . 4$ </td><td> $2 0 . 2 \pm 0 . 3$ </td><td> $1 5 . 6 \pm 0 . 2$ </td><td> $1 6 . 9 \pm 0 . 3$ </td><td> $1 9 . 7 \pm 0 . 5$ </td></tr><tr><td>0.5</td><td> $2 1 . 5 \pm 0 . 3$ </td><td> $1 6 . 8 \pm 0 . 2$ </td><td> $1 7 . 4 \pm 0 . 4$ </td><td> $2 1 . 7 \pm 0 . 6$ </td><td> $1 6 . 3 \pm 0 . 5$ </td><td> $1 7 . 7 \pm 0 . 3$ </td><td> $2 1 . 2 \pm 0 . 6$ </td></tr><tr><td>0.4</td><td> $2 3 . 0 \pm 0 . 9$ </td><td> $1 7 . 3 \pm 0 . 7$ </td><td> $1 8 . 3 \pm 0 . 5$ </td><td> $2 3 . 2 \pm 0 . 5$ </td><td> $1 7 . 1 \pm 0 . 5$ </td><td> $1 8 . 0 \pm 1 . 1$ </td><td> $2 2 . 9 \pm 0 . 6$ </td></tr><tr><td>0.3</td><td> $2 3 . 9 \pm 1 . 0$ </td><td> $1 7 . 3 \pm 0 . 6$ </td><td> $1 8 . 4 \pm 0 . 2$ </td><td> $2 2 . 3 \pm 0 . 6$ </td><td> $1 7 . 4 \pm 0 . 6$ </td><td> $1 8 . 5 \pm 0 . 6$ </td><td> $2 3 . 0 \pm 0 . 5$ </td></tr><tr><td>0.2</td><td> $2 5 . 1 \pm 0 . 9$ </td><td> $1 7 . 5 \pm 0 . 3$ </td><td> $1 8 . 7 \pm 0 . 1 $ </td><td> $1 6 . 6 \pm 1 . 1$ </td><td> $1 7 . 6 \pm 0 . 3$ </td><td> $1 8 . 3 \pm 0 . 1$ </td><td> $1 8 . 3 \pm 0 . 9$ </td></tr><tr><td>0.1</td><td> $2 7 . 0 \pm 1 . 0$ </td><td> $1 7 . 5 \pm 0 . 5$ </td><td> $1 8 . 8 \pm 0 . 1$ </td><td> $3 . 9 \pm 0 . 5$ </td><td> $1 7 . 6 \pm 0 . 5$ </td><td> $1 8 . 3 \pm 0 . 1$ </td><td> $5 . 0 \pm 0 . 4$ </td></tr><tr><td colspan="8">DiT</td></tr><tr><td>1</td><td> $1 1 . 2 \pm 1 . 3$ </td><td> $1 1 . 7 \pm 1 . 2$ </td><td> $1 1 . 7 \pm 1 . 2$ </td><td> $1 1 . 7 \pm 1 . 2$ </td><td> $1 2 . 0 \pm 1 . 0$ </td><td> $1 2 . 0 \pm 1 . 0$ </td><td> $1 2 . 0 \pm 1 . 0$ </td></tr><tr><td>0.8</td><td> $1 6 . 2 \pm 0 . 8$ </td><td> $1 4 . 3 \pm 0 . 4$ </td><td> $1 5 . 2 \pm 1 . 0$ </td><td> $1 7 . 0 \pm 0 . 4$ </td><td> $1 4 . 0 \pm 0 . 6$ </td><td> $1 4 . 8 \pm 0 . 9$ </td><td> $1 7 . 3 \pm 0 . 5$ </td></tr><tr><td>0.6</td><td> $2 0 . 2 \pm 0 . 5$ </td><td> $1 6 . 5 \pm 0 . 1$ </td><td> $1 7 . 9 \pm 0 . 4$ </td><td> $2 0 . 8 \pm 0 . 6$ </td><td> $1 6 . 4 \pm 0 . 1$ </td><td> $1 8 . 1 \pm 0 . 6 $ </td><td> $2 0 . 8 \pm 0 . 9$ </td></tr><tr><td>0.5</td><td> $2 2 . 0 \pm 0 . 2$ </td><td> $1 6 . 6 \pm 1 . 1$ </td><td> $1 8 . 9 \pm 1 . 0$ </td><td> $2 3 . 9 \pm 0 . 9$ </td><td> $1 6 . 9 \pm 0 . 7$ </td><td> $1 8 . 9 \pm 1 . 6$ </td><td> $2 3 . 8 \pm 0 . 9$ </td></tr><tr><td>0.4</td><td> $2 4 . 1 \pm 0 . 3$ </td><td> $1 7 . 0 { \pm } 0 . 6 $ </td><td> $1 9 . 3 \pm 0 . 3$ </td><td> $2 6 . 2 \pm 0 . 6$ </td><td> $1 7 . 0 \pm 0 . 9$ </td><td> $1 9 . 6 \pm 1 . 1$ </td><td> $2 5 . 7 \pm 0 . 7$ </td></tr><tr><td>0.3</td><td> $2 5 . 2 \pm 1 . 1$ </td><td> $1 7 . 8 \pm 0 . 9$ </td><td> $1 9 . 7 \pm 0 . 6 $ </td><td> $2 6 . 7 \pm 0 . 2$ </td><td> $1 7 . 2 \pm 0 . 9$ </td><td> $1 9 . 4 \pm 0 . 5$ </td><td> $2 6 . 7 \pm 0 . 7$ </td></tr><tr><td>0.2</td><td> $2 5 . 4 \pm 0 . 5$ </td><td> $1 8 . 0 \pm 0 . 7$ </td><td> $1 9 . 5 \pm 0 . 7$ </td><td> $2 7 . 8 \pm 0 . 6$ </td><td> $1 7 . 2 \pm 0 . 8$ </td><td> $1 9 . 5 \pm 0 . 8$ </td><td> $2 7 . 3 \pm 0 . 6 $ </td></tr><tr><td>0.1</td><td> $2 6 . 0 \pm 0 . 6$ </td><td> $1 7 . 3 \pm 0 . 2$ </td><td> $1 9 . 4 \pm 0 . 6$ </td><td> $1 7 . 0 { \pm } 0 . 6 $ </td><td> $1 7 . 2 \pm 0 . 7$ </td><td> $1 9 . 2 \pm 0 . 6$ </td><td> $1 6 . 7 \pm 0 . 4$ </td></tr></table>

BCT-L makes low-temperature decoding effective. With conventional training, lowering the temperature helps little: standard tempering stays at $1 7 - 1 8 \%$ for every $T \le 0 . 4$ on both backbones, close to the Top-1 reference, its $T \stackrel { - } {  } ($ 0 limit. BCT-L instead improves steadily as the temperature decreases, reaching $2 7 . 0 \pm 1 . 0 \%$ on S-arch and $2 6 . 0 \pm 0 . 6 \%$ on DiT at $T = 0 . 1$ , and it is more accurate than standard tempering at every $T < 1$

Matching the dynamics at inference gives large gains at moderate temperatures. Without training, BCT-P already improves substantially on standard tempering, raising its best accuracy by about 6 points on S-arch and 10 points on DiT, and at moderate temperatures it is comparable to BCT-L, slightly ahead on DiT.

Strong tempering requires training on the tempered bridge. Under strong tempering, however, BCT-P collapses and falls below standard tempering at $T = 0 . 1$ on both backbones. This is consistent with its train–inference mismatch (Section 3.2): the latents follow the tempered bridge, on whose states the denoiser was not trained, and the mismatch grows as T decreases. BCT-L, whose denoiser is trained on the tempered bridge, has no such mismatch and is the most accurate sampler at $T = 0 . 1$ . The results at $T = 0 . 1$ thus particularly highlight the importance of removing the mismatch between the intermediate states seen in training and those visited during sampling.

Keeping the bridge is stable but gains less. BCT-E keeps the bridge on which the denoiser was trained and therefore does not degrade at low temperature, but it improves only modestly on standard tempering and stays below BCT-L at every $T < 1$ , increasingly so as the temperature decreases.

## 4.3 GAINS BEYOND REDUCED DIVERSITY

Lowering the temperature concentrates the sampler, so the low-temperature gain of BCT-L could merely trade sample diversity for single-sample accuracy (Chen et al., 2021), which a less concentrated sampler would recover with more samples. We test this on S-arch by comparing BCT-L with standard tempering on Cont. in Pass@K for $K \in \{ 1 , 2 , 4 , 8 \}$ , estimated from eight generations per problem with the unbiased estimator of Chen et al. (2021) (Figure 3; details in Appendix D).

For every $T < 1$ and $K \leq 8$ , BCT-L has the higher Pass@K, and the 95% paired bootstrap interval of the difference lies above zero. More samples raise Pass@K for both samplers, but not enough for standard tempering to catch up. The low-temperature gain of BCT-L is therefore not explained by reduced diversity alone.

![](images/dfa44dd773a6d280d50274691e6f90cdf50d4111a062c07d9372589443ce11bf.jpg)  
Figure 3: Pass@K on GSM8K (S-arch, $\mathrm { N F E } = 1 0 2 4 )$ , estimated from eight generations per problem. Bands are 95% bootstrap intervals over problems; the panels share the vertical axis.

## 4.4 BEYOND GSM8K: SUDOKU-HARD

Sudoku-Hard tests whether our conclusions carry over to tightly constrained generation, in which each puzzle has a single valid completion. Because no Sudoku checkpoint is released, we train the base model from scratch and obtain TFT by continuing it; as in Section 4.2, the fair baseline for BCT-L is therefore Cont. (Figure 4). Following the released configuration of S-FLM (Deschenaux & Gulcehre, 2026), we sample with 180 NFE.

Lowering the temperature slightly hurts standard tempering, whereas BCT-L improves monotonically to 46.8%. The training-free variants also benefit from lower temperatures, and on this task BCT-P does not collapse; still, BCT-L is the most accurate at every $T ~ \leq ~ 0 . 6$ , with a margin that widens as T decreases, and the base checkpoint shows the same pattern (Appendix F).

![](images/bd7c7eee0aca2186a0600c75b4c80d56b603325aa0efcb406d307bccc809c297.jpg)  
Figure 4: Sudoku-Hard exact solve rate (NFE $= ~ 1 8 0 ) \colon$ mean and ± one sample standard deviation over three generation seeds; basecheckpoint results are in Table 5.

## 4.5 BEYOND THE GEODESIC BRIDGE: VON MISES–FISHER FLOWS

We finally apply BCT to SFCD (Chemseddine et al., 2026), whose von Mises–Fisher bridge replaces the geodesic bridge of S-FLM (Section 3.3). We use its released TinyGSM checkpoint and sampler with $\mathrm { N F E } = 1 0 2 \dot { 4 }$ , and updates for TFT and Cont., and average over five generation seeds $( \mathsf { A p - }$ pendix G). Unlike S-FLM, this flow already benefits from standard tempering: lowering the temperature raises its accuracy by 9 points between $\bar { \boldsymbol { T } } = 1$ and $T ~ = ~ 0 . 3$ (Figure 5), suggesting that inconsistent tempering harms the vMF bridge less. We therefore ask whether consistent tempering still improves a model whose standard tempering is already effective.

It does, mainly at moderate temperatures. BCT-L improves on standard tempering at every $T \in$ [0.2, 0.6] and attains the highest accuracy among the evaluated samplers, $3 4 . 5 \pm 0 . 5 \%$ at $T = 0 . 3 ,$ against $3 2 . 6 \pm 0 . 6 \%$ for standard tempering, a margin well beyond the seed standard deviations.

![](images/8a1037c141b9c62cc8e100a43c810ff74c5aa3efd726110275b52976d3653cc7.jpg)  
Figure 5: GSM8K accuracy of BCT on the vMF flow LM (NFE = 1024): mean and ± one sample standard deviation over five generation seeds. BCT-E falls below the plotted range for $T \leq 0 . 3 ;$ more detailed results are in Table 7.

Under strong tempering, the learned realization remains stable while the training-free variants fail. BCT-L declines at the lowest temperature, a trend absent on S-FLM, but only to the level of standard tempering (31.7% vs. 31.3% at $\dot { T } = 0 . 1 )$ . Both training-free variants instead fall below it: BCT-P as on S-FLM (28.5%), and BCT-E, which is stable on S-FLM, collapses to essentially zero accuracy for $T \leq 0 . 2$ , likely because on this bridge its correction $- ( \beta - 1 ) \kappa _ { t } \langle e _ { x } , z \rangle$ amplifies any mismatch between the learned logits and the analytic bridge term by β and $\kappa _ { t } .$ . BCT-L is thus the only variant that never falls below standard tempering for $T < 1 \colon$ : consistent tempering improves even a flow that already responds well to temperature, and training on the tempered bridge keeps it reliable under strong tempering.

## 5 RELATED WORK

Posterior-induced flow language models. Non-autoregressive language models include discrete diffusion (Austin et al., 2021; Lou et al., 2024; Sahoo et al., 2024; Shi et al., 2024; Sahoo et al., 2025; Arriola et al., 2025; Nie et al., 2025) and continuous diffusion and flow models, which map tokens to continuous representations such as learned embeddings or one-hot vectors (Li et al., 2022; Dieleman et al., 2022; Gulrajani & Hashimoto, 2023; Lee et al., 2026; Hu et al., 2026). In many continuous models, a denoiser trained with cross-entropy predicts the clean token, and its posterior weights conditional velocities, drifts, or scores. Among spherical constructions (Davis et al., 2024; Jo & Hwang, 2025), S-FLM moves latents along geodesics of the hypersphere (Deschenaux & Gulcehre, 2026; Chen & Lipman, 2024), whereas SFCD uses von Mises–Fisher bridges (Chemseddine et al., 2026); we instantiate BCT for both.

Temperature and consistency. In autoregressive models and discrete diffusion, temperature rescales the categorical distribution from which tokens are sampled (Ackley et al., 1985; Holtzman et al., 2020), and lowering it substantially improves discrete diffusion language models on GSM8K (Deschenaux & Gulcehre, 2026); we reproduce this in Appendix E. Continuous diffusion and flow models instead temper the score or the token posterior (Dieleman et al., 2022; Gulrajani & Hashimoto, 2023; Deschenaux & Gulcehre, 2026; Azangulov & Zhang, 2026), and for continuous data tempering is known not to commute with noising: the scaled score is not the score of the noised tempered distribution (Du et al., 2023). Work on this inconsistency targets other settings: Feynman–Kac correctors reweight and resample populations of particles along annealed marginals (Skreta et al., 2025; Hasan et al., 2026), Li et al. (2026) correct score scaling under Gaussian noising for flattening $( T > 1 )$ , and temporal score rescaling controls a local temperature that leaves the mixture weights unchanged (Xu et al., 2026). None of them addresses flows induced by a token posterior over a finite vocabulary, where tempering the posterior implicitly changes the conditional bridge while leaving the conditional velocities untempered. BCT corrects this through the bridge itself, along a single sampling trajectory without particles, and extends the correction to training.

Train–inference mismatch. Models evaluated on states that differ from those seen in training suffer from exposure bias, both in autoregressive models (Ranzato et al., 2016; Bengio et al., 2015) and in diffusion models, where it is reduced by perturbing training inputs or shifting sampling time steps (Ning et al., 2023; Li et al., 2024). There, the mismatch arises from the model’s own prediction errors. Under tempering, it is instead introduced by a known change of the bridge, so TFT can draw its training states exactly from the tempered bridge that the sampler follows.

## 6 CONCLUSION

We introduced Bayes-Consistent Tempering (BCT), which treats temperature as part of the generative flow rather than as a post-hoc transformation of the denoising posterior. BCT aligns the tempered endpoint distribution, intermediate-state distribution, training distribution, and sampling dynamics; its training-free variants recover parts of this consistency for conventionally trained checkpoints. Across hyperspherical flow LMs with two denoiser architectures and the von Mises–Fisher flow LM, BCT improves GSM8K generation over broad temperature ranges, while its learned realization remains robust under aggressive tempering where the post-hoc variants deteriorate. The gains also extend to Sudoku, suggesting that BCT’s benefits extend to tightly constrained generation. Overall, our results indicate that temperature in posterior-induced flow language models is better treated as part of the flow itself, across both training and inference.

## LIMITATIONS AND DISCUSSION

BCT applies to posterior-induced flows for which the tempered bridge and its corresponding sampling dynamics can be derived or approximated; the concrete derivations in this work cover the hyperspherical bridge of S-FLM and the von Mises–Fisher bridge of SFCD. Extending BCT to other bridge families is an important direction for future work.

## ACKNOWLEDGMENTS

This research was conducted by AIST and Institute of Science Tokyo under FRONTia, a Japanese national program led by the Ministry of Economy, Trade and Industry of Japan (METI) and the New Energy and Industrial Technology Development Organization (NEDO), aimed at developing a domestic multimodal foundation model for AI robots and physical AI. This work was also supported in part by JSPS KAKENHI Grant Number 25H01137 and JST K Program Japan Grant Number JPMJKP24C3.

## REFERENCES

David H. Ackley, Geoffrey E. Hinton, and Terrence J. Sejnowski. A learning algorithm for Boltzmann machines. Cognitive Science, 9(1):147–169, 1985.

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. In The Thirteenth International Conference on Learning Representations, 2025.

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, 2021.

Iskander Azangulov and Leo Zhang. Sampling from flow language models via marginal-conditioned bridges. arXiv preprint arXiv:2605.13681, 2026.

Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas Nolte, and Brian Karrer. Accelerated sampling from masked diffusion models via entropy bounded unmasking, 2025. URL https://arxiv. org/abs/2505.24857.

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In Advances in Neural Information Processing Systems, 2015.

Jannis Chemseddine, Gregor Kornhardt, and Gabriele Steidl. Spherical flows for sampling categorical data. arXiv preprint arXiv:2605.05629, 2026.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Ricky T. Q. Chen and Yaron Lipman. Flow matching on general geometries. In The Twelfth International Conference on Learning Representations, 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Oscar Davis, Samuel Kessler, Mircea Petrache, <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan, Michael Bronstein, and Avishek Joey Bose. Fisher flow matching for generative modeling over discrete data. In Advances in Neural Information Processing Systems, 2024.

Justin Deschenaux and Caglar Gulcehre. Language modeling with hyperspherical flows. arXiv preprint arXiv:2605.11125, 2026.

Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H. Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, Curtis Hawthorne, Remi´ Leblond, Will Grathwohl, and Jonas Adler. Continuous diffusion for categorical data. arXiv preprint arXiv:2211.15089, 2022.

Yilun Du, Conor Durkan, Robin Strudel, Joshua B. Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, and Will Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and MCMC. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 8489–8510, 2023.

Ishaan Gulrajani and Tatsunori B. Hashimoto. Likelihood-based diffusion language models. In Advances in Neural Information Processing Systems, volume 36, 2023.

Mohsin Hasan, Viktor Ohanesian, Artem Gazizov, Yoshua Bengio, Alan Aspuru-Guzik, Roberto´ Bondesan, Marta Skreta, and Kirill Neklyudov. Discrete Feynman-Kac correctors. arXiv preprint arXiv:2601.10403, 2026.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In International Conference on Learning Representations, 2020.

Keya Hu, Linlu Qiu, Yiyang Lu, Hanhong Zhao, Tianhong Li, Yoon Kim, Jacob Andreas, and Kaiming He. Elf: Embedded language flows. arXiv preprint arXiv:2605.10938, 2026.

Jaehyeong Jo and Sung Ju Hwang. Continuous diffusion model for language modeling. In Advances in Neural Information Processing Systems, 2025.

Jaeyeon Kim, Seunggeun Kim, Taekyun Lee, David Z. Pan, Hyeji Kim, Sham Kakade, and Sitan Chen. Fine-tuning masked diffusion for provable self-correction. In International Conference on Machine Learning, 2026.

Chanhyuk Lee, Jaehoon Yoo, Manan Agarwal, Sheel Shah, Jerry Huang, Aditi Raghunathan, Seunghoon Hong, Nicholas M Boffi, and Jinwoo Kim. Flow map language models: One-step language modeling via continuous denoising. arXiv preprint arXiv:2602.16813, 2026.

Mingxiao Li, Tingyu Qu, Ruicong Yao, Wei Sun, and Marie-Francine Moens. Alleviating exposure bias in diffusion models through sampling with shifted time steps. In International Conference on Learning Representations, 2024.

Peizhuo Li, Emre Aksan, Alexandru-Eugen Ichim, Thabo Beeler, and Olga Sorkine-Hornung. Diversify diffusion with temperature sampling and variance-corrective time shifting. arXiv preprint arXiv:2607.10853, 2026.

Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusionlm improves controllable text generation. Advances in neural information processing systems, 35: 4328–4343, 2022.

Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, and Yi Zhang. TinyGSM: achieving >80% on GSM8K with small language models. arXiv preprint arXiv:2312.09241, 2023.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning, 2024.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

Mang Ning, Enver Sangineto, Angelo Porrello, Simone Calderara, and Rita Cucchiara. Input perturbation reduces exposure bias in diffusion models. In International Conference on Machine Learning, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. URL https: //arxiv.org/abs/2212.09748.

Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. Sequence level training with recurrent neural networks. In International Conference on Learning Representations, 2016.

Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.

Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin T Chiu, and Volodymyr Kuleshov. The diffusion duality. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=9P9Y8FOSOk.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37: 103131–103167, 2024.

Marta Skreta, Tara Akhound-Sadegh, Viktor Ohanesian, Roberto Bondesan, Alan Aspuru-Guzik,´ Arnaud Doucet, Rob Brekelmans, Alexander Tong, and Kirill Neklyudov. Feynman-Kac correctors in diffusion: Annealing, guidance, and product of experts. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learn ing Research, pp. 55906–55949, 2025.

Hannes Stark, Bowen Jing, Chenyu Wang, Gabriele Corso, Bonnie Berger, Regina Barzilay, and Tommi Jaakkola. Dirichlet flow matching with applications to dna sequence design. In Proceedings of the 41st International Conference on Machine Learning, pp. 46495–46513, 2024.

Yanbo Xu, Yu Wu, Sungjae Park, Zhizhuo Zhou, and Shubham Tulsiani. Temporal score rescaling for temperature sampling in diffusion and flow models. In International Conference on Machine Learning, 2026.

## A SEQUENCES

For a sequence $\boldsymbol { x } = ( x ^ { 1 } , \dots , x ^ { L } )$ with joint endpoint distribution $\pi ( x )$ and a bridge that factorizes over positions, the posterior at position l marginalizes the other positions:

$$
r _ { t } ( x ^ { l } \mid z ) = \frac { q _ { t } ( z ^ { l } \mid x ^ { l } ) \widetilde { \pi } _ { t } ( x ^ { l } ; z ^ { - l } ) } { \sum _ { y } q _ { t } ( z ^ { l } \mid y ) \widetilde { \pi } _ { t } ( y ; z ^ { - l } ) } , \qquad \widetilde { \pi } _ { t } ( x ^ { l } ; z ^ { - l } ) = \sum _ { x ^ { - l } } \pi ( x ) \prod _ { j \neq l } q _ { t } ( z ^ { j } \mid x ^ { j } ) .\tag{19}
$$

This is Eq. (1) at position l with π replaced by the context-conditioned endpoint $\tilde { \pi } _ { t } ( \cdot ; z ^ { - l } )$ , which does not depend on $z ^ { l }$

Every construction of Section 3 therefore applies position-wise and tempers $\tilde { \pi } _ { t } ;$ under $\mathrm { T F T } , \tilde { \pi } _ { t }$ is formed with the tempered bridges $q _ { t , T }$ at the other positions. For any fixed endpoint sequence $\mathbf { \tau } ( \mathbf { x } ) .$ , the product of the position-wise conditional bridges is transported by the concatenation of their corresponding conditional velocities, so the joint conditional continuity equation holds by the product rule. None of them tempers the joint endpoint: that would require the context factor $\begin{array} { r } { \sum _ { x ^ { - l } } \pi ( x ) ^ { \beta } \prod _ { j \not = l } q _ { t } ^ { \prime } ( z ^ { j } \mid x ^ { j } ) } \end{array}$ , with $q _ { t } ^ { \prime }$ the bridge of the target pair, which in general is not proportional to the power of π˜ that every variant uses. Per-token temperature in autoregressive models has the same property.

## B THE HYPERSPHERICAL BRIDGE

Throughout, $\theta = \theta _ { x } ( z )$ is the polar angle about $e _ { x } , \nu = d - 2 , \sigma$ is the surface measure on $\mathbb { S } ^ { d - 1 }$ and $a = a _ { t } \in ( 0 , 1 ]$ , which equals 1 at the source and decreases in $t ;$ the bridge depends on t only through $a ,$ and the results below hold for every $a \in ( 0 , 1 ]$

## B.1 BRIDGE DENSITY

S-FLM draws $z _ { \mathrm { 0 } }$ uniformly on $\mathbb { S } ^ { d - 1 }$ and sets $z _ { t } = \mathrm { S L E R P } ( z _ { 0 } , e _ { x } ; \alpha _ { t } )$ . About $e _ { x }$ , write $z =$ cos $\theta e _ { x } +$ sin $\theta \xi$ with $\boldsymbol { \xi }$ in the unit sphere $\mathbb { S } ^ { d - 2 }$ of the orthogonal complement of $e _ { x }$ . The surface measure factorizes as

$$
d \sigma ( z ) = \sin ^ { \nu } \theta d \theta d \sigma _ { d - 2 } ( \pmb { \xi } ) .\tag{20}
$$

Let $\omega = \theta _ { x } ( z _ { 0 } )$ . Then $\mathrm { S L E R P } ( z _ { 0 } , e _ { x } ; \alpha ) =  \lceil \sin ( a \omega ) z _ { 0 } + \sin ( ( 1 - a ) \omega ) e _ { x } \rceil /$ sin ω, and substi tuting z = cos ω $e _ { x } + \sin \omega \xi { \mathrm { g i v e s } } z _ { t } = \cos ( a \omega ) e _ { x } + \sin ( a \omega ) \xi { : } { \mathrm { S } }$ LERP keeps $\boldsymbol { \xi }$ and contracts the polar angle by the factor $a .$ . For uniform $z _ { \mathrm { 0 } } , \mathrm { } \left( \omega , \pmb { \xi } \right)$ has density $\sin ^ { \nu } \omega / | \mathbb { S } ^ { d - 1 } |$ with respect to $d \omega d \sigma _ { d - 2 } ( \pmb { \xi } )$ . The map ω 7→ aω has Jacobian $^ { a , }$ and dividing by the factor sin $\nu \theta$ of Eq. (20) gives

$$
q _ { t } ( z \mid x ) = \frac { \sin ^ { \nu } ( \theta / a ) } { a \left| \mathbb { S } ^ { d - 1 } \right| \sin ^ { \nu } \theta } { \bf 1 } \{ \theta < a \pi \} ,\tag{21}
$$

which is Eq. (15).

## B.2 TEMPERED BRIDGE

By Eq. (20), under $q _ { t , T } ( \cdot \mid x ) \propto q _ { t } ( \cdot \mid x ) ^ { \beta }$ the direction $\boldsymbol { \xi }$ is uniform on $\mathbb { S } ^ { d - 2 }$ and independent of the polar angle, whose marginal is

$$
m _ { a , T } ( \theta ) = \frac { g _ { a , T } ( \theta ) } { C _ { a , T } } , \qquad g _ { a , T } ( \theta ) = \sin ^ { \nu \beta } \Bigl ( \frac { \theta } { a } \Bigr ) \sin ^ { \nu ( 1 - \beta ) } \theta , \qquad \theta \in ( 0 , a \pi ) ,\tag{22}
$$

with $\begin{array} { l } { { C _ { a , T } = \int _ { 0 } ^ { a \pi } g _ { a , T } } } \end{array}$ . Rotations preserve σ and map the bridges of different tokens onto one another, so $Z _ { t , T }$ depends only on $( a , T , d )$ . Since $g _ { a , T }$ behaves like $\theta ^ { \nu }$ as $\theta  0$ and like $( a \pi - \theta ) ^ { \nu \beta }$ as θ → aπ (a < 1), $Z _ { t , T } <$ ∞ and $q _ { t , T }$ is a valid density for every $T > 0$ . Its support does not depend on $T .$ At $a = 1 , g _ { 1 , T } = \sin ^ { \nu } \theta$ for every $T .$ , so the tempered source $q _ { 0 , T }$ is the uniform source of S-FLM.

## B.3 TEMPERED CONDITIONAL VELOCITY

We write the polar dynamics as $d \theta / d a = v _ { a , T } ( \theta )$

Symmetry reduction. $q _ { t , T } ( \cdot \mid x )$ is invariant under the orthogonal transformations fixing $e _ { x } ,$ and we take its conditional velocity to be equivariant, $u ( R z ) = R u ( z )$ for all such R. The transformations fixing both $e _ { x }$ and z therefore fix $u ( z )$ , and for $d \geq 3$ the only tangent vectors at z they fix are multiples of $\mathbf { e } _ { \theta } .$ , the unit tangent pointing away from $e _ { x }$ along the geodesic. Equivariance under transformations that move $\boldsymbol { \xi }$ makes the magnitude a function of θ alone, so $u = { \dot { \theta } } \mathbf { e } _ { \theta }$

One-dimensional continuity equation. For a density $\rho$ with respect to $\sigma , \ \nabla \cdot ( \rho v \mathbf { e } _ { \theta } ) \ =$ $\sin ^ { - \nu } \theta \ \partial _ { \theta } ( \sin ^ { \nu } \theta \rho v )$ . Multiplying $\partial _ { a } \rho + \nabla \cdot ( \rho v \mathbf { e } _ { \theta } ) \ { \stackrel { } { = } } \ 0 \ \mathbf { b y } \ | \mathbb { S } ^ { d - { \bar { 2 } } } | \sin ^ { \nu } \theta ,$ , which turns $\rho$ into the polar marginal $m _ { a , T }$ , gives

$$
\partial _ { a } m _ { a , T } ( \theta ) + \partial _ { \theta } \big ( m _ { a , T } ( \theta ) v _ { a , T } ( \theta ) \big ) = 0 .\tag{23}
$$

With no flux through the pole (m $v  0$ as $\theta  0 )$ , integrating over $( 0 , \theta )$ gives the unique solution $v _ { a , T } = - \partial _ { a } M _ { a , T } / m _ { a , T }$ , where $\begin{array} { r } { M _ { a , T } ( \theta ) = \int _ { 0 } ^ { \theta } m _ { a , T } } \end{array}$

Closed form. From Eq. (22), $\partial _ { a } g _ { a , T } ( \theta ) = - ( \nu \beta / a ^ { 2 } ) \theta \cot ( \theta / a ) g _ { a , T } ( \theta )$ . Since $g _ { a , T } ( a \pi ) = 0 \quad$ differentiating $C _ { a , T }$ produces no boundary term, and $\partial _ { a } C _ { a , T } = - ( \nu \beta / a ^ { 2 } ) \bar { H } _ { a , T } C _ { a , T }$ with $\bar { H } _ { a , T } =$ $\begin{array} { r } { \mathbb E _ { \theta \sim m _ { a , T } } [ \theta \cot ( \theta / a ) ] } \end{array}$ . Hence

$$
v _ { a , T } ( \theta ) = \frac { \nu \beta } { a ^ { 2 } } \frac { \int _ { 0 } ^ { \theta } \left[ \vartheta \cot ( \vartheta / a ) - \bar { H } _ { a , T } \right] g _ { a , T } ( \vartheta ) d \vartheta } { g _ { a , T } ( \theta ) } .\tag{24}
$$

Velocity field. As t increases, d $a / d t = - \dot { \alpha } _ { t } , \mathrm { s o } \dot { z } = - \dot { \alpha } _ { t } v _ { a , T } ( \theta ) \mathbf { e } _ { \theta }$ . Since $\operatorname { L o g } _ { z } ( e _ { x } ) = - \theta \mathbf { e } _ { \theta }$ this equals

$$
u _ { t , T } ( z \mid x ) = w _ { t , T } ( \theta ) u _ { t } ( z \mid x ) , \qquad w _ { t , T } ( \theta ) = \frac { a _ { t } v _ { a _ { t } , T } ( \theta ) } { \theta } , \qquad u _ { t } ( z \mid x ) = \frac { \dot { \alpha } _ { t } } { a _ { t } } \mathrm { L o g } _ { z } ( e _ { x } ) ,
$$

which is Eq. (16). $\mathbf { A } \mathbf { t } \ T = 1$ , integration by parts with $s = \theta / a$ gives $\textstyle \int _ { 0 } ^ { \pi }$ s cos s sin $\iota ^ { - 1 } s d s =$ $\begin{array} { r } { - \frac { 1 } { \nu } \int _ { 0 } ^ { \pi } \sin ^ { \nu } s d s . } \end{array}$ , hence $\bar { H } _ { a , 1 } = - a / \nu ,$ and $\begin{array} { r } { \frac { d } { d \theta } \big [ \frac { \theta } { a } g _ { a , 1 } ( \theta ) \big ] = \frac { \nu } { a ^ { 2 } } \big [ \theta \cot ( \theta / a ) - \bar { H } _ { a , 1 } \big ] g _ { a , 1 } ( \theta ) } \end{array}$ . The integral in Eq. (24) thus equals $\begin{array} { r } { { \frac { a } { \nu } } \theta g _ { a , 1 } ( \theta ) } \end{array}$ , so $v _ { a , 1 } = \theta / a$ and $w _ { t , 1 } \equiv 1 \colon u _ { t }$ is the S-FLM conditional velocity, and the tempered construction reduces to it at $T = 1$

## B.4 EXACT SAMPLING OF THE TEMPERED BRIDGE

Use the normalized angles $\psi = \omega / \pi$ and $\phi = \theta / ( a \pi )$ in $( 0 , 1 )$ . Let $F _ { \mathrm { s r c } }$ be the CDF of the density $\propto \sin ^ { \nu } ( \pi \psi )$ , and let $F _ { a , T }$ be the CDF of the density $\propto \sin ^ { \nu \beta } ( \pi \phi ) \sin ^ { \nu ( 1 - \beta ) } ( a \pi \phi )$ , which is Eq. (22) in the variable $\phi .$ Define

$$
\begin{array} { r } { Q _ { a , T } ( \omega ) = a \pi F _ { a , T } ^ { - 1 } \big ( F _ { \mathrm { s r c } } ( \omega / \pi ) \big ) . } \end{array}\tag{25}
$$

Under both the source and the tempered bridge, ξ is uniform on $\mathbb { S } ^ { d - 2 }$ and independent of the polar angle. Hence $z =$ cos $\theta e _ { x } + \sin \theta \xi$ with $\theta = Q _ { a , T } ( \omega )$ is an exact sample of $q _ { t , T } ( \cdot \mid x )$ . At $\beta \stackrel { \cdot } { = } 1 , F _ { a , 1 } = F _ { \mathrm { s r c } }$ and $\theta = a \omega$ , which reproduces SLERP for the same $z _ { \mathrm { 0 } }$ . Equivalently, $z =$ SLER $\mathrm { P } ( z _ { 0 } , e _ { x } ; 1 - \theta / \omega )$ , a SLERP with a position-specific parameter. For TFT we build the target CDF for each $( a _ { t } , T )$ pair in the batch on a grid refined geometrically toward both ends of $[ 0 , a _ { t } \pi ]$ and invert it at the drawn angles, with no interpolation table over $( a , T )$

## C HELD-OUT SELECTION OF λ

Held-out set. The TinyGSM data used by S-FLM are split into a training part, which the TFT continuation reads, and a validation part of 118,216 problems, which it does not. From a random subset of 768 validation problems, we remove 111 prompts that also occur verbatim in the training part and 30 problems without a reliably recoverable reference answer (seven fall in both groups), leaving 634 problems, none of which matches a GSM8K test problem. The reference answer of each problem is the value returned by its TinyGSM reference program, and generated programs are graded as on GSM8K.

Protocol. For each backbone, the sweep uses the TFT checkpoint, generation seed 1 and the GSM8K sampling protocol. Each $\lambda \in \{ 0 , 0 . \dot { 2 } 5 , 0 . 5 , 0 . 7 5 , 1 \}$ is scored by its held-out accuracy averaged over $T \stackrel {  } { \in } \{ 0 . 4 , 0 . 2 , 0 . 1 \}$ ; the maximizer, with ties broken toward the larger λ, is used at every temperature. This selects $\lambda = 0 . 5$ for S-arch and $\lambda = 0 . 2 5$ for DiT (Figure 6, Table 2).

![](images/9d94db867d45ace971ece98b1f3757fab2e5daaf5ad0823c31099ad273575d25.jpg)

![](images/e8a3357de8c3c41cae8b7553991a9a6156af1d8a7a761885515edfce2af5f77e.jpg)  
Figure 6: Held-out selection of λ for BCT-L (634 held-out TinyGSM problems, TFT, seed 1). Thin lines: accuracy at each selection temperature; thick line: their mean, the selection score. The dotted line marks the selected $\lambda . \ \lambda = 0$ applies no Bayes correction and $\lambda = 1$ the full correction. Values in Table 2.

Table 2: Held-out λ sweep: correct answers out of 634 at each selection temperature and the selection score (mean accuracy over the three temperatures, %); bold: selected value. The last column is a DiT TFT checkpoint after 500 updates, not used for selection.
<table><tr><td></td><td colspan="4">S-arch</td><td colspan="4">DiT</td><td>DiT (500 upd.)</td></tr><tr><td>λ</td><td> $T { = } 0 . 4$ </td><td>0.2</td><td>0.1</td><td>Score</td><td> $T { = } 0 . 4$ </td><td>0.2</td><td>0.1</td><td>Score</td><td>Score</td></tr><tr><td>0</td><td>279</td><td>285</td><td>282</td><td>44.48</td><td>271</td><td>278</td><td>287</td><td>43.95</td><td>42.27</td></tr><tr><td>0.25</td><td>279</td><td>292</td><td>290</td><td>45.27</td><td>278</td><td>295</td><td>316</td><td>46.74</td><td>48.74</td></tr><tr><td>0.5</td><td>283</td><td>304</td><td>303</td><td>46.79</td><td>281</td><td>299</td><td>295</td><td>46.00</td><td>48.32</td></tr><tr><td>0.75</td><td>284</td><td>305</td><td>295</td><td>46.48</td><td>282</td><td>295</td><td>257</td><td>43.85</td><td>45.79</td></tr><tr><td>1</td><td>286</td><td>300</td><td>289</td><td>46.00</td><td>279</td><td>279</td><td>181</td><td>38.85</td><td>43.22</td></tr></table>

## D PASS@K ON S-ARCH

This appendix gives the setup and values behind Section 4.3.

Setup. We compare Cont. with standard tempering and TFT with BCT-L $( \lambda = 0 . 5 )$ on S-arch, using the checkpoints, sampler, prompts and grader of Table 1 at the same eight temperatures. The sampler draws one generation per problem, so the eight generations per problem come from eight runs with generation seeds $1 + \mathbf { \bar { 1 0 } } j , j = 0 , \ldots , 7 ;$ the first three are the seeds of Table 1 and reproduce its generations exactly. For a problem with c correct generations out of $n = 8 ,$ , we use the unbiased estimator $1 - \textstyle { \binom { n - { \overline { { c } } } } { k } } / \binom { n } { k }$ (Chen et al., 2021), averaged over the 1319 problems. Intervals are 95% percentile bootstrap intervals over problems (10,000 resamples), paired for differences.

Table 3: Pass@K (%) of Figure 3: unbiased estimator over $n = 8$ generations per problem (generation seeds 1, $1 1 , \ldots , 7 1 )$ , S-arch, 1319 GSM8K problems. Pass@1 is the mean single-generation accuracy over the eight draws and agrees with the three-seed mean of Table 1 to within 0.6 points.
<table><tr><td></td><td colspan="4">Cont. standard tempering</td><td colspan="4">TFT BCT-L</td></tr><tr><td>T</td><td> $K = 1$ </td><td>2</td><td>4</td><td>8</td><td> $K = 1$ </td><td>2</td><td>4</td><td>8</td></tr><tr><td>1</td><td>12.3</td><td>19.6</td><td>28.3</td><td>37.5</td><td>12.7</td><td>20.2</td><td>29.0</td><td>37.7</td></tr><tr><td>0.8</td><td>14.4</td><td>22.5</td><td>32.1</td><td>42.0</td><td>16.5</td><td>25.1</td><td>34.7</td><td>44.5</td></tr><tr><td>0.6</td><td>15.8</td><td>24.3</td><td>34.1</td><td>44.5</td><td>20.2</td><td>29.8</td><td>40.3</td><td>51.3</td></tr><tr><td>0.5</td><td>16.9</td><td>25.8</td><td>36.0</td><td>46.8</td><td>21.8</td><td>31.3</td><td>41.2</td><td>51.1</td></tr><tr><td>0.4</td><td>17.4</td><td>26.3</td><td>36.7</td><td>47.5</td><td>23.3</td><td>32.9</td><td>42.8</td><td>52.5</td></tr><tr><td>0.3</td><td>17.6</td><td>26.7</td><td>36.9</td><td>47.3</td><td>24.4</td><td>33.7</td><td>43.2</td><td>52.5</td></tr><tr><td>0.2</td><td>17.7</td><td>26.6</td><td>36.6</td><td>47.2</td><td>25.3</td><td>34.5</td><td>43.4</td><td>51.9</td></tr><tr><td>0.1</td><td>17.8</td><td>26.8</td><td>37.0</td><td>47.5</td><td>27.1</td><td>35.7</td><td>43.9</td><td>51.1</td></tr></table>

Differences. For every $T < 1$ and $K \leq 8 ,$ the paired interval of the difference (BCT-L minus standard tempering) lies above zero. $\mathrm { A t } T = 0 . 1$ the difference is 9.2 [8.1, 10.3] points at $K = 1$ and $3 . 6 [ 1 . 4 , 5 . 8 ] \mathrm { a t } K = 8 .$ The number of problems solved by at least one of the eight generations is 693 and 674 for BCT-L a $T = 0 . 3$ and 0.1, against 624 and 627 for standard tempering.

## E STANDARD TEMPERATURE SCALING ON DISCRETE DIFFUSION BASELINES

We sweep the sampling temperature of the MDLM (Sahoo et al., 2024) and Duo (Sahoo et al., 2025) TinyGSM checkpoints released by Deschenaux & Gulcehre (2026), without any training or tuning. Both use the ancestral samplers of that release: at every step the denoiser logits are divided by T before the softmax, and the tempered clean-token distribution enters the unchanged reverse transition (absorbing state for MDLM, uniform state for Duo). At $T = 1$ this reproduces the released sampler exactly (1319 of 1319 identical programs for both models, checked with 32 steps). We use 1024 steps, generation seed 1 and the GSM8K evaluation of Section 4.1; the S-FLM values are the standard-tempering results at seed 1.

![](images/120157eb798d9828e144dc914d754c0c178dd709adb2e4c0a94a3e245d27324a.jpg)  
Figure 7: Standard temperature scaling on the TinyGSM baselines released with S-FLM (NFE = 1024, seed 1). Gray: S-FLM (Section 4.2). Values in Table 4.

Table 4: Accuracy (%) and correct answers out of 1319 for Figure 7. S-FLM uses the exact velocity.
<table><tr><td> $T$ </td><td>MDLM</td><td>Duo</td><td>S-FLM (S-arch)</td><td>S-FLM (DiT)</td></tr><tr><td>1</td><td>15.0 (198)</td><td>18.6 (245)</td><td>13.0 (172)</td><td>11.7 (154)</td></tr><tr><td>0.8</td><td>22.5 (297)</td><td>23.0 (304)</td><td>15.0 (198)</td><td>13.9 (183)</td></tr><tr><td>0.6</td><td>28.5 (376)</td><td>28.3 (373)</td><td>15.8 (208)</td><td>16.3 (215)</td></tr><tr><td>0.5</td><td>29.3 (386)</td><td>29.5 (389)</td><td>16.5 (218)</td><td>17.3 (228)</td></tr><tr><td>0.4</td><td>30.2 (398)</td><td>32.1 (423)</td><td>17.5 (231)</td><td>17.0 (224)</td></tr><tr><td>0.3</td><td>32.6 (430)</td><td>34.0 (448)</td><td>18.0 (238)</td><td>16.9 (223)</td></tr><tr><td>0.2</td><td>33.7 (445)</td><td>35.3 (465)</td><td>18.0 (237)</td><td>17.1 (226)</td></tr><tr><td>0.1</td><td>32.7 (431)</td><td>36.2 (477)</td><td>17.3 (228)</td><td>16.9 (223)</td></tr></table>

At $T = 0 . 1$ , MDLM and Duo solve 32.7% and 36.2%, in line with the 33% and 36% reported by Deschenaux & Gulcehre (2026); at T = 1 they solve 15.0% and 18.6%, against the reported 18.0% and 17.2%.

## F SUDOKU-HARD

This appendix gives the setup and values behind Section 4.4.

Task. We use the Sudoku-Hard task and puzzle generator of S-FLM (Deschenaux & Gulcehre, 2026) unchanged. Every puzzle has 30 clues and a unique solution (checked with a solution counter), and the 48,000 training and 2,000 test puzzles share no puzzle or solution. A sequence is the puzzle followed by its solution, each written as 81 digits with row separators (a vocabulary of 12 tokens). The puzzle is a fixed prefix during sampling, and a test puzzle counts as solved only if the generated solution equals the reference.

Model and training. We train with the released Sudoku recipe of S-FLM unchanged: a DiT denoiser with 8 blocks and hidden size 512, the geodesic bridge with its truncated adaptive noise schedule, 20,000 updates with a batch of 256 puzzles on one GPU, AdamW with a constant learning rate of $3 \times 1 0 ^ { - 4 }$ after warmup, EMA decay 0.9999, and training seed 1. Our code reproduces the released code bit for bit on this recipe (weights, EMA weights and optimizer state after 1,200 updates). We call the result the base checkpoint. Cont. and TFT resume its complete training state for 2,000 updates; TFT uses the TFT setting of Section 4.1 with T-conditioning. Each checkpoint is a single training run.

Evaluation. All samplers use the released Sudoku sampler (180 NFE, exact velocity, argmax decoding, EMA weights, batches of 64 puzzles on one GPU) with generation seeds 1, 11 and 21; only the temperature and the BCT rule change between runs, and for a given seed all samplers start from the same noise. $\mathbf { A } \mathbf { t } T = 1$ the training-free samplers coincide up to floating-point differences, which change the number of solved puzzles by at most 2 per seed.

Selection of λ. We select λ with the protocol of Appendix C on 2,000 new puzzles from the same generator with a different seed, disjoint from the training and test puzzles. Held-out accuracy increases with λ at every selection temperature (Table 6), so $\lambda = 1$

Table 5: Sudoku-Hard exact solve rate (%), mean ± standard deviation over generation seeds 1, 11 and 21 (2,000 test puzzles, NFE = 180). BCT-L uses the held-out $\lambda = 1$ . The seed standard deviation is dominated by a shift shared by all samplers.
<table><tr><td rowspan="2">T</td><td colspan="3">Base checkpoint</td><td colspan="3">Cont.</td><td colspan="2">TFT</td></tr><tr><td>Standard</td><td>BCT-E</td><td>BCT-P</td><td>Standard</td><td>BCT-E</td><td></td><td>BCT-P</td><td>BCT-L</td></tr><tr><td>1</td><td> $4 2 . 9 \pm 1 . 5$ </td><td> $4 2 . 9 \pm 1 . 6$ </td><td> $4 2 . 9 \pm 1 . 5$ </td><td> $4 1 . 7 \pm 1 . 2$ </td><td> $4 1 . 7 \pm 1 . 2$ </td><td> $4 1 . 7 \pm 1 . 2$ </td><td></td><td> $4 1 . 7 \pm 1 . 1$ </td></tr><tr><td>0.8</td><td> $4 2 . 6 \pm 1 . 6$ </td><td> $4 3 . 7 \pm 1 . 4$ </td><td> $4 3 . 6 \pm 1 . 0$ </td><td> $4 1 . 6 \pm 1 . 2$ </td><td></td><td> $4 2 . 6 \pm 1 . 2$ </td><td> $4 2 . 4 \pm 1 . 0$ </td><td> $4 2 . 7 \pm 0 . 7$ </td></tr><tr><td>0.6</td><td> $4 2 . 3 \pm 1 . 3$ </td><td> $4 4 . 1 \pm 1 . 2$ </td><td> $4 4 . 2 \pm 1 . 1$ </td><td> $4 1 . 4 \pm 1 . 1$ </td><td> $4 3 . 1 \pm 1 . 0$ </td><td></td><td> $4 2 . 7 \pm 1 . 2$ </td><td> $4 4 . 3 \pm 1 . 2$ </td></tr><tr><td>0.5</td><td> $4 2 . 2 \pm 1 . 3$ </td><td> $4 4 . 4 \pm 1 . 3$ </td><td> $4 4 . 4 \pm 1 . 2$ </td><td> $4 1 . 3 \pm { 1 . 0 }$ </td><td> $4 3 . 3 \pm 1 . 2$ </td><td></td><td> $4 3 . 0 \pm 0 . 8$ </td><td> $4 5 . 1 \pm 1 . 3$ </td></tr><tr><td>0.4</td><td> $4 2 . 1 \pm 1 . 2$ </td><td> $4 4 . 8 \pm 1 . 4$ </td><td> $4 4 . 8 \pm 1 . 1$ </td><td> $4 1 . 2 \pm { 1 . 0 }$ </td><td> $4 3 . 5 \pm 1 . 2$ </td><td></td><td> $4 3 . 2 \pm 0 . 5$ </td><td> $4 5 . 6 \pm 1 . 3$ </td></tr><tr><td>0.3</td><td> $4 2 . 0 \pm 1 . 2$ </td><td> $4 5 . 0 \pm 1 . 4$ </td><td> $4 4 . 7 \pm 0 . 9$ </td><td> $4 1 . 2 \pm { 1 . 0 }$ </td><td> $4 3 . 7 \pm 1 . 2$ </td><td></td><td> $4 3 . 3 \pm 0 . 5$ </td><td> $4 5 . 9 \pm 1 . 1$ </td></tr><tr><td>0.2</td><td> $4 1 . 9 \pm 1 . 3$ </td><td> $4 5 . 2 \pm 1 . 4$ </td><td> $4 4 . 8 \pm 0 . 4$ </td><td> $4 1 . 2 \pm { 1 . 0 }$ </td><td> $4 4 . 0 \pm 1 . 2$ </td><td></td><td> $4 3 . 6 \pm 0 . 1$ </td><td> $4 6 . 4 \pm 1 . 1$ </td></tr><tr><td>0.1</td><td> $4 1 . 7 \pm 1 . 3$ </td><td> $4 5 . 5 \pm 1 . 4$ </td><td> $4 4 . 9 \pm 0 . 4$ </td><td> $4 1 . 0 \pm 1 . 0$ </td><td></td><td> $4 4 . 2 \pm 1 . 4$ </td><td> $4 3 . 7 \pm 0 . 5$ </td><td> $4 6 . 8 \pm 1 . 2$ </td></tr></table>

Table 6: Held-out selection of λ for BCT-L on Sudoku-Hard: exact solve rate (%) of TFT on 2,000 held-out puzzles (generation seed 1, NFE = 180). Mean: the selection score. Bold: selected. No test puzzle was used.
<table><tr><td>λ</td><td> $T = 0 . 4$ </td><td> $T = 0 . 2$ </td><td> $T = 0 . 1$ </td><td>Mean</td></tr><tr><td>0</td><td>41.8</td><td>40.8</td><td>40.1</td><td>40.9</td></tr><tr><td>0.25</td><td>42.8</td><td>43.5</td><td>43.3</td><td>43.2</td></tr><tr><td>0.5</td><td>44.0</td><td>44.5</td><td>44.9</td><td>44.5</td></tr><tr><td>0.75</td><td>44.9</td><td>45.4</td><td>46.9</td><td>45.7</td></tr><tr><td>1</td><td>45.6</td><td>46.3</td><td>47.5</td><td>46.5</td></tr></table>

## G BCT FOR VMF FLOWS

This appendix gives the instantiation and setup of Section 4.5. SFCD (Chemseddine et al., 2026) keeps learned token embeddings on the unit sphere but uses the von Mises–Fisher bridge $q _ { t } ( z \mid x ) =$ $\mathrm { v M F } ( e _ { x } , \kappa _ { t } )$ with increasing concentration and $\kappa _ { 0 } = 0$ . Its sampler integrates the posterior-induced velocity Eq. (2) on a time warp chosen by the release. We use the released TinyGSM checkpoint (250k updates) with its architecture, warp and sampler unchanged, and change only the velocity.

Tempering the vMF bridge. The tempered bridge is $\mathrm { v M F } ( e _ { x } , \kappa _ { t } / T )$ Eq. (18), whose normalizer does not depend on x, so BCT-P keeps the posterior weights of standard tempering. Because the vMF bridge is indexed by its concentration, the released conditional velocity has the form $u _ { t } ( z \mid$ $x ) = \dot { \kappa } _ { t } V _ { \kappa _ { t } } ( z \mid x )$ , where $V _ { \kappa } ( \cdot \mid x )$ transports $\mathrm { v M F } ( e _ { x } , \kappa )$ as κ increases. The tempered bridge is the same family at concentration ${ \kappa _ { t } } / T$ , so its conditional velocity is

$$
u _ { t , T } ( z \mid x ) = \frac { \dot { \kappa } _ { t } } { T } V _ { \kappa _ { t } / T } ( z \mid x ) :\tag{26}
$$

the released velocity evaluated at concentration $\kappa _ { t } / T$ , with the concentration advancing $1 / T$ times as fast. BCT-P and BCT-L integrate this velocity on the warp grid of the release. BCT-E keeps the original bridge and dynamics and, since log $q _ { t } ( z \mid x ) = \kappa _ { t } \langle e _ { x } , z \rangle$ up to a constant in x, adds $( 1 - \beta ) \kappa _ { t } \langle e _ { x } , z \rangle$ to the tempered logits. At $T = 1$ every variant reduces to the released sampler.

TFT and BCT-L. TFT uses the parameterization of Section 3.3: the network outputs a remainder $^ { a _ { \theta } , }$ , the posterior is softmax $( { \pmb a } _ { \theta } + { \pmb g } )$ with the analytic term $g _ { x } = ( \kappa _ { t } / T ) \langle e _ { x } , z \rangle$ , and training uses the cross-entropy of Eq. (6). The temperature enters the network through $\kappa _ { t } / ( T \kappa _ { \mathrm { m a x } } )$ . BCT-L Eq. (9) then takes the form $\ell _ { x } = \beta a _ { \theta , x } + \left[ \beta + \lambda ( 1 - \beta ) \right] g _ { x }$

Continued training. Both continuations resume the released checkpoint for 20,000 updates: Cont. with the original recipe (with the warp frozen, as in TFT) and TFT with the temperature distribution of Section 4.1 and latents drawn from $q _ { t , T }$ . Both are evaluated with an EMA accumulated over the continuation (decay 0.9999); the released checkpoint uses its released EMA weights. Each continuation is a single training run.

Evaluation. We use the released SFCD evaluation of the 1319 GSM8K test problems (warp-aware ODE sampler, $\mathrm { N F E } = 1 0 2 4$ , batches of 32, and its own answer checker) with generation seeds 42– 46; every cell of Table 7 is complete over the five seeds. The separate $T = 1$ runs of BCT-E on the released checkpoint reproduce standard tempering exactly.

Selection of λ. We select λ with the protocol of Appendix C on 1024 TinyGSM validation problems that the continuation did not read and that share no problem with the GSM8K test set. The score increases with λ (51.92, 53.19, 53.87, 54.07 and 54.39% for $\lambda = 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 )$ , so $\lambda = 1$

Table 7: BCT on SFCD: GSM8K accuracy (%), mean ± sample standard deviation over generation seeds 42–46 (1319 problems, $\mathrm { N F E } = 1 0 2 4 )$ . Released: the SFCD TinyGSM checkpoint with its released EMA weights. Cont. and TFT: 20,000 updates from it, evaluated with an adaptation-local EMA. BCT-L uses the held-out $\lambda = 1$ . At $T = 1$ every correction is the identity, so the three columns of a model coincide.
<table><tr><td rowspan="2">T</td><td colspan="3">Released</td><td colspan="3">Cont.</td><td>TFT</td></tr><tr><td>Standard</td><td>BCT-E</td><td>BCT-P</td><td>Standard</td><td>BCT-E</td><td>BCT-P</td><td>BCT-L</td></tr><tr><td>1</td><td> $2 3 . 5 5 \pm 0 . 7 9$ </td><td> $2 3 . 5 5 \pm 0 . 7 9$ </td><td> $2 3 . 5 5 \pm 0 . 7 9$ </td><td> $2 3 . 5 9 \pm 0 . 7 1$ </td><td> $2 3 . 5 9 \pm 0 . 7 1$ </td><td> $2 3 . 5 9 \pm 0 . 7 1$ </td><td> $2 1 . 5 5 \pm 0 . 8 6$ </td></tr><tr><td>0.8</td><td> $2 7 . 7 9 \pm 0 . 8 3$ </td><td> $2 9 . 1 9 \pm 1 . 4 3$ </td><td> $2 9 . 3 9 \pm 0 . 4 8$ </td><td> $2 7 . 9 9 \pm 0 . 2 7$ </td><td> $2 9 . 8 0 \pm 0 . 7 5$ </td><td> $2 9 . 2 8 \pm 0 . 7 4$ </td><td> $2 8 . 1 9 \pm 0 . 9 7$ </td></tr><tr><td>0.6</td><td> $3 0 . 3 4 \pm 0 . 6 3$ </td><td> $3 1 . 7 1 \pm 0 . 6 4$ </td><td> $3 1 . 6 3 \pm 0 . 7 7$ </td><td> $3 0 . 6 1 \pm 0 . 7 7$ </td><td> $3 1 . 5 4 \pm 0 . 7 8$ </td><td> $3 2 . 4 3 \pm 0 . 4 5$ </td><td> $3 2 . 0 5 \pm 0 . 4 7$ </td></tr><tr><td>0.5</td><td> $3 0 . 8 0 \pm 0 . 8 0$ </td><td> $3 1 . 4 0 \pm 0 . 6 2$ </td><td> $3 2 . 3 6 \pm 1 . 0 3$ </td><td> $3 1 . 8 0 \pm 0 . 7 7$ </td><td> $3 1 . 2 1 \pm 0 . 4 7$ </td><td> $3 3 . 2 8 \pm 0 . 3 9$ </td><td> $3 3 . 5 4 \pm 0 . 6 3$ </td></tr><tr><td>0.4</td><td> $3 1 . 1 9 \pm 0 . 9 9$ </td><td> $2 8 . 2 6 \pm 0 . 5 2$ </td><td> $3 3 . 3 9 \pm 0 . 9 0$ </td><td> $3 2 . 5 7 \pm 0 . 8 3$ </td><td> $2 6 . 3 7 \pm 0 . 3 2$ </td><td> $3 3 . 9 3 \pm 0 . 5 8$ </td><td> $3 4 . 3 3 \pm 0 . 9 7$ </td></tr><tr><td>0.3</td><td> $3 1 . 7 5 \pm 0 . 9 1$ </td><td> $1 5 . 3 0 \pm 0 . 6 7$ </td><td> $3 3 . 6 3 \pm 0 . 6 9$ </td><td> $3 2 . 6 2 \pm 0 . 6 1$ </td><td> $2 . 9 3 \pm 0 . 2 5$ </td><td> $3 3 . 5 4 \pm 0 . 7 5$ </td><td> $3 4 . 4 7 \pm 0 . 5 5$ </td></tr><tr><td>0.2</td><td> $3 1 . 7 1 \pm 0 . 4 0$ </td><td> $0 . 0 8 \pm 0 . 1 1$ </td><td> $3 2 . 0 5 \pm 0 . 6 7$ </td><td> $3 2 . 1 6 \pm 0 . 4 0$ </td><td> $0 . 0 2 \pm 0 . 0 3$ </td><td> $3 2 . 2 4 \pm 0 . 2 8$ </td><td> $3 3 . 2 4 \pm 1 . 1 2$ </td></tr><tr><td>0.1</td><td> $3 0 . 9 9 \pm 0 . 5 5$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $2 6 . 6 3 \pm 0 . 8 5$ </td><td> $3 1 . 2 5 \pm 0 . 4 2$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $2 8 . 4 9 \pm 0 . 5 4$ </td><td> $3 1 . 6 8 \pm 0 . 5 3$ </td></tr></table>