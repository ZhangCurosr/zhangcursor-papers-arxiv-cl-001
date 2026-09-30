# Reconstructing the Vocal Tract with Differentiable Acoustic Simulation

Eric M. Chen MIT CSAIL echen01@mit.edu

Jin Woo Lee MIT RLE & KAIST GSCT jnlee@mit.edu

Vincent Sitzmann MIT CSAIL sitzmann@mit.edu

## Abstract

The vocal tract is the region of the human body responsible for filtering one’s voice to create speech. In this paper, we present a differentiable and GPU accelerated acoustic simulator for the vocal tract. The differentiable simulator synthesizes speech by propagating sound along an acoustic tube model of the vocal tract, and via its gradients, can solve the inverse problem: reconstructing the shape of the vocal tract solely from the sound it produces. Although the inverse mapping between geometry and sound is notoriously non-convex, we discover that gradient descent succeeds with three technical contributions: (1) we design a frequency domain formulation of the vocal tract’s fluid dynamics that is 70x more GPU parallelizable than finite differences in time, (2) we integrate a differentiable model for turbulence to synthesize consonants, and (3) similar to prior work in implicit neural representations (INRs) and neural fields, we find that parameterizing the geometry with a neural network accelerates convergence and escapes local minima that trap discrete representations. Because the simulator is differentiable, it is readily integrated with other deep learning pipelines to enable novel linguistics and medical imaging applications. (1) We demonstrate self-supervised autoencoding of vocal tract shapes across 11 languages, and (2) we couple our simulator with a generative model of MRI (magnetic resonance imaging) images to reconstruct one’s moving vocal tract from only their speech without paired data.

## 1 Introduction

Learning to speak is fundamentally an inverse problem. An infant babbles to discover the non-linear mapping between their articulatory motor commands and their speech, learning the controls required to produce language. Most remarkably, children solve this inverse problem without ever seeing paired data between the muscle configuration of their vocal tract and the speech it produces.

In domains like robotics, this type of sensorimotor learning is accelerated by differentiable simulators. Differentiable simulators, such as Brax [22], Taichi [25], and Warp [38], are computer graphics engines which allow gradients from a physical environment to backpropagate directly to a policy, enabling efficient learning of locomotion and manipulation. But to picture the physical actions of your mouth, tongue, etc. that form language, no equivalent computational infrastructure exists.

Such a method to visualize one’s vocal tract would have significant impact in language, such as studying the cognitive processes of language acquisition [2, 8], in music, as a tool for voice coaching or singing synthesis, as well as in healthcare such as speech pathology and speech brain-computer interfaces [41]. Although several differentiable acoustic simulators [60, 20, 34, 35] have been recently proposed for room acoustics, these rely on ray-based methods for stationary systems that are ill-suited for capturing the time-varying morphological transformations of the vocal tract [33, 23]. On the other hand, while neural vocoders [45, 30, 61] achieve high acoustic realism, they operate as black boxes that ignore the underlying physics. Conversely, classical vocal tract simulators [27, 39] are computationally expensive and CPU-bound, rendering them incompatible with modern deep learning pipelines and making them ineffective at solving the speech-to-motor inverse problem.

![](images/3db26606affc339facd13057bf33e09fbe20875e393055b00e3a400c211b3dc6.jpg)  
Figure 1: Method Overview. We introduce a method to reconstruct a MRI video of a person’s moving vocal tract from their speech alone. (a) Given an MRI image of a person’s vocal tract, we extract cross-sectional areas along the length of the tract. (b) These areas define an acoustic tube. Our differentiable simulator Φ propagates sound through the acoustic tube to synthesize speech. Gradients from the loss flow back through Φ and the generative model into a set of latent vectors, reconstructing the MRI video from the target speech signal.

We address this gap by introducing a differentiable, GPU-accelerated physical simulator of speech production. Our simulator produces speech by propagating sound along an acoustic tube representation of a person’s vocal tract, and via its gradients, can visualize the shape of the vocal tract from only audio. We adopt a frequency-domain solution tailored for GPUs. This approach is 70 more parallelizable on GPUs and, crucially, avoids high-frequency noise inherent to sequential time-stepping (Section 5.1). And to support end-to-end learning, we introduce a gradient-friendly formulation of turbulent noise for modeling consonants (Section 5.2). Although acoustic inverse problems are non-convex, leading to sub-optimal solutions [57, 47], we find that, much like in neural rendering, representing the geometry with a neural network acts as a regularizer that stabilizes the learning dynamics (Section 6.1). We validate these design choices empirically.

We demonstrate the effectiveness of our simulator through two novel applications. First, we use our simulator to train a self-supervised autoencoder which maps raw audio to the vocal tract shape (Section 6.2). Second, we provide a tool to visualize an MRI (magnetic resonance imaging) video of a person’s moving vocal tract from their speech, without training on any paired speech-MRI data. This is a task, to the best of our knowledge, that has not been previously attempted. As outlined in Figure 1, we connect a generative model of MRI images to our differentiable acoustic simulator. Our method then jointly recovers an MRI video and audio. Because our model is geometrically grounded, it outperforms the closest prior work trained on paired data [43].

All in all, by establishing a first-of-its-kind differentiable link between vocal tract geometry and speech, we provide the infrastructure to treat speech as a physically-grounded learning problem.

## 2 Related Work

Articulatory Synthesis. Many physics-based simulators for the vocal tract have been introduced over the last century [27, 39, 55, 4], the most well-known being the Kelly-Lochbaum method [27], and the state of the art being VocalTractLab (VTL) [4]. However, these models are difficult to directly backpropagate through because they rely on finite differences to model a differential equation of sound propagation. Furthermore, they use separate forward models for consonants and vowels, causing a discontinuity in their control flow.

Black Box Methods. To avoid backpropagating through an acoustic simulation, recent works have proposed to solve the inverse problem by training neural networks on paired data between speech and articulatory measurements. However, because these networks are black boxes, they are difficult to interpret physically. Furthermore, their quality fundamentally depends on paired training data, which are sparse. TensorTract2 [31] trains a neural network to map articulatory controls to synthetic sounds generated from VocalTractLab [4]. Black box models have also been trained on biomedical data such as vocal tract MRI [43] and electromagnetic articulography (EMA) [9]. In contrast, our method does not require paired articulatory and speech data.

Another related line of work is differentiable digital signal processing (DDSP) [15]. DDSP models been used to model human speech, such as in neural source filter [61] and Schulze-Forster et al. [51]. However, unlike our method, these models are not constrained by physics. Schulze-Forster et al. [51], for instance, model the vocal tract filter with an all-pole filter via line spectral frequencies. On the other hand, our physical simulator can model both poles and zeros, allowing for realistic modeling of radiation and viscous losses.

Differentiable Acoustic Simulation with Neural Fields. Following the success of differentiable rendering and neural fields for solving inverse problems in 3D computer vision [42, 54], several works have also proposed differentiable simulators and neural fields for sound propagation [37, 60, 34, 35]. These works focus primarily on room-scale environments, where sound waves are approximated as rays for efficiency. However, such ray-based approximations are inappropriate for the human voice. This is because the fundamental pitch of the human voice (80–255 Hz) has much longer wavelength than the diameter of the vocal tract (< 10 cm), so the wave effects become dominant. Instead, our simulator explicitly leverages physics-based modeling suitable for time-varying acoustic tubes such as the human vocal tract.

Südholt et al. [57] have also recently proposed to reconstruct vocal tract areas by backpropagating through a differentiable simulator. However, instead of parameterizing the vocal tract’s area with an implicit neural representation, they adopt an explicit parameterization. Furthermore, they only demonstrate results on the limited setting of static vowels and consonants. Meanwhile, our work shows that neural fields are an effective way to reconstruct both the geometry and movement of the vocal tract, enabling the synthesis of full words.

## 3 Background: The Acoustics of Speech

Speech production is commonly framed as a source-filter process. First, a source signal is produced by either vibrations of the vocal folds or turbulent noise formed at narrow constrictions. Then, the vocal tract’s geometry acts as an acoustic resonator, filtering the audio source to shape the different phonemes we interpret as speech.

Acoustic Tubes. Although the space of all possible speech sounds is large, the vocal tract’s geometry moves in relatively fewer dimensions over time. Following many prior works [3, 27, 39, 4], we represent the vocal tract as an acoustic tube: a one-dimensional tube whose cross-sectional area $A ( x )$ varies along its length. Pictured in Figure 1, the size of this tube is dynamically controlled by articulators such as the tongue, mouth and jaw. Sound propagates as waves along this tube, whose physics are governed by the linearized Euler equations:

$$
{ \frac { A ( x ) } { \rho c ^ { 2 } } } { \frac { \partial } { \partial t } } p ( x , t ) = - { \frac { \partial } { \partial x } } u ( x , t ) \qquad { \frac { \rho } { A ( x ) } } { \frac { \partial } { \partial t } } u ( x , t ) = - { \frac { \partial } { \partial x } } p ( x , t )\tag{1}
$$

$\boldsymbol { p } ( \boldsymbol { x } , t )$ is sound (pressure), $u ( x , t )$ is the volume velocity of airflow, $\rho$ is the density of air, and c is the speed of sound. x ranges over $[ 0 , L ]$ , where L is the length of the vocal tract, and t ranges over [0, T], where T is the length of the speech signal. Most importantly, $p ( L , t )$ , which we name $p _ { \mathrm { o u t } } ( t )$ , is what we perceive as speech. For brevity, material terms are omitted, but the equations can be augmented with viscous damping and wall vibration terms, summarized in Appendix A. To synthesize speech, boundary conditions are set at each end of the tract as Eq. (22)-(23). At $x = 0$ the vocal folds vibrate, leading to a glottal pulse: $u ( 0 , t ) \triangleq u _ { \mathrm { i n } } ( t )$ . We represent $u _ { \mathrm { i n } } ( t )$ with the Liljencrants–Fant (LF) model [18]. $\mathrm { A t } x = L$ , a radiation boundary is set for the lips.

Acoustic Tubes as Source-Filter Models. In this problem, the primary object of interest is determining the lip radiation pressure $\boldsymbol { p } _ { \mathrm { o u t } } ( t )$ given a source $u _ { \mathrm { i n } } ( t )$ . Considering a sufficiently small time window compared to the changes in oral cavity, $A ( x )$ can be regarded as quasi-static, and the relationship between $p _ { \mathrm { o u t } }$ and $u _ { \mathrm { i n } }$ is time-invariant. Solving the linearized Euler equations for a given $A ( x )$ is thus analogous to deriving a linear time-invariant (LTI) filter $h _ { \mathrm { t r a c t } } ,$ such that $p _ { \mathrm { o u t } } = h _ { \mathrm { t r a c t } } * u _ { \mathrm { i n } } ,$ or equivalently $P _ { \mathrm { o u t } } = H _ { \mathrm { t r a c t } } U _ { \mathrm { i n } }$ in the frequency domain. The peaks of $H _ { \mathrm { t r a c t } } ^ { } \mathbf { \vec { s } }$ frequency response function are called its formants, which are crucial auditory cues for humans to perceive vowels (illustrated in Figure 1). Although representing the system in either the time domain or frequency domain may seem equivalent, from an optimization perspective, the choice of domain significantly impacts the convergence of gradient-based approaches as we shall see in Section 5.

![](images/ed7199764780d2d7416111843665a523177d63fbeee2c00a9f0db79a1f69675e.jpg)  
Figure 2: FDS Leads to Smoother Loss Landscapes. (a) Starting from the vowel /u/, we interpolate its area in the direction of the /a/ geometry and /e/ geometry. (b) We plot the log Mel spectrogram loss between the sound produced by the interpolated geometries and the sound produced by /u/. The loss from time domain synthesis is noiser than the loss from frequency domain synthesis.

Discretization. In accordance with the conventions of prior studies [27, 12], the area function $A ( x , t )$ of the dynamic oral cavity is simplified into a quasi-static and piecewise-constant geometry. Specifically, the area function $A ( x , t )$ of a vocal tract is discretized into uniform $N \times K$ lattice with spatial sections of length $L / N$ and temporal windows of length $T / K$ , where each windowed section is considered to have constant area, $\begin{array} { r } { i . e . , \stackrel { \bf { \bar { A } } } { A } [ n , k ] = A \left( \frac { n L } { N } , \frac { k \bar { T } } { K } \right) } \end{array}$ const. We call the simulator which solves the Euler equations in the time-varying case Φ : $( u _ { \mathrm { i n } } , A ) \mapsto p _ { \mathrm { o u t } } .$

## 4 Problem Statement and Method Overview

We outline our method in Figure 1. Given a target speech sample $p _ { \mathrm { t g t } } .$ , our objective is to infer the geometry of the vocal tract that produced $p _ { \mathrm { t g t } }$ . We first estimate the vocal fold vibration $u _ { \mathrm { i n } } ( t )$ with a pitch tracker (Appendix C), and set it as the input boundary condition. Then, our strategy is to leverage gradients from a differentiable simulator Φ to find an area function $A _ { \theta }$ that minimizes the squared distance, $\mathcal { L } _ { 2 } ,$ , between spectrograms of the output $p _ { \mathrm { o u t } } \triangleq \Phi ( u _ { \mathrm { i n } } , A _ { \theta } )$ and the target $p _ { \mathrm { t g t } }$ Enabling both efficient forward simulation and gradient-based reconstruction requires addressing a series of challenges, each of which motivates a technical contribution of our work.

How do we make Φ and its gradients efficient to compute? Section 5.1 demonstrates how solving the governing equations in the frequency domain enables GPU parallelization which is 70 faster than time domain synthesis, and more importantly, enables stable gradient flow.

How do we model non-linear effects? Turbulence, which forms consonants, is a non-linear effect not captured by the linearized Euler equations. Section 5.2 introduces a differentiable turbulence model that unifies vowel and consonant synthesis in a single forward pass.

How do we avoid sub-optimal solutions for $A _ { \theta } ?$ The speech-to-geometry inverse problem is non-convex, and a differentiable forward model alone does not guarantee a good solution. Drawing on recent work in neural fields, we find that parameterizing $A _ { \theta }$ with a neural network accelerates convergence. Furthermore, coupling our simulator with neural networks unlocks new applications in linguistics and medical imaging, described in Chapter 6.

## 5 Differentiable Speech Simulation

## 5.1 Frequency Domain Synthesis (FDS)

Various numerical methods such as finite differences can be employed to obtain solutions of the linearized Euler equations. Even though backpropagating the gradients through the finite differences is possible, we find that the feasibility of this gradient signal is limited by challenges such as noise accumulation from extensive temporal recursion, leading to noisy loss landscapes that cause neural network training to fail in many cases. Assuming the linear time invariance as described in Section 3, the system admits separable eigensolutions $p ( { \bar { x } } , t ) = P ( x , \omega ) e ^ { - j \omega t }$ and $u ( x , t ) = U ( x , \omega ) e ^ { - j \omega t }$ and solving the linearized Euler’s equations essentially reduces to the Helmholtz eigenvalue problem:

Table 1: Speech Reconstruction Metrics. When reconstructing real speech examples from LibriTTS-R [29], the audio from frequency domain synthesis (FDS) is significantly more perceptible than the results from time domain synthesis (TDS). To balance fidelity and efficiency, we use a window of 25 ms and a hop ratio of $1 / 4$ . FDS is also 71.5x faster than real time (RT) while TDS is only 1.08x faster.  
![](images/ed5487e6f532f22c4d9dbf72c82190a3de83edab2dea07dcce316cee16236acc.jpg)

<table><tr><td rowspan="2"></td><td>TDS</td><td colspan="2">FDS (25 ms window)</td><td colspan="2">FDS (50 ms window)</td></tr><tr><td></td><td>1/8 Hop</td><td>1/4 Hop</td><td>1/8 Hop</td><td>1/4 Hop</td></tr><tr><td>SI-SDR</td><td> $7 . 3 2 \pm 2 . 4 2$ </td><td> $1 7 . 8 3 \pm 2 . 1 9$ </td><td> $1 6 . 3 9 \pm 2 . 4 2$ </td><td> $1 7 . 4 7 \pm 1 . 9 4$ </td><td>15.82 ±2.47</td></tr><tr><td>STOI</td><td> $0 . 7 7 \pm 0 . 0 3$ </td><td>0.93 ±0.02</td><td> $0 . 9 3 \pm 0 . 0 2$ </td><td> $0 . 9 3 { \scriptstyle \pm 0 . 0 2 }$ </td><td>0.93 ±0.02</td></tr><tr><td>PESQ</td><td>1.55 ±0.14</td><td>2.02 ±0.31</td><td> $1 . 9 8 \pm 0 . 2 6 $ </td><td> $1 . 9 7 \pm 0 . 2 6 $ </td><td>1.98 ±0.25</td></tr><tr><td>RT Factor</td><td>1.08x</td><td>35.0x</td><td> $7 1 . 5 \mathbf { x }$ </td><td>36.0x</td><td>70.0x</td></tr></table>

Sample Duration (ms)  
Figure 3: Runtime. FDS is 70x faster than TDS on GPU. Tested on a RTX 4090 with a 25 ms window and 1/4 hop.

$$
{ \frac { j \omega A ( x ) } { \rho c ^ { 2 } } } P ( x , \omega ) = { \frac { \partial } { \partial x } } U ( x , \omega ) { \frac { j \omega \rho } { A ( x ) } } U ( x , \omega ) = { \frac { \partial } { \partial x } } P ( x , \omega )\tag{2}
$$

The boundary conditions simplify to $U ( 0 , \omega ) = U _ { \mathrm { i n } } ( \omega )$ , and $P ( L , \omega ) = Z _ { \mathrm { l i p s } } ( \omega ) U ( L , \omega )$ , where $Z _ { \mathrm { l i p s } } ( \omega )$ is the impedance defined in Eq. (36).The time-independence achieved by transforming the governing equation into a spatial ODE offers distinct advantages by not only eliminating temporal recursion but also enabling independent analysis of the geometries for each frequency ω. When using central differences for the spatial derivatives, as $\begin{array} { r } { \frac { \partial } { \partial x } U ( x , \omega ) \mid _ { n + \frac { 1 } { 2 } } = \frac { U _ { n + 1 } - U _ { n } } { \Delta x } } \end{array}$ and $\scriptstyle { \frac { \partial } { \partial x } } P ( x , \omega ) \mid _ { n + { \frac { 1 } { 2 } } } =$ $\frac { P _ { n + 1 } - P _ { n } } { \Delta x }$ , the relationship between neighboring sections n and n+1 can be expressed as $4 2 \times 2$ chain matrix, ${ \bf K } _ { n }$ . Multiplying successive ${ \bf K } _ { n }$ matrices over $N$ sections derives a closed-form solution for the vocal tract filter, $H _ { \mathrm { t r a c t } } ( \omega )$

$$
\left[ { \begin{array} { l } { P _ { n + 1 } } \\ { U _ { n + 1 } } \end{array} } \right] = \mathbf { K } _ { n } \left[ { \begin{array} { l } { P _ { n } } \\ { U _ { n } } \end{array} } \right] , \qquad \mathbf { K } _ { \mathrm { t o t } } ( \omega ) = \prod _ { n = 0 } ^ { N } \mathbf { K } _ { n } ( \omega ) , \qquad \left[ { \begin{array} { l l } { P _ { N + 1 } } \\ { U _ { N + 1 } } \end{array} } \right] = \underbrace { \left[ { \begin{array} { l l } { \mathbf { A } } & { \mathbf { B } } \\ { \mathbf { C } } & { \mathbf { D } } \end{array} } \right] } _ { \mathbf { D } } \left[ { \begin{array} { l } { P _ { 0 } } \\ { U _ { 0 } } \end{array} } \right]\tag{3}
$$

$$
H _ { \mathrm { t r a c t } } ( \omega ) = \frac { Z _ { \mathrm { l i p s } } ( \omega ) } { \mathbf { A } ( \omega ) - \mathbf { C } ( \omega ) Z _ { \mathrm { l i p s } } ( \omega ) } = \frac { P _ { \mathrm { o u t } } ( \omega ) } { U _ { \mathrm { i n } } ( \omega ) }\tag{4}
$$

The output pressure spectrum is then $P _ { \mathrm { o u t } } = H _ { \mathrm { t r a c t } } U _ { \mathrm { i n } } ,$ , and the synthesized speech $p _ { \mathrm { o u t } }$ can be recovered with an inverse Fourier transform. This technique is an instance of the transmission line matrix (TLM) method [5], because its equations describe an equivalent acoustic circuit consisting of $\mathbf { K _ { n } }$ components. We detail its derivation in Appendix A.3. Two properties of this formulation are critical. First, the closed-form solution for $H _ { \mathrm { t r a c t } }$ is independent of temporal discretization. Second, the ${ \bf K } _ { n }$ matrices can be solved simultaneously for each frequency, enabling parallelization.

Modeling Time-Varying Vocal Tracts. To synthesize speech from time-varying vocal tracts, we compute $p _ { \mathrm { o u t } }$ independently for successive windows then combine them using the overlap-add method. We apply a Tukey analysis window with $\alpha = 0 . 2 5$ to $u _ { \mathrm { i n } } .$ , and a Hann synthesis window to $p _ { \mathrm { o u t } }$

## 5.2 Differentiable Gating of Consonants

Consonants as Turbulence. Although the linearized Euler equations can be solved efficiently in the frequency domain, they do not describe consonant production, which is non-linear. Consonants are formed when steady air is pushed through narrow constrictions in the vocal tract, breaking down into turbulence. This transition from steady to turbulent flow is modeled by the squared Reynolds number. At section $A _ { n }$ it is defined as

$$
\mathbf { R e } _ { n } ^ { 2 } = \frac { 4 \rho ^ { 2 } } { \pi \mu ^ { 2 } } \frac { u _ { n } ( t ) ^ { 2 } } { A _ { n } } ,\tag{5}
$$

where $\mu$ is the dynamic viscosity of air, and $u _ { n }$ is the volume velocity at section $n , \ u _ { n }$ is calculated in the frequency domain as $U _ { n } \stackrel { . } { = } H _ { 0 \mapsto n } U _ { \mathrm { i n } } { } ^ { 1 }$ . Notice that as $A _ { n }$ becomes smaller, such as the case when forming consonants by constricting one’s teeth, tongue, or lips, the Reynolds number increases.

Discontinuities in Turbulence Models. Since directly solving the compressible Navier–Stokes equations to capture turbulence is computationally prohibitive, a widely adopted approach is to use auxiliary models for the turbulence. However these auxiliary models have discontinuities that prevent gradient-based optimization. Prior works [6, 55, 39] adopt consonant formation by injecting a noise source into the vocal tract at the smallest point of constriction: $A _ { \operatorname* { m i n } } = \operatorname* { m i n } \{ A _ { 1 } , \dots , A _ { N } \}$ , with amplitude $u _ { \mathrm { n o i s e } } = \operatorname* { m a x } \{ 0 , \alpha \cdot z _ { n } ( \mathbf { R e } _ { n } ^ { 2 } - \dot { \mathbf { R e } } _ { \mathrm { c r i t } } ^ { 2 } ) \}$ . α is a gain, $z _ { n }$ is noise, and $\mathrm { R e } _ { \mathrm { c r i t } }$ is the critical Reynolds number. $\mathrm { R e } _ { \mathrm { c r i t } }$ is a constant that gates the effect of turbulence.

Removing the Discontinuities. We replace the min and max operations with smooth approximations. First, to differentiate through the position of the constriction, we simply replace the argmin operator with softmax $\left( - A _ { n } \right)$ . By doing so, gradients flow backward into every area element, and turbulence is weighted by how small the area is at $A _ { n }$ . Second, we replace the max gate with a softplus function, so the overall noise injected at section n is

$$
u _ { \mathrm { n o i s e } _ { n } } = \alpha \cdot z _ { n } \cdot \mathrm { s o f t m a x } ( - A _ { n } ) \cdot \mathrm { s o f t p l u s } ( \mathrm { R e } _ { n } ^ { 2 } - \mathrm { R e } _ { \mathrm { c r i t } } ^ { 2 } )\tag{6}
$$

We use fractal Perlin noise [49] as the noise source, which has been widely adopted to add turbulent textures in various fields such as computer graphics [7, 56] and sound synthesis [24]. $ { \mathrm { R e } } _ { \mathrm { c r i t } }$ is set to 3500 as experimentally determined by Sondhi and Schroeter [55].

Unified Vowel and Consonant Synthesis. We represent the final form of our vocal tract transfer function as follows.

$$
P _ { \mathrm { o u t } } ( \omega ) = H _ { \mathrm { t r a c t } } ( \omega ) U _ { \mathrm { i n } } ( \omega ) + \sum _ { n = 1 } ^ { N } H _ { n \to N + 1 } ( \omega ) U _ { \mathrm { n o i s e } _ { n } } ( \omega )\tag{7}
$$

This harmonic-plus-noise spectral modeling can also be viewed as encapsulating our vocal tract transfer function within a differentiable DSP pipeline [15], thereby not only enabling dynamic transition between vowels and consonants, but also preserving differentiability. Although turbulence is computed in the time domain, this technique is still 70x faster than real time (Figure 3).

## 5.3 Comparing Frequency Domain Synthesis (FDS) to Time Domain Synthesis (TDS)

FDS Smooths Gradients. To validate our choice of formulation, we compare frequency domain synthesis (FDS) against time domain synthesis (TDS) based on the methods of [5]. The TDS method is a semi-implicit scheme described in Appendix B. While the two methods may be theoretically equivalent in continuous time, we find that TDS has significantly noiser gradients in implementation. To illustrate this, in Figure 2, we visualize the loss landscapes of $\mathcal { L } _ { 2 }$ when interpolating between three vowels. FDS produces a consistently smooth loss landscape, while TDS suffers from noise.

FDS Improves Acoustic Reconstruction. The differences in loss landscape has direct consequences for optimization. Table 1 reports perceptual speech quality metrics for area functions fit to 25 speech samples from LibriTTS-R [29]. The experimental setup is described in Appendix C. Using TorchAudio-Squim [32], we measure SI-SDR (scale-invariant signal-to-distortion ratio), STOI (short time objective intelligibility), and PESQ (wideband perceptual evaluation of speech quality). Across all metrics, FDS substantially outperforms TDS. The SI-SDR for TDS is almost 10 dB lower than FDS across all configurations. Notably, FDS is also robust to window and hop length. To balance fidelity and efficiency, we use 25 ms windows and a $1 / 4$ hop, resulting in a hop length of 6.25 ms.

FDS Enables GPU Parallelization. Because solutions for $H _ { \mathrm { t r a c t } }$ are independent for each frequency $\omega ,$ FDS can be efficiently parallelized on a GPU. We compare the forward runtimes for FDS and TDS on an RTX 4090 with a sampling rate of 16 kHz and $N = 3 2$ sections in Figure 3. Given a glottal pulse $u _ { \mathrm { i n } }$ , TDS takes 1.08 seconds to synthesize a 1-second utterance. However, in the frequency domain, this takes only 14.2 ms. The runtime tends to increase linearly as the number of windows in the time-domain lattice grows. On average, FDS is 70x faster with a 25 ms window and $1 / 4$ hop.

![](images/156d2c654e8ef71f1ddb66853e3340c9e540426bcf5751a449d5f457e12a53d1.jpg)  
Figure 4: Neural Fields Smoothly Reconstruct Acoustic Tubes. Directly optimizing a discrete acoustic tube fails to fit basic vowels. Optimization is trapped around the initialization, and the recovered geometries sometimes suffer from high-frequency oscillation. In comparison, neural fields couple the geometry of each section with shared weights leading to smooth reconstructions.

## 6 Experiments

We now describe how our differentiable simulator can be used for solving the speech-to-geometry inverse problem. Because the mapping $\Phi ( u _ { \mathrm { i n } } , A _ { \theta } ) \mapsto p _ { \mathrm { o u t } }$ is non-convex, classical reconstruction methods require techniques like second-order optimizers, regularizers, and discrete codebooks to find good solutions [47]. However, inspired by recent work in neural rendering [42, 54], we discover that we can simply use first-order gradient descent without regularizers by parameterizing $A _ { \theta }$ with a neural network. Learning from this, we propose three neural network parameterizations for the vocal tract:

1. A neuralfield, which demonstrates how parameterizing the area function with a neural network can escape the local minima which trap discrete optimization.

2. A self-supervised autoencoder, that encodes an unlabeled speech sample into an area function, then decodes it back to speech with our simulator.

3. A GAN, used to synthesize realistic MRI videos of a person while speaking. To our knowledge, this is the first method to infer a MRI video from speech audio without paired speech-MRI data.

Each enables novel applications in language and medical imaging. For results of over 30 reconstructed utterances, area functions, MRIs, and singing samples, please find our videos on our website.

## 6.1 Neural Networks Escape Sub-optimality

Neural Fields Break Gradient Deadlock. As common in classical iterative optimization, we initially attempted to directly optimize a discrete array representing the area function $A _ { \theta } [ n , k ]$ across N spatial segments and K time windows. However, this approach proved unsuccessful. Upon further investigation, we find that parameterizing $A _ { \theta } ( x , t )$ with a neural network regularizes the optimization space and better fits high-frequency detail, thereby enabling efficient optimization. Consider, for instance, the example shown in Figure 4. In this example, we initialize the model with a uniform tube and employ the Adam optimizer [28] to fit $A _ { \theta }$ to four distinct vowel shapes. The ground truth is from the X-ray data of Fant [16]. When using discrete reconstruction, the optimization stalls as the solutions are often stuck in oscillations around the initial state. It is almost as if the individual

$A _ { n }$ sections of the tube are competing against one another with conflicting gradient directions. Conversely, with a continuous neural field, $A _ { \theta } = F _ { \theta } ( x , t )$ , the model is able to recover the general morphology of the area function. Even when the recovered shapes are not flawless, the formants align accurately with the ground truth. We attribute this outcome to the fact that neural fields, and neural networks more generally, share weights between the spatial sections, allowing optimization to succeed in a coarse-to-fine manner. We test two neural fields for $F _ { \theta } \colon$ : a fully-connected network with random Fourier feature (RFF) encoding [59], and a multiplicative filter network (MFN) [19]. The results in Table 2 show that both neural fields significantly outperform the discrete baseline.

Table 2: Area Reconstruction. We report the mean squared error of each area parameterization in cm<sup>4</sup>. Both neural field methods, RFF and MFN, significantly outperform the discrete baseline.
<table><tr><td></td><td> $/ \mathrm { a } /$ </td><td>/el</td><td> $/ \mathrm { i } /$ </td><td> $/ \mathbf { \dot { 1 } } /$ </td><td> $/ \mathrm { o } /$ </td><td>/u/</td></tr><tr><td>Discrete</td><td> $1 4 . 6 6 \pm 2 . 2 5$ </td><td> $2 6 . 2 2 \pm 1 . 3 6$ </td><td> $2 7 . 6 6 \pm 2 . 0 4$ </td><td> $2 9 . 6 5 \pm 2 . 3 4$ </td><td> $2 7 . 4 2 \pm 3 . 6 0$ </td><td> $2 8 . 1 9 \pm 2 . 5 4$ </td></tr><tr><td>RFF</td><td> ${ \bf 1 . 9 6 \pm 4 . 1 7 }$ </td><td> ${ \bf 1 . 2 9 \pm 0 . 9 0 }$ </td><td> $2 . 4 6 \pm 4 . 0 1$ </td><td> ${ \bf 0 . 4 7 \pm 0 . 3 2 }$ </td><td> ${ \bf 1 . 6 3 \pm 0 . 4 0 }$ </td><td> $2 7 . 2 6 \pm 2 2 . 3 0$ </td></tr><tr><td>MFN</td><td> $5 . 0 6 \pm 3 . 6 4$ </td><td> $1 5 . 0 0 \pm 7 . 0 6$ </td><td> $5 . 9 6 \pm 1 . 3 7$ </td><td> $5 . 7 9 \pm 1 0 . 5 2$ </td><td> $8 . 8 3 \pm 3 . 2 2$ </td><td> $\mathbf { 1 9 . 9 8 \ : \pm 5 . 0 4 }$ </td></tr></table>

## 6.2 Self-supervised Autoencoding of Speech

Since the process of fitting a new neural field to each utterance is computationally expensive, we also train an encoder $\mathcal { E } _ { \theta } : p _ { \mathrm { t g t } } \mapsto A$ that maps a speech recording directly to an area function. The area function is then decoded back to speech with our simulator, Φ. We use the Wav2Vec 2.0 architecture [1] for ${ \mathcal { E } } _ { \theta }$ . The encoder is trained end-to-end with the loss $\mathcal { L } _ { 2 } ( p _ { \mathrm { t g t } } , \Phi ( u _ { \mathrm { i n } } , \mathcal { E } _ { \theta } ( p _ { \mathrm { t g t } } ) )$ . No paired data between $p _ { \mathrm { t g t } }$ and $A ( x , t )$ is required since Φ itself physically constrains the latent space of the autoencoder to be plausible. Note that unlike prior articulatory autoencoders [31, 9] that require paired speech and articulatory data, our model is self-supervised.

Analysis-by-Synthesis Enables Scalable Training. While there is no prior self-supervised articula tory autoencoder to compare to, we can still evaluate our work against TensorTract2 (TT2) [31], an articulatory autoencoder trained on paired data. TT2 is trained on synthetic consonant-vowel clusters generated by VocalTractLab [4]. In comparison, our autoencoder benefits from training on real-world speech datasets because it is self-supervised. Described in Appendix C.5, we train our autoencoder on 11 different languages. 200 synthesized utterances from the test set of each language are then input to Whisper [50] to evaluate their word error rates (WER) and character error rates (CER). We also report cosine similarity between ECAPA-TDNN [13] speaker identity embeddings. Across 11 languages, Table 3 demonstrates that our simulator outputs utterances that are more intelligible and better preserve identity.

Vocal Tract Area Functions Encode a Linguistic Spectrum. Consisting of only 32 spatial sections, area functions are an exceptionally compact representation of speech. To investigate what linguistic information they retain—if any—we plot a UMAP [40] projection of the area functions and color each point by its phoneme (Figure 5). Despite receiving no phonetic labels during training, the area functions remarkably form a continuous spectrum of phonemes, ordered by place of articulation. On the left are vowels like /a/, /i/, and $/ \mathrm { u } / ,$ while constrictions like /s/ and /k/ occupy the right. Although phonemes are often thought of as discrete symbols, our encoder naturally embeds them in a low-dimensional continuous space. This is a fully anticipated phenomenon, considering that the transitions in oral structure between phonemes occur smoothly in spoken language.

![](images/d242e0bfa72abab94a53e8ff8d59eab73facf9b546c0348fc4d99d6811a9af01.jpg)  
Figure 5: Encoded areas form a spectrum.

## 6.3 Physically-Grounded MRI Visualization from Speech

A visual representation of how one’s lip position, tongue geometry, etc. affect their speech would be useful in applications such as language learning and singing instruction. However, directly capturing this data with an MRI machine is impractical outside of specialized settings. Our differentiable simulator thus enables a task that, to our knowledge, has not previously been attempted: visualizing an MRI of a person’s vocal tract from their speech, without any paired speech-MRI training data.

Our method is shown in Figure 1. First, we pre-train a StyleGAN2 network [26] on MRI scans from [36]. The trained GAN G maps a latent vector w $\in \mathbb { R } ^ { 5 1 2 }$ to an MRI I. For reconstruction, we start by initializing a sequence of latents, $\theta = w _ { 1 } , \dotsc , w _ { K } .$ corresponding to a video $I _ { 1 } , \ldots , I _ { K } =$ $G ( w _ { 1 } ) , \dots , G ( w _ { K } )$ . We then extract the cross-sectional area from each I, denoted with $\psi ( I _ { k } ) =$ $A [ 1 { : } N , k ]$ . And finally, we run simulation on A to generate an utterance $p _ { \mathrm { o u t } } .$ . Overall, $\ p _ { \mathrm { o u t } } =$ $\Phi \dot { ( } u _ { \mathrm { i n } } , \psi ( \dot { G } ( w _ { 1 } ) ) , \dots , \dot { \psi } ( G ( w _ { K } ) ) )$ . To fit an utterance, we backpropagate the entire process end-toend, deriving gradients for each $w _ { k }$ . We differentiably segment the cross-sectional areas from each MRI frame by first softly binarizing it with a sigmoid function, then extracting the areas along a guide. The guide, pictured in orange in Figure 1, simply consists of a quarter circle arc and a line segment, connecting the glottis to the lips. Along each orange line, the cross-sectional areas are approximated by summing the binarized pixels. Because this technique does not require paired speech-MRI data it is fully self-supervised.

Table 3: Speech Autoencoding Metrics. We report WER, CER and ID metrics, with their 95% confidence intervals, across 11 languages. Our self-supervised autoencoder better encodes intelligible utterances than TensorTract2, an autoencoder trained on synthetic consonant-vowel clusters.
<table><tr><td rowspan="2"></td><td colspan="2">Ground Truth</td><td colspan="3">TensorTract2 [31]</td><td colspan="3">Ours</td></tr><tr><td>WER</td><td>CER</td><td>WER</td><td>CER</td><td>Speaker ID</td><td>WER</td><td>CER</td><td>Speaker ID</td></tr><tr><td>English</td><td>3.91 ±0.84</td><td> $2 . 0 5 \pm 0 . 7 0$ </td><td> $1 3 . 0 0 \pm 1 . 8 2$ </td><td> $7 . 3 7 \pm 1 . 1 1$ </td><td> $1 6 . 8 6 \pm 1 . 0 1$ </td><td> ${ \pm } . 3 0 \pm 0 . 9 4$ </td><td>2.82 ±0.75</td><td> ${ \pm 2 . 0 \pm 1 . 3 9 }$ </td></tr><tr><td>German</td><td>5.53 ±0.78</td><td> $2 . 4 0 \pm 0 . 4 7$ </td><td> $1 7 . 5 9 \pm 1 . 1 5$ </td><td> $1 0 . 8 5 \pm 0 . 7 3$ </td><td> $1 4 . 0 7 \pm 1 . 4 2$ </td><td> $\pm 2 . 4 9 \pm 1 . 5 4$ </td><td> ${ \bf 7 . 8 9 \pm 0 . 7 4 }$ </td><td> ${ \pm 4 . 9 \pm 1 . 4 5 }$ </td></tr><tr><td>Dutch</td><td>9.24 ±0.87</td><td> $2 . 5 9 \pm 0 . 3 0$ </td><td> $2 7 . 0 8 \pm 1 . 6 3$ </td><td> $1 2 . 8 9 \pm 0 . 9 1$ </td><td> $2 6 . 9 7 \pm 1 . 1 5$ </td><td> ${ \bf 1 7 . 1 9 \pm 1 . 2 7 }$ </td><td> $7 . 7 7 \pm 0 . 6 5$ </td><td> ${ \pm \bf 9 . 3 \pm 1 . 1 7 }$ </td></tr><tr><td>French</td><td>4.72 ±0.79</td><td> $2 . 2 7 \pm 0 . 5 9$ </td><td> $1 9 . 6 5 \pm 1 . 7 1 $ </td><td> $1 1 . 3 7 \pm 1 . 1 7$ </td><td> $1 5 . 1 1 \pm 1 . 3 4$ </td><td> ${ \pm 2 . 9 1 \pm 1 . 3 8 }$ </td><td> $\mathbf { 7 . 6 9 \ \pm 0 . 8 5 }$ </td><td> ${ 4 7 . 4 \pm 1 . 4 6 }$ </td></tr><tr><td>Spanish</td><td>3.35 ±0.69</td><td>1.37 ±0.44</td><td> $9 . 7 1 \pm 0 . 9 3 $ </td><td> $5 . 1 3 \pm 0 . 4 4$ </td><td> $1 4 . 7 4 \pm 1 . 4 9$ </td><td> ${ \bf 9 . 2 7 \pm 1 . 2 5 }$ </td><td> $4 . 4 3 \pm 0 . 5 9$ </td><td> ${ \pm 2 . 9 \pm 1 . 8 1 }$ </td></tr><tr><td>Italian</td><td>9.96 ±1.18</td><td>2.28 ±0.32</td><td> $2 7 . 6 1 \pm 1 . 9 6$ </td><td> $1 0 . 1 0 \pm 0 . 9 5$ </td><td> $1 0 . 3 4 \pm 1 . 2 1$ </td><td> $2 2 . 0 1 \pm 1 . 9 4$ </td><td> ${ \bf 6 . 9 8 \pm 0 . 7 0 }$ </td><td> ${ \pm 3 . 9 \pm 1 . 4 4 }$ </td></tr><tr><td>Portuguese</td><td>6.23 ±0.79</td><td>2.43 ±0.44</td><td> $2 6 . 6 1 \pm 2 . 5 0$ </td><td> $1 4 . 1 5 \pm 2 . 4 0$ </td><td> $1 7 . 3 2 \pm 0 . 9 9$ </td><td> ${ \pm 3 . 3 0 \pm 9 . 3 3 }$ </td><td> $1 2 . 6 8 \pm 5 . 9 0$ </td><td> ${ \bf 4 9 . 1 \pm 1 . 2 4 }$ </td></tr><tr><td>Polish</td><td> $4 . 1 4 \pm 0 . 6 3$ </td><td> $0 . 9 8 \pm 0 . 2 5$ </td><td> $2 0 . 9 5 \pm 1 . 3 0$ </td><td> $7 . 4 4 \pm 0 . 5 0 $ </td><td> $1 8 . 6 7 \pm 0 . 8 8$ </td><td> $1 2 . 8 8 \pm 1 . 0 8$ </td><td> ${ \bf 4 . 7 9 \pm 0 . 4 2 }$ </td><td> ${ \bf 4 9 . 6 \pm } 2 . 4 3$ </td></tr><tr><td>Korean</td><td> $8 . 0 0 \pm 2 . 4 2$ </td><td> $2 . 1 0 \pm 0 . 7 9$ </td><td> $3 2 . 3 2 \pm 4 . 2 7$ </td><td> $1 4 . 5 8 \pm 2 . 4 2$ </td><td> $2 3 . 5 1 \pm 0 . 7 7$ </td><td> $\mathbf { 1 9 . 4 5 \ : \pm 3 . 4 9 }$ </td><td> ${ \bf 6 . 0 5 \pm 1 . 2 5 }$ </td><td> ${ \bf 6 3 . 4 \pm 0 . 9 4 }$ </td></tr><tr><td>Chinese</td><td></td><td> $1 3 . 5 4 \pm 3 . 6 3$ </td><td></td><td> $5 3 . 2 7 \pm 4 2 . 9 7$ </td><td> $1 7 . 0 6 \pm 1 . 1 2$ </td><td></td><td> ${ \bf 3 0 . 1 1 \pm } 1 3 . 0$ </td><td> ${ \bf 4 7 . 9 \pm 1 . 5 2 }$ </td></tr><tr><td>Japanese</td><td></td><td> $6 . 8 9 \pm 1 . 1 9$ </td><td></td><td> $1 5 . 0 1 \pm 2 . 3 8$ </td><td> $2 3 . 7 6 \pm 1 . 4 1$ </td><td></td><td> ${ \bf 1 1 . 8 8 \pm } 1 . 5 7$ </td><td> ${ \pm 3 . 9 \pm 1 . 4 4 }$ </td></tr></table>

![](images/1e5fa964ece8454ce11de78052f7bc63b019bc5b80b902285210917d910189f9.jpg)

![](images/cd8665df5789b9de81254491251b0841f0f79b69bcf97c5af827eab17d756737.jpg)

![](images/8bcbfa50061f997ad0a0c595c4d4b9b7e1f33e35934aca5954a6f9af6cc2815f.jpg)  
Figure 6: Phonetic Structure Emerges without Supervision. We randomly generate 1,000 MRIs from our GAN, then compute their formants with our simulator. The convex hull of formants resemble a trapezoid corresponding almost exactly to the IPA vowel chart. The anatomy of each generated MRI corresponds closely to textbook illustrations of vowels from Fant [17].

Phonetic Structure Emerges without Supervision. We explore how physically grounded the audio-visual representation is. In Figure 6, we visualize the formant space generated from our learned representation. We randomly sample 1,000 MRI frames from the GAN, use our simulator to synthesize vowels extracted from their cross-sectional areas, then plot their first two formants. Despite the absence of any explicit supervision, the formant space’s convex hull forms a trapezoid almost exactly resembling the trapezoidal vowel chart from the International Phonetic Alphabet (IPA) chart. When compared to vowel examples from [17], we observe a very close correspondence in tongue positioning as well. The fact that the IPA emerges spontaneously from this audio-visual representation gives us confidence that our simulator captures geometrically meaningful vocal tract configurations.

Acoustic Grounding Prevents Video Collapse.Without a physical model to connect geometry to speech, prior works that tackle this task act as black boxes. For example, Speech2rtMRI [43], the closest prior work, trains a diffusion model on paired data between speech and MRI. The quality of this diffusion model is limited by MRI data, which is difficult to collect. Furthermore, because Speech2rtMRI does not leverage a physical representation, the generated videos may lack physical plausibility. It can only generate 10-20 frames of video. Our “analysis by synthesis” technique on the other hand makes MRI visualization tractable for the first time by physically constraining the video generations with audio. We compare our methods on 40 sequences of 3 seconds, corresponding to 9,900 video frames.

![](images/cfe06fd3991eed59939a77bbcb19cc868bebee9c754d02b3d45d25fa0171b341.jpg)  
Figure 7: Acoustic Grounding Prevents Video Collapse. Because Speech2rtMRI [43] does not leverage a physical representation, it tends to degenerate over time. Our method remains stable.

We evaluate each method on their acoustic and visual quality. According to the experimental results, Speech2rtMRI suffers from noise degradation. As illustrated in Figure 7, the diffusion model collapses and the generated images exhibit severe noise and unrealistic artifacts. The quantitative results show that the diffusion model achieves a Fréchet Video Distance (FVD) of 2949, an order of magnitude higher than our value of 623. Despite begin trained on unlabeled data, our method still outperforms Speech2rtMRI on SSIM (structural similarity index measure) and LPIPS (learned perceptual image patch similarity) [62] as well. These two image metrics measure how accurately the MRI’s structure matches the ground truth. Our method also simultaneously simulates audio. Despite the challenging task of backpropagating through an MRI generator, it is still able to produce intelligible audio, something not previously possible.

Table 4: MRI Metrics
<table><tr><td></td><td>Ours</td><td>Speech 2rtMRI</td></tr><tr><td>Audio</td><td></td><td></td></tr><tr><td>SI-SDR ↑ STOI↑</td><td>13.08 0.92</td><td>N/A N/A</td></tr><tr><td>PESQ↑</td><td>2.09</td><td>N/A</td></tr><tr><td>Visual</td><td></td><td></td></tr><tr><td>FVD↓</td><td>623</td><td>2949</td></tr><tr><td>SSIM↑</td><td></td><td></td></tr><tr><td>LPIPS↓</td><td>0.352 0.159</td><td>0.317 0.355</td></tr></table>

## 7 Conclusion

We have introduced a differentiable and GPU parallelizable physics-based model of human speech. We have discovered several technical insights along the way: (1) Modeling acoustics in the frequency domain enables stable gradient flow, and is 70x more GPU parallelizable than finite differences in time. (2) We introduce a differentiable model for consonant formation. (3) Parameterizing the area functions with a neural field alleviates the sub-optimality failures common in acoustic inverse problems. Beyond the simulator itself, we enable two applications that were not previously possible. (1) We train a self-supervised autoencoder that predicts the vocal tract’s area function directly from raw audio, without any paired articulatory data. (2) We provide a way to visualize an MRI video of one’s vocal tract from their speech, without paired speech-MRI training data.

Limitations. We note several directions for future improvement. As shown in Figure 7, our MRI images sometimes do not correspond directly to the ground truth. Part of the reason is that the inverse problem is inherently ill-posed: different geometries can produce the same sound. Furthermore, the MRI dataset used does not image the nasal tract. Reconstructing the nasal tract would produce more faithful formants and anti-formants.

Broader Impacts. While this work is primarily focused on acoustics, we believe that this work opens the door to exciting applications beyond acoustics. Our acoustic simulator can be used, for instance, in music, as a tool for voice coaching or singing synthesis; in cognitive science, such as the study of language acquisition [2]; and in medicine, such as speech pathology. Another direction for future work is to extend our model to other animals, such as sperm whales [52] or zebra finches.

## Acknowledgments

This work would not have been possible without discussions with Morris Alper about phonetics; with Mark Rau about acoustic simulation; and with Matthew Caren and Kartik Chandra about their work on vocal imitation. EC is supported by the NSF Graduate Research Fellowship Program.

Special thanks to our foreign language contributors: Clément Jambon for French, David Charatan for German, Chonghyuk (Andrew) Song for Korean, and Emerald Liu for Chinese.

## References

[1] Alexei Baevski, Henry Zhou, Abdel rahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. ArXiv, abs/2006.11477, 2020. URL https://api.semanticscholar.org/CorpusID:219966759. 8, 22

[2] Gašper Beguš, Alan Zhou, Peter Wu, and Gopala K Anumanchipalli. Articulation gan: Unsupervised modeling of articulatory learning. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023. 1, 10

[3] Stefan Bilbao. Acoustic Tubes, chapter 9, pages 249–286. John Wiley & Sons, Ltd, 2009. ISBN 9780470749012. doi: https://doi.org/10.1002/9780470749012.ch9. 3, 16

[4] Peter Birkholz. Modeling consonant-vowel coarticulation for articulatory speech synthesis. PLOS ONE, 8(4):1–17, 04 2013. doi: 10.1371/journal.pone.0060603. URL https://doi. org/10.1371/journal.pone.0060603. 2, 3, 8, 16

[5] Peter Birkholz and Dietmar Jackel. Influence of temporal discretization schemes on formant frequencies and bandwidths in time domain simulations of the vocal tract system. In Interspeech, 2004. URL https://api.semanticscholar.org/CorpusID:15404079. 5, 6, 20

[6] Peter Birkholz and Dietmar Jackel. Noise sources and area functions for the synthesis of fricative consonants. 2006. URL https://api.semanticscholar.org/CorpusID:12604988. 6

[7] Robert Bridson, Jim Houriham, and Marcus Nordenstam. Curl-noise for procedural fluid flow. ACM SIGGRAPH 2007 papers, 2007. URL https://api.semanticscholar.org/ CorpusID:10174968. 6

[8] Matthew Caren, Kartik Chandra, Joshua Tenenbaum, Jonathan Ragan-Kelley, and Karima Ma. Sketching with your voice: "non-phonorealistic" rendering of sounds via vocal imitation. In SIGGRAPH Asia 2024 Conference Papers, SA ’24, New York, NY, USA, 2024. Association for Computing Machinery. URL https://doi.org/10.1145/3680528.3687679. 1

[9] Cheol Jun Cho, Peter Wu, Tejas S. Prabhune, Dhruv Agarwal, and Gopala K. Anumanchipalli. Coding speech through vocal tract kinematics. IEEE Journal of Selected Topics in Signal Processing, 18(8):1427–1440, 2024. doi: 10.1109/JSTSP.2024.3497655. 3, 8

[10] Eleanor Chodroff, Blaz Pazon, Annie Baker, and Steven Moran. Phonetic segmentation of the ucla phonetics lab archive. In International Conference on Language Resources and Evaluation, 2024. URL https://api.semanticscholar.org/CorpusID:268732610. 22

[11] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/ forum?id=OnD9zGAGT0k. 22

[12] John R Deller Jr, John G Proakis, and John H Hansen. Discrete time processing of speech signals. Prentice Hall PTR, 1993. 4

[13] Brecht Desplanques, Jenthe Thienpondt, and Kris Demuynck. ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification. In Interspeech 2020, pages 3830–3834, 2020. 8

[14] Hugh K Dunn. The calculation of vowel resonances, and an electrical vocal tract. The Journal ofthe Acoustical Society ofAmerica, 22(6):740–753, 1950. 17

[15] Jesse Engel, Lamtharn Hantrakul, Chenjie Gu, and Adam Roberts. Ddsp: Differentiable digital signal processing. ArXiv, abs/2001.04643, 2020. URL https://api.semanticscholar. org/CorpusID:210473083. 3, 6

[16] Gunnar Fant. Acoustic theory of speech production, with calculations based on X-ray studies of Russian articulations. Mouton and Co. N.V., The Hague, 1960. 7

[17] Gunnar Fant. Speech Acoustics and Phonetics. Text, Speech and Language Technology. Springer Dordrecht, 2004. doi: 10.1007/978-1-4020-5746-5. URL https://api.semanticscholar. org/CorpusID:60121294. 9

[18] Gunnar Fant, Johan Liljencrants, and Qi-guang Lin. A four-parameter model of glottal flow. STL-QPSR, 4(1985):1–13, 1985. 3, 17, 21

[19] Rizal Fathony, Anit Kumar Sahu, Devin Willmott, and J. Zico Kolter. Multiplicative filter networks. In International Conference on Learning Representations, 2021. URL https: //api.semanticscholar.org/CorpusID:235613628. 8, 21, 22

[20] Ugo Finnendahl, Markus Worchel, Tobias Jüterbock, Daniel Wujecki, Fabian Brinkmann, Stefan Weinzierl, and Marc Alexa. Differentiable geometric acoustic path tracing using timeresolved path replay backpropagation. ACM Transactions on Graphics (TOG), 44(4), 2025. doi: 10.1145/3730900. 1

[21] James L. Flanagan. Speech Analysis Synthesis and Perception. Communication and Cybernetics. Springer Berlin, Heidelberg, 2 edition, 1972. doi: 10.1007/978-3-662-01562-9. 16, 17

[22] C. Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier Bachem. Brax - a differentiable physics engine for large scale rigid body simulation, 2021. URL http://github.com/google/brax. 1

[23] Andrew S Glassner. An introduction to ray tracing. Morgan Kaufmann, 1989. 1

[24] James K Hahn, Joe Geigel, Jong Won Lee, Larry Gritz, Tapio Takala, and Suneil Mishra. An integrated approach to motion and sound. The Journal of Visualization and Computer Animation, 6(2):109–123, 1995. 6

[25] Yuanming Hu, Luke Anderson, Tzu-Mao Li, Qi Sun, Nathan A. Carr, Jonathan Ragan-Kelley, and Frédo Durand. Difftaichi: Differentiable programming for physical simulation. ArXiv, abs/1910.00935, 2019. URL https://api.semanticscholar.org/CorpusID:203626832. 1

[26] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8107–8116, 2019. URL https: //api.semanticscholar.org/CorpusID:209202273. 8, 22

[27] J. L. Jr. Kelly and C. C. Lochbaum. Speech synthesis. In Proceedings of the Stockholm Speech Communication Seminar, 1962. 1, 2, 3, 4, 16

[28] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014. URL https://api.semanticscholar.org/CorpusID:6628106. 7, 21, 22

[29] Yuma Koizumi, Heiga Zen, Shigeki Karita, Yifan Ding, Kohei Yatabe, Nobuyuki Morioka, Michiel Bacchiani, Yu Zhang, Wei Han, and Ankur Bapna. Libritts-r: A restored multi-speaker text-to-speech corpus. ArXiv, abs/2305.18802, 2023. URL https://api.semanticscholar. org/CorpusID:258967444. 5, 6, 21, 22

[30] Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. In NeurIPS, volume 33, pages 17022–17033, 2020. 1

[31] Paul Konstantin Krug, Christoph Wagner, Peter Birkholz, and Timo Stich. Precisely controllable neural speech synthesis. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2025. doi: 10.1109/ICASSP49660.2025. 10890772. 3, 8, 9

[32] Anurag Kumar, Ke Tan, Zhaoheng Ni, Pranay Manocha, Xiaohui Zhang, Ethan Henderson, and Buye Xu. Torchaudio-squim: Reference-less speech quality and intelligibility measures in torchaudio. ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2023. URL https://api.semanticscholar.org/ CorpusID:257921409. 6, 21

[33] Heinrich Kuttruff. Room acoustics. Crc Press, 2016. 1

[34] Zitong Lan, Chenhao Zheng, Zhiwei Zheng, and Mingmin Zhao. Acoustic volume rendering for neural impulse response fields. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 1, 3

[35] Susan Liang, Chao Huang, Yapeng Tian, Anurag Kumar, and Chenliang Xu. Av-nerf: Learning neural fields for real-world audio-visual scene synthesis. Advances in Neural Information Processing Systems, 36:37472–37490, 2023. 1, 3

[36] Yongwan Lim, Asterios Toutios, Yannick Bliesener, Ye Tian, Sajan Goud Lingala, Colin Vaz, Tanner Sorensen, Miran Oh, Sarah Harper, Weiyi Chen, Yoonjeong Lee, Johannes Töger, Mairym Lloréns Montesserin, Caitlin Smith, Bianca Godinez, Louis Goldstein, Dani Byrd, Krishna S Nayak, and Shrikanth Narayanan. A multispeaker dataset of raw and reconstructed speech production real-time MRI video and 3D volumetric images. 2 2021. doi: 10.6084/m9.figshare.13725546.v1. URL https://figshare. com/articles/dataset/A\_multispeaker\_dataset\_of\_raw\_and\_reconstructed\_ speech\_production\_real-time\_MRI\_video\_and\_3D\_volumetric\_images/13725546. 8, 22

[37] Andrew Luo, Yilun Du, Michael Tarr, Josh Tenenbaum, Antonio Torralba, and Chuang Gan. Learning neural acoustic fields. Advances in Neural Information Processing Systems, 35: 3165–3177, 2022. 3

[38] Miles Macklin. Warp: A high-performance python framework for gpu simulation and graphics. https://github.com/nvidia/warp, March 2022. NVIDIA GPU Technology Conference (GTC). 1

[39] Shinji Maeda. A digital simulation method of the vocal-tract system. Speech Communication, 1 (3):199–229, 1982. ISSN 0167-6393. doi: https://doi.org/10.1016/0167-6393(82)90017-6. URL https://www.sciencedirect.com/science/article/pii/0167639382900176. 1, 2, 3, 6, 16, 20

[40] Leland McInnes, John Healy, Nathaniel Saul, and Lukas Grossberger. Umap: Uniform manifold approximation and projection. The Journal ofOpen Source Software, 3(29):861, 2018. 8, 22

[41] Sean L. Metzger, Kaylo T Littlejohn, Alexander B. Silva, David Aaron Moses, Margaret P. Seaton, Ran Wang, Maximilian E. Dougherty, Jessie R. Liu, Peter Wu, Michael Berger, Inga Zhuravleva, Adelyn P. Tu-Chan, Karunesh Ganguly, Gopala Krishna Anumanchipalli, and Edward F. Chang. A high-performance neuroprosthesis for speech decoding and avatar control. Nature, 620:1037–1046, 2023. URL https://api.semanticscholar.org/CorpusID:261098775. 1

[42] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision, pages 405–421. Springer, 2020. 3, 7

[43] Hong Nguyen, Sean Foley, Kevin Huang, Xuan Shi, Tiantian Feng, and Shrikanth S. Narayanan. Speech2rtmri: Speech-guided diffusion model for real-time mri video of the vocal tract during speech. ArXiv, abs/2409.15525, 2024. URL https://api.semanticscholar.org/ CorpusID:272832441. 2, 3, 9, 10

[44] Lars Nieradzik. Swiftf0: Fast and accurate monophonic pitch detection, 2025. URL https: //arxiv.org/abs/2508.18440. 21

[45] Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, 2016. 1

[46] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: An asr corpus based on public domain audio books. 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5206–5210, 2015. URL https: //api.semanticscholar.org/CorpusID:2191379. 22

[47] Sankaran Panchapagesan and Abeer Alwan. A study of acoustic-to-articulatory inversion of speech by analysis-by-synthesis using chain matrices and the maeda articulatory model. The Journal of the Acoustical Society of America, 129 4:2144–62, 2011. URL https://api. semanticscholar.org/CorpusID:18781420. 2, 7

[48] Kyubyong Park. Kss dataset: Korean single speaker speech dataset, 2018. URL https: //kaggle.com/bryanpark/korean-single-speaker-speech-dataset. 22

[49] Ken Perlin. An image synthesizer. SIGGRAPH Comput. Graph., 19(3):287–296, July 1985. ISSN 0097-8930. doi: 10.1145/325165.325247. URL https://doi.org/10.1145/325165. 325247. 6

[50] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International Confer ence on Machine Learning, 2022. URL https://api.semanticscholar.org/CorpusID: 252923993. 8

[51] Kilian Schulze-Forster, Gaël Richard, Liam Kelley, Clement S. J. Doire, and Roland Badeau. Unsupervised music source separation using differentiable parametric source models. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:1276–1289, 2022. 3

[52] Pratyusha Sharma, Shane Gero, Roger Payne, David F Gruber, Daniela Rus, Antonio Torralba, and Jacob Andreas. Contextual and combinatorial structure in sperm whale vocalisations. Nature Communications, 15, 2023. URL https://api.semanticscholar.org/CorpusID: 266150065. 10

[53] Yao Shi, Hui Bu, Xin Xu, Shaoji Zhang, and Ming Li. Aishell-3: A multi-speaker mandarin tts corpus and the baselines. 2015. URL https://arxiv.org/abs/2010.11567. 22

[54] Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. Advances in neural information processing systems (NeurIPS 2020), 33:7462–7473, 2020. 3, 7

[55] Man Sondhi and J. Schroeter. A hybrid time-frequency domain articulatory speech synthesizer. IEEE Transactions on Acoustics, Speech, and Signal Processing, 35(7):955–967, 1987. doi: 10.1109/TASSP.1987.1165240. 2, 6, 16, 20

[56] Jos Stam and Eugene Fiume. Turbulent wind fields for gaseous phenomena. Proceedings of the 20th annual conference on Computer graphics and interactive techniques, 1993. URL https://api.semanticscholar.org/CorpusID:1618202. 6

[57] David Südholt, Mateo Cámara, Zhiyuan Xu, and Joshua D Reiss. Vocal tract area estimation by gradient descent. 2023. 2, 3

[58] Shinnosuke Takamichi, Kentaro Mitsui, Yuki Saito, Tomoki Koriyama, Naoko Tanji, and Hiroshi Saruwatari. Jvs corpus: free japanese multi-speaker voice corpus. ArXiv, abs/1908.06248, 2019. URL https://api.semanticscholar.org/CorpusID:201070145. 22

[59] Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. NeurIPS, 2020. 8, 22

[60] Mason Wang, Ryosuke Sawata, Samuel Clarke, Ruohan Gao, Shangzhe Wu, and Jiajun Wu. Hearing anything anywhere. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 3

[61] Xin Wang, Shinji Takaki, and Junichi Yamagishi. Neural source-filter waveform models for statistical parametric speech synthesis. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:402–415, 2019. 1, 3

[62] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018. URL https: //api.semanticscholar.org/CorpusID:4766599. 10

## A Physics of the Vocal Tract

We understand that the physics of speech production may be unfamiliar to the broader machine learning community, so we provide a derivation of the speech simulation method in this section. For a comprehensive introduction to speech, we refer the reader to Flanagan [21], or Bilbao [3] for a more general overview on physical modeling synthesis.

## A.1 Derivation of the Linearized Euler Equations

In the most general form, we can model the fluid dynamics of the vocal tract with Euler’s equations for compressible flow, over 3D space $\mathbf { x } \in \mathbb { R } ^ { 3 }$ and time $t \in \mathbb { R }$ . For a pressure field $p : ( \mathbf { x } , t ) \mapsto$ R, velocity field $\mathbf { v } : ( \mathbf { x } , t ) \mapsto \mathbb { R } ^ { 3 }$ , and density field $\rho : ( \mathbf { x } , t ) \mapsto \mathbb { R } _ { \geq 0 }$ , the first equation represents conservation of mass, while the second represents conservation of momentum. Subscripts and $_ x$ denote partial derivatives in time and space respectively.

$$
\rho _ { t } = - \nabla \cdot ( \rho \mathbf { v } )
$$

mass

$$
- \nabla p = \rho ( \mathbf { v } _ { t } + ( \mathbf { v } \cdot \nabla ) \mathbf { v } )\tag{8}
$$

momentum

(9)

The pressure at the outlet of the vocal tract, which we will denote with $p _ { \mathrm { o u t } } ,$ is what we perceive as speech. These equations are difficult to model numerically, but by applying a series of simplifying assumptions, we can derive an entire hierarchy of governing equations for vocal tract acoustics.

First, when modeling sound propagation, $p$ and $\rho$ may be defined as perturbations around equilibrium states: $\rho = \rho _ { 0 } + \rho ^ { \prime } , p = p _ { 0 } + p ^ { \prime }$ . Substituting these linearizations into Euler’s equations, we have

$$
( \rho _ { 0 } + \rho ^ { \prime } ) _ { t } = - \nabla \cdot ( ( \rho _ { 0 } + \rho ^ { \prime } ) \mathbf { v } )\tag{10}
$$

$$
- \nabla ( p _ { 0 } + p ^ { \prime } ) = ( \rho _ { 0 } + \rho ^ { \prime } ) ( { \bf v } _ { t } + ( { \bf v } \cdot \nabla ) { \bf v } ) .\tag{11}
$$

Because $v , \rho ^ { \prime }$ and $p ^ { \prime }$ are assumed to be small, all second-order terms may be eliminated. Further, we may assume that $\bar { p ^ { \prime } } = c ^ { 2 } \rho ^ { \prime }$ , where c is the speed of sound, eliminating the need for $\rho ^ { \prime } .$ . The resulting equations are a common form of Euler’s equations used in acoustics. Note that by linearization, we lose the ability to synthesize turbulent flow.

$$
\frac { 1 } { c ^ { 2 } } p _ { t } ^ { \prime } = - \rho _ { 0 } \nabla \cdot \mathbf { v }
$$

mass

$$
- \nabla p ^ { \prime } = \rho _ { 0 } \mathbf { v } _ { t }\tag{12}
$$

momentum

(13)

For frequencies less than 4 kHz, the acoustics of the vocal tract are dominated by plane waves: waves traveling perpendicular to the vocal tract’s cross section $\mathrm { { d A } = \mathrm { { A } ( x ) } }$ dˆn. This is because the wavelength of a 4 kHz wave traveling at the speed of sound (343 m/s) is 8.6 cm, significantly larger than the diameter of the vocal tract (2-3 cm). This is a common simplification used in prior works [27, 39, 55, 4]. We can model planar flow by rewriting Eq. 12 and Eq. 13 in terms of a one-dimensional flux, $\begin{array} { r } { \boldsymbol { u } ( \boldsymbol { x } , t ) = \int _ { A ( x ) } \boldsymbol { \bar { \mathbf { v } } } \cdot \mathrm { d } \boldsymbol { \mathrm { A } } } \end{array}$ . To model this flux (also known as volume velocity), we integrate the governing equations over a control volume V.

$$
\int _ { V } { \frac { 1 } { c ^ { 2 } } } p _ { t } ^ { \prime } \mathrm { d V } = - \rho _ { 0 } \int _ { V } ( \nabla \cdot \mathbf { v } ) \mathrm { d V }\tag{14}
$$

$$
- \int _ { V } \nabla p ^ { \prime } { \mathrm { d V } } = \rho _ { 0 } \int _ { V } \mathbf { v } _ { t } { \mathrm { d V } } .\tag{15}
$$

In the limit, $\mathrm { d } \mathrm { V } = \mathrm { d } \mathrm { A }$ dx. Further assuming that $p ^ { \prime } ( \mathbf { x } , t )$ is constant over each dA, the 3D field $\boldsymbol { p } ^ { \prime } ( \mathbf { x } , t )$ can be replaced with a simpler 1D field: $\boldsymbol { p } ^ { \prime } ( \boldsymbol { x } , t )$ . By applying the divergence theorem and expanding the integrals, we find that

$$
A \Delta x { \frac { 1 } { c ^ { 2 } } } p _ { t } ^ { \prime } = - \rho _ { 0 } \int _ { A ( x ) + \Delta x } \mathbf { v } \cdot \mathrm { d } \mathbf { A } - \rho _ { 0 } \int _ { \mathrm { A ( x ) } } \mathbf { v } \cdot \mathrm { d } \mathbf { A }\tag{16}
$$

$$
- A \Delta x p _ { x } ^ { \prime } = \rho _ { 0 } \int _ { x } ^ { x + \Delta x } \int _ { A } \mathbf { v } _ { t } \cdot \mathrm { d A } \mathrm { d x } .\tag{17}
$$

Finally, substituting $u ,$ and taking the limit $\Delta x \mapsto 0 ,$ we derive Euler’s equations in terms of u and A.   
For simplicity, $p ^ { \prime }$ is often written as just $p ,$ and $\rho _ { 0 }$ is written as $\rho ,$ the ambient density of air.

$$
\frac { A } { c ^ { 2 } } p _ { t } = - \rho u _ { x }
$$

mass

$$
- A p _ { x } = \rho u _ { t }\tag{18}
$$

momentum

(19)

These are the basic equations for modeling acoustics in the vocal tract, and can be easily augmented with additional dampening parameters for the vocal tract’s material properties [21]. The equations can be written even more compactly by differentiating the mass equation by t, the momentum equation by x, then combining them a single equation, known as Webster’s equation:

$$
( A ( x ) p _ { x } ) _ { x } = \frac { A ( x ) } { c ^ { 2 } } p _ { t t }\tag{20}
$$

Importantly, Webster’s equation resembles the wave equation, demonstrating how the geometry of the vocal tract, controlled by $A ( x )$ , affects the wave propagation of speech. After applying the frequency domain separation of variables $p ( x , t ) = P ( x ) e ^ { - j \omega \overline { { t } } } , u ( x , t ) \ : \stackrel { . } { = } \ : U ( x ) e ^ { - j \omega t }$ , Webster’s equation becomes

$$
( A ( x ) P ( x ) _ { x } ) _ { x } = - \frac { \omega ^ { 2 } } { c ^ { 2 } } A ( x ) P ( x ) ,\tag{21}
$$

resembling the Helmholtz equation. Solving the linearized Euler equations or Webster’s equation is thus equivalent to solving the Helmholtz eigenvalue problem in the frequency domain.

To generate vowels, a glottal pulse $u _ { \mathrm { i n } }$ parameterized by the LF model [18] is set as the initial condition, and is propagated to the other end of the tract to synthesize $p _ { \mathrm { o u t } }$ . Solving the linearized Euler equations or Webster’s equation derives a filter $h _ { \mathrm { t r a c t } }$ , such that $p _ { \mathrm { o u t } } = h _ { \mathrm { t r a c t } } * u _ { \mathrm { i n } }$

The following boundary conditions at the glottis and the lip are typically adopted to solve the system.

$$
u ( 0 , t ) = u _ { \mathrm { i n } } ( t )
$$

$$
\mathrm { G l o t t i s \ : b o u n d a r y }\tag{22}
$$

$$
R _ { \mathrm { l i p s } } p ( L , t ) + L _ { \mathrm { l i p s } } p _ { t } ( L , t ) = R _ { \mathrm { l i p s } } L _ { \mathrm { l i p s } } u _ { t } ( L , t )
$$

Lip boundary

(23)

The lip opening area, $A ( L )$ , controls how much sound is reflected back into the tract versus how much is dissipated outside via an inductance term $\begin{array} { r } { L _ { \mathrm { l i p s } } = \frac { 8 \rho } { 3 \pi \sqrt { \pi A ( L ) } } } \end{array}$ and resistance term $\begin{array} { r } { R _ { \mathrm { l i p s } } = \frac { 1 2 8 \rho c } { 9 \pi ^ { 2 } A ( L ) } } \end{array}$ (with coefficients derived by Flanagan [21]).

## A.2 Circuit Interpretation and the Transmission Line Model

Interpreting the fluid dynamics of the vocal tract as a circuit model has been long studied [14]. Considering a piecewise-constant area function with N cylindrical sections where each n-th section has of length $l _ { n }$ and area $A _ { n }$ and taking the Fourier transform of Eq. (18)-(19) gives the following frequency-domain governing equations:

$$
j \omega C _ { n } P _ { n } ( \omega ) = U _ { n } ( \omega ) - U _ { n + 1 } ( \omega ) ,\tag{24}
$$

$$
j \omega L _ { n } U _ { n } ( \omega ) = P _ { n - 1 } ( \omega ) - P _ { n } ( \omega ) ,\tag{25}
$$

where $C _ { n } = A _ { n } l _ { n } / ( \rho _ { 0 } c ^ { 2 } )$ is the acoustic compliance and $L _ { n } = \rho _ { 0 } l _ { n } / ( 2 A _ { n } )$ is the acoustic inductance of the half-section. This describes a lossless ideal fluid flow with purely imaginary impedances.

Extension with viscous losses. More realistic fluid result from a viscous boundary layer at the cylinder wall. This can be modeled by imposing the no-slip condition at the wall, so the resulting wall shear stress acts as a drag on the fluid. This ‘wall friction’ due to this boundary layer is known to be well-modeled by introducing an additional dissipative term in the momentum equation:

$$
- \nabla p = L _ { n } \dot { u } + r _ { \mathrm { w f } } ( t ) * u\tag{26}
$$

where denotes convolution and $r _ { \mathrm { w f } } ( t )$ is the impulse response of the wall-friction operator. For a circular cylinder of perimeter $S _ { n }$ and cross-sectional area $A _ { n } ,$ the viscous boundary-layer theory gives the frequency-domain expression of this wall-friction operator as:

$$
R _ { n } ( x , \omega ) = \frac { S _ { n } ( x ) } { 2 A _ { n } ^ { 2 } ( x ) } \sqrt { \frac { \rho _ { 0 } \omega \mu } { 2 } }\tag{27}
$$

which can be analogously interpreted as a frequency-dependent resistance $R _ { n } ( \omega ) \propto \sqrt { \omega }$ . Taking the Fourier transform to Eq. (26) gives the follows.

$$
P _ { n - 1 } ( \omega ) - P _ { n } ( \omega ) = ( j \omega L _ { n } + R _ { n } ( \omega ) ) U _ { n } ( \omega )
$$

The series impedance of the cylinder at the n-th section is therefore

$$
Z _ { s , n } ( \omega ) = R _ { n } ( \omega ) + j \omega L _ { n } .\tag{28}
$$

Extension with yielding walls. In addition to modeling the fluid flow, the acoustic modeling of the vocal tract wall can be approached using a damped harmonic oscillator (equivalently the mass-spring-damper). Denoting the wall mass, damping coefficient, and stiffness coefficient as $m _ { w }$ $b _ { w }$ , and $k _ { w }$ , respectively, the normal displacement of the wall $\xi _ { n }$ satisfies

$$
\begin{array} { r } { m _ { w } \ddot { \xi } _ { n } + b _ { w } \dot { \xi } _ { n } + k _ { w } \xi _ { n } = p ( x , t ) . } \end{array}\tag{29}
$$

In the frequency domain, Eq. (29) gives a wall displacement $\Xi _ { n } = P / ( - \omega ^ { 2 } m _ { w } + j \omega b _ { w } + k _ { w } )$ , so the wall presents a shunt impedance to the acoustic field. For n-th section with lateral surface area $S _ { n } l _ { n }$ , the lumped wall impedance is

$$
Z _ { w , n } = \frac { - \omega ^ { 2 } m _ { w } + j \omega b _ { w } + k _ { w } } { S _ { n } l _ { n } \omega ^ { 2 } } = \frac { R _ { w , n } + j \omega L _ { w , n } + \frac { 1 } { j \omega C _ { w , n } } } { j \omega }
$$

Combined with the acoustic compliance $C _ { n }$ in parallel, the total shunt admittance is

$$
Z _ { w , n } ( \omega ) = \left( R _ { w , n } + j \omega L _ { w , n } + \frac { 1 } { j \omega C _ { w , n } } \right) \parallel \frac { 1 } { j \omega C _ { n } }\tag{30}
$$

where $L _ { w , n } = m _ { w } / ( S _ { n } l _ { n } ) , R _ { w , n } = b _ { w } / ( S _ { n } l _ { n } ) , \mathrm { a n d } C _ { w , n } = ( S _ { n } l _ { n } ) / k _ { w } .$

## A.3 Transfer Functions

Combining Eq. (28) and Eq. (30), each n-th section of the acoustic tube is represented in the frequency domain as the T-network where a series arm impedance $Z _ { s , n }$ carries momentum losses and a shunt arm $Z _ { w , n }$ stores compliance and wall losses. Finally, as described in Eq. (3), the two-port transfer matrix for section n is

$$
\Big [ P _ { n + 1 } \Big ] = \underbrace { \Big [ 1 + Z _ { s , n } / Z _ { w , n } } _ { - 1 / Z _ { w , n } } \quad \underbrace { - 2 Z _ { s , n } - Z _ { s , n } ^ { 2 } / Z _ { w , n } } _ { 1 + Z _ { s , n } / Z _ { w , n } } \Big ]  _ { \bf { K } \quad \underbrace { - 1 } _ { ( \sigma , \sigma ) } } \Big [ P _ { n } \Big ]\tag{31}
$$

with a section matrix

$$
\mathbf { K } _ { n } = \left[ \begin{array} { c c } { \mathbf { A } _ { n } } & { \mathbf { B } _ { n } } \\ { \mathbf { C } _ { n } } & { \mathbf { D } _ { n } } \end{array} \right] .
$$

This can be used to construct partial chain matrices by concatenating the individual section matrices:

$$
{ \bf K } _ { 0 \mapsto n } : = { \bf K } _ { n } { \bf K } _ { n - 1 } \cdot \cdot \cdot { \bf K } _ { 1 } = \left[ { \begin{array} { c c } { { \bf A } _ { 0 n } } & { { \bf B } _ { 0 n } } \\ { { \bf C } _ { 0 n } } & { { \bf D } _ { 0 n } } \end{array} } \right] ,\tag{32}
$$

$$
{ \bf K } _ { n \mapsto N + 1 } : = { \bf K } _ { N } \cdot \cdot \cdot { \bf K } _ { n + 1 } = \left[ { \bf A } _ { n N } \quad { \bf B } _ { n N } \right] .\tag{33}
$$

Note that chain matrices have unit determinant for reciprocal media, i.e., det $\begin{array} { r l } { \mathbf { K } _ { 0 \mapsto n } } & { { } = } \end{array}$ det ${ \bf K } _ { n \mapsto N + 1 } = 1$

Transfer function from glottis to section n. To compute $H _ { 0 \mapsto n }$ , the transfer function from glottis to section n, apply Eq. (31) for $\mathbf { K } _ { 0 \mapsto n } \colon$

$$
\left[ { \begin{array} { c } { P _ { n } } \\ { U _ { n } } \end{array} } \right] = \left[ { \begin{array} { c c } { \mathbf { A } _ { 0 n } } & { \mathbf { B } _ { 0 n } } \\ { \mathbf { C } _ { 0 n } } & { \mathbf { D } _ { 0 n } } \end{array} } \right] \left[ { \begin{array} { c } { P _ { \mathrm { i n } } } \\ { U _ { \mathrm { i n } } } \end{array} } \right]\tag{34}
$$

where $P _ { \mathrm { i n } }$ is the pressure at the glottal plane and $U _ { \mathrm { i n } }$ is the glottal volume velocity. Imposing the boundary condition at the glottis, $P _ { \mathrm { i n } } = Z _ { \mathrm { i n } } U _ { \mathrm { i n } }$

$$
U _ { n } = ( { \bf C } _ { 0 n } Z _ { \mathrm { i n } } + { \bf D } _ { 0 n } ) U _ { \mathrm { i n } } ,
$$

$$
G _ { 0 \mapsto n } ( \omega ) = \frac { U _ { n } } { U _ { \mathrm { i n } } } = \mathbf { C } _ { 0 n } Z _ { \mathrm { i n } } + \mathbf { D } _ { 0 n } .\tag{35}
$$

The backward impedance seen from section n looking toward the glottis follows from the same substitution.

$$
Z _ { 1 } = - \frac { P _ { n } } { U _ { n } } = - \frac { \mathbf { A } _ { 0 n } Z _ { \mathrm { i n } } + \mathbf { B } _ { 0 n } } { \mathbf { C } _ { 0 n } Z _ { \mathrm { i n } } + \mathbf { D } _ { 0 n } }
$$

The sign convention fo $\cdot - U _ { n }$ indicates that $U _ { n }$ is flowing away from glottis and thus into the $Z _ { 1 }$ load when viewed from section n.

Transfer function from section n to lips. From boundary condition Eq. (23), the radiation impedance $Z _ { \mathrm { l i p s } } = R _ { \mathrm { l i p s } } \ \parallel \ j \omega L _ { \mathrm { l i p s } }$ can be derived with

$$
L _ { \mathrm { l i p s } } = \frac { 8 \rho _ { 0 } } { 3 \pi \sqrt { \pi A _ { N } } } \qquad \mathrm { a n d } \qquad R _ { \mathrm { l i p s } } = \frac { 1 2 8 \rho _ { 0 } c } { 9 \pi ^ { 2 } A _ { N } } .\tag{36}
$$

Now, apply ${ \bf K } _ { n \mapsto N + 1 }$ from section n to the lip plane, where the boundary condition $P _ { N + 1 } =$ $Z _ { { \mathrm { l i p s } } } U _ { N + 1 }$ must be satisfied:

$$
\begin{array} { r } { \Big [ P _ { N + 1 } \Big ] = \Big [ \mathbf { A } _ { n N } \quad \mathbf { B } _ { n N } \Big ] \Big [ P _ { n } \Big ] , } \\ { \Big [ U _ { N + 1 } \Big ] = \Big [ \mathbf { C } _ { n N } \quad \mathbf { D } _ { n N } \Big ] \Big [ U _ { n } \Big ] , } \end{array}\tag{37}
$$

giving

$$
\begin{array} { r } { P _ { N + 1 } = \mathbf { A } _ { n N } P _ { n } + \mathbf { B } _ { n N } U _ { n } , } \end{array}\tag{38}
$$

$$
\begin{array} { r } { U _ { N + 1 } = \mathbf { C } _ { n N } P _ { n } + \mathbf { D } _ { n N } U _ { n } . } \end{array}\tag{39}
$$

Substituting $P _ { N + 1 } = Z _ { \mathrm { l i p s } } U _ { N + 1 }$ into Eq. (38) and (39) to eliminate $P _ { N + 1 } \mathrm { : }$

$$
Z _ { \mathrm { l i p s } } U _ { N + 1 } = { \bf A } _ { n N } P _ { n } + { \bf B } _ { n N } U _ { n } .\tag{40}
$$

Divide Eq. (40) by (39):

$$
Z _ { \mathrm { l i p s } } = { \frac { \mathbf { A } _ { n N } P _ { n } + \mathbf { B } _ { n N } U _ { n } } { \mathbf { C } _ { n N } P _ { n } + \mathbf { D } _ { n N } U _ { n } } } \qquad { \implies } \qquad Z _ { 2 } : = { \frac { P _ { n } } { U _ { n } } } = { \frac { \mathbf { D } _ { n N } Z _ { \mathrm { l i p s } } - \mathbf { B } _ { n N } } { \mathbf { A } _ { n N } - \mathbf { C } _ { n N } Z _ { \mathrm { l i p s } } } }\tag{41}
$$

which is the forward impedance seen from section n towards the lips. Eliminating $P _ { n }$ using $P _ { n } =$ $Z _ { 2 } U _ { n }$ yields:

$$
U _ { N + 1 } = \left( \mathbf { C } _ { n N } Z _ { 2 } + \mathbf { D } _ { n N } \right) U _ { n } = \frac { \mathbf { C } _ { n N } \left( \mathbf { D } _ { n N } Z _ { \mathrm { l i p s } } - \mathbf { B } _ { n N } \right) + \mathbf { D } _ { n N } \left( \mathbf { A } _ { n N } - \mathbf { C } _ { n N } Z _ { \mathrm { l i p s } } \right) } { \mathbf { A } _ { n N } - \mathbf { C } _ { n N } Z _ { \mathrm { l i p s } } } U _ { n } .
$$

Because $\mathbf { A } _ { n N } \mathbf { D } _ { n N } - \mathbf { B } _ { n N } \mathbf { C } _ { n N } = \operatorname* { d e t } \mathbf { K } _ { n \mapsto N + 1 } = 1$ , the numerator simplifies, leaving

$$
G _ { n \mapsto N + 1 } = \frac { U _ { N + 1 } } { U _ { n } } = \frac { 1 } { \mathbf { A } _ { n N } - \mathbf { C } _ { n N } Z _ { \mathrm { l i p s } } } .\tag{42}
$$

The pressure transfer function from n to lips is therefore

$$
H _ { n \mapsto N + 1 } = { \frac { P _ { \mathrm { o u t } } } { U _ { n } } } = { \frac { Z _ { \mathrm { l i p s } } } { \mathbf { A } _ { n N } - \mathbf { C } _ { n N } Z _ { \mathrm { l i p s } } } } .\tag{43}
$$

Combining the full and noise transfer functions. The glottis-to-lips volume velocity transfer is obtained by applying $\mathbf { K } _ { \mathrm { t o t } } = \mathbf { K } _ { N } \mathbf { K } _ { N - 1 } \cdot \cdot \cdot \mathbf { K } _ { 1 }$ as

$$
\begin{array} { r } { \Big [ P _ { N + 1 } \Big ] = \Big [ \mathbf { A } _ { 0 N } \quad \mathbf { B } _ { 0 N } \Big ] \Big [ P _ { \mathrm { i n } } \Big ] . } \\ { \Big [ U _ { N + 1 } \Big ] = \Big [ \mathbf { C } _ { 0 N } \quad \mathbf { D } _ { 0 N } \Big ] \Big [ U _ { \mathrm { i n } } \Big ] . } \end{array}\tag{44}
$$

Setting $P _ { \mathrm { i n } } = Z _ { \mathrm { i n } } U _ { \mathrm { i n } }$ at the glottis and imposing $P _ { N + 1 } = Z _ { \mathrm { l i p s } } U _ { N + 1 }$ at the lips gives

$$
\mathbf { A } _ { 0 N } Z _ { \mathrm { i n } } + \mathbf { B } _ { 0 N } = Z _ { \mathrm { l i p s } } ( \mathbf { C } _ { 0 N } Z _ { \mathrm { i n } } + \mathbf { D } _ { 0 N } )
$$

therefore the input impedance of the tract seen from the glottis is

$$
Z _ { \mathrm { i n } } = \frac { \mathbf { D } _ { 0 N } Z _ { \mathrm { l i p s } } - \mathbf { B } _ { 0 N } } { \mathbf { A } _ { 0 N } - \mathbf { C } _ { 0 N } Z _ { \mathrm { l i p s } } } .\tag{45}
$$

With $Z _ { \mathrm { i n } }$ established, substitute Eq. (45) into $H _ { \mathrm { t r a c t } } = P _ { N + 1 } / U _ { \mathrm { i n } } = \mathbf { A } _ { 0 N } Z _ { \mathrm { i n } } + \mathbf { B } _ { 0 N } { \mathrm { g i v e } }$ s:

$$
H _ { \mathrm { t r a c t } } ( \omega ) = \mathbf { A } _ { 0 N } \left( \frac { \mathbf { D } _ { 0 N } Z _ { \mathrm { l i p s } } - \mathbf { B } _ { 0 N } } { \mathbf { A } _ { 0 N } - \mathbf { C } _ { 0 N } Z _ { \mathrm { l i p s } } } \right) + \mathbf { B } _ { 0 N } ,\tag{46}
$$

$$
= \frac { \mathbf { A } _ { 0 N } ( \mathbf { D } _ { 0 N } Z _ { \mathrm { l i p s } } - \mathbf { B } _ { 0 N } ) + \mathbf { B } _ { 0 N } ( \mathbf { A } _ { 0 N } - \mathbf { C } _ { 0 N } Z _ { \mathrm { l i p s } } ) } { \mathbf { A } _ { 0 N } - \mathbf { C } _ { 0 N } Z _ { \mathrm { l i p s } } } .\tag{47}
$$

The numerator expands to $( \mathbf { A } _ { 0 N } \mathbf { D } _ { 0 N } - \mathbf { B } _ { 0 N } \mathbf { C } _ { 0 N } ) Z _ { \mathrm { l i p s } } = \operatorname* { d e t } \mathbf { K } _ { \mathrm { t o t } } \cdot Z _ { \mathrm { l i p s } } = Z _ { \mathrm { l i p s } }$ . Therefore, the total pressure transfer function is

$$
H _ { \mathrm { t r a c t } } = { \frac { Z _ { \mathrm { l i p s } } } { { \bf A } _ { n N } - { \bf C } _ { n N } Z _ { \mathrm { l i p s } } } }\tag{48}
$$

As described in Subsection $5 . 2 ,$ the propagation of the volume-velocity noise source $U _ { \mathrm { n o i s e } _ { n } } ( \omega )$ to the lips is is only governed by ${ \bf K } _ { n \mapsto N + 1 }$ , and therefore the pressure transfer function for the noise from the section n is identical to $H _ { n \mapsto N + 1 }$ . This confirms that the total combined transfer function from the glottal pulse and the noise to the lips as Eq. (7).

## A.4 The Nasal Tract

Although we do not pursue nasal tract reconstruction because the nasal tract does not appear in the MRI frames, the transmission line model can be used to model it. In fact, with the TLM method, the nasal effect can be modeled as a bifurcated transmission line, and the combination of these two LTI systems can be represented as an equivalent circuit consisting of a single transmission line. In the present work as well, although the nasal cavity is not explicitly reconstructed, it can be viewed as being modeled using a lumped equivalent area function. Examples of this technique are provided in Sondhi and Schroeter [55].

## B Finite Difference Techniques

## B.1 Time Domain Governing Equation

Solving Eq. (18) with the wall-friction momentum equation Eq. (26) requires a convolution with a half-order integro-differential operator for frequency-dependent resistance $R ( \omega ) \propto { \sqrt { \omega } } .$ . In practice, the resistance is implemented using Hagen–Poiseuille DC resistance [39]

$$
R _ { n } ^ { \mathrm { { D C } } } = { \frac { 4 \mu l _ { n } \pi } { A _ { n } ^ { 2 } } }
$$

which is the Stokes flow resistance for a circular duct in the zero-frequency limit. With this substitu tion, all terms in the circuit equations for the N-section vocal tract model reduce to the following coupled ODEs.

$$
\dot { p } _ { n } = \frac { 1 } { C _ { n } } ( u _ { n } - u _ { n + 1 } - u _ { w , n } )\tag{mass}
$$

$$
( L _ { n - 1 } + L _ { n } ) { \dot { u } } _ { n } + ( R _ { n - 1 } + R _ { n } ) u _ { n } = p _ { n - 1 } - p _ { n }\tag{49}
$$

momentum

$$
L _ { w , n } \ddot { u } _ { w , n } + R _ { w , n } \dot { u } _ { w , n } + \frac { u _ { w , n } } { C _ { w , n } } = \dot { p } _ { n }\tag{50}
$$

wall vibration

(51)

The expression for the wall vibration can be deduced from Eq. (29) by taking time derivative and denoting $u _ { w , n } = S _ { n } l _ { n } \dot { \xi } _ { n }$ . The last section $i = N$ terminates into the radiation impedance $Z _ { \mathrm { l i p s } } = R _ { \mathrm { l i p s } } \parallel j \omega L _ { \mathrm { l i p s } }$ . In the time domain this parallel R–L load introduces two additional flow unknowns $u _ { N + 1 }$ (total lip flow) and $u _ { N + 2 }$ (inductive branch flow):

$$
\dot { u } _ { N + 1 } L _ { N } + u _ { N + 1 } ( R _ { N } + R _ { \mathrm { l i p s } } ) - u _ { N + 2 } R _ { \mathrm { l i p s } } = p _ { N } ,\tag{52}
$$

$$
\dot { u } _ { N + 2 } L _ { \mathrm { l i p s } } + u _ { N + 2 } R _ { \mathrm { l i p s } } - u _ { N + 1 } R _ { \mathrm { l i p s } } = 0 .\tag{53}
$$

The radiated pressure (voltage across $R _ { \mathrm { l i p s } } )$ is

$$
p _ { \mathrm { o u t } } ( t ) = R _ { \mathrm { l i p s } } \big ( u _ { N + 1 } ( t ) - u _ { N + 2 } ( t ) \big ) = L _ { \mathrm { l i p s } } \dot { u } _ { N + 2 } ( t ) .\tag{54}
$$

These equations describe the governing equations in continuous time-domain. The only departure from the frequency-domain expressions is the substitution of $R _ { n } ( \omega )$ by $R _ { n } ^ { \mathrm { D C } }$

## B.2 Numerical Scheme

In order to solve the time-domain governing equation on a discrete stencil, the semi-implicit scheme has been employed:

$$
\dot { f } = \frac { f - f ^ { \prime } } { \Delta t \vartheta } - \frac { \bar { \vartheta } } { \vartheta } \dot { f } ^ { \prime } , \qquad \bar { \vartheta } = 1 - \vartheta .\tag{55}
$$

While various schemes may result depending on the value of $\vartheta ,$ in practice, we set $\vartheta = 0 . 5$ (trapezoidal) to perform time-domain simulations. The process of solving Subsection B.1 with this scheme reduces to a tridiagonal linear system, which can be resolved through the time-domain recursion and jax.linalg.tridiagonal\_solve. Please note that in this study, this time-domain simulation serves only as a baseline for validating the performance of the employed frequency-domain simulator. For further details regarding the time-domain implementation, please refer to Birkholz and Jackel [5].

![](images/0249512addcb9c02799f02a89b6a5873ed5bd70fffa00f7982b88a028947b34c.jpg)  
Figure 8: A schematic diagram illustrating three application cases of the differentiable Φ, as described in Section 6. The problem of estimating $A ( x , t )$ from speech can thus be solved through various gradient-based optimization techniques, and none of these methods presuppose a ground truth A, but rather resolve this inverse problem solely by utilizing the real-world speech signal $p _ { \mathrm { t g t } }$

## C Experimental Details

## C.1 Simulation Details

The simulation is modeled in centimeters. For all experiments, we use $N = 3 2$ spatial segments. For the benchmarks in the paper, and in general, we run our simulation at 16 kHz. Higher fidelity can however be achieved with higher sample rates. For the singing examples in the supplementary material, we use sample rates of 32 kHz - 44.1 kHz.

Glottal Pulse. To estimate the glottal pulse $u _ { \mathrm { i n } }$ of each signal, we first extract fundamental frequencies from the speech sample using Swift-F0 [44]. The fundamental frequencies are then used to construct a glottal pulse using the LF model [18]. Its time-varying amplitude is determined by the RMS energy of the signal, measured in Librosa. The pulse is normalized so it has a maximum value of 1 $\mathrm { c m ^ { 3 } / \bar { s } }$ Generally, we find that gradient-based optimization is insensitive to the glottal pulse used. The spectrogram loss focuses gradients on formants, which are a property of the vocal tract’s resonances and not the glottal pulse. To show robustness to glottal pulse, we include a vocal tract reconstruction of a song played on a piano in the supplementary material. The model is able to create an acapella reconstruction of the song, Ode to Joy, while maintaining a human-like timbre.

## C.2 Loss Function

As described in Section 4, we use a multiscale log Mel spectrogram loss to fit the speech samples. We use $n _ { \mathrm { m e l } } = 1 2 8$ Mel bins, FFT sizes of (512, 1024, 2048), window lengths of (160, 400, 800), and hop lengths of (40, 80, 160). This builds three configurations for the spectrogram resolution: $\mathcal { C } = \{ ( 5 1 2 , 1 6 0 , 4 0 )$ , (1024, 400, 80), (2048, 800, 160) . For a target pressure $p _ { \mathrm { t g t } }$ and an output pressure $p _ { \mathrm { o u t } } \triangleq \Phi ( u _ { \mathrm { i n } } , A _ { \theta } )$ , the loss is computed using the $L _ { 2 }$ norm of the log Mel spectrogram distances under three configurations $( n _ { \mathrm { f f t } } , n _ { \mathrm { w i n } } , n _ { \mathrm { h o p } } ) \in \mathcal { C }$ averaged over the choices

$$
\mathcal { L } _ { 2 } ( p _ { \mathrm { o u t } } , p _ { \mathrm { t g t } } ) = \frac { 1 } { | \mathcal { C } | } \sum _ { \mathcal { C } } \left\| \log \operatorname { m e l } ( p _ { \mathrm { o u t } } ) - \log \operatorname { m e l } ( p _ { \mathrm { t g t } } ) \right\| _ { 2 } .\tag{56}
$$

Here, log mel is the log Mel spectrogram transformation using the Mel filterbank $\mathbf { M } \in \mathbb { R } ^ { n }$ <sub>mel</sub>×n<sub>freq</sub> and the short-time Fourier transformation (STFT) that outputs the spectrogram $\mathbf { p } \in \mathbb { R } ^ { n _ { \mathrm { f r e q } } \times K }$ , where $n _ { \mathrm { f r e q } } = \lfloor n _ { \mathrm { f f t } } / 2 \rfloor + 1$ is the number of STFT frequency bins and $K = \lceil ( n _ { \mathrm { s a m p l e } } - n _ { \mathrm { w i n d o w } } ) / n _ { \mathrm { h o p } } \rceil + 1$ is the number of its window chunks. The m-th Mel bin of the k-th window of the log Mel spectrogram is computed as log $\begin{array} { r } { \mathrm { m e l } ( p ) [ m , k ] = \log ( \sum _ { j } \mathbf { M } [ m , j ] \cdot | \mathbf { p } [ j , k ] | ) } \end{array}$ . It is well-known that this trick of employing a multi-resolution spectrogram can distribute the resolution dependency of the loss calculation, thereby facilitating smoother optimization.

## C.3 Evaluation Setup for Perceptual Speech Quality Metrics

We evaluate how well each method can fit 25 random speech samples from LibriTTS-R [29]. For the results in Table 1, we parametrize each area function with a multiplicative filter network [19] (this choice is explained in Subsection 6.1). Gradient descent is done with the Adam optimizer [28] with a learning rate of $1 0 ^ { - 2 }$ for 200 steps.

Using TorchAudio-Squim [32], we report SI-SDR (scale-invariant signal-to-distortion ratio), STOI (short-time objective intelligibility), and PESQ (wideband perceptual evaluation of speech quality).

## C.4 Neural Fields

We study two neural field parameterizations $F _ { \theta }$ in the paper, a 4-layer fully connected network with random Fourier feature (RFF) positional encoding [59], and a 3-layer multiplicative filter network (MFN) [19]. The RFF network has a 64-dimensional encoding, and a 256-dimensional hidden dimension. The MFN network had an input scale of 256, and 256-dimenaional hidden dimension. To constrain the area functions to be positive, we apply a softmax to the output, and clip the minimum values at $1 0 ^ { - 2 }$ . All models were trained with the Adam optimizer [28] for 200 steps and a learning rate of $1 0 ^ { - 2 }$ . To fit a 5 second speech sample, training takes approximately 1 minute.

Neural Fields Improve Acoustic Reconstruction. We also study how each field performs at reconstructing real-world speech in Table 5. Using the experimental setup in Appendix C.3, we reconstruct the area functions for 25 speech samples from [29]. We find that both neural fields significantly outperform the discrete case across all speech metrics.

Table 5: Neural Field Reconstruction
<table><tr><td></td><td>Discrete</td><td>RFF [59]</td><td>MFN [19]</td></tr><tr><td>SI-SDR</td><td>10.78 ±2.22</td><td>14.51 ±3.78</td><td>16.39 ±2.42</td></tr><tr><td>STOI</td><td>0.81 ±0.02</td><td>0.84 ±0.05</td><td>0.93 ±0.02</td></tr><tr><td>PESQ</td><td>1.57 ±0.12</td><td>1.76 ±0.26</td><td>1.98 ±0.26</td></tr></table>

## C.5 Autoencoding

We use the Wav2Vec 2.0 [1] architecture for the autoencoder. Raw waveforms are input into a CNN with the same dimension as the original Wav2Vec2 model. The CNN embeddings are then passed into a 12 layer transformer with 12 heads, and a hidden dimension of 768. The model is trained with the Adam optimizer [28] at a learning rate of $1 0 ^ { - 4 } .$ . The model is trained on 3 second segments of English and a batch size of 64 for 15,000 iterations, equating to 800 hours of training audio. It is then fine-tuned for each other language for 10 epochs.

We train our encoder on LibriTTS-R [29] for English, Multilingual LibriSpeech [46] for other European languages, KSS [48] for Korean, JVS [58] for Japanese, and AISHELL-3 [53] for Chinese.

## C.5.1 UMAP Projection

UMAP [40] projections are performed on area functions encoded from the VoxAngeles dataset [10]. To generate the UMAP projections, we used 12 neighbors and a minimum distance of 0.2. However, we found that the UMAP projections are generally robust to these hyperparameters. In Figure 9, we show an UMAP projection on an even larger set of IPA phonemes. The area functions remain disentangled by phoneme, and still form a continuous spectrum.

## C.6 MRI Generation

A StyleGAN2 [26] network is trained on the MRI data of [36]. Using the identity labels from the dataset, the network is conditioned on these labels to preserve identity during reconstruction. to The model is trained using the default configurations for 64x64 images. During reconstruction, we use K = 30 latent vectors to reconstruct 3 second clips at a time.

The audio recorded from the MRI dataset [36] is extremely noisy due to the spinning magnets in the MRI machine. To remove noise from the recordings, we use the Adobe Express speech enhancement tool. We then use the enhanced speech samples as the target utterances when reconstructing the MRI video.

## C.6.1 MRI Reconstruction with Diffusion Models

We have also tried using Diffusion Posterior Sampling (DPS) [11] for MRI reconstruction, but we have found that its performance was worse than using the GAN. The diffusion model was not able to make large geometric changes such as tongue movement. DPS and its follow-ups mainly focus on low-level inverse problems such as deblurring. Compared to a diffusion model, a GAN has a semantically meaningful latent space which allows better optimization of the vocal tract shape during speech. Solving blind inverse problems with diffusion models is still an open problem, and we believe that combining diffusion with acoustic simulation is an exciting extension of our work.

![](images/00ee1f2e38c167e4fac92f14e26caf247a4676dd961a9bf26683340f4208a5a5.jpg)  
Figure 9: A Larger UMAP Projection of Encoded Vocal Tract Area Functions. We provide a UMAP projection of an even larger set of phonemes. The embedded area functions form a continuous space spanning the set of IPA symbols.

## D Ethical Discussion

While the simulator presented in this paper is intended for realistic speech reconstruction and not speech generation, we acknowledge that generative speech models—and the tasks that they enable, such as voice cloning—have become serious ethical issues. As with other speech models, responsible use is essential. All samples produced with our simulator should be clearly labeled as such.