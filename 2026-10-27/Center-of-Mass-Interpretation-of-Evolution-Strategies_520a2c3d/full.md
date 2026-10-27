# Center-of-Mass Interpretation of Evolution Strategies

Akshat Gupta UC Berkeley akshat.gupta@berkeley.edu

## Abstract

Evolution strategies (ES) update a model by sampling a population of perturbed copies, scoring each one, and moving in the rewardweighted direction of the perturbations. This update is usually presented as an estimate of a smoothed gradient. We present a geometrical interpretation of this update instead. Treat each perturbed model as a particle and its re ward as the particle’s mass. Before scoring, every particle has the same mass and the population balances at its centroid. Once rewards are assigned, it balances at its reward-weighted center of mass instead. We show that the mean subtracted ES update is exactly this displacement, scaled by a scalar factor that depends on the mean reward, for any finite population. The physical reading needs nonnegative rewards, which can always be arranged by adding a constant to every reward without changing the update. In the limit of a large population or under antithetic sampling, the centroid sits on the model itself, so ES has an even simpler geometric interpretation: from its current location, the model moves towards the reward-weighted center of mass of its own population. The same picture explains what subtracting the mean reward does: it changes the point from which the displacement is measured, from the model to the sampled centroid. Our contribution is an intuitive, geometric way to visualize an ES update, exact for any finite population.

## 1 Introduction and Related Work

Evolution strategies (ES) have recently returned as a practical alternative to backpropagation for fine-tuning large language models (Qiu et al., 2026; Abdi et al., 2026). The algorithm is simple. Perturb the current model N times with Gaussian noise, score each perturbed copy, and merge the perturbations weighted by their rewards. The standard way to understand this update is as a Monte Carlo estimate of the gradient of a smoothed objective (Nesterov and Spokoiny, 2017; Salimans et al., 2017). That view is correct, but says little about what a single update does to the model in parameter space.

In this paper, we give the ES update a geometrical interpretation grounded in a physical intuition. Think of the N perturbed models as particles sitting around the current model (Figure 1(a)). Before scoring, every particle has the same mass and the population balances at its centroid, which is also its center-of-mass (COM). After scoring, give each particle a mass equal to its reward. The COM then shifts away from the centroid. We show that the mean-subtracted ES update is exactly this displacement, scaled by a scalar factor that depends on the mean reward, for any finite population (Figure 1). For the masses to be physical, the rewards should be nonnegative with a positive total. This is not a restriction: adding a constant to every reward makes it so and leaves the update unchanged. In the limit of a large population or at any N under antithetic sampling (mirrored sampling), the centroid sits on the model itself, and the picture becomes simpler and even more intuitive: from its current location in parameter space, the model moves towards the reward-weighted center of mass of its sampled population.

The center of mass has appeared in the ES literature before, but in a different role. In classical evolution strategies the best offspring are selected and the parent is moved to their centroid, a point that Beyer and Schwefel (2002) describe as a center of mass. CMA-ES does the same with rank-based weights in place of equal ones (Hansen, 2016), and the selection term of Price’s equation gives the change in a population’s mean trait as the covariance between trait and fitness (Price, 1970). In each case the object described is the new mean, written as a weighted average of the candidates. The mean-subtracted update used in modern practice, for instance by Qiu et al. (2026) with an additional standard-deviation normalization, is not of this form. It takes a step, and we show that this step is the displacement between two centers of mass of the same population, one computed with equal masses and one with rewards as masses. This is also what makes the role of reward centering visible. Subtracting the mean reward changes the point from which the displacement is measured, from the model location to the sampled centroid.

![](images/795b704d8b37eab9cd574b2bb811ddd2b74f27acf50f5b9ab6e2388ad26eccc4.jpg)

![](images/fe60d2adb5bfb6f97d38e25d2035cfb450caffa44a3261b11a86815cb05e2a15.jpg)

![](images/9796165720d4e5bf28f8b0ba6fd71490a527ea6668d07d72344aeb703c4110aa.jpg)  
Figure 1: One ES update as a change of center of mass. (a) Candidates are sampled around the model θ in antithetic pairs, so with equal masses the population’s centroid coincides with θ. (b) Giving each candidate a mass equal to its reward moves the center of mass (COM) towards the high-reward side. (c) The update is proportional to the displacement from the centroid to the COM, $\Delta \pmb { \theta } = ( \alpha \overline { { r } } / \sigma ) ( \mathbf { p } _ { \mathrm { C O M } } - \bar { \mathbf { p } } )$ , which here starts at θ. The same sixteen candidates are shown in every panel; displacements are exaggerated for visibility.

## 2 Background: Anatomy of an ES Update

Let the current model parameters be represented by a point $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$ in a d−dimensional parameter space. An ES iteration usually consists of three steps: creating perturbed models, scoring them, and combining the sampled perturbations to update the current parameters.

1. Perturb. First, we independently sample N random directions and create perturbed versions of the original model (Salimans et al., 2017). We represent these perturbed versions by p<sub>i</sub>, where

$$
p _ { i } = \theta + \sigma \epsilon _ { i } , \quad \epsilon _ { i } \sim \mathcal { N } ( 0 , I _ { d } ) , \quad i = 1 , \dots , N .\tag{1}
$$

Here, $\epsilon _ { i }$ is the sampled direction, $\sigma > 0$ controls the scale of the perturbation, $p _ { i }$ is the resulting perturbed model, and N represents the total number of perturbations involved in one ES update step. Geometrically, the points $p _ { 1 } , \ldots , p _ { N }$ can be visualized to form a population of points in parameter space as shown in Figure 1 (a), sampled from a normal distribution centered at θ.

2. Score. Second, we evaluate each perturbed model $p _ { i }$ to obtain a reward $r _ { i } = f ( p _ { i } )$ , where a higher reward indicates better performance. We denote the mean reward by $\begin{array} { r } { \overline { { r } } = \frac { \bar { 1 } } { N } \sum _ { i = 1 } ^ { N } r _ { i } } \end{array}$ and the standard deviation of rewards within the population by $\begin{array} { r } { s _ { r } = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( r _ { i } - \overline { { r } } ) ^ { 2 } } } \end{array}$

3. Combine. Third, we obtain the model update by taking a weighted sum of the sampled directions,

$$
\Delta \theta = \frac { \alpha } { N } \sum _ { i = 1 } ^ { N } w _ { i } \epsilon _ { i } ,\tag{2}
$$

where $\alpha > 0$ is a learning rate which also absorbs the perturbation scale following Qiu et al. (2026) and each coefficient $w _ { i }$ is obtained from a transformation of the rewards. The update $\Delta \theta$ is then added to the current parameters θ. The choice of transformation determines how each candidate contributes to this update.

The Weighting Rules. ES implementations differ in how they convert rewards into the coefficients $w _ { i } .$ . Common choices include subtracting the mean reward, standardizing the rewards, and assigning coefficients based on the ranks of the candidates.

Our main derivation covers weighting rules whose coefficients sum to zero, $\textstyle \sum _ { i = 1 } ^ { N } { w _ { i } = 0 }$ . Several variants of the ES algorithm that share this condition, as shown in Table 1. Rules whose coefficients do not sum to zero, such as using raw rewards directly, are discussed separately in 4.1.

## 3 The Sampled Population and Its Centroid

The N perturbed models in eq. (3) are sampled around the current model θ, as shown in fig. 1(a). The centroid of this sampled population can be written as

$$
\bar { \bf p } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \bf p } _ { i } = { \pmb \theta } + \sigma \bar { \bf \varepsilon } , \qquad \bar { \bf \varepsilon } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \pmb \varepsilon } _ { i } ,\tag{3}
$$

The centroid has a simple physical interpretation. Suppose every sampled model is a point particle with the same mass m. The center of mass of the population is

$$
{ \frac { \sum _ { i = 1 } ^ { N } m { \bf p } _ { i } } { \sum _ { i = 1 } ^ { N } m } } = { \frac { m \sum _ { i = 1 } ^ { N } { \bf p } _ { i } } { N m } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } { \bf p } _ { i } = { \bar { \bf p } } .\tag{4}
$$

So the centroid of the sampled population is exactly its equal-weight center of mass.

Where does the centroid lie relative to the model? Taking expectations in Equation 3 and using $\mathbb { E } [ \varepsilon _ { i } ] = 0$

$$
\mathbb { E } [ \bar { \mathbf { p } } ] = \pmb { \theta } + \frac { \sigma } { N } \sum _ { i = 1 } ^ { N } \mathbb { E } [ \varepsilon _ { i } ] = \pmb { \theta } ,\tag{5}
$$

so, averaged over repeated populations, the centroid lies exactly at the current model. For one finite population it is displaced from θ by $\sigma { \bar { \varepsilon } } .$ . Since the perturbations are independent with zero mean, the cross terms $\mathbb { E } [ \varepsilon _ { i } ^ { \top } \varepsilon _ { j } ]$ vanish for $i \neq j ,$ , and with $\mathbb { E } \Vert \varepsilon _ { i } \Vert ^ { 2 } = d ,$

$$
\mathbb { E } \| \bar { \mathbf { p } } - \pmb { \theta } \| ^ { 2 } = \frac { \sigma ^ { 2 } } { N ^ { 2 } } \sum _ { i = 1 } ^ { N } \mathbb { E } \| \pmb { \varepsilon } _ { i } \| ^ { 2 } = \frac { \sigma ^ { 2 } d } { N }\tag{6}
$$

As the population grows, the centroid converges to the model,

$$
\bar { \varepsilon } \longrightarrow 0 , \qquad \bar { \bf p } \longrightarrow \theta \qquad \mathrm { a s } \ N  \infty ,\tag{7}
$$

and in this limit the current model is itself the equalweight center of mass of the population. If the perturbations are drawn in mirrored pairs $\pm \varepsilon _ { i }$ (Salimans et al., 2017), then $\bar { \varepsilon } = 0$ and $\bar { \bf p } = \pmb \theta$ exactly, even for a finite population.

## 4 ES Moves the Model Along a Center-of-Mass Shift

In the previous section we ignored the rewards and gave every perturbed model the same mass. This is the state of the population “particles” before they are evaluated. Once each perturbed model $\mathbf { p } _ { i }$ is scored, the center-of-mass interpretation sets its mass equal to its reward $\boldsymbol { r } _ { i } ,$ and the question becomes where the population balances now. We derive this in exact form next.

We start with the ES update in eq. (2) for the mean-subtracted rule $w _ { i } = r _ { i } - \bar { r }$ . This rule, often with a further rescaling, is the most common choice in practice (Wierstra et al., 2014; Mania et al., 2018; Qiu et al., 2026). We first rewrite the update in terms of the positions $\mathbf { p } _ { i }$ instead of the perturbations $\varepsilon _ { i }$ . Using $\pmb { \varepsilon } _ { i } = ( \mathbf { p } _ { i } - \pmb { \theta } ) / \sigma$ from eq. (1), the summation becomes

$$
\begin{array} { l } { { \displaystyle \sum _ { i } w _ { i } \varepsilon _ { i } = \sum _ { i } ( r _ { i } - \overline { { r } } ) \varepsilon _ { i } = \frac { 1 } { \sigma } \sum _ { i } ( r _ { i } - \overline { { r } } ) ( { \bf p } _ { i } - \pmb \theta ) } } \\ { ~ } \\ { { \displaystyle ~ = \frac { 1 } { \sigma } \left[ \sum _ { i } ( r _ { i } - \overline { { r } } ) { \bf p } _ { i } - \pmb \theta \sum _ { i } ( r _ { i } - \overline { { r } } ) \right] } } \\ { { \displaystyle ~ = \frac { 1 } { \sigma } \sum _ { i } ( r _ { i } - \overline { { r } } ) { \bf p } _ { i } . } } \end{array}
$$

We see that the terms containing θ cancel because $\begin{array} { r } { \sum _ { i } ( r _ { i } - { \overline { { r } } } ) ~ = ~ 0 . } \end{array}$ For the same reason, any fixed point can be subtracted from every p<sub>i</sub> in eq. (8) without changing the summation. Substituting eq. (8) into eq. (2) and subtracting the sampled centroid p¯ from eq. (3), we get

$$
\begin{array} { l } { \displaystyle \Delta \theta = \frac { \alpha } { N \sigma } et { } { ' } \sum _ { i } ( r _ { i } - \bar { r } ) { \bf p } _ { i } } \\ { = \frac { \alpha } { N \sigma } et { } { ' } \sum _ { i } ( r _ { i } - \bar { r } ) ( { \bf p } _ { i } - \bar { \bf p } ) } \\ { = \frac { \alpha } { N \sigma } \left[ \sum _ { i } r _ { i } ( { \bf p } _ { i } - \bar { \bf p } ) - \bar { r } \sum _ { i } ( { \bf p } _ { i } - \bar { \bf p } ) \right] } \\ { = \frac { \alpha } { N \sigma } et { } { ' } \sum _ { i } r _ { i } ( { \bf p } _ { i } - \bar { \bf p } ) } \end{array}\tag{}
$$

because $\begin{array} { r } { \sum _ { i } ( \mathbf { p } _ { i } - \bar { \mathbf { p } } ) = 0 } \end{array}$ . Let the total reward given to all perturbations be $\begin{array} { r } { R = \sum _ { i = 1 } ^ { N } r _ { i } = N \overline { { r } } } \end{array}$ Continuing with the physical analogy, R is the total mass of the system, and when $R \neq 0 ^ { 1 }$ , the rewardweighted COM of the sampled population is

$$
\mathbf { p } _ { \mathrm { C O M } } = \frac { 1 } { R } \sum _ { i = 1 } ^ { N } r _ { i } \mathbf { p } _ { i } .\tag{10}
$$

We now have all the tools to rewrite the ES update in terms of the two centers of mass:

$$
\begin{array} { l } { { \displaystyle \Delta \theta = \frac { \alpha } { N \sigma } \sum _ { i } r _ { i } ( { \bf p } _ { i } - { \bar { \bf p } } ) } } \\ { ~ } \\ { { \displaystyle ~ = \frac { \alpha } { N \sigma } \left[ \sum _ { i } r _ { i } { \bf p } _ { i } - { \bar { \bf p } } \sum _ { i } r _ { i } \right] } } \\ { { \displaystyle ~ = \frac { \alpha } { N \sigma } \left[ R { \bf p } _ { \mathrm { C O M } } - R { \bar { \bf p } } \right] } . } \end{array}\tag{11}
$$

Therefore, we get

$$
\left| \Delta \pmb \theta = \frac { \alpha R } { N \sigma } \left( \mathbf p _ { \mathrm { C O M } } - \bar { \mathbf p } \right) = \frac { \alpha \bar { r } } { \sigma } \left( \mathbf p _ { \mathrm { C O M } } - \bar { \mathbf p } \right) . \right|\tag{12}
$$

This is the central observation of the paper. For mean-subtracted ES, the update step has a simple geometric interpretation. The update vector is the displacement from the equal-weight balance point of the system, which is the centroid p¯, to its rewardweighted COM p<sub>COM</sub>, scaled by $\alpha \overline { { r } } / \sigma$ and applied at the current model θ (fig. 1(b,c)). Here, system refers to the sampled population of models around θ. In the infinite population regime, or at any N under antithetic sampling, the centroid coincides with the current location of the model (eq. (7)), which means that the update literally moves the model at θ towards the reward-weighted COM of the population. Put simply, the ES update can be understood with a physical intuition—treat reward as mass, and in the infinite population regime the model moves towards the COM of its own system of particles.

## 4.1 Centered and Raw Rewards Use Different Reference Points

The appearance of $\bar { \bf p }$ in the center-of-mass update is a direct consequence of subtracting the mean reward. To see this, consider the same ES update using the raw rewards directly:

$$
\begin{array} { l } { { \displaystyle \Delta \pmb { \theta } _ { \mathrm { r a w } } = \frac { \alpha } { N } \sum _ { i } r _ { i } \varepsilon _ { i } = \frac { \alpha } { N \sigma } \sum _ { i } r _ { i } \big ( \mathbf { p } _ { i } - \pmb { \theta } \big ) } } \\ { { \displaystyle ~ = \frac { \alpha } { N \sigma } \left[ \sum _ { i } r _ { i } \mathbf { p } _ { i } - \pmb { \theta } \sum _ { i } r _ { i } \right] = \frac { \alpha R } { N \sigma } \left( \mathbf { p } _ { \mathrm { C O M } } - \mathbf { \frac { \alpha } { \alpha } } \right) } } \end{array}\tag{θ}
$$

Compare this with the centered update from eq. (12), the two updates use the same rewardweighted COM, but they measure its displacement from different reference points:

<table><tr><td>raw rewards: θ  pcoM,</td><td></td><td></td></tr><tr><td>centered rewards:</td><td></td><td>¯ → pCoM.</td></tr></table>

(13)

Therefore, without centering, the update measures the displacement from the current model θ to the reward-weighted COM. After subtracting the mean reward, the reference point moves from θ to the equal-weight COM of the realized sample, p¯.

We can also write the raw-reward update in terms of the centered-reward update of eq. (12), which we now denote $\Delta \theta _ { \mathrm { c e n t e r e d } }$ as follows:

$$
\Delta \pmb { \theta } _ { \mathrm { r a w } } = \Delta \pmb { \theta } _ { \mathrm { c e n t e r e d } } + \frac { \alpha R } { N \sigma } \left( \bar { \bf p } - \pmb { \theta } \right) .\tag{14}
$$

This shows that the raw-reward update is the centered-reward update plus an additional term that depends on two things: (1) the total reward obtained by all the perturbations, and (2) the distance between the centroid of the sampled population and the current model, which exists only because the population is finite. Neither of these has anything to do with the differences between rewards, which is the only thing that carries information about which direction is better. This is what mean subtraction does geometrically. It removes the part of the update caused by the sampled centroid not sitting on the model, a part that is dominated by noise.<sup>2</sup> The extra term vanishes as the population grows, and mean subtraction removes it exactly at any N. Antithetic sampling removes it too, since then $\bar { \bf p } = \pmb \theta$ and the raw and centered updates coincide.

## 5 Conclusion

In this paper, we gave the ES update a geometrical interpretation grounded in physical intuition. We treat each perturbed model as a particle and its reward as the particle’s mass. Before scoring, all particles have the same mass and the center of mass of the system is simply the centroid of the sampled population. After scoring, the reward-weighted COM shifts away from the centroid, and we show that the mean-subtracted ES update is exactly this displacement, scaled by a factor that depends on the mean reward. This holds for any finite population. In the infinite population regime, or under antithetic sampling, the centroid coincides with the model and the interpretation becomes even simpler. From its current location, the model moves towards the reward-weighted COM of its own system of particles. The same picture also explains the role of mean subtraction.

## Limitations

This paper presents an interpretation of the ES update and not a new method. The identity in eq. (12) is exact and needs no experimental validation. The physical reading of rewards as masses needs nonnegative rewards with a positive total. This is always achievable by adding a constant to every reward, but the identity itself holds without it, so the physical picture is a convenience rather than a requirement. Our analysis of the term removed by mean subtraction assumes independent Gaussian perturbations with a fixed σ, and does not cover adaptive step sizes or covariance updates such as those in CMA-ES. Finally, the claim that this term is dominated by noise is made for d ≫ N, which is the regime of LLM fine-tuning, and may not hold for small problems.

## References

Immanuel Abdi, Akshat Gupta, Micah Mok, Alex Lu, Nicholas Lee, and Gopala Anumanchipalli. 2026. Evolutionary strategies at scale lead to catastrophic forgetting. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 194–204, San Diego, California, United States. Association for Computational Linguistics.

Hans-Georg Beyer and Hans-Paul Schwefel. 2002. Evolution strategies—a comprehensive introduction. Natural Computing, 1(1):3–52.

Patryk Chrabaszcz, Ilya Loshchilov, and Frank Hutter. 2018. Back to basics: Benchmarking canonical evolution strategies for playing atari. arXiv preprint arXiv:1802.08842.

Nikolaus Hansen. 2016. The CMA evolution strategy: A tutorial. ArXiv:1604.00772.

Horia Mania, Aurelia Guy, and Benjamin Recht. 2018. Simple random search provides a competitive approach to reinforcement learning. arXiv preprint arXiv:1803.07055.

Yurii Nesterov and Vladimir Spokoiny. 2017. Random gradient-free minimization of convex functions. Foundations of Computational Mathematics, 17(2):527–566.

George R. Price. 1970. Selection and covariance. Nature, 227(5257):520–521.

Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Yinggan Xu, Roberto Dailey, Elliot Meyerson, Babak Hodjat, and Risto Miikkulainen. 2026. Evolution strategies at scale: LLM fine-tuning beyond reinforcement learning. In Proceedings of the 43rd International Conference on Machine Learning. ArXiv:2509.24372.

Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. 2017. Evolution strategies as a scalable alternative to reinforcement learning. ArXiv:1703.03864.

Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jan Peters, and Jürgen Schmidhuber. 2014. Natural evolution strategies. Journal of Machine Learning Research, 15(27):949–980.

A Appendix

<table><tr><td>Weighting rule</td><td>Coefficient  $w _ { i }$ </td><td>Used in</td></tr><tr><td>Mean-subtracted reward</td><td> $r _ { i } - \bar { r }$ </td><td>NES fitness baseline (Wierstra et al., 2014)</td></tr><tr><td>Standardized reward</td><td> $( r _ { i } - \overline { { r } } ) / s _ { r }$ </td><td>ARS (Mania et al., 2018); LLM fine-tuning (Qiu et al., 2026)</td></tr><tr><td>Centered rank transform</td><td> $q _ { i } - 1 \mathrm { ~  ~ { ~ 1 ~ } ~ }$   $\overline { { N - 1 } } ^ { - } \overline { { 2 } }$ </td><td>OpenAI-ES (Salimans et al., 2017; Chrabaszcz et al., 2018)</td></tr><tr><td>Rank utilities</td><td> $u _ { q _ { i } } - 1 / N$ </td><td>NES (Wierstra et al., 2014)</td></tr></table>

Table 1: Common ES weighting rules. Under the conventions stated in the text, each rule produces coefficients that sum to zero. The centered-rank formula is the one in the released OpenAI-ES code; the paper itself specifies only “a rank transformation” (Salimans et al., 2017). CMA-ES uses the positive, truncated analogue of the rank utilities, $w _ { i } \propto \log ( \mu + \textstyle \frac 1 2 ) - \log i ,$ , which sums to one rather than zero (Hansen, 2016).