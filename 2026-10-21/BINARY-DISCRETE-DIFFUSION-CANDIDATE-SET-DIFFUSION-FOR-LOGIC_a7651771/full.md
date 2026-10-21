# BINARY DISCRETE DIFFUSION: CANDIDATE-SET DIFFUSION FOR LOGIC AND PLANNING TASKS

Rodrigo Casado Noguerales<sup>1,2,∗</sup> Dimitri von Rutte¨ <sup>1</sup> Bernhard Scholkopf¨ <sup>1,2,3</sup> Thomas Hofmann<sup>1</sup> Aran Raoufi<sup>1</sup>

<sup>1</sup>ETH Zurich <sup>2</sup>Max Planck Institute for Intelligent Systems <sup>3</sup>ELLIS Institute Tubingen ¨

## ABSTRACT

Discrete diffusion models iteratively denoise a noisy sequence of categorical data such as text, enabling any-order generation with self-correction and in-filling capabilities. This makes them promising candidates for reasoning, planning, and constrained-generation tasks like coding and math. Yet masked and uniform diffusion erase a token’s identity in a single corruption event, so their noisy states cannot explicitly represent partially resolved uncertainty in which several alternatives remain plausible. We close this gap with Binary Discrete Diffusion, a family of continuous-time discrete-diffusion processes whose noisy state is a set of candidate tokens per position, represented as a binary membership vector. Binary diffusion changes candidate sets through additions and removals, allowing information to be lost and recovered gradually while the diffusion state itself represents partial uncertainty. On Sudoku-Extreme, Boxoban, and graph coloring, the symmetric formulation of binary diffusion consistently outperforms strong, welltuned masked and uniform baselines under matched architectures and controlled training, sampling, and tuning budgets, with its largest gains on the hardest instances, while remaining competitive on language modeling (Text8, TinyStories, OpenWebText). These results support candidate sets as an effective intermediate representation for iterative reasoning and show that the corruption kernel can materially change the computation learned by the reverse model.

![](images/0180914194d80b5369a4a814bfe0e36ef08389c09bbad32917f85f3981888724.jpg)  
Figure 1: Denoising process of each kernel over the candidate sets of a three-token vocabulary {A,B,C}, with its allowed moves faded. Each panel highlights one possible denoising path to A.

## 1 INTRODUCTION

Discrete diffusion models generate by iteratively refining a corrupted sequence. Unlike autoregressive generation, decisions need not follow a fixed left-to-right order and can be revised later, while many positions can be updated in parallel (Schiff et al., 2025; Sahoo et al., 2024; Gong et al., 2025). These properties make discrete diffusion attractive for reasoning, planning, and constrained generation (Ye et al., 2025; Wewer et al., 2025). Yet sample quality at finite inference budgets remains challenging, motivating work on ordering, remasking, planned denoising, and search (Kim et al., 2025; Wang et al., 2025b; Liu et al., 2025; Bai et al., 2026). We instead focus on a complementary axis: the corruption process itself. How much can be gained by changing the corruption process, and in particular the intermediate representation on which denoising operates? Logic and planning are useful testbeds: they impose global constraints, admit exact solve-rate evaluation, and scale cleanly in difficulty at manageable computational cost, while sharing structure with coding, mathematics, and biological design (Sahoo et al., 2026; Avdeyev et al., 2023; Stark et al., 2024). They also expose¨ a mismatch in the dominant masked and uniform kernels (Sahoo et al., 2024; Shi et al., 2024; Schiff et al., 2025; Sahoo et al., 2025): a position cannot naturally express that several alternatives remain plausible. A model solving Sudoku, for example, might narrow a cell from {2, 7, 9} to {7, 9}, later reopen 2 as neighboring decisions are revised, and eventually settle on 7.

We propose to make such candidate sets themselves the diffusion states. A clean token is a singleton, while during reverse generation intermediate states can carry several live hypotheses that may be eliminated or restored. More broadly, the corruption kernel does more than specify how examples are noised: it determines what information survives at intermediate times and how the model’s prediction is translated into reverse moves and their rates. Identical architectures can therefore induce qualitatively different iterative computations under different kernels. We instantiate this idea with Binary Discrete Diffusion, representing candidate sets by binary membership vectors in $\{ 0 , 1 \} ^ { V }$ (Figure 1). In binary symmetric diffusion, membership bits can switch both on and off, allowing hypotheses to be eliminated and later restored. In binary absorbing diffusion, corruption only adds candidates, so reverse generation progressively prunes the set toward a singleton.

Although each position has a nominal state space of size $2 ^ { V }$ , every state has only V single-bit neigh bors, and all neighboring reverse rates can be computed in $O ( V )$ time from the model’s usual V-way prediction, without enumerating subsets. The model learns these dynamics from clean solutions without candidate labels, task-specific deduction rules, or symbolic search. The richer state also enables candidate-aware sampling: we derive exact ancestral updates through a shared clean-token proposal, adapt re-corrupt sampling, and, building on Ψ-sampling and confidence-guided remasking (Deschenaux et al., 2026; Wang et al., 2025b), develop revision rules that choose both which positions to reconsider and which hypotheses to reopen.

We evaluate Binary Discrete Diffusion on Sudoku-Extreme (Wang et al., 2025a), Boxoban (Guez et al., 2018), and graph coloring under controlled architectures, training budgets, and sampler tuning, using direct single-attempt solutions without verifier-guided selection. Across all three tasks, a consistent pattern emerges: binary symmetric diffusion outperforms uniform diffusion, and both outperform masked and binary absorbing diffusion. The advantage of binary symmetric over uniform grows on harder regimes, including low-clue Sudoku, long-horizon Boxoban, and graph coloring far beyond the training size. On Text8, TinyStories, and OpenWebText (Mahoney, 2011; Eldan & Li, 2023; Gokaslan & Cohen, 2019), binary diffusion remains competitive with established kernels. These results support candidate sets as an effective intermediate representation for iterative reasoning and show that the corruption process can materially change the computation learned by the reverse model.

## 2 DISCRETE DIFFUSION MODELING

We consider a data distribution $p _ { \mathrm { d a t a } }$ on sequences of length $L$ over a finite vocabulary $[ V ] =$ $\{ 1 , \ldots , V \}$ . Starting from $Z _ { 0 } \sim p _ { \mathrm { d a t a } } .$ , we corrupt its positions independently with a continuoustime Markov chain (CTMC) $\hat { Z _ { t } } = \hat { ( Z _ { t } ^ { 1 } , \ldots , Z _ { t } ^ { L } ) }$ , whose distribution approaches an easily sampled noise distribution as $t \uparrow 1$ . We write $q _ { t \mid s } ( y \mid z )$ for the one-position transition distribution or kernel between times $s < t$ and states $z , y$ . For masked and uniform diffusion we use (Austin et al., 2021)

$$
q _ { t \mid 0 } ( y \mid c ) = { \left\{ \begin{array} { l l } { ( 1 - t ) \mathbf { 1 } \{ y = c \} + t \mathbf { 1 } \{ y = \mathfrak { m } \} , } & { { \mathrm { m a s k e d } } , } \\ { ( 1 - t ) \mathbf { 1 } \{ y = c \} + t / V , } & { { \mathrm { u n i f o r m } } . } \end{array} \right. }\tag{1}
$$

Thus $Z _ { t }$ can be sampled directly at any time t, without simulating the intervening jumps.

We write $Q _ { t } ( z , y )$ for the forward jump rate from z to y, so that $q _ { t + h | t } ( y \mid z ) = \mathbf { 1 } \{ y = z \} +$ $h Q _ { t } ( z , y ) + \upsilon ( h )$ as $h \downarrow 0$ . The reverse rates depend on $p _ { \mathrm { d a t a } } .$ , but conditioning on the clean sample $Z _ { 0 }$ makes them computable from the forward process (Campbell et al., 2022). For a change at position i from noisy token $Z _ { t } ^ { i }$ to another $y \neq Z _ { t } ^ { \bar { i } }$ , the clean-conditioned reverse rates are given by

$$
a _ { i } ( Z _ { 0 } , Z _ { t } , y ; t ) \stackrel { \mathrm { d e f } } { = } Q _ { t } ( y , Z _ { t } ^ { i } ) \frac { q _ { t | 0 } ( y \mid Z _ { 0 } ^ { i } ) } { q _ { t | 0 } ( Z _ { t } ^ { i } \mid Z _ { 0 } ^ { i } ) } .\tag{2}
$$

The true reverse rates are obtained by averaging these clean-conditioned rates over $q ( Z _ { 0 } \mid Z _ { t } )$ Let $a _ { i } ^ { \theta } ( Z _ { t } , y ; t )$ denote learned reverse rates that depend on the noisy sequence $Z _ { t }$ and time t but not on $Z _ { 0 }$ . We train them against the clean-conditioned targets by minimizing the negative of the continuous-time evidence lower bound (ELBO) (Campbell et al., 2022; Casado Noguerales et al., 2026):

$$
- \operatorname { E L B O } ( \theta ) = \int _ { 0 } ^ { 1 } \mathbb { E } _ { Z _ { 0 } , Z _ { t } } [ \mathcal { I } _ { t } ( \theta ; Z _ { 0 } , Z _ { t } ) ] d t ,
$$

$$
\mathcal { I } _ { t } ( \theta ; Z _ { 0 } , Z _ { t } ) = \sum _ { i = 1 } ^ { L } \sum _ { y \neq Z _ { t } ^ { i } } \Phi ( a _ { i } , a _ { i } ^ { \theta } ) , \qquad \Phi ( a , b ) = a \log \frac { a } { b } - a + b .\tag{3}
$$

This upper-bounds the expected negative log-likelihood. More sharply, its excess above the dataentropy floor is exactly the path-space KL between the exact reverse process and the learned reverse process (Casado Noguerales et al., 2026). In practice, the jump rates can become singular near $t = 0$ or $t = 1$ , so we train on an interior window $0 < t _ { 1 } < t _ { 2 } < 1$ . The corresponding finite-window objective includes reconstruction and terminal-prior boundary terms, given in App. C.1.

Training. We parameterize the model reverse rates through a network that takes $( Z _ { t } , t )$ as input and outputs, at each position i, a strictly positive distribution $\lceil \boldsymbol { \mu } _ { i } ^ { \theta } ( \boldsymbol { c } \mid Z _ { t } ; t )$ over clean tokens $c \in \bar { | V | }$ This induces a noisy-state distribution $\mathbf { \dot { \rho } } _ { m _ { i } ^ { \theta } }$ , which determines the model reverse rates for $y \neq Z _ { t } ^ { i } .$

$$
a _ { i } ^ { \theta } ( Z _ { t } , y ; t ) \stackrel { \mathrm { d e f } } { = } Q _ { t } ( y , Z _ { t } ^ { i } ) \frac { m _ { i } ^ { \theta } ( y \mid Z _ { t } ; t ) } { m _ { i } ^ { \theta } ( Z _ { t } ^ { i } \mid Z _ { t } ; t ) } , \qquad m _ { i } ^ { \theta } ( y \mid Z _ { t } ; t ) \stackrel { \mathrm { d e f } } { = } \sum _ { c = 1 } ^ { V } \mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) q _ { t \mid 0 } ( y \mid c ) .\tag{4}
$$

Thus one $V \mathrm { - w a y }$ network output determines all allowed reverse moves at a position. Reweighting $\mu _ { i } ^ { \theta }$ by the kernel $q _ { t | 0 } ( Z _ { t } ^ { i } \mid c )$ gives a new distribution which we denote by

$$
\pi _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) \stackrel { \mathrm { d e f } } { = } \frac { \mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) q _ { t \mid 0 } ( Z _ { t } ^ { i } \mid c ) } { m _ { i } ^ { \theta } ( Z _ { t } ^ { i } \mid Z _ { t } ; t ) } .\tag{5}
$$

In terms of this distribution, the model rate $a _ { i } ^ { \theta } ( Z _ { t } , y ; t )$ from (4) becomes an average of the cleanconditioned rate $a _ { i } ( Z _ { 0 } , Z _ { t } , y ; t )$ from (2), where tokens with $\pi _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) = 0$ contribute zero:

$$
a _ { i } ^ { \theta } ( Z _ { t } , y ; t ) = Q _ { t } ( y , Z _ { t } ^ { i } ) \sum _ { c = 1 } ^ { V } \pi _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) \frac { q _ { t \mid 0 } ( y \mid c ) } { q _ { t \mid 0 } ( Z _ { t } ^ { i } \mid c ) } = \mathbb { E } _ { Z _ { 0 } ^ { i } \sim \pi _ { i } ^ { \theta } ( \cdot \mid Z _ { t } ; t ) } \left[ a _ { i } ( Z _ { 0 } , Z _ { t } , y ; t ) \right] .\tag{6}
$$

Training then minimizes an empirical estimate of the path integral in (3) by sampling $( t , Z _ { 0 } , Z _ { t } )$

Remark 2.1. This is the cavity parameterization (Casado Noguerales et al., 2026). Since the true reverse rates take the averaged form (6) with the posterior $q ( Z _ { 0 } ^ { i } \ | \ Z _ { t } )$ in place of $\pi _ { i } ^ { \theta } .$ , the model rates equal the true rates, and thus minimize the negative ELBO, when $\pi _ { i } ^ { \theta } ( c \bar { \mid } Z _ { t } ; t ) = q ( Z _ { 0 } ^ { i } = c \mid Z _ { t } )$ This is attained by setting $\mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) = q ( Z _ { 0 } ^ { \breve { i } } = c \mid Z _ { t } ^ { - i } )$ , the distribution of the clean token given the rest of the sequence $\breve { Z } _ { t } ^ { - i }$ (that is, $Z _ { t }$ excluding $\dot { Z } _ { t } ^ { i } )$ , even though the network is evaluated on the full sequence $Z _ { t }$ . We call $\overline { { \pi _ { i } ^ { \theta } } }$ the denoiser and $\mu _ { i } ^ { \theta }$ the cavity predictions or distributions.

## 3 BINARY DISCRETE DIFFUSION

In binary diffusion, the noisy state at each position is a set ofcandidate tokens. We represent a clean token $c \in [ V ]$ by the singleton {c}, and corruption changes this set by adding or removing candidates, so that an intermediate state can retain several alternatives at a single position. Representing each set by its binary membership vector identifies the noisy state $Z _ { t } ^ { i }$ with a vertex of the binary cube $\{ 0 , 1 \check  \} ^ { V }$ , and we use the same symbol for a noisy set z and its membership vector, including for a clean singleton $\{ c \}$ and its one-hot vector. We consider two corruption processes on this space: binary symmetric diffusion, under which candidates can be both removed and restored, and binary absorbing diffusion, which only adds candidates. We focus primarily on the symmetric process and treat the absorbing process as a simpler monotone counterpart.

Although the bits evolve independently when conditioned on the clean sequence $Z _ { 0 } ,$ the true and learned reverse dynamics are generally correlated across positions and depend on the full noisy sequence. Crucially, though, the network still produces only the usual V-way distribution $\mu _ { i } ^ { \theta } ( c \mid \dot { Z } _ { t } ; t )$

over clean tokens at each position, and the construction of Section 2 converts this single prediction into all allowed reverse rates on the binary cube. This is what makes candidate-set diffusion tractable, as we explain in more detail in Section 3.3.

## 3.1 BINARY SYMMETRIC DIFFUSION

In binary symmetric diffusion, each membership bit retains its clean value with probability $1 - t$ and is otherwise resampled uniformly from {0, 1}. A resampled bit keeps its value half of the time, so a bit differs from its clean value with probability $t / 2$ , and given $Z _ { 0 }$ the forward process factorizes over positions and bits with the one-position kernel

$$
q _ { t | 0 } ^ { \mathrm { s y m } } ( z \mid c ) = \Big ( 1 - \frac { t } { 2 } \Big ) ^ { V - d _ { H } ( z , c ) } \Big ( \frac { t } { 2 } \Big ) ^ { d _ { H } ( z , c ) } ,\tag{7}
$$

where $d _ { H }$ is the Hamming distance. A state at time t thus contains the clean token c with probability $1 - t / 2$ and every other token independently with probability $t / 2$ , so it holds $( V - \dot { 1 } ) t / 2$ false candidates on average, and it approaches independent fair bits, a uniformly random subset of the vocabulary, as $t \uparrow 1$ . More generally, between times $0 \leq s < t < 1$ each bit is resampled with probability $( t - s ) / ( 1 - s )$ , which gives the transition kernel

$$
q _ { t | s } ^ { \mathrm { s y m } } ( y \mid z ) = ( 1 - \varepsilon _ { s , t } ) ^ { V - d _ { H } ( y , z ) } \varepsilon _ { s , t } ^ { d _ { H } ( y , z ) } , \qquad \varepsilon _ { s , t } \stackrel { \mathrm { d e f } } { = } \frac { t - s } { 2 ( 1 - s ) } .\tag{8}
$$

Forward rates. Each bit is resampled at rate $1 / ( 1 - t )$ and changes its value half of the time, so differentiating (8) gives the flip rate

$$
Q _ { t } ^ { \mathrm { s y m } } \big ( z , z ^ { ( u ) } \big ) = r _ { t } \stackrel { \mathrm { d e f } } { = } \frac 1 { 2 ( 1 - t ) } ,\tag{9}
$$

where $z ^ { ( u ) }$ is the state z with a single bit u flipped, and every other jump has rate zero. The forward process thus moves along the edges of the binary cube: from z, it can only reach the V Hamming neighbors $z ^ { ( 1 ) } , \dots , z ^ { ( V ) }$ of z in $\{ 0 , 1 \} ^ { V }$

Clean-conditioned reverse rates. The reverse process traverses the same edges, so each move flips one bit. We write $a _ { i , u } ^ { \mathrm { s y m } }$ for the rate of flipping bit u of $Z _ { t } ^ { i }$ . Since the forward kernel (7) factorizes over bits, all unchanged-bit factors cancel in the kernel ratio in $( 2 )$ . The remaining ratio is $\rho _ { t } = ( 1 - t / 2 ) / ( t / 2 ) = ( 2 \bar { ~ } t ) / t$ when the flip repairs a mismatch with the clean bit $( Z _ { 0 } ^ { i } ) _ { u } ^ { - }$ , and $\rho _ { t } ^ { - 1 }$ when it creates one. With the forward rate $r _ { t }$ introduced above, the clean-conditioned rate is therefore

$$
a _ { i , u } ^ { \mathrm { s y m } } ( Z _ { 0 } , Z _ { t } ; t ) = r _ { t } \big [ \rho _ { t } \mathbf { 1 } \{ ( Z _ { t } ^ { i } ) _ { u } \neq ( Z _ { 0 } ^ { i } ) _ { u } \} + \rho _ { t } ^ { - 1 } \mathbf { 1 } \{ ( Z _ { t } ^ { i } ) _ { u } = ( Z _ { 0 } ^ { i } ) _ { u } \} \big ] .\tag{10}
$$

A repairing flip is therefore favored over a corrupting flip by a factor $\rho _ { t } ^ { 2 } .$ . This preference strengthens as $t \downarrow$ 0 and vanishes as $t \uparrow 1$ , while both directions remain possible for $0 < t < 1$

Model reverse rates. Averaging over the denoiser as in (6) replaces each indicator in (10) by its probability, so that, with $\mathbb { P } _ { i } ^ { \theta }$ the probability under $Z _ { 0 } ^ { i } \sim \pi _ { i } ^ { \theta } ( \cdot \mid Z _ { t } ^ { \cdot } ; t )$ , the model reverse rates are

$$
a _ { i , u } ^ { \theta , \mathrm { s y m } } ( Z _ { t } ; t ) = r _ { t } \big [ \rho _ { t } \mathbb { P } _ { i } ^ { \theta } \{ ( Z _ { t } ^ { i } ) _ { u } \neq ( Z _ { 0 } ^ { i } ) _ { u } \} + \rho _ { t } ^ { - 1 } \mathbb { P } _ { i } ^ { \theta } \{ ( Z _ { t } ^ { i } ) _ { u } = ( Z _ { 0 } ^ { i } ) _ { u } \} \big ] .\tag{11}
$$

The bit of an active candidate u is wrong unless u is the clean token, and the bit of an inactive token is wrong only if it is the clean token, so removing and restoring a candidate have model reverse rates

$$
a _ { i , u } ^ { \theta , \mathrm { s y m } } ( Z _ { t } ; t ) = \left\{ { r _ { t } } \left[ \rho _ { t } \left( 1 - \pi _ { i } ^ { \theta } ( u \mid Z _ { t } ; t ) \right) + \rho _ { t } ^ { - 1 } \pi _ { i } ^ { \theta } ( u \mid Z _ { t } ; t ) \right] , \quad u \in Z _ { t } ^ { i } , \right.\tag{12}
$$

where the denoiser $\pi _ { i } ^ { \theta } ( \cdot \mid Z _ { t } ; t )$ is the cavity $\mu _ { i } ^ { \theta } ( \cdot \mid Z _ { t } ; t )$ multiplied by $\rho _ { t } ^ { 2 }$ on the active candidates of $Z _ { t } ^ { i }$ and renormalized.

Interpretation. Each rate lies between the slow rate $r _ { t } / \rho _ { t }$ and the fast rate $r _ { t } \rho _ { t }$ of (10). Candidates assigned higher probability by the denoiser are harder to remove when present and faster to restore when absent.

## 3.2 BINARY ABSORBING DIFFUSION

Binary absorbing diffusion only adds candidates: between times $0 \leq s < t < 1$ , each inactive bit activates independently with probability $( t - s ) / ( 1 - s )$ , and active bits never flip. Its kernel and forward rates are therefore

$$
q _ { t | s } ^ { \mathrm { a b s } } ( y \mid z ) = { \bf 1 } \{ z \subseteq y \} \Big ( \frac { 1 - t } { 1 - s } \Big ) ^ { V - | y | } \Big ( \frac { t - s } { 1 - s } \Big ) ^ { | y | - | z | } , \qquad Q _ { t } ^ { \mathrm { a b s } } \big ( z , z ^ { ( u ) } \big ) = \frac { { \bf 1 } \{ u \notin z \} } { 1 - t } ,\tag{13}
$$

and every other jump has rate zero. In particular, $q _ { t \mid 0 } ^ { \mathrm { a b s } } ( z ~ | ~ c ) ~ = ~ { \bf 1 } \{ c ~ \in ~ z \} ( 1 - t ) ^ { V - | z | } t ^ { | z | - 1 } $ the clean token c always remains a candidate and every other token becomes one independently with probability t, so the state holds $( V - 1 ) t$ false candidates on average and approaches the full vocabulary as $t \uparrow 1$

Since the forward process only adds candidates, the reverse process only removes them. Removing a false candidate gives the kernel ratio $( 1 - t ) / t$ in the clean-conditioned rate $( 2 )$ , while the clean token cannot be removed, so this rate and its average (6) over the denoiser $\pi _ { i } ^ { \theta }$ are

$$
a _ { i , u } ^ { \mathrm { a b s } } ( Z _ { 0 } , Z _ { t } ; t ) = \frac { \mathbf { 1 } \{ u \in Z _ { t } ^ { i } \setminus Z _ { 0 } ^ { i } \} } { t } , \qquad a _ { i , u } ^ { \theta , \mathrm { a b s } } ( Z _ { t } ; t ) = \frac { \mathbf { 1 } \{ u \in Z _ { t } ^ { i } \} } { t } \big ( 1 - \pi _ { i } ^ { \theta } ( u \mid Z _ { t } ; t ) \big ) ,\tag{14}
$$

where $\pi _ { i } ^ { \theta } ( \cdot \mid Z _ { t } ; t )$ is the renormalized restriction of $\mu _ { i } ^ { \theta } ( \cdot \mid Z _ { t } ; t )$ to the active candidates of $Z _ { t } ^ { i }$

Interpretation. A candidate receiving a larger share of the model probability within the current candidate set has a smaller removal rate. Denoising therefore prunes unlikely candidates while protecting plausible ones.

## 3.3 EFFICIENT COMPUTATION AND EMBEDDING WITH CANDIDATE SETS

Candidate-set diffusion has $2 ^ { V }$ possible states per position, which a priori require work exponential in V for computing the model rates and the ELBO loss. However, parameterizing in terms of a clean-data distribution allows us to work in $O ( V )$ per position, without enumerating the exponentially many possible candidate sets. For the rates, every reverse jump flips a single bit, so the reverse process can only move from a state z to its $V$ Hamming neighbors $\bar { \mathbf { \chi } } _ { z } ( u )$ . After one network evaluation, the denoiser $\pi _ { i } ^ { \theta }$ of a position follows from $\mu _ { i } ^ { \theta }$ by reweighting and normalization in $O ( V )$ time, and each of the V flip rates $a _ { i , u } ^ { \theta } ,$ (12) or (14), then takes constant work. For the same reason, the path loss in (3) sums only over these V moves per position $\begin{array} { r } { i , \mathcal { T } _ { t } = \sum _ { i } \sum _ { u = 1 } ^ { V } \Phi ( a _ { i , u } , a _ { i , u } ^ { \theta } ) } \end{array}$ . App. B derives all rates and losses in full detail.

Similarly, to avoid materializing an embedding table over the $2 ^ { V }$ states, we embed a candidate set by applying the input embedding matrix $E \in \breve { \mathbb { R } } ^ { V \times d }$ , whose row $E _ { c }$ embeds clean token $c \in [ V ]$ , to its $\ell _ { 2 }$ -normalized membership vector,

$$
e ( z ) = E ^ { \top } \frac { z } { \| z \| _ { 2 } } = \frac { 1 } { \sqrt { | z | } } \sum _ { c \in z } E _ { c } , \qquad e ( \emptyset ) = 0 .\tag{15}
$$

This costs $O ( V )$ and reduces to the ordinary token embedding $E _ { c }$ for a singleton $\{ c \}$ . Apart from this input layer, we therefore use the same architecture as for the masked and uniform baselines.

## 3.4 INFORMATION LOSS AND THE TIME SCHEDULE

In binary diffusion, the first false candidate doubles the number of candidates for the clean token, whereas each later one enlarges an already large set. Since false candidates appear at a rate proportional to V , the binary kernels thus erase the clean token much faster than masked and uniform diffusion at large vocabularies.

We quantify the retained information by the mutual information $I ( Z _ { 0 } ^ { i } ; Z _ { t } ^ { i } ) = H ( Z _ { 0 } ^ { i } ) - H ( Z _ { 0 } ^ { i } \mid Z _ { t } ^ { i } )$ the reduction in uncertainty about the clean token from observing its noisy state, where H denotes Shannon entropy. Since $\dot { I } ( Z _ { 0 } ^ { i } ; Z _ { t } ^ { i } )$ is intractable in general, we do so on a reference distribution of independent, uniformly distributed tokens $( \mathrm { A p p . ~ C . 4 } )$ . Masked and uniform diffusion then lose information roughly linearly in t, whereas the binary kernels lose it roughly logarithmically, so that a uniform sample of t places most training times where little information is left (Figure 2, left).

![](images/183226a52f9366e76920d61b8a86e8b8e273a0be0f2ccf9f57ced2f3ab7516dd.jpg)  
Figure 2: Fraction of the information retained for independent, uniformly distributed tokens with $V = 1 0 2 4$ , against the reparameterized time $\xi$ of the linear (left) and logarithmic (right) schedules.

This reference calculation motivates a change of clock (Casado Noguerales et al., 2026): a time reparameterization $\xi \mapsto t ,$ , with $\xi$ drawn uniformly from [0, 1], to control how fast information is destroyed. Reweighting the path integrand $\mathcal { T } _ { t }$ by $d t / d { \bar { \xi } }$ then leaves the ELBO objective (3) unchanged, so that only the allocation of training times changes. With a logarithmic map instead of the linear one,

$$
t _ { \mathrm { l i n } } ( \xi ) = t _ { 1 } + \left( t _ { 2 } - t _ { 1 } \right) \xi , \qquad t _ { \mathrm { l o g } } ( \xi ) = t _ { 1 } { \left( \frac { t _ { 2 } } { t _ { 1 } } \right) } ^ { \xi } ,\tag{16}
$$

the binary kernels lose their information approximately linearly in ξ (Figure 2, right). We also give ξ to the network as its time input rather than t. For $V \dot { \geq } 1 0 2 4$ , the logarithmic schedule substantially improved optimization. For the smaller vocabularies $( V \leq 9 )$ of our logic tasks, this choice made no difference and all kernels trained successfully with the linear schedule. We further refine all time schedules with an adaptive time sampler (see Apps. C.3 and C.5 for the details).

## 4 SAMPLING

Generation runs the learned reverse process on a grid $t _ { 2 } = \tau _ { N } > \cdot \cdot \cdot > \tau _ { 0 } = t _ { 1 }$ , starting from a prior distribution $p _ { t _ { 2 } }$ and ending with a final decoding step via the denoiser $\pi _ { i } ^ { \theta } ( \cdot \mid Z _ { t _ { 1 } } ; t _ { 1 } )$ . As priors we take the limiting forward distributions, a uniformly random subset of the vocabulary for binary symmetric and the full vocabulary, given by the all-ones vector $\mathbb { 1 } _ { V }$ , for binary absorbing:

$$
p _ { t _ { 2 } } ^ { \mathrm { s y m } } ( z ) \stackrel { \mathrm { d e f } } { = } 2 ^ { - V } , \qquad p _ { t _ { 2 } } ^ { \mathrm { a b s } } ( z ) \stackrel { \mathrm { d e f } } { = } 1 \{ z = 1 _ { V } \} .\tag{17}
$$

Ancestral sampling. Ancestral sampling from a time t to the next time $s < t$ of the grid follows the reverse process learned through the ELBO, so that as $t \downarrow s ,$ its jump rates approach the model rates (6). At each step, it updates the positions independently given $\bar { Z } _ { t }$ by averaging the bridge $q _ { s \mid t , 0 } ( y \mid z , c )$ , the distribution of $Z _ { s } ^ { i } = y$ given $Z _ { t } ^ { i } = \bar { z }$ and $Z _ { 0 } ^ { i } = c _ { : }$ , over the denoiser:

$$
p _ { s | t , i } ^ { \theta } ( y \mid Z _ { t } ) = \sum _ { c = 1 } ^ { V } \pi _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) q _ { s | t , 0 } ( y \mid Z _ { t } ^ { i } , c ) .\tag{18}
$$

As in Section 3.3, we avoid enumerating the $2 ^ { V }$ states: given the clean token, the forward process corrupts the bits independently, so the bridge factors over bits,

$$
q _ { s \mid t , 0 } ( y \mid z , c ) = \prod _ { u = 1 } ^ { V } q _ { s \mid t , 0 } { \bigl ( } y _ { u } \mid z _ { u } , \mathbf { 1 } \{ u = c \} { \bigr ) } .\tag{19}
$$

Thus, each bit’s factor accounts for every sequence of jumps between s and t, and a step can change many bits without simulating jumps or Poisson counts. The mixture (18) does not factor over bits, but we can still sample it exactly at a cost of $O ( V )$ per position without enumerating the $2 ^ { V }$ possible states: we draw a clean proposal $\widehat { Z } _ { 0 } ^ { i } \sim \pi _ { i } ^ { \theta } \dot { ( \cdot \mid Z _ { t } ; t ) }$ and then sample each bit independently from (19) with $c = \widehat { Z } _ { 0 } ^ { i } \left( \mathrm { A p p . } \ \hat { \mathbf { B } } . 3 \right)$ . Under the symmetric bridge, each bit favors its current value for s close to t and the proposal’s value as $s \downarrow 0$ , so removed candidates can return, whereas the absorbing bridge always retains the clean proposal and independently retains each other active candidate with probability $s / t$

Revision by re-corruption. Ancestral sampling conditions each step on the current state $Z _ { t } .$ which biases it toward what earlier steps have decided, including their mistakes. Re-corrupt sampling instead forgets the current state: it averages the kernel $q _ { s | 0 }$ rather than the bridge $q _ { s \mid t , 0 }$

$$
p _ { s | t , i } ^ { \theta , \mathrm { r e - c o r r } } ( y \mid Z _ { t } ) = \sum _ { c = 1 } ^ { V } \pi _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) q _ { s | 0 } ( y \mid c ) ,\tag{20}
$$

which we again sample in two steps by drawing a clean proposal $\widetilde { Z } _ { 0 } ^ { i } \sim \pi _ { i } ^ { \theta } ( \cdot \ : | \ : Z _ { t } ; t )$ and then corrupting it afresh to noise level s. With an exact denoiser, both updates preserve the per-position forward marginals, but re-corruption makes every position revisable at every step. Interestingly, re-corruption alone markedly improves solve rates on logic tasks (Table 10, App. F.4).

Confidence-guided Ψ-sampler. As in the Ψ-sampler of Deschenaux et al. (2026), we mix the ancestral and re-corrupt updates, but independently at each position i with weight $\kappa _ { i } \mathbf { : }$

$$
\widehat { Z } _ { 0 } ^ { i } , \widetilde { Z } _ { 0 } ^ { i } \sim \pi _ { i } ^ { \theta } ( \cdot \mid Z _ { t } ; t ) , \qquad Z _ { s } ^ { i } \sim ( 1 - \kappa _ { i } ) q _ { s \mid t , 0 } ( \cdot \mid Z _ { t } ^ { i } , \widehat { Z } _ { 0 } ^ { i } ) + \kappa _ { i } q _ { s \mid 0 } ( \cdot \mid \widetilde { Z } _ { 0 } ^ { i } ) ,\tag{21}
$$

where we draw the two proposals independently and where $\kappa _ { i }$ may depend on the re-corrupted proposal $\widetilde { Z } _ { 0 } ^ { i }$ . A constant $\kappa _ { i } = \eta$ interpolates from ancestral sampling at $\eta = 0$ to re-corrupt sampling at $\eta = 1$ and preserves the per-position marginals, since both updates do. For masked diffusion, this mixture is the remasking sampler ReMDM (Wang et al., 2025b). We extend it with two confidence mechanisms, position confidence with strengths γ and λ and candidate confidence with strength $\beta ,$ which recover the plain Ψ-sampler at zero (Algorithm 1; full details specified in App. D).

• Position confidence: where to reconsider. Following ReMDM-conf, we use the model’s confidence in its clean proposal $\hat { Z } _ { 0 } ^ { i }$ , read from the cavity or the denoiser, to concentrate revision on uncertain positions. We let $\kappa _ { i }$ decrease with confidence at strength γ, keeping its mean equal to η across positions. We also tilt the noise level $s _ { i }$ to which re-corrupted positions return with the same confidence at strength λ, keeping the mean noise level at $s .$

• Candidate confidence: which candidates to reconsider. Plain re-corruption reopens alternative candidates uniformly at random, as the forward kernel does. For binary diffusion, we instead reopen candidate u with probability proportional to $p _ { i , u } ^ { \beta } .$ , keeping the expected number of reopened candidates fixed, where $p _ { i , u }$ is the denoiser probability of u when it was last in the candidate set. Uniform diffusion analogously draws the new token $\boldsymbol { u } \ne Z _ { 0 } ^ { i }$ in proportion to $\pi _ { i } ^ { \theta } ( u \mid Z _ { t } ; t ) ^ { \beta }$ , while masked diffusion has no candidate to reopen. Plausible candidates are thus preferentially restored.

Algorithm 1 Confidence-guided Ψ-sampler (full details specified in App. D).   
Require: Cavity model, kernel, grid $t _ { 2 } = \tau _ { N } > \cdot \cdot \cdot > \tau _ { 0 } = t _ { 1 }$ , re-corrupt fraction η, strengths   
$\gamma , \lambda , \beta ,$ temperature $T$   
1: Draw $Z _ { t _ { 2 } } ^ { i } \sim p _ { t _ { 2 } }$ and set $p _ { i , u } = 1 / V$ for all $i , u .$   
2: for $j = \mathsf { \bar { N } } , \hdots , 1$ do   
3: Set $t ~ = ~ \tau _ { j } , s ~ = ~ \tau _ { j - 1 }$ , evaluate $\mu _ { i } ^ { \theta } ( \cdot ~ | ~ Z _ { t } ; t )$ , compute $\pi _ { i } ^ { \theta }$ by (5), and record $p _ { i , u } ~ =$   
$\pi _ { i } ^ { \theta } ( u \mid Z _ { t } ; t )$ for u $\in Z _ { t } ^ { i }$ (every u for uniform diffusion).   
4: Draw a proposal $\tilde { Z } _ { 0 } ^ { i } \propto \pi _ { i } ^ { \theta } \dot { ( \cdot \mid Z _ { t } ; t ) } ^ { 1 / T }$ and read its confidence con $\mathrm { f } _ { i }$ from $\mu _ { i } ^ { \theta }$ or $\pi _ { i } ^ { \theta }$   
5: Set $\kappa _ { i } \propto \exp ( - \gamma$ conf <sub>i</sub>) and $s _ { i }$ ∝ exp(−λ conf <sub>i</sub>), capped at one, with means η and s over   
positions (using equations (65) and (66), App. D).   
6: With probability $\kappa _ { i } .$ , re-corrupt $\widetilde { Z } _ { 0 } ^ { i }$ to $s _ { i }$ , reopening candidates u with probability ∝ $p _ { i , u } ^ { \beta }$ , and   
otherwise draw $Z _ { s } ^ { i }$ from the ancestral update (18) with its own proposal $\widehat { Z } _ { 0 } ^ { i }$   
7: end for   
8: Sample each position ∝ $\pi _ { i } ^ { \theta } ( \cdot \mid Z _ { t _ { 1 } } ; t _ { 1 } ) ^ { 1 / T }$ , only the masked ones for masked diffusion.

## 5 EXPERIMENTS

We focus on solve rate and sample quality and report negative ELBOs only for completeness: the negative ELBO measures the path-space KL to the exact reverse process (Casado Noguerales et al., 2026), but it need not track the quality of the final samples, especially under non-ancestral samplers, as uniform versus masked diffusion shows (Schiff et al., 2025; Sahoo et al., 2025).

## 5.1 LOGIC AND PLANNING TASKS

We evaluate on Sudoku-Extreme (Wang et al., 2025a), Boxoban (Guez et al., 2018; Garriga-Alonso et al., 2024), and four-coloring of random graphs, using matched 43M-parameter diffusion transformers and five training seeds per kernel and task. Besides the full test sets, we report hard subsets to de-saturate the benchmarks: low-clue Sudoku, long-solution Boxoban, and graphs ten times the training size. A sample counts as solved only if it satisfies every constraint, with one attempt per instance and no verifier. For the full and hard settings separately, we select each kernel’s sampler on validation at a reference budget of 256 network evaluations (NFE) for Sudoku and Boxoban and 64 for graph coloring, with the same tuning budget for every kernel and candidate confidence for all but masked diffusion. Parentheses below are standard errors over the five training seeds. Ful experimental details are given in App. F, and the test negative ELBOs in Table 6.

Table 1: Solve rate (%) on the full test sets and hard subsets at the reference NFE.
<table><tr><td></td><td colspan="2">Sudoku</td><td colspan="2">Boxoban</td><td colspan="2">Graph coloring</td></tr><tr><td>Kernel</td><td>All</td><td>Hard</td><td>All</td><td>Hard</td><td>All</td><td>Hard</td></tr><tr><td>Binary symmetric</td><td>96.27 (0.21)</td><td>92.66 (0.39)</td><td>94.29 (0.30)</td><td>84.65 (0.56)</td><td>99.35 (0.00)</td><td>64.13 (1.10)</td></tr><tr><td>Uniform</td><td>95.47 (0.21)</td><td>89.81 (0.47)</td><td>92.38 (0.64)</td><td>78.73 (1.20)</td><td>99.01 (0.02)</td><td>61.41 (0.34)</td></tr><tr><td>Binary absorbing</td><td>91.24 (0.11)</td><td>81.08 (0.26)</td><td>87.30 (0.81)</td><td>69.13 (1.66)</td><td>98.83 (0.02)</td><td>54.77 (0.39)</td></tr><tr><td>Masked</td><td>87.96 (0.16)</td><td>76.12 (0.10)</td><td>91.26 (0.31)</td><td>75.47 (0.69)</td><td>97.44 (0.04)</td><td>4.85 (0.33)</td></tr></table>

![](images/ec1606f9bbbe29969a2a0d4cc1342c32e8f0e75be7d2acd46bc8607d703356a7.jpg)  
Figure 3: Solve rate against inference budget (NFE, left) and training step (right).

Table 1 reports the solve rates at the reference NFE, while Figure 3 shows their dependence on inference budget and training progress. Binary symmetric diffusion is the strongest kernel on every task, followed by uniform diffusion, with masked and binary absorbing diffusion behind both. Its margin widens on the hard subsets, especially on long-horizon Boxoban and on large graphs. On the large graphs, masked diffusion nearly fails. Binary symmetric diffusion leads at almost all budgets and training steps (Figure 3), rather than depending on a single NFE or checkpoint. Table 10 ablates the sampling components, while Figure 5 in App. F.5 shows that the kernel comparison persists across difficulty bins and graph sizes. We show how each kernel solves a Sudoku board in App. G.

## 5.2 TEXT

We compare the four kernels using unconditional generation on three corpora of increasing sequence length $L \colon$ Text8 (Mahoney, 2011) at $L = 2 5 6$ , TinyStories (Eldan & Li, 2023) at $L = 5 1 2$ , and OpenWebText (Gokaslan & Cohen, 2019) at L = 1024 (App. F.6). We train every model on 131B tokens, on Text8 at the character level with $V = 2 7$ and on TinyStories and OpenWebText with byte-pair-encoding tokenizers of $V = 1 0 2 4$ , using as training objective the path term of the negative ELBO (3) (except for masked and binary absorbing diffusion on Text8, see App. C.2). We report MAUVE (Pillutla et al., 2021) against GPT-2 token unigram entropy, with GPT-2 Large (Radford et al., 2019) supplying the features. MAUVE compares the distribution of generated samples with held-out text, and we use it as the primary metric given known failure modes of generative perplexity (GenPPL) (Franca & Tong, 2026; Zheng et al., 2025b), which we report with the negative ELBOs in App. F.6.

Table 2 gives the test MAUVE of each kernel at the sampler setting selected on validation (App. F.4), while Figure 4 traces quality against diversity using sampling settings selected on validation at representative entropies and evaluated on the test split. No kernel leads on all three corpora: binary symmetric diffusion reaches the best MAUVE on Text8, uniform diffusion on TinyStories, and masked diffusion on OpenWebText, each by a small margin. The kernels peak at slightly different entropies, so Table 12 in App. F.6 also compares them at one common entropy per corpus, where binary symmetric diffusion reaches the best MAUVE on both Text8 and TinyStories.

Table 2: Test MAUVE of tuned samplers (standard error over two generation seeds).
<table><tr><td>Dataset</td><td>Binary symmetric</td><td>Uniform</td><td>Binary absorbing</td><td>Masked</td></tr><tr><td>Text8</td><td>0.913 (0.001)</td><td>0.896 (0.006)</td><td>0.821 (0.000)</td><td>0.854 (0.008)</td></tr><tr><td>TinyStories</td><td>0.882 (0.006)</td><td>0.904 (0.001)</td><td>0.585 (0.011)</td><td>0.850 (0.009)</td></tr><tr><td>OpenWebText</td><td>0.840 (0.001)</td><td>0.822 (0.004)</td><td>0.747 (0.004)</td><td>0.871 (0.013)</td></tr></table>

![](images/12ed2daef952e1a89ad39f3608e8008f72cf0baa3d0bec358cd81a3eb2e87b7e.jpg)  
Figure 4: MAUVE against token entropy with GPT-2 Large as the scorer. We select each point’s sampling configuration on validation at a representative entropy and evaluate it on the test split.

## 6 RELATED WORK

Discrete diffusion was first formulated on a discrete time grid, modeling the reverse transitions directly (Hoogeboom et al., 2021; Austin et al., 2021), and its continuous-time formulation instead learns to reverse a categorical jump process (Campbell et al., 2022; Sun et al., 2023). Since then, methods have differed mainly in how they parameterize the reverse rates (Lou et al., 2024; Ou et al., 2025) and in which paths the forward process allows (von Rutte et al., 2025; Shaul et al.,¨ 2025). Closer to our setting, simplex-valued methods represent categorical uncertainty in a continuous state (Floto et al., 2023; Avdeyev et al., 2023), while partial and semantic masking keep residual information about the clean token (Chao et al., 2025; Zhou et al., 2025). Binary diffusion instead changes the state space: each position follows a jump process on the binary cube, whose coordinates express candidate membership. Similar bit-flipping processes have been explored to model binary data, where the cube is the data space (Sohl-Dickstein et al., 2015; Pham et al., 2025). The closest predecessor is the ShortListing Model, which also grows candidate sets, in discrete time (Song et al., 2025). Under the right noise schedule, its corruption process provides a discrete-time formulation of binary absorbing diffusion (App. E). Our main contributions are the non-absorbing symmetric process, which allows candidates to be both removed and restored during denoising, and the empirical evaluation on logic and planning tasks.

Discrete diffusion has been applied to Sudoku and other structured problems before (Ye et al., 2025; Wewer et al., 2025), and follow-up work improves solving through adaptive token order, remasking, explicit planning, or test-time search (Kim et al., 2025; Wang et al., 2025b; Liu et al., 2025; Zhang et al., 2026). Sudoku is also a standard benchmark for small recurrent reasoners such as HRM and TRM (Wang et al., 2025a; Jolicoeur-Martineau, 2025), and Lattice Deduction Transformers likewise maintain per-variable candidate sets but train deduction on-policy with search and abstention heuristics (Davis et al., 2026). All of these change the sampler, the architecture, or the training signal. Our experiments are instead designed to isolate the effect of the corruption kernel and evaluate direct, verifier-free samples from each kernel’s tuned sampler.

## 7 CONCLUSION

We have introduced Binary Discrete Diffusion, a family of continuous-time discrete-diffusion processes whose noisy states are sets of candidate tokens, represented as binary membership vectors. Parameterizing the reverse process through clean-token predictions lets a single V-way output determine all reverse rates and sampling updates, avoiding any 2<sup>V</sup> enumeration in training and generation. We also introduced a confidence-guided self-correcting sampler that chooses which positions and candidates to reopen. Under controlled architectures, training budgets, and sampler tuning, binary symmetric diffusion outperforms uniform diffusion, and both outperform masked and binary absorbing diffusion on logic and planning tasks, with the largest gains on the hardest instances. Binary diffusion also remains competitive on language modeling. Our results thus show that candidate sets can be an effective intermediate representation for iterative reasoning, and that the corruption process can materially change the computation learned by the model.

Binary diffusion also opens a broader design space for candidate-set and bitwise corruption processes. Future kernels could exploit semantic or task-specific structure to introduce plausible candi dates rather than random ones. The bits could also encode tokens differently, for example through their binary indices or an error-correcting code, although each such code defines a different state space whose states are no longer candidate sets. Richer representations of candidate sets than the normalized embedding sum used here may also prove useful at larger scales. Scaling these ideas to larger models and datasets, and to domains such as code, mathematics, and biological sequence design, is a natural next step.

## AI USE STATEMENT

Large language models were used to aid and polish writing (editing and reviewing the manuscript), for retrieval and discovery (finding and summarizing related work), and for research ideation and execution (implementing, running, and analyzing experiments). All AI-assisted work was reviewed by the authors, who verified every technical claim and result and take full responsibility for the final content.

## REPRODUCIBILITY STATEMENT

App. F specifies the datasets with their splits and hard subsets, the architectures and training settings, every evaluation procedure, and the selection of learning rates, checkpoints, and sampler settings. App. C describes the training objectives and time sampling, Apps. A and B derive the rates, losses, and ancestral updates of every kernel, and App. D specifies the sampler.

## REFERENCES

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, volume 34, pp. 17981–17993, 2021. URL https://openreview.net/ forum?id=h7-XixPCAL.

Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, and Jian Zhou. Dirichlet diffusion score model for biological sequence generation. In Proceedings ofthe 40th International Conference on Machine Learning (ICML), volume 202 of Proceedings ofMachine Learning Research, pp. 1276– 1301. PMLR, 2023. URL https://proceedings.mlr.press/v202/avdeyev23a. html.

Jinbin Bai, Yixuan Li, Yuchen Zhu, Yi Xin, Qingyu Shi, Aosong Feng, Xiaohong Liu, Molei Tao, Jianru Xue, Xiangtai Li, and Ming-Hsuan Yang. Prism: Efficient test-time scaling via hierarchical search and self-verification for discrete diffusion language models. In Forty-third International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2602.01842.

Chris Cameron, Wangzheng Wang, Nikita Ivanov, Ashmita Bhattacharyya, Didier Chetelat, and´ Yingxue Zhang. One step forward and K steps back: Better reasoning with denoising recursion models, 2026. URL https://arxiv.org/abs/2604.18839.

Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://openreview. net/forum?id=DmT862YAieY.

Rodrigo Casado Noguerales, Bernhard Scholkopf, Thomas Hofmann, and Aran Raoufi. What does¨ a discrete diffusion model learn?, 2026. URL https://arxiv.org/abs/2607.05381.

Nuria Chandra, Yucen Li, Alan Amin, Alex Ali, Joshua Rollins, Sebastian W. Ober, Aniruddh Raghu, and Andrew Gordon Wilson. A unification of discrete, Gaussian, and simplicial diffusion. In International Conference on Learning Representations (ICLR), 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ 45d5823950cab9a88e10c6e8dbff351c-Abstract-Conference.html.

Chen-Hao Chao, Wei-Fang Sun, Hanwen Liang, Chun-Yi Lee, and Rahul G. Krishnan. Beyond masked and unmasked: Discrete diffusion models via partial masking. In Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://openreview.net/forum? id=nqpbbvEZwF.

Tao Chen, Chenhui Wang, and Hongming Shan. BerDiff: Conditional Bernoulli diffusion model for medical image segmentation. In Medical Image Computing and Computer Assisted Intervention (MICCAI), volume 14223 of Lecture Notes in Computer Science, pp. 491–501. Springer, 2023a. doi: 10.1007/978-3-031-43901-8 47. URL https://doi.org/10.1007/ 978-3-031-43901-8\_47.

Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. In International Conference on Learning Representations (ICLR), 2023b. URL https://openreview.net/forum?id=3itjR9QxFw.

Liam Davis, Leopold Haller, Alberto Alfarano, and Mark Santolucito. Lattice deduction transformers, 2026. URL https://arxiv.org/abs/2605.08605.

Justin Deschenaux, Caglar Gulcehre, and Subham Sekhar Sahoo. The diffusion duality, chapter II: Ψ-samplers and efficient curriculum. In International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/abs/2602.21185.

Ronen Eldan and Yuanzhi Li. TinyStories: How small can language models be and still speak coherent English?, 2023. URL https://arxiv.org/abs/2305.07759.

Jacob Fein-Ashley and Paria Rashidinejad. Solve the loop: Attractor models for language and reasoning, 2026. URL https://arxiv.org/abs/2605.12466.

Griffin Floto, Thorsteinn Jonsson, Mihai Nica, Scott Sanner, and Eric Zhengyu Zhu. Diffusion on the probability simplex, 2023. URL https://arxiv.org/abs/2309.02530.

Antonio Franca and Alexander Tong. Hacking generative perplexity: Why unconditional text evaluation needs distributional metrics, 2026. URL https://arxiv.org/abs/2606.08417.

Adria Garriga-Alonso, Mohammad Taufeeque, and Adam Gleave. Planning behavior in a recurrent\` neural network that plays Sokoban. In ICML 2024 Workshop on Mechanistic Interpretability, 2024. URL https://openreview.net/forum?id=T9sB3S2hok.

Aaron Gokaslan and Vanya Cohen. OpenWebText corpus. http://Skylion007.github. io/OpenWebTextCorpus, 2019.

Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, Hao Peng, and Lingpeng Kong. Scaling diffusion language models via adaptation from autoregressive models. In International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/forum?id=j1tSLYKwg8.

Alex Graves, Rupesh Kumar Srivastava, Timothy Atkinson, and Faustino Gomez. Bayesian flow networks, 2023. URL https://arxiv.org/abs/2308.07037.

Arthur Guez, Mehdi Mirza, Karol Gregor, Rishabh Kabra, Sebastien Racaniere, Theophane Weber, David Raposo, Adam Santoro, Laurent Orseau, Tom Eccles, Greg Wayne, David Silver, Timothy Lillicrap, and Victor Valdes. An investigation of model-free planning: Boxoban levels. https: //github.com/google-deepmind/boxoban-levels/, 2018.

Alec Helbling, Andrey Bryutkin, Mauro Martino, Duen Horng Chau, Nima Dehmamy, and Hendrik Strobelt. Flow reasoning models: Turning discrete flows into efficient recurrent reasoners, 2026. URL https://arxiv.org/abs/2606.29150.

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forre, and Max Welling. Argmax flows´ and multinomial diffusion: Learning categorical distributions. In Advances in Neural Information Processing Systems, volume 34, pp. 12454–12465, 2021. URL https://openreview.net/ forum?id=6nbpPqUCIi7.

Benhao Huang, Zhengyang Geng, and J. Zico Kolter. Equilibrium reasoners: Learning attractors enables scalable reasoning. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=lh95PnOlpM.

Alexia Jolicoeur-Martineau. Less is more: Recursive reasoning with tiny networks, 2025. URL https://arxiv.org/abs/2510.04871.

Pascal Jutras-Dube, Patrick Pynadath, Jeremy Lu, Yuan Gao, and Ruqi Zhang. Sticky jump dif-´ fusions: A unifying view of masked, continuous, and hybrid diffusion, 2026. URL https: //arxiv.org/abs/2607.10951.

Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham M. Kakade, and Sitan Chen. Train for the worst, plan for the best: Understanding token ordering in masked diffusions. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML), volume 267 of Proceedings ofMachine Learning Research, pp. 30749–30768. PMLR, 2025. URL https://openreview.net/ forum?id=DjJmre5IkP. Oral presentation.

Jaeyeon Kim, Seunggeun Kim, Taekyun Lee, David Z. Pan, Hyeji Kim, Sham Kakade, and Sitan Chen. Fine-tuning masked diffusion for provable self-correction. In Forty-third International Conference on Machine Learning (ICML), 2026. URL https://arxiv.org/abs/2510. 01384.

Sanghyun Lee, Chunsan Hong, Seungryong Kim, Jonghyun Lee, Jongho Park, and Dongmin Park. Looped diffusion language models, 2026. URL https://arxiv.org/abs/2605.26106.

Sulin Liu, Juno Nam, Andrew Campbell, Hannes Stark, Yilun Xu, Tommi Jaakkola,¨ and Rafael Gomez-Bombarelli. Think while you generate: Discrete diffusion with´ planned denoising. In International Conference on Learning Representations (ICLR), 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ cbf883a744952d4a40591271a58ab9d0-Abstract-Conference.html.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In Proceedings ofthe 41st International Conference on Machine Learning (ICML), volume 235 of Proceedings of Machine Learning Research, pp. 32819–32848. PMLR, 2024. URL https://openreview.net/forum?id=CNicRIVIPA.

Matt Mahoney. Large text compression benchmark. http://mattmahoney.net/dc/ textdata, 2011.

Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. In International Conference on Learning Representations (ICLR), 2025. URL https: //openreview.net/forum?id=sMyXP8Tanm.

Fred Zhangzhi Peng, Zachary Bezemek, Sawan Patel, Jarrid Rector-Brooks, Sherwood Yao, Avishek Joey Bose, Alexander Tong, and Pranam Chatterjee. Path planning for masked diffu sion model sampling, 2025. URL https://arxiv.org/abs/2502.03540.

Le-Tuyet-Nhi Pham, Dario Shariatian, Antonio Ocello, Giovanni Conforti, and Alain Oliviero Durmus. Discrete Markov probabilistic models: An improved discrete score-based framework with sharp convergence bounds under minimal assumptions. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML), volume 267 of Proceedings of Machine Learning Research, pp. 49195–49258. PMLR, 2025. URL https://proceedings.mlr.press/ v267/pham25a.html.

Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. MAUVE: Measuring the gap between neural text and human text using divergence frontiers. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, pp. 4816–4828, 2021. URL https://openreview.net/forum?id=Tqx7nJp7PR.

Patrick Pynadath, Jiaxin Shi, and Ruqi Zhang. CANDI: Hybrid discrete-continuous diffusion models, 2025. URL https://arxiv.org/abs/2510.22510.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Technical report, OpenAI, 2019. URL https://cdn.openai.com/better-language-models/language\_ models\_are\_unsupervised\_multitask\_learners.pdf.

Pierre H. Richemond, Sander Dieleman, and Arnaud Doucet. Categorical SDEs with simplex diffusion, 2022. URL https://arxiv.org/abs/2210.14784.

Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https: //openreview.net/forum?id=L4uaAR4ArM.

Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin T. Chiu, and Volodymyr Kuleshov. The diffusion duality. In Proceedings of the 42nd International Conference on Machine Learning (ICML), volume 267 of Proceedings of Machine Learning Research, pp. 52584–52619. PMLR, 2025. URL https://openreview.net/forum?id= 9P9Y8FOSOk.

Subham Sekhar Sahoo, Jean-Marie Lemercier, Zhihan Yang, Justin Deschenaux, Jingyu Liu, John Thickstun, and Ante Jukic. Scaling beyond masked diffusion language models. In International Conference on Machine Learning (ICML), 2026. URL https://openreview.net/ forum?id=xKJG1CNtQM.

Jinya Sakurai, Patrick Pynadath, Satoshi Hayakawa, Jaehong Yoon, Xulei Yang, Nancy F. Chen, and Xun Xu. Simplex relaxation for discrete diffusion, 2026. URL https://arxiv.org/abs/ 2608.10615.

Javier E. Santos, Zachary R. Fox, Nicholas Lubbers, and Yen Ting Lin. Blackout diffusion: Generative diffusion models in discrete-state spaces. In Proceedings of the 40th International Conference on Machine Learning (ICML), volume 202 of Proceedings of Machine Learning Research, pp. 9034–9059. PMLR, 2023. URL https://openreview.net/forum?id=lpc5vlfxp8.

Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dallatorre, Bernardo P. de Almeida, Alexander Rush, Thomas Pierrot, and Volodymyr Kuleshov. Simple guidance mechanisms for discrete diffusion models. In International Conference on

Learning Representations (ICLR), 2025. URL https://openreview.net/forum?id= i5MrJ6g5G1.

Amin Sghaier, Ali Parviz, and Alexia Jolicoeur-Martineau. Probabilistic tiny recursive model, 2026. URL https://arxiv.org/abs/2605.19943.

Neta Shaul, Itai Gat, Marton Havasi, Daniel Severo, Anuroop Sriram, Peter Holderrieth, Brian Karrer, Yaron Lipman, and Ricky T. Q. Chen. Flow matching with general discrete paths: A kineticoptimal perspective. In International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/forum?id=tcvMzR2NrP. Oral presentation.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. Simplified and generalized masked diffusion for discrete data. In Advances in Neural Information Processing Systems, volume 37, pp. 103131–103167, 2024. URL https://openreview.net/forum? id=xcqSOfHt4g.

Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning (ICML), volume 37 of Proceedings of Machine Learning Research, pp. 2256–2265. PMLR, 2015. URL https://proceedings.mlr.press/v37/ sohl-dickstein15.html.

Yuxuan Song, Zhe Zhang, Yu Pei, Jingjing Gong, Qiying Yu, Zheng Zhang, Mingxuan Wang, Hao Zhou, Jingjing Liu, and Wei-Ying Ma. ShortListing model: A streamlined simplex diffusion for discrete variable generation. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, 2025. doi: 10.52202/085713-1558. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 42c4ca570eec68981c20094de1cc760e-Abstract-Conference.html.

Hannes Stark, Bowen Jing, Chenyu Wang, Gabriele Corso, Bonnie Berger, Regina Barzilay, and ¨ Tommi Jaakkola. Dirichlet flow matching with applications to DNA sequence design. In Proceedings of the 41st International Conference on Machine Learning (ICML), volume 235 of Proceedings of Machine Learning Research, pp. 46495–46513. PMLR, 2024. URL https: //proceedings.mlr.press/v235/stark24b.html.

Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuous-time discrete diffusion models. In International Conference on Learning Representations (ICLR), 2023. URL https://openreview.net/forum?id=BYWWwSY2G5s.

Anej Svete and Ashish Sabharwal. On the reasoning abilities of masked diffusion language models. In International Conference on Learning Representations (ICLR), 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ e20a65c7308b7b94ed1178eebc45bf76-Abstract-Conference.html.

Huu Binh Ta, Michael Cardei, Alvaro Velasquez, and Ferdinando Fioretto. Search-augmented masked diffusion models for constrained generation, 2026. URL https://arxiv.org/ abs/2602.02727.

Alex M. Tseng, Nathaniel Diamant, Tommaso Biancalani, and Gabriele Scalia. GraphGUIDE: Interpretable and controllable conditional graph generation with discrete Bernoulli diffusion, 2023. URL https://arxiv.org/abs/2302.03790.

Dimitri von Rutte, Janis Fluri, Yuhui Ding, Antonio Orvieto, Bernhard Sch¨ olkopf, and Thomas¨ Hofmann. Generalized interpolating discrete diffusion. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML), volume 267 of Proceedings of Machine Learning Research, pp. 61810–61843. PMLR, 2025. URL https://openreview.net/forum?id= rvZv7sDPV9.

Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori. Hierarchical reasoning model, 2025a. URL https://arxiv.org/ abs/2506.21734.

Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling. In Advances in Neural Information Processing Systems (NeurIPS), 2025b. URL https://openreview.net/forum?id=IJryQAOy0p.

Yuxiang Wang, Yu Xiang, Baojian Zhou, Qifang Zhao, Keyue Jiang, Yanghua Xiao, and Xiaoxiao Xu. On the trainability of masked diffusion language models via blockwise locality, 2026. URL https://arxiv.org/abs/2604.24832.

Christopher Wewer, Bartlomiej Pogodzinski, Bernt Schiele, and Jan Eric Lenssen. Spatial reasoning with denoising models. In Proceedings of the 42nd International Conference on Machine Learning (ICML), volume 267 of Proceedings ofMachine Learning Research, pp. 66706–66725. PMLR, 2025. URL https://proceedings.mlr.press/v267/wewer25a.html.

Jiawei Xu, Minghui Liu, Aakriti Agrawal, Yifan Chen, and Furong Huang. Scheduling thoughts: Learning the order of thought in diffusion language models, 2026. URL https://arxiv. org/abs/2606.23567.

Jiacheng Ye, Jiahui Gao, Shansan Gong, Lin Zheng, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Beyond autoregression: Discrete diffusion for complex reasoning and planning. In International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/ forum?id=NRYgUzSPZz.

Tao Zhang, Jia-Shu Pan, Ruiqi Feng, and Tailin Wu. VFScale: Intrinsic reasoning through verifierfree test-time scalable diffusion model. In International Conference on Learning Representations (ICLR), 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/ hash/6188c02ccc16a7587716de2efd754033-Abstract-Conference.html.

Huangjie Zheng, Shansan Gong, Ruixiang Zhang, Tianrong Chen, Jiatao Gu, Mingyuan Zhou, Navdeep Jaitly, and Yizhe Zhang. Continuously augmented discrete diffusion model for categorical generative modeling, 2025a. URL https://arxiv.org/abs/2510.01329.

Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. In International Conference on Learning Representations (ICLR), 2025b. URL https://openreview.net/forum?id=CTC7CmirNr.

Cai Zhou, Chenyu Wang, Dinghuai Zhang, Shangyuan Tong, Yifei Wang, Stephen Bates, and Tommi Jaakkola. Next semantic scale prediction via hierarchical diffusion language models. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/ 085713-1385. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/hash/3b3d7aafe5f36995a8320059a858e4b9-Abstract-Conference. html.

## APPENDIX CONTENTS

A Masked and uniform diffusion 17   
A.1 Masked diffusion 17   
A.2 Uniform diffusion 17   
B Mathematical details of binary diffusion 18   
B.1 Binary symmetric diffusion . 18   
B.2 Binary absorbing diffusion 19   
B.3 Binary ancestral bridges 20   
C Training objectives and time sampling 21   
C.1 Finite-window negative ELBO and boundary terms 21   
C.2 Training objectives 21   
C.3 Time schedules 22   
C.4 Information retained under each kernel . 23   
C.5 Adaptive gradient-RMS importance sampling 23   
D Confidence-guided Ψ-sampler 24   
E Extended related work 25   
F Experimental details 26   
F.1 Optimization details . 26   
F.2 Learning-rate and checkpoint selection . 26   
F.3 Evaluation 28   
F.4 Sampler tuning procedure . 29   
F.5 Logic tasks 30   
F.6 Language modeling . 32   
G Sudoku solve trajectories 34   
H Generated samples 36   
H.1 Text8 36   
H.2 TinyStories 36   
H.3 OpenWebText . 38

## A MASKED AND UNIFORM DIFFUSION

We specialize the results of Campbell et al. (2022) and Casado Noguerales et al. (2026) using the notation of Sections 2–4. Local noisy states are written $z , y ,$ and the priors are the limiting distributions from which generation starts. We call the noisy-state distribution $m _ { i } ^ { \theta }$ in (4) the local model mass.

## A.1 MASKED DIFFUSION

For the masked kernel in (1), the off-diagonal forward rates are (Casado Noguerales et al., 2026)

$$
Q _ { t } ( z , y ) = { \frac { \mathbf { 1 } \{ z \in [ V ] , y = \mathfrak { m } \} } { 1 - t } } , \qquad y \neq z .\tag{22}
$$

For $0 \leq s < t < 1$ , the transition kernel is

$$
q _ { t \mid s } ( y \mid z ) = \frac { 1 - t } { 1 - s } { \bf 1 } \{ y = z \} + \frac { t - s } { 1 - s } { \bf 1 } \{ y = \mathtt { m } \} .\tag{23}
$$

The local model masses and denoiser are

$$
\begin{array} { r l } & { m _ { i } ^ { \theta } ( y \mid Z _ { t } ; t ) = \{ ( 1 - t ) \mu _ { i } ^ { \theta } ( y \mid Z _ { t } ; t ) , \quad y \in [ V ] ,  } \\ & { \quad y = \mathrm { \textmu } , } \\ & { \quad \pi _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) = \{ \mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) , \quad Z _ { t } ^ { i } = \mathrm { \textmu } ,  } \\ & { \quad  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}\tag{24}
$$

For $y \in \left[ V \right]$ with $y \neq Z _ { t } ^ { i }$ , the reverse rates are

$$
a _ { i } ( Z _ { 0 } , Z _ { t } , y ; t ) = { \frac { \mathbf { 1 } \{ Z _ { t } ^ { i } = \mathtt { m } , \ y = Z _ { 0 } ^ { i } \} } { t } } , \qquad a _ { i } ^ { \theta } ( Z _ { t } , y ; t ) = { \frac { \mathbf { 1 } \{ Z _ { t } ^ { i } = \mathtt { m } \} \mu _ { i } ^ { \theta } ( y \mid Z _ { t } ; t ) } { t } } .\tag{25}
$$

All other off-diagonal reverse rates vanish. The path loss reduces to masked cross-entropy:

$$
\mathcal { T } _ { t } ( \theta ; Z _ { 0 } , Z _ { t } ) = - \frac { 1 } { t } \sum _ { i } \mathbf { 1 } \{ Z _ { t } ^ { i } = \mathfrak { m } \} \log \mu _ { i } ^ { \theta } ( Z _ { 0 } ^ { i } \mid Z _ { t } ; t ) .\tag{26}
$$

Plugging $\mu _ { i } ^ { \theta }$ into the bridge (47) gives the ancestral update of Sahoo et al. (2024): visible tokens are copied, and a masked position unmasks to c with probability $( 1 - s / t ) \mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t )$

## A.2 UNIFORM DIFFUSION

For the uniform kernel in (1), the off-diagonal forward rates are (Casado Noguerales et al., 2026)

$$
Q _ { t } ( z , y ) = \frac { 1 } { V ( 1 - t ) } , \qquad y \neq z .\tag{27}
$$

For $0 \leq s < t < 1$ , the transition kernel is

$$
q _ { t | s } ( y \mid z ) = { \frac { 1 - t } { 1 - s } } { \bf 1 } \{ y = z \} + { \frac { t - s } { V ( 1 - s ) } } .\tag{28}
$$

The local model mass and induced denoiser are

$$
\begin{array} { r l } & { m _ { i } ^ { \theta } ( y \mid Z _ { t } ; t ) = \left( 1 - t \right) \mu _ { i } ^ { \theta } ( y \mid Z _ { t } ; t ) + t / V , } \\ & { ~ \pi _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) = \displaystyle \frac { \mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) \left[ \left( 1 - t \right) \mathbf { 1 } \{ Z _ { t } ^ { i } = c \} + t / V \right] } { \left( 1 - t \right) \mu _ { i } ^ { \theta } \left( Z _ { t } ^ { i } \mid Z _ { t } ; t \right) + t / V } . } \end{array}\tag{29}
$$

For $y \neq Z _ { t } ^ { i }$ , the clean-conditioned and model reverse rates are

$$
\begin{array} { r } { a _ { i } ( Z _ { 0 } , Z _ { t } , y ; t ) = \cfrac { 1 } { V ( 1 - t ) } \frac { ( 1 - t ) \mathbf { 1 } \{ y = Z _ { 0 } ^ { i } \} + t / V } { ( 1 - t ) \mathbf { 1 } \{ Z _ { t } ^ { i } = Z _ { 0 } ^ { i } \} + t / V } , } \\ { a _ { i } ^ { \theta } ( Z _ { t } , y ; t ) = \cfrac { 1 } { V ( 1 - t ) } \frac { ( 1 - t ) \mu _ { i } ^ { \theta } ( y \mid Z _ { t } ; t ) + t / V } { ( 1 - t ) \mu _ { i } ^ { \theta } ( Z _ { t } ^ { i } \mid Z _ { t } ; t ) + t / V } . } \end{array}\tag{30}
$$

The path loss is

$$
\mathcal { I } _ { t } ( \theta ; Z _ { 0 } , Z _ { t } ) = \frac { 1 } { V ( 1 - t ) } \sum _ { i } \sum _ { y \neq Z _ { t } ^ { i } } \Phi \left( \frac { q _ { t | 0 } ( y \mid Z _ { 0 } ^ { i } ) } { q _ { t | 0 } ( Z _ { t } ^ { i } \mid Z _ { 0 } ^ { i } ) } , \frac { m _ { i } ^ { \theta } ( y \mid Z _ { t } ; t ) } { m _ { i } ^ { \theta } ( Z _ { t } ^ { i } \mid Z _ { t } ; t ) } \right) .\tag{31}
$$

All masses in these quotients are positive on an interior time interval.

Plugging $\mu _ { i } ^ { \theta }$ into the bridge (47) gives the ancestral update of Schiff et al. (2025), which, unlike the masked update, can change a visible token.

Prior term. Corrupting a uniform clean token leaves its distribution uniform. The finite-time and limiting priors thus coincide:

$$
p _ { t _ { 2 } } ( y ) = 1 / V , \qquad y \in [ V ] .\tag{32}
$$

For the product sequence prior,

$$
\mathbb { E } _ { Z _ { 0 } } \mathrm { K L } \left( q _ { t _ { 2 } | 0 } ( \cdot \ | \ Z _ { 0 } ) \parallel p _ { t _ { 2 } } \right) = L \left[ ( 1 - t _ { 2 } + t _ { 2 } / V ) \log \left( V ( 1 - t _ { 2 } ) + t _ { 2 } \right) + \frac { ( V - 1 ) t _ { 2 } } { V } \log t _ { 2 } \right] .
$$

The uniform prior has full support at every finite cutoff.

(33)

## B MATHEMATICAL DETAILS OF BINARY DIFFUSION

We use the candidate-set notation and the bitwise loss from Section 3. The network probability mass of the candidates in a set z is

$$
\nu _ { i } ^ { \theta } ( z \mid Z _ { t } ; t ) \stackrel { \mathrm { d e f } } { = } \displaystyle \sum _ { c \in z } \mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) ,\tag{34}
$$

and $m _ { i } ^ { \theta }$ denotes the local model mass of (4). Superscripts sym and abs specify the kernel, and we omit them in derivations where it is fixed. The cavity $\mu _ { i } ^ { \theta }$ , the masses $m _ { i } ^ { \theta }$ , and the denoiser $\pi _ { i } ^ { \theta }$ always refer to that kernel.

## B.1 BINARY SYMMETRIC DIFFUSION

Proposition B.1 (Symmetric kernel and reverse rates). Let each bit flip at rate $1 / [ 2 ( 1 - t ) ]$ ]. For $0 \leq s < t < 1$ , the probability of a bit differing from its value at s is

$$
\varepsilon _ { s , t } = \frac { t - s } { 2 ( 1 - s ) } .\tag{35}
$$

The local kernel is

$$
q _ { t \mid s } ^ { \mathrm { s y m } } ( y \mid z ) = \left( 1 - \varepsilon _ { s , t } \right) ^ { V - d _ { H } ( y , z ) } \left( \varepsilon _ { s , t } \right) ^ { d _ { H } ( y , z ) } .\tag{36}
$$

Using the odds $\rho _ { t }$ from (10) and writing $z = Z _ { t } ^ { i }$ , the reverse rates are

$$
\begin{array} { r l } & { a _ { i , u } ^ { \mathrm { s y m } } ( Z _ { 0 } , Z _ { t } ; t ) = \displaystyle \frac { 1 } { 2 ( 1 - t ) } \left\{ \rho _ { t } , \quad \boldsymbol { z } _ { u } \neq ( Z _ { 0 } ^ { i } ) _ { u } , \right. } \\ & { ~ \quad a _ { i , u } ^ { \theta , \mathrm { s y m } } ( Z _ { t } ; t ) = \displaystyle \frac { \rho _ { t } ^ { 2 z _ { u } - 1 } } { 2 ( 1 - t ) } \frac { 1 + ( \rho _ { t } ^ { 2 } - 1 ) \nu _ { i } ^ { \theta } ( z ^ { ( u ) } \mid Z _ { t } ; t ) } { 1 + ( \rho _ { t } ^ { 2 } - 1 ) \nu _ { i } ^ { \theta } ( z \mid Z _ { t } ; t ) } . } \end{array}\tag{37}
$$

Proof. The bit-flip probability solves

$$
\partial _ { t } \varepsilon _ { s , t } = \frac { 1 - 2 \varepsilon _ { s , t } } { 2 ( 1 - t ) } , \qquad \varepsilon _ { s , s } = 0 .
$$

The stated flip probability satisfies this equation and initial condition, and multiplying the independent bit kernels proves (36). The clean-conditioned reverse rates follow from the change in Hamming distance: a flip toward the clean bit multiplies the clean-conditioned mass by $\rho _ { t } .$ , while a flip away multiplies it by $\rho _ { t } ^ { - 1 }$ . Multiplying these ratios by the forward flip rate gives the formula for $a _ { i , u }$

To compute the model mass, put

$$
C _ { t } ( z ) = ( 1 - t / 2 ) ^ { V - | z | } ( t / 2 ) ^ { | z | } .
$$

As a function of the clean token, the forward kernel is

$$
q _ { t | 0 } ( z \mid c ) = C _ { t } ( z ) \rho _ { t } ^ { 2 z _ { c } - 1 } , \qquad m _ { i } ^ { \theta } ( z \mid Z _ { t } ; t ) = \frac { C _ { t } ( z ) } { \rho _ { t } } [ 1 + ( \rho _ { t } ^ { 2 } - 1 ) \nu _ { i } ^ { \theta } ( z \mid Z _ { t } ; t ) ] .
$$

Toggling bit u gives

$$
\frac { C _ { t } ( z ^ { ( u ) } ) } { C _ { t } ( z ) } = \rho _ { t } ^ { 2 z _ { u } - 1 } , \qquad \nu _ { i } ^ { \theta } ( z ^ { ( u ) } \mid Z _ { t } ; t ) = \nu _ { i } ^ { \theta } ( z \mid Z _ { t } ; t ) + ( 1 - 2 z _ { u } ) \mu _ { i } ^ { \theta } ( u \mid Z _ { t } ; t ) .
$$

The quotient of the two local masses, multiplied by $1 / [ 2 ( 1 - t ) ]$ , gives the stated model rate. All local masses are positive for $0 < t < 1$ , so every bit flip has positive model rate. □

The denoiser used for decoding and sampling is

$$
\pi _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) = \frac { \mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) \rho _ { t } ^ { 2 ( Z _ { t } ^ { i } ) _ { c } } } { 1 + \left( \rho _ { t } ^ { 2 } - 1 \right) \nu _ { i } ^ { \theta } ( Z _ { t } ^ { i } \mid Z _ { t } ; t ) } .\tag{38}
$$

Substituting $\mu _ { i } ^ { \theta } ( u \mid Z _ { t } ; t ) = \rho _ { t } ^ { - 2 z _ { u } } \pi _ { u } [ 1 + ( \rho _ { t } ^ { 2 } - 1 ) \nu _ { i } ^ { \theta } ( z \mid Z _ { t } ; t ) ]$ , with $\pi _ { u } = \pi _ { i } ^ { \theta } ( u \mid Z _ { t } ; t )$ , turns the ratio of masses in (37) into $1 - ( 1 - \rho _ { t } ^ { - 2 } ) \pi _ { u }$ for u $\in ~ z$ and $1 + ( \rho _ { t } ^ { 2 } - 1 ) \bar { \pi _ { u } }$ for u $\notin \boldsymbol { z }$ . The model rate is therefore $[ \rho _ { t } ( 1 - \pi _ { u } ) + \rho _ { t } ^ { - \mathrm { I } } \pi _ { u } ] / [ 2 ( 1 - t ) ]$ for removing an active candidate and $[ \rho _ { t } \pi _ { u } + \rho _ { t } ^ { - 1 } ( 1 - \pi _ { u } ) ] / [ \ddot { 2 } ( 1 - t ) ]$ ] for adding an inactive one, which is (12).

Prior. The prior draws independent fair bits, $p _ { t _ { 2 } } ^ { \mathrm { s y m } } ( z ) = 2 ^ { - V }$ . For Bernoulli distributions with success probabilities $a , b \in ( 0 , 1 )$ , the KL divergence is

$$
D _ { \mathrm { B e r } } ( a \| b ) = a \log { \frac { a } { b } } + ( 1 - a ) \log { \frac { 1 - a } { 1 - b } } .\tag{39}
$$

Conditional on the clean token $c ,$ the bits of $Z _ { t _ { 2 } } ^ { i }$ are independent, with bit c active with probability $1 - t _ { 2 } / 2$ and every other bit with probability $t _ { 2 } ^ { - } / 2$ . The prior also factors over bits, so the KL is a sum of Bernoulli terms, which coincide because $D _ { \mathrm { B e r } } ( 1 - a \Vert 1 / 2 ) = D _ { \mathrm { B e r } } ( a \Vert 1 / 2 )$

$$
\mathrm { K L } \left( q _ { t _ { 2 } | 0 } ^ { \mathrm { s y m } } ( \cdot \mid c ) \Vert p _ { t _ { 2 } } ^ { \mathrm { s y m } } \right) = V D _ { \mathrm { B e r } } ( t _ { 2 } / 2 \Vert 1 / 2 ) .\tag{40}
$$

This does not depend on c and vanishes as $t _ { 2 } \uparrow 1$

## B.2 BINARY ABSORBING DIFFUSION

Proposition B.2 (Absorbing kernel and rates). Let each zero bit activate at rate $1 / ( 1 - t )$ and each active bit remain active. For $0 \leq s < t < 1$ and nonempty binary vectors $z , y ,$

$$
q _ { t | s } ^ { \mathrm { a b s } } ( y \mid z ) = { \bf 1 } \{ z \subseteq y \} \left( \frac { 1 - t } { 1 - s } \right) ^ { V - | y | } \left( \frac { t - s } { 1 - s } \right) ^ { | y | - | z | } .\tag{41}
$$

The expression is zero when the subset condition fails. For the clean singleton {c}, it reduces to

$$
q _ { t | 0 } ^ { \mathrm { a b s } } ( z | c ) = \mathbf { 1 } \{ c \in z \} ( 1 - t ) ^ { V - | z | } t ^ { | z | - 1 } .\tag{42}
$$

The clean-conditioned and model reverse rates are

$$
a _ { i , u } ^ { \mathrm { a b s } } ( Z _ { 0 } , Z _ { t } ; t ) = \frac { \mathbf { 1 } \{ u \in Z _ { t } ^ { i } \setminus Z _ { 0 } ^ { i } \} } { t } , \qquad a _ { i , u } ^ { \theta , \mathrm { a b s } } ( Z _ { t } ; t ) = \frac { \mathbf { 1 } \{ u \in Z _ { t } ^ { i } \} } { t } \frac { \nu _ { i } ^ { \theta } ( Z _ { t } ^ { i } \setminus \{ u \} \mid Z _ { t } ; t ) } { \nu _ { i } ^ { \theta } ( Z _ { t } ^ { i } \mid Z _ { t } ; t ) } .\tag{43}
$$

Proof. A bit that is zero at s remains zero at t with probability

$$
\exp \left( - \int _ { s } ^ { t } { \frac { d v } { 1 - v } } \right) = { \frac { 1 - t } { 1 - s } } .
$$

Once activated, it remains one. Multiplication over bits proves (41), and $z = \{ c \} , s = 0 \mathrm { g i v e s } ( 4 2 )$ For a nonempty state z,

$$
m _ { i } ^ { \theta } ( z \mid Z _ { t } ; t ) = ( 1 - t ) ^ { V - \mid z \mid } t ^ { \mid z \mid - 1 } \nu _ { i } ^ { \theta } ( z \mid Z _ { t } ; t ) .
$$

Deleting an active bit u changes this mass by the factor

$$
\frac { ( 1 - t ) \nu _ { i } ^ { \theta } ( z \setminus \{ u \} \mid Z _ { t } ; t ) } { t \nu _ { i } ^ { \theta } ( z \mid Z _ { t } ; t ) } .
$$

Multiplying by the reversed forward activation rate $1 / ( 1 - t )$ gives the model rate. For $Z _ { 0 } ^ { i } = c ,$ , the substitution

$$
\mu _ { i } ^ { \theta } ( u \mid Z _ { t } ; t ) = \mathbf { 1 } \{ u = c \}
$$

gives the clean-conditioned rate. Deleting the last bit has zero rate because the empty vector has zero mass. □

Proposition B.3 (Absorbing path loss). Under the assumptions of Proposition $B . 2 ,$

$$
\mathcal { I } _ { t } ^ { \mathrm { a b s } } ( \theta ; Z _ { 0 } , Z _ { t } ) = - \frac { 1 } { t } \sum _ { i } \sum _ { u \in Z _ { t } ^ { i } \setminus Z _ { 0 } ^ { i } } \log \left( 1 - \frac { \mu _ { i } ^ { \theta } ( u \mid Z _ { t } ; t ) } { \nu _ { i } ^ { \theta } ( Z _ { t } ^ { i } \mid Z _ { t } ; t ) } \right) .\tag{44}
$$

Proof. Fix $i ,$ write $Z _ { 0 } ^ { i } = c ,$ and put

$$
h _ { u } = \frac { \mu _ { i } ^ { \theta } ( u \mid Z _ { t } ; t ) } { \nu _ { i } ^ { \theta } ( Z _ { t } ^ { i } \mid Z _ { t } ; t ) } , \quad u \in Z _ { t } ^ { i } , \quad \quad \sum _ { u \in Z _ { t } ^ { i } } h _ { u } = 1 .
$$

The homogeneity identity $\Phi ( a / t , b / t ) = \Phi ( a , b ) / t$ gives the local loss times t as

$$
\Phi ( 0 , 1 - h _ { c } ) + \sum _ { u \in Z _ { t } ^ { i } \backslash \{ c \} } \Phi ( 1 , 1 - h _ { u } ) = 1 - h _ { c } + \sum _ { u \neq c , \ u \in Z _ { t } ^ { i } } [ - h _ { u } - \log ( 1 - h _ { u } ) ] .
$$

Because the $h _ { u }$ sum to one, the linear terms cancel, leaving (44). For a singleton, the remaining sum is empty and the loss is zero. □

For absorbing diffusion, the local kernel is constant over clean tokens in $Z _ { t } ^ { i }$ and zero outside it. Bayes reweighting gives the denoiser

$$
\pi _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) = { \frac { \mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) \mathbf { 1 } \{ c \in Z _ { t } ^ { i } \} } { \nu _ { i } ^ { \theta } ( Z _ { t } ^ { i } \mid Z _ { t } ; t ) } } .\tag{45}
$$

Since $\nu _ { i } ^ { \theta } ( Z _ { t } ^ { i } \setminus \{ u \} \mid Z _ { t } ; t ) / \nu _ { i } ^ { \theta } ( Z _ { t } ^ { i } \mid Z _ { t } ; t ) = 1 - \pi _ { i } ^ { \theta } ( u \mid Z _ { t } ; t )$ for $u \in Z _ { t } ^ { i }$ , the model rate in (43) equals (14). The cavity parameterization applies this support restriction analytically, while a separately predicted denoiser must respect it through explicit masking or learning.

## B.3 BINARY ANCESTRAL BRIDGES

By Bayes’ rule and the Markov property, the forward bridge of Section 4 is

$$
q _ { s \mid t , 0 } ( y \mid z , c ) = \frac { q _ { t \mid s } ( z \mid y ) q _ { s \mid 0 } ( y \mid c ) } { q _ { t \mid 0 } ( z \mid c ) } .\tag{46}
$$

Averaging it over the denoiser (5) cancels $q _ { t | 0 } ( z \mid c )$ , which gives the plug-in form of the ancestral update (18):

$$
p _ { s | t , i } ^ { \theta } ( y \mid Z _ { t } ) = \frac { q _ { t | s } ( Z _ { t } ^ { i } \mid y ) \sum _ { c } \mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) q _ { s | 0 } ( y \mid c ) } { \sum _ { c } \mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) q _ { t | 0 } ( Z _ { t } ^ { i } \mid c ) } .\tag{47}
$$

This is the bridge with the cavity in place of the clean token. We now derive the conditional bit bridges from (46).

Symmetric bridge. Conditional on the clean proposal $\widehat Z _ { 0 } ^ { i } = c .$ , the bits are independent. With $\varepsilon _ { s , t }$ from (35), bit u with clean value $b _ { u } = \mathbf { 1 } \{ u = c \}$ and current value $\boldsymbol { z } _ { u } = ( Z _ { t } ^ { i } ) _ { \imath }$ differs from $b _ { u }$ at s with probability

$$
q ^ { \mathrm { s y m } } \big ( ( Z _ { s } ^ { i } ) _ { u } \neq b _ { u } \ | \ z _ { u } , b _ { u } \big ) = \left\{ { s \varepsilon _ { s , t } } / ( 2 - t ) , \ z _ { u } = b _ { u } , \ \right.\tag{48}
$$

the chance of differing from the clean bit at s and then flipping, or not, to the current value at t. It favors the current value for s close to t and the clean bit as $s \downarrow 0$

A simpler symmetric sampler applies a one-bit Bayes update to each cavity bit marginal and draws the bits independently. This ignores the dependence between the bits through the shared clean token: at $s = 0$ , the exact bridge draws a single one-hot state from $\pi _ { i } ^ { \theta }$ , whereas independent bit draws can leave zero or several candidates active. It performed worse in early experiments, and we dropped it.

Absorbing bridge. Conditional on $\widehat Z _ { 0 } ^ { i } = c$ , the bit c stays active, and every bit inactive in $Z _ { t } ^ { i }$ remains inactive at s. For u $\in Z _ { t } ^ { i } \setminus \{ c \}$

$$
q ^ { \mathrm { a b s } } ( ( Z _ { s } ^ { i } ) _ { u } = 1 \mid ( Z _ { t } ^ { i } ) _ { u } = 1 , Z _ { 0 } ^ { i } = c ) = \frac { s } { t } .\tag{49}
$$

Indeed, a false candidate is active at s with probability $s ,$ active at t with probability t, and activation by s implies activation by t. These bit events are independent conditional on c and the current state. The update therefore preserves nonemptiness and produces a subset of the current active set.

## C TRAINING OBJECTIVES AND TIME SAMPLING

## C.1 FINITE-WINDOW NEGATIVE ELBO AND BOUNDARY TERMS

The idealized objective (3) integrates the reverse process from clean data at $t = 0$ to the terminal noise distribution at $t = 1$ . In practice we use an interior window $0 < t _ { 1 } < t _ { 2 } < 1$ to avoid singular jump rates near the endpoints. The corresponding finite-window negative ELBO, abbreviated NELBO in the appendices, is

$$
\mathrm { N E L B O } ( \theta ; t _ { 1 } , t _ { 2 } ) = \mathcal { R } _ { t _ { 1 } } ( \theta ) + K _ { t _ { 2 } } + \int _ { t _ { 1 } } ^ { t _ { 2 } } \mathbb { E } _ { Z _ { 0 } , Z _ { t } } [ \mathcal { I } _ { t } ( \theta ; Z _ { 0 } , Z _ { t } ) ] d t ,\tag{50}
$$

where

$$
\begin{array} { r } { \mathcal { R } _ { t _ { 1 } } ( \theta ) = \mathbb { E } _ { Z _ { 0 } , Z _ { t _ { 1 } } } \bigl [ - \log p _ { 0 | t _ { 1 } } ^ { \theta } ( Z _ { 0 } \mid Z _ { t _ { 1 } } ) \bigr ] , \qquad K _ { t _ { 2 } } = \mathbb { E } _ { Z _ { 0 } } \mathrm { K L } \bigl ( q _ { t _ { 2 } | 0 } ( \cdot \mid Z _ { 0 } ) \parallel p _ { t _ { 2 } } \bigr ) . } \end{array}\tag{51}
$$

The first term, with the denoiser as decoder, $\begin{array} { r } { p _ { 0 \mid t _ { 1 } } ^ { \theta } \left( Z _ { 0 } \mid Z _ { t _ { 1 } } \right) = \prod _ { i } \pi _ { i } ^ { \theta } ( Z _ { 0 } ^ { i } \mid Z _ { t _ { 1 } } ; t _ { 1 } ) } \end{array}$ , accounts for the residual corruption at the clean end of the truncated path, while the second measures the mismatch between the forward distribution at $t _ { 2 }$ and the prior from which generation starts.

For uniform and binary symmetric diffusion, the terminal-prior term is finite and available analytically (Apps. A.2 and B.1). With the prior matched to the limiting forward distribution, the boundary contributions vanish as $t _ { 1 } \downarrow 0$ and $t _ { 2 } \uparrow 1$ , recovering (3). For masked and binary absorbing diffusion, the limiting prior is a point mass, so the terminal KL is infinite for every $t _ { 2 } < 1$ but zero at $t _ { 2 } = 1$ . The terminal endpoint therefore requires separate treatment, and App. F.3 describes our likelihood-evaluation convention.

## C.2 TRAINING OBJECTIVES

Alongside the path ELBO, we compared a flat ELBO, cross-entropy (xent), and weighted crossentropy (wxent). For the kernels considered here, the reverse rates share a scalar prefactor, which extends the flip rate $r _ { t }$ of (9) to every kernel,

$$
r _ { t } = \left\{ \begin{array} { l l } { 1 / [ 2 ( 1 - t ) ] , } & { \mathrm { b i n a r y ~ s y m m e t r i c } , } \\ { 1 / [ V ( 1 - t ) ] , } & { \mathrm { u n i f o r m } , } \\ { 1 / t , } & { \mathrm { b i n a r y ~ a b s o r b i n g } , } \\ { 1 / t , } & { \mathrm { m a s k e d } , } \end{array} \right.\tag{52}
$$

and the remaining factors are ratios of the forward kernel $q _ { t | 0 }$ at the current noisy token. Since $\Phi ( r a , r b ) = r \Phi ( a , b )$ , the prefactor is also the weight of the jump penalties in the path integrand $\mathcal { T } _ { t }$ of (3). The flat ELBO drops $\mathrm { i t } , \mathcal { T } _ { t } ^ { \mathrm { f l a t } } = \mathcal { T } _ { t } / r _ { t }$ , which changes the relative weight of the noise levels. Cross-entropy trains the network output directly as a denoiser,

$$
\mathcal { I } _ { t } ^ { \mathrm { x e n t } } ( \theta ; Z _ { 0 } , Z _ { t } ) = - \sum _ { i = 1 } ^ { L } \log \mu _ { i } ^ { \theta } ( Z _ { 0 } ^ { i } \mid Z _ { t } ; t ) ,\tag{53}
$$

and weighted cross-entropy multiplies it by the prefactor, $\mathcal { T } _ { t } ^ { \mathrm { w x e n t } } = r _ { t } \mathcal { T } _ { t } ^ { \mathrm { x e n t } }$ . For masked diffusion the sums run over masked positions. The ELBO and wxent are integrated over time as in (3), while the flat ELBO and xent are averaged over the base schedule of App. C.3. All four are estimated with the inverse-density weight defined there, which the adaptive proposal of App. C.5 preserves.

We train the logic models on the path ELBO. For text, we selected the objective of each kernel and corpus on the validation NELBO after 10% of the training budget. The path ELBO was best everywhere except on Text8, where we train masked diffusion on xent, which coincides with its flat ELBO, and binary absorbing diffusion on wxent. The latter still bounds the likelihood, as we show next.

Absorbing cross-entropy bound. For binary absorbing diffusion, wxent upper-bounds the path integrand. Fix a position with clean token $Z _ { 0 } ^ { i } = c .$ The denoiser is supported on the active set $Z _ { t } ^ { i }$ which contains c, so

$$
\prod _ { u \in Z _ { t } ^ { i } \backslash \{ c \} } \left( 1 - \pi _ { i } ^ { \theta } ( u \mid Z _ { t } ; t ) \right) \geq 1 - \sum _ { u \in Z _ { t } ^ { i } \backslash \{ c \} } \pi _ { i } ^ { \theta } ( u \mid Z _ { t } ; t ) = \pi _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) .\tag{54}
$$

Since $\pi _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) = \mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) / \nu _ { i } ^ { \theta } ( Z _ { t } ^ { i } \mid Z _ { t } ; t ) \geq \mu _ { i } ^ { \theta } ( c \mid Z _ { t } ; t )$ , taking negative logarithms, multiplying by $1 / t ,$ and summing over positions in (44) gives $\mathcal { T } _ { t } ^ { \mathrm { a b s } } \leq \mathcal { T } _ { t } ^ { \mathrm { w x e n t } }$

## C.3 TIME SCHEDULES

The base schedule reparameterizes corruption time on the clipped window $[ t _ { 1 } , t _ { 2 } ]$ by $\xi \in [ 0 , 1 ]$ which we draw uniformly during training. The linear, logarithmic, and cosine maps are

$$
t _ { \mathrm { l i n } } ( \xi ) = t _ { 1 } + ( t _ { 2 } - t _ { 1 } ) \xi , \qquad t _ { \mathrm { l o g } } ( \xi ) = t _ { 1 } \left( \frac { t _ { 2 } } { t _ { 1 } } \right) ^ { \xi } , \qquad t _ { \mathrm { c o s } } ( \xi ) = t _ { 1 } + ( t _ { 2 } - t _ { 1 } ) \sin ^ { 2 } \left( \frac { \pi \xi } { 2 } \right) ,\tag{55}
$$

which give the base sampling densities on $[ t _ { 1 } , t _ { 2 } ]$

$$
p _ { \mathrm { l i n } } ( t ) = \frac { 1 } { t _ { 2 } - t _ { 1 } } , \qquad p _ { \mathrm { l o g } } ( t ) = \frac { 1 } { t \log ( t _ { 2 } / t _ { 1 } ) } , \qquad p _ { \mathrm { c o s } } ( t ) = \frac { 1 } { \pi \sqrt { ( t - t _ { 1 } ) ( t _ { 2 } - t ) } } .\tag{56}
$$

Logarithmic sampling assigns equal probability to intervals with the same ratio of endpoints, giving more resolution near clean data. At large vocabularies, this matches how the binary kernels lose information (App. C.4). Cosine sampling instead gives more resolution near both ends of the window. To estimate the path integral without bias, we weight each sequence’s loss by the inverse density of its sampled time, namely $t _ { 2 } - t _ { 1 }$ for the linear schedule, $t \log ( t _ { 2 } / t _ { 1 } )$ for the logarithmic schedule, and $\pi \sqrt { ( t - t _ { 1 } ) ( t _ { 2 } - t ) }$ for the cosine schedule.

We give ξ to the network as its time input:

$$
\xi ( t ) = \left\{ \begin{array} { l l } { { \displaystyle { \frac { t - t _ { 1 } } { t _ { 2 } - t _ { 1 } } } } , } & { { \mathrm { l i n e a r ~ s c h e d u l e } , } } \\ { { \displaystyle { \frac { \log ( t / t _ { 1 } ) } { \log ( t _ { 2 } / t _ { 1 } ) } } } , } & { { \mathrm { l o g a r i t h m i c ~ s c h e d u l e } , } } \\ { { \displaystyle { \frac { 2 } { \pi } } \arcsin \sqrt { \frac { t - t _ { 1 } } { t _ { 2 } - t _ { 1 } } } } , } & { { \mathrm { c o s i n e ~ s c h e d u l e } . } } \end{array} \right.\tag{57}
$$

The adaptive proposal below also acts on $\xi ,$ while the kernel and rates retain their definitions in corruption time t. We choose inference grids separately from training-time sampling. Applying any map in (55) to $j / N _ { \parallel }$ , for $j = 0 , \ldots , N ,$ , gives a linear, logarithmic, or cosine grid for N reverse steps, traversed in decreasing j as in Algorithm 1. The logarithmic grid has smaller absolute steps near $t _ { 1 }$ and the cosine grid near both ends of the window. We train with the linear and logarithmic schedules (Tables 3 and 4) and sample on grids of all three kinds (Table 7).

## C.4 INFORMATION RETAINED UNDER EACH KERNEL

Section 3.4 and Figure 2 consider independent positions whose clean tokens are uniform on $[ V ]$ The other positions then carry no information about $Z _ { 0 } ^ { i }$ , so the true cavity is uniform, $q ( Z _ { 0 } ^ { i } = c \ |$ $Z _ { t } ^ { - i } ) = 1 / V$ , and the information in the noisy sequence is a sum over positions of

$$
I ( Z _ { 0 } ^ { i } ; Z _ { t } ^ { i } ) = \log V - \mathbb { E } \left[ H \big ( q ( Z _ { 0 } ^ { i } \mid Z _ { t } ^ { i } ) \big ) \right] ,\tag{58}
$$

where the posterior of the clean token is the denoiser (5) with a uniform $\mu _ { i } ^ { \theta }$ , the forward kernel $q _ { t | 0 } ( Z _ { t } ^ { i } \mid c )$ normalized over c.

• Masked. The posterior is a point mass at an unmasked token and uniform at a mask, so $I =$ $\left( 1 - t \right) \log V$

• Uniform. Given the noisy token, the posterior assigns $a = 1 - t + t / V$ to it and $b = t / V$ to every other token, so $I = \log \dot { V } + a \log a + \dot { ( } V - 1 ) b \log \dot { b } .$

• Binary absorbing. The posterior is uniform on the active set, whose size is $1 + K$ with $K \sim$ Bi $\mathsf { \Omega } _ { 1 } ( V - 1 , t )$ , so I = log V − E[log(1 + K)] ≥ log $V - \log ( 1 + ( V - 1 ) t )$ by Jensen’s inequality.

• Binary symmetric. A state z differs from the clean token c in $d _ { H } ( z , c ) = | z | + 1 - 2 z _ { c }$ bits, so the likelihood $q _ { t | 0 } ^ { \mathrm { s y m } } ( z \mid c )$ is proportional to $\rho _ { t } ^ { - d _ { H } ( z , c ) } \propto \rho _ { t } ^ { 2 z _ { c } }$ as a function of $c ,$ and with a uniform prior the posterior weighs active candidates by $\rho _ { t } ^ { 2 }$ and inactive tokens by one, as (38) with a uniform $\mu _ { i } ^ { \theta } . \mathrm { A }$ state with k active candidates therefore has posterior entropy

$$
H _ { k } = \log ( k \rho _ { t } ^ { 2 } + V - k ) - \frac { k \rho _ { t } ^ { 2 } \log \rho _ { t } ^ { 2 } } { k \rho _ { t } ^ { 2 } + V - k } ,\tag{59}
$$

and $I = \log V - \mathbb { E } H _ { B + K }$ with independent $B \sim \mathrm { B e r } ( 1 - t / 2 )$ , the clean bit, and $K \sim \operatorname { B i n } ( V -$ $1 , t / 2 )$ , the false candidates. For small $t , \rho _ { t } ^ { 2 }$ is large and the posterior is nearly uniform on the active set, so that $I \approx$ log $V - \log ( 1 + ( \dot { V } - 1 ) t / \bar { 2 } )$ , as in the absorbing case with half as many false candidates.

These expressions give the curves of Figure 2. With $V = 1 0 2 4$ on the window $\left[ 5 \cdot 1 0 ^ { - 4 } , 0 . 9 9 9 9 \right]$ , the linear schedule draws 95% of its times where less than half of the information is left under binary symmetric diffusion and 97% under binary absorbing diffusion, against 50% under masked and 60% under uniform diffusion. The logarithmic schedule lowers these fractions to 39% and 46% for the binary kernels. On the other hand, with $V = 9$ , half of the information is lost at $t \approx 0 . 2 7$ under binary symmetric, 0.28 under binary absorbing, and 0.29 under uniform diffusion, so the binary kernels lose information about as fast as uniform diffusion.

## C.5 ADAPTIVE GRADIENT-RMS IMPORTANCE SAMPLING

Adaptive importance sampling replaces the base schedule by a proposal that draws more samples where loss gradients are large. It improved over the base schedule alone in all our experiments, so we train every model, for all four kernels and all tasks, with it. The statistic is the root-mean-square norm of the gradient with respect to the network’s output logits, used as an inexpensive proxy for parameter-gradient norms.

We divide the range of $\xi$ into $K = 3 2$ equal-width bins. Let $g _ { \theta }$ denote a sequence’s contribution to the loss under the base schedule, including token normalization and the base-schedule weight, and let ℓ denote its output logits. For the path objective with L scored tokens per sequence, this contribution is

$$
g _ { \theta } = \frac { \mathcal { I } _ { t } ( \theta ; Z _ { 0 } , Z _ { t } ) } { L p _ { \mathrm { b a s e } } ( t ) } , \qquad p _ { \mathrm { b a s e } } \in \{ p _ { \mathrm { l i n } } , p _ { \mathrm { l o g } } , p _ { \mathrm { c o s } } \} .\tag{60}
$$

When scored token counts vary across sequences, we replace L by their minibatch mean. In each bin $k ,$ we maintain exponentially decayed sums $S _ { k }$ of $\| \nabla _ { \ell } g _ { \theta } \| _ { 2 } ^ { 2 }$ and counts $C _ { k } ,$ , both with decay 0.99. Their ratio gives the mean squared gradient norm, from which we form the RMS estimate and the bin probability:

$$
G _ { k } = \sqrt { \frac { S _ { k } } { C _ { k } } } , \qquad P _ { k } = 0 . 9 \frac { G _ { k } } { \sum _ { j = 1 } ^ { K } G _ { j } } + \frac { 0 . 1 } { K } .\tag{61}
$$

Sampling uniformly within each selected bin gives the time density and path-loss weight

$$
p ( t ) = K P _ { k } p _ { \mathrm { b a s e } } ( t ) , \qquad \frac { 1 } { p ( t ) } = \frac { 1 } { K P _ { k } p _ { \mathrm { b a s e } } ( t ) } , \qquad \xi ( t ) \ \mathrm { i n } \ \mathrm { b i n } \ k .\tag{62}
$$

The additional factor $1 / ( K P _ { k } )$ preserves the objective under the base schedule. Mixing in 10% of the base schedule keeps every bin accessible and bounds this factor by 10.

For the first 100 minibatches we sample from the base schedule while collecting statistics. Each minibatch uses the preceding proposal, held fixed during differentiation, and updates the statistics for the next minibatch. We stratify the times within a minibatch in the proposal’s cumulative-probability coordinate. For validation, we draw times from the base schedule.

## D CONFIDENCE-GUIDED Ψ-SAMPLER

This appendix specifies each setting of Algorithm 1 for all four kernels. One step maps the state $Z _ { t }$ to $\bar { Z } _ { s } , s < \bar { t } .$ , on a sampling grid with linear, cosine, or logarithmic spacing (App. C.3). We hold conditioning positions, such as Sudoku clues, fixed and exclude them from every average over positions below.

Temperature $T$ and nucleus top-p. For every draw from the denoiser, for the clean proposal, inside the ancestral update, and in the final decoding, we use the tempered denoiser

$$
\pi _ { i } ^ { \theta , T } ( c \mid Z _ { t } ; t ) \propto \pi _ { i } ^ { \theta } ( c \mid Z _ { t } ; t ) ^ { 1 / T } ,\tag{63}
$$

which on text we further restrict to its top-p nucleus and renormalize, and $T = 0$ selects the most probable token. The clean proposal $\widetilde { Z } _ { 0 } ^ { i } \sim \pi _ { i } ^ { \theta , T } ( . \vert Z _ { t } ; t )$ is drawn independently at each position, and the ancestral update draws its own proposal $Z _ { 0 } ^ { i }$ in the same way.

Re-corrupt fraction η. Each position is re-corrupted with probability $\kappa _ { i }$ and otherwise takes the ancestral update (18), as in (21). Without position confidence, $\kappa _ { i } = \eta$ at every position, so that $\eta = 0$ is ancestral sampling and $\eta = 1$ re-corrupts every position at every step.

Confidence coordinate. Position confidence and the ordering tilt read the confidence in the proposal from the cavity or from the denoiser,

$$
\mathrm { c o n f } _ { i } = \mu _ { i } ^ { \theta } ( \widetilde { Z } _ { 0 } ^ { i } \mid Z _ { t } ; t ) \qquad \mathrm { o r } \qquad \mathrm { c o n f } _ { i } = \pi _ { i } ^ { \theta } ( \widetilde { Z } _ { 0 } ^ { i } \mid Z _ { t } ; t ) .\tag{64}
$$

The cavity reflects the context alone, while the denoiser also accounts for the position’s own noisy state.

Position confidence γ. The re-corruption probabilities decrease with confidence,

$$
\kappa _ { i } = \operatorname* { m i n } \{ 1 , \zeta e ^ { - \gamma \mathrm { c o n f } _ { i } } \} ,\tag{65}
$$

where we set the constant ζ by bisection so that the mean of $\kappa _ { i }$ over positions equals $\eta .$ Positions capped at one thus pass their excess to the others, and $\gamma = 0$ gives $\kappa _ { i } = \eta$ . For masked diffusion, masked positions keep $\kappa _ { i } = \eta .$ , and the redistribution runs over the unmasked positions only, each with the confidence in its token recorded when it was unmasked, since the network is not trained at unmasked positions.

Ordering tilt λ. A re-corrupted position returns to its own noise level

$$
s _ { i } = \operatorname* { m i n } \{ 1 , \zeta ^ { \prime } e ^ { - \lambda \mathrm { c o n f } _ { i } } \}\tag{66}
$$

instead of $s ,$ with $\zeta ^ { \prime }$ set so that the mean of $s _ { i }$ over positions equals s. Less confident positions are thus corrupted more while the mean noise level follows the grid, and $\lambda = 0$ gives $s _ { i } = s$

Candidate confidence $\beta .$ . Plain re-corruption opens alternatives uniformly at random, as the forward kernel does. With candidate confidence, the binary kernels corrupt the proposal’s own bit as the forward kernel does, dropping it with probability $s _ { i } / 2$ under binary symmetric diffusion and keeping it under binary absorbing diffusion, and open each other token $\dot { u } \ne \bar { Z } _ { 0 } ^ { i }$ independently with probability min $\{ 1 , \zeta _ { i } p _ { i , u } ^ { \beta } \}$ , where $\zeta _ { i }$ is set so that

$$
\sum _ { u \ne \widetilde { Z } _ { 0 } ^ { i } } \operatorname* { m i n } \{ 1 , \ : \zeta _ { i } p _ { i , u } ^ { \beta } \} = \biggl \{ ( V - 1 ) s _ { i } / 2 , \ : \ : \mathrm { b i n a r y \ s y m m e t r i c } , \ : \ :\tag{67}
$$

The expected number of opened alternatives is thus that of the forward kernel, which $\beta = 0$ recovers. The weight $p _ { i , u }$ is the denoiser probability of u at the most recent step at which u was a candidate at position i, starting from $1 / V$ , because the binary absorbing denoiser vanishes outside the current set. Without this memory, reopening in proportion to the current cavity or denoiser probability lowers binary symmetric diffusion’s best Sudoku validation solve rate over the same grid to 89.6% and 85.2%, below position confidence alone (95.2%). Uniform diffusion re-corrupts a position by changing its token with the forward probability $( V - 1 ) s _ { i } / V$ and draws the new token u $\neq \widetilde { Z } _ { 0 } ^ { i }$ with probability proportional to $\pi _ { i } ^ { \theta } ( u ^ { \star } | ~ Z _ { t } ; t ) ^ { \beta }$ , so that $\beta = 0$ is again the forward kernel. Masked diffusion has no candidate to reopen.

Final decoding. After the last step, we decode each position from the tempered denoiser at $t _ { 1 }$ keeping the unmasked tokens of masked diffusion.

## E EXTENDED RELATED WORK

Discrete diffusion and corruption-state design. Several lines of work broaden the forward path without introducing candidate sets. Generalized interpolating diffusion and kinetic path design admit richer categorical couplings (von Rutte et al., 2025; Shaul et al., 2025), partial masking inserts¨ intermediate corruption states between a token and a full mask (Chao et al., 2025), and sticky jump processes combine discrete jumps with continuous coordinates (Jutras-Dube et al., 2026). CADD´ and CANDI go further and augment or hybridize the categorical state with a continuous one (Zheng et al., 2025a; Pynadath et al., 2025). Even at scale, and with architectures and training budgets matched, the choice between masking, uniform corruption, and interpolating paths remains conse quential (Sahoo et al., 2026).

There is also a substantial simplex-valued lineage, which lifts categories to continuous points on the simplex or updates distribution parameters directly (Richemond et al., 2022; Floto et al., 2023; Avdeyev et al., 2023; Stark et al., 2024; Graves et al., 2023; Chandra et al., 2026). Simplax is the¨ closest of these to our setting: it keeps uniform categorical diffusion as its marginal and adds an exact Dirichlet auxiliary variable to the reverse bridge (Sakurai et al., 2026). Its denoiser, however, still observes a corrupted category, whereas binary diffusion exposes the candidate set to the denoiser as the noisy state.

Binary diffusion and candidate-set representations. Prior binary diffusion methods treat each bit as part of the object being generated. Bit-flip diffusion of binary data goes back to binomial diffusion (Sohl-Dickstein et al., 2015), and DMPM studies its continuous-time form (Pham et al., 2025). BerDiff applies a Bernoulli kernel to binary segmentation masks (Chen et al., 2023a), GraphGUIDE adds, removes, or flips graph edges through Bernoulli transitions (Tseng et al., 2023), and Blackout diffusion models discrete counts through birth–death processes (Santos et al., 2023). Analog Bits instead encodes one category in $\lceil \log _ { 2 } ^ { - } V \rceil$ real-valued bit coordinates and applies continuous diffusion (Chen et al., 2023b). In binary diffusion, by contrast, there are $V$ binary coordinates per position and an active bit denotes membership in the noisy subset.

The ShortListing Model (SLM) is the closest predecessor (Song et al., 2025). It represents a categorical variable by a nonempty candidate set, starts the forward process from the clean singleton, activates false candidates until reaching the full vocabulary, and samples in reverse by progressive candidate pruning. However, the model is formulated in discrete time and identifies sets with simplex-face centroids. To see the correspondence with binary absorbing diffusion, let $n _ { t }$ be SLM’s expected candidate-set size at time t and take the linear noise schedule

$$
n _ { t } = 1 + ( V - 1 ) t .\tag{68}
$$

The probability of adding a candidate between times $s < t$ is then

$$
{ \frac { n _ { t } - n _ { s } } { V - n _ { s } } } = { \frac { t - s } { 1 - s } } ,\tag{69}
$$

and the clean-conditioned probability of retaining a false candidate is

$$
\frac { n _ { s } - 1 } { n _ { t } - 1 } = \frac { s } { t } ,\tag{70}
$$

which are the absorbing kernel (41) and bridge (49).

Diffusion for structured reasoning. Beyond Autoregression established masked diffusion baselines on Sudoku and other reasoning and planning tasks (Ye et al., 2025). Dirichlet diffusion solves Sudoku in a continuous simplex state (Avdeyev et al., 2023), and Simplax reports Sudoku generalization across clue densities (Sakurai et al., 2026). Several analyses show that the corruption and the order of updates matter as much as the model: Spatial Reasoning Models find that the order of variable updates strongly affects accuracy (Wewer et al., 2025), and the locality induced by the corruption and factorization changes trainability on Sudoku and graph path-finding (Wang et al., 2026). On the theoretical side, masked diffusion models can simulate chain-of-thought under finite precision (Svete & Sabharwal, 2026).

A separate line of work changes the reverse-time policy rather than the forward state, through token ordering, planned denoising and path planning, remasking, learned reveal schedules, or search during and around sampling (Kim et al., 2025; Liu et al., 2025; Peng et al., 2025; Wang et al., 2025b; Kim et al., 2026; Xu et al., 2026; Ta et al., 2026; Bai et al., 2026; Zhang et al., 2026). These methods compose with any corruption kernel, and beyond the confidence-guided re-corruption of Section 4 we use none of them: our verifier-free solve rate refers to direct samples from each kernel’s tuned sampler, without search, reranking, or a symbolic solver.

Recurrent reasoning. Sudoku and related benchmarks have also driven a rapidly growing literature on small recurrent reasoners, which allocate computation through repeated latent updates, fixed-point dynamics, or looped layers (Wang et al., 2025a; Jolicoeur-Martineau, 2025; Sghaier et al., 2026; Huang et al., 2026; Fein-Ashley & Rashidinejad, 2026; Helbling et al., 2026; Cameron et al., 2026; Lee et al., 2026). These approaches change the architecture, the training dynamics, or the recurrent depth, which is a separate axis from the corruption state that binary diffusion changes, so we do not compare against them directly.

## F EXPERIMENTAL DETAILS

## F.1 OPTIMIZATION DETAILS

Tables 3 and 4 give the architectures and optimization settings of the models. We embed candidate sets by (15). Dividing the sum by $| z |$ instead, or embedding the signed vector $( 2 z - 1 ) / V ,$ made little difference in our experiments.

## F.2 LEARNING-RATE AND CHECKPOINT SELECTION

We selected the learning rate of each task and kernel from $\{ 1 0 ^ { - 4 } , 3 \cdot 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ on the validation NELBO of App. F.3, computed on 51,200 validation draws and averaged over two training seeds (Table 5). Within each task, kernel, and seed, all learning rates share the data, times, and corruptions, so their differences have a Monte Carlo standard error below $3 \cdot 1 0 ^ { - 4 }$ , and the variation between seeds dominates. We tune Sudoku and graph coloring at their full training budget and Boxoban at 1M steps. On Boxoban uniform and binary absorbing, the two lowest learning rates are within seed noise, and in both cases we keep the middle rate $3 \cdot 1 0 ^ { - 4 }$ . Since Boxoban validation NELBO was still improving slowly at that point, we trained the selected rates for 2.5M steps and selected a checkpoint for each run on the validation NELBO, evaluated every 100k steps from 1M. We keep the final checkpoint unless an earlier one is lower by more than two standard errors of the paired difference, and only two binary absorbing seeds select an earlier checkpoint, at 2.0M and 1.0M steps.

Table 3: Training settings of the logic models.
<table><tr><td colspan="2">Architecture</td></tr><tr><td>transformer layers</td><td>12</td></tr><tr><td>hidden width</td><td>512</td></tr><tr><td>attention heads</td><td>8</td></tr><tr><td>MLP width</td><td>2048</td></tr><tr><td>time-conditioning width</td><td>128</td></tr><tr><td>conditioning</td><td>Sudoku: clue embedding; Boxoban: board prefix; graph coloring: adja- cency in attention</td></tr><tr><td>sequence length</td><td>Sudoku 81; Boxoban  $1 0 0 + 1 2 8$  (flattened board + solution); graph col- oring 64</td></tr><tr><td>precision</td><td>mixed: float32 parameters, normalization, logits, and loss; bfloat16 mat- muls, attention, and residual stream</td></tr><tr><td>dropout</td><td>0</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>optimizer</td><td>AdamW,  $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ </td></tr><tr><td>learning rate</td><td> $1 0 ^ { - 4 }$  Boxoban  $3 \cdot 1 0 ^ { - 4 }$  except binary symmetric</td></tr><tr><td>schedule</td><td>constant after 1,000 warm-up steps</td></tr><tr><td>batch size</td><td>512</td></tr><tr><td>training steps</td><td>Sudoku 1M, Boxoban 2.5M, graph coloring 300k</td></tr><tr><td>weight decay</td><td>0.05, excluding norms and embeddings</td></tr><tr><td>gradient clip</td><td>1</td></tr><tr><td>EMA decay</td><td>0.9999</td></tr><tr><td>training seeds</td><td>5</td></tr><tr><td colspan="2">Diffusion</td></tr><tr><td>objective</td><td></td></tr><tr><td>time window</td><td>path ELBO, cavity parameterization [0.01, 0.9999]</td></tr><tr><td>base schedule</td><td>linear, low-discrepancy sampling</td></tr><tr><td>adaptive sampling</td><td>gradient RMS, EMA 0.99</td></tr></table>

Table 4: Training settings of the text models.
<table><tr><td colspan="2">Architecture</td></tr><tr><td>transformer layers hidden width</td><td>12; OpenWebText 16</td></tr><tr><td>attention heads</td><td>768; OpenWebText 1024</td></tr><tr><td>MLP width</td><td>12; OpenWebText 16</td></tr><tr><td></td><td>3072; OpenWebText 4096</td></tr><tr><td>time-conditioning width sequence length</td><td>128</td></tr><tr><td>vocabulary</td><td>Text8 256, TinyStories 512, OpenWebText 1024 Text8 27, TinyStories 1024, OpenWebText 1024</td></tr><tr><td>precision</td><td>mixed: float32 parameters, normalization, logits, and loss; bfloat16 mat-</td></tr><tr><td>dropout</td><td>muls, attention, and residual stream 0</td></tr><tr><td>Optimization</td><td></td></tr><tr><td>optimizer</td><td></td></tr><tr><td>learning rate</td><td>AdamW, (β1, β2) = (0.9, 0.999)</td></tr><tr><td>schedule</td><td>3 · 10−4</td></tr><tr><td>batch size</td><td>constant after 2,500 warm-up steps</td></tr><tr><td>training steps</td><td>512</td></tr><tr><td>weight decay</td><td>Text8 1M, TinyStories 500k, OpenWebText 250k 0.02, excluding norms and embeddings</td></tr><tr><td>gradient clip</td><td>1</td></tr><tr><td>EMA decay</td><td>0.9999</td></tr><tr><td>training seeds</td><td>1</td></tr><tr><td></td><td></td></tr><tr><td>Diffusion</td><td></td></tr><tr><td>objective</td><td>path ELBO, cavity parameterization; Text8: xent for masked, wxent for</td></tr><tr><td></td><td>binary absorbing</td></tr><tr><td>time window</td><td>Text8 [0.001, 0.9999]; TinyStories and OpenWebText [0.0005, 0.9999]</td></tr><tr><td>base schedule</td><td>linear; binary kernels logarithmic; low-discrepancy sampling</td></tr></table>

Table 5: Validation NELBOs for the logic learning-rate sweep in nats per scored token, mean over two training seeds with half their difference in parentheses.
<table><tr><td>Task</td><td>Kernel</td><td> $1 0 ^ { - 4 }$ </td><td> $3 { \cdot } 1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Sudoku</td><td>Binary symmetric</td><td>0.0163 (0.0005)</td><td>0.0175 (0.0002)</td><td>0.0230 (0.0007)</td></tr><tr><td></td><td>Uniform</td><td>0.0162 (0.0013)</td><td>0.0179 (0.0001)</td><td>0.0226 (0.0005)</td></tr><tr><td></td><td>Binary absorbing</td><td>0.0148 (0.0001)</td><td>0.0178 (0.0010)</td><td>0.0229 (0.0011)</td></tr><tr><td></td><td>Masked</td><td>0.0154 (0.0004)</td><td>0.0176 (0.0002)</td><td>0.0245 (0.0009)</td></tr><tr><td>Boxoban</td><td>Binary symmetric</td><td>0.0286 (0.0017)</td><td>0.0306 (0.0034)</td><td>0.0358 (0.0006)</td></tr><tr><td></td><td>Uniform</td><td>0.0267 (0.0026)</td><td>0.0256 (0.0009)</td><td>0.0332 (0.0007)</td></tr><tr><td></td><td>Binary absorbing</td><td>0.0290 (0.0005)</td><td>0.0309 (0.0005)</td><td>0.0394 (0.0002)</td></tr><tr><td></td><td>Masked</td><td>0.0294 (0.0001)</td><td>0.0266 (0.0002)</td><td>0.0338 (0.0009)</td></tr><tr><td>Graph coloring</td><td>Binary symmetric</td><td>0.0863 (0.0001)</td><td>0.0873 (0.0000)</td><td>0.0905 (0.0000)</td></tr><tr><td></td><td>Uniform</td><td>0.0857 (0.0000)</td><td>0.0865 (0.0002)</td><td>0.0896 (0.0002)</td></tr><tr><td></td><td>Binary absorbing</td><td>0.0881 (0.0001)</td><td>0.0898 (0.0001)</td><td>0.0951 (0.0003)</td></tr><tr><td></td><td>Masked</td><td>0.0837 (0.0001)</td><td>0.0845 (0.0001)</td><td>0.0880 (0.0005)</td></tr></table>

## F.3 EVALUATION

NELBO. We evaluate the finite-window NELBO (50) on the training window with the estimators used in training. For the path term, we draw one time per sequence from the training time schedule  
Table 6: Test NELBOs of the final logic models in nats per scored token, mean and standard error over the five training seeds.
<table><tr><td>Task</td><td>Binary symmetric</td><td>Uniform</td><td>Binary absorbing</td><td>Masked</td></tr><tr><td>Sudoku</td><td>0.0165 (0.0003)</td><td>0.0160 (0.0004)</td><td>0.0161 (0.0001)</td><td>0.0166 (0.0002)</td></tr><tr><td>Boxoban</td><td>0.0224 (0.0002)</td><td>0.0235 (0.0005)</td><td>0.0278 (0.0004)</td><td>0.0236 (0.0002)</td></tr><tr><td>Graph coloring</td><td>0.0860 (0.0001)</td><td>0.0857 (0.0001)</td><td>0.0877 (0.0001)</td><td>0.0837 (0.0000)</td></tr></table>

with inverse-density weighting. We estimate reconstruction from forward corruptions at $t _ { 1 }$ with the denoiser as decoder, and visible tokens in masked diffusion are copied exactly and have zero reconstruction loss. The prior is the limiting distribution from which generation starts. Its KL from the $\mathrm { t i m e } { \cdot } t _ { 2 }$ corruption is analytic for uniform and binary symmetric diffusion (Apps. A.2 and B.1). For masked and binary absorbing diffusion the prior is a point mass, so that the KL is infinite for $t _ { 2 } ~ < ~ 1$ and zero at $t _ { 2 } = 1$ , and we omit it. We aggregate the total NELBO over scored tokens, excluding padding and conditioning clues.

Logic solve rate. We score logic models on direct samples, with one attempt per instance and no verifier or selection among attempts, so the reported solve rate is single-attempt success. Sudoku success requires the completed reference grid, and Boxoban success requires legal actions that end with every box on a goal, with the action sequence read up to the first STAY, the no-operation action that pads solutions. Graph coloring accepts any proper coloring. We evaluate the selected EMA checkpoint of each of the five training seeds (App. F.2) with a linear sampling grid on [0.01, 0.9999] and the sampler selected on validation at the reference NFE, 256 for Sudoku and Boxoban and 64 for graph coloring (App. F.4). The tables use the full test splits, and the NFE and training curves use 8,192 and 4,096 test instances, except in their hard columns, which use every hard test instance. We evaluate every kernel on the same instances and fix the difficulty bins on validation data. The hard subsets are defined with the data in App. F.5. We report the mean and standard error over training seeds.

Text sample quality. We tune every sampler setting on the validation split and evaluate on the test split of each corpus (App. F.6). After removing special tokens, we retokenize complete decoded strings with GPT-2 Large for GenPPL and MAUVE features. Each sample set holds 2,000 samples from one generation seed, and each reported value averages the two generation seeds of a sampler setting. Entropy is the unigram entropy, in nats, of the GPT-2 tokens pooled over a sample set rather than a within-string average, so it measures the diversity of the set as a whole. GenPPL is the exponential of the total GPT-2 Large negative log-likelihood of the sample set divided by its total token count, so it is pooled over tokens rather than averaged over strings. We compute MAUVE with 200 buckets and average it over four reference draws and five clustering seeds, with reference subsets sampled without replacement and of the size of the generated sample set. Error bars span the two generation seeds and condition on the checkpoint.

## F.4 SAMPLER TUNING PROCEDURE

We sample every kernel with the confidence-guided Ψ-sampler of App. D. Its settings are the temperature T, the re-corrupt fraction η, the confidence coordinate, the position-confidence strength $\gamma ,$ the ordering tilt λ, and the candidate-confidence strength β, which we tune for every kernel but masked diffusion on the logic tasks and for the binary kernels on text. On text, we also tune the nucleus top-p and the spacing of the sampling grid. We apply temperature and top-p to the denoiser rather than to the cavity (Casado Noguerales et al., 2026).

For each task and kernel, we select these settings on validation over the grids of Table 7. On the logic tasks the score is the mean solve rate over the training seeds on a fixed validation split of 2,048 instances (1,024 for the large graphs) at the reference NFE. On text it is MAUVE at NFE $= L ,$ screened with 512 samples and confirmed with 2,000 samples for the three best candidates. We then freeze the selection for every test measurement, budget, and training seed unless otherwise stated. We select the sampler of the hard columns separately on the hard subsets of the validation splits. The grids are not full products: the ordering and confidence strengths enter through a few sub-grids, and we give masked diffusion, which has no candidate to reopen, a matching number of extra ordering and re-corruption points in place of the candidate-confidence settings (Table 7), so we tune every kernel with the same budget. Because the cavity and the denoiser coincide for masked diffusion, the confidence coordinate has no effect there. Table 10 selects over nested subsets of the logic grid: ancestral sampling, re-corruption alone, the Ψ-sampler without confidence, with position confidence, and with every setting, so each row reports what its settings add. Candidate confidence lifts binary symmetric diffusion on Sudoku and binary absorbing diffusion on the large graphs, whereas it is only selected for uniform diffusion on the Boxoban hard subset. Tables 8 and 9 list the selected settings.

Figure 4 measures each kernel at several entropies: we bin the sampler settings screened on validation by GPT-2 token entropy, evaluate the best of each bin on the test split with 2,000 samples and two generation seeds, and draw the upper envelope of these points over entropy, on which the validation-selected setting lies for every kernel.

Table 7: Sampler settings tuned on validation.
<table><tr><td>Setting</td><td>Logic tasks</td><td>Text</td></tr><tr><td>Temperature T</td><td>{0, 0.5, 1}</td><td>{0.8, 0.9, 1, 1.1}</td></tr><tr><td>Re-corrupt fraction η masked also</td><td>{0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1}{0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1} {0.9}</td><td></td></tr><tr><td>Position confidence γ</td><td>{0, 1, 3, 10}</td><td>{0, 1, 3, 5, 10}</td></tr><tr><td>Ordering tilt λ</td><td>{0, 1, 3, 10, 30, 100}</td><td>{0, 1, 3, 10}</td></tr><tr><td>masked also Candidate confidence β</td><td>{2, 5, 7, 15, 20, 50, 300, 1000} {0, 0.5, 1, 2, 3}, all but masked</td><td>{0, 0.5, 1, 2}, binary</td></tr><tr><td>Confidence coordinate</td><td>cavity, denoiser</td><td>cavity, denoiser</td></tr><tr><td>Grid spacing</td><td>linear</td><td>cosine, logarithmic</td></tr><tr><td>Nucleus top-p</td><td>1</td><td>{0.9, 1}</td></tr></table>

Table 8: Selected sampler settings on the logic tasks, on the full validation splits and on the hard subsets. The confidence coordinate (coord.) is the cavity (c) or the denoiser (d).
<table><tr><td>Kernel</td><td>Setting</td><td>Sudoku All</td><td>Sudoku Hard</td><td>Boxoban All</td><td>Boxoban Hard</td><td>Graph coloring All</td><td>Graph coloring Hard</td></tr><tr><td rowspan="6">Binary symmetric</td><td>T</td><td>0</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td></tr><tr><td>η</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.8</td><td>1</td></tr><tr><td>coord.</td><td>C</td><td>c</td><td>c</td><td>c</td><td>c</td><td>c</td></tr><tr><td>γ</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>λ</td><td>0</td><td>0</td><td>3</td><td>3</td><td>0</td><td>3</td></tr><tr><td>β</td><td>0.5</td><td>0.5</td><td>3</td><td>0.5</td><td>0</td><td>0</td></tr><tr><td rowspan="6">Uniform</td><td>T</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>η</td><td>0.5</td><td>0.5</td><td>0.5</td><td>1</td><td>0.5</td><td>1</td></tr><tr><td>coord.</td><td>C</td><td>C</td><td>c</td><td>c</td><td>d</td><td>C</td></tr><tr><td>γ</td><td>10</td><td>10</td><td>3</td><td>0</td><td>3</td><td>0</td></tr><tr><td>λ</td><td>0</td><td>0</td><td>3</td><td>3</td><td>0</td><td>30</td></tr><tr><td>β</td><td>0</td><td>0</td><td>0</td><td>2</td><td>0</td><td>0</td></tr><tr><td rowspan="6">Binary absorbing</td><td>T</td><td>0.5</td><td>0</td><td>0</td><td>0</td><td>0.5</td><td>0</td></tr><tr><td>η</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0.5</td><td>1</td></tr><tr><td>coord.</td><td>d</td><td>d</td><td>C</td><td>C</td><td>c</td><td>C</td></tr><tr><td>γ</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td></tr><tr><td>λ</td><td>1</td><td>1</td><td>1</td><td>0</td><td>3</td><td>0</td></tr><tr><td>β</td><td>0</td><td>0</td><td>0</td><td>0.5</td><td>0</td><td>0.5</td></tr><tr><td rowspan="6">Masked</td><td>T</td><td>0</td><td>0.5</td><td>0</td><td>0</td><td>0.5</td><td>0.5</td></tr><tr><td>η</td><td>0.9</td><td>0.9</td><td>0.5</td><td>0.8</td><td>0.8</td><td>1</td></tr><tr><td>coord.</td><td>C</td><td>c</td><td>C</td><td>C</td><td>C</td><td>C</td></tr><tr><td>γ</td><td>0</td><td>0</td><td>10</td><td>0</td><td>3</td><td>0</td></tr><tr><td>λ</td><td>3</td><td>3</td><td>0</td><td>3</td><td>3</td><td>2</td></tr><tr><td>β</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

## F.5 LOGIC TASKS

In every task, we condition the model on the given part of the instance: we show it clean, never corrupt it, and exclude it from the loss, so all randomness and revision act on the generated positions.

Sudoku. For Sudoku, we use the Sudoku-Extreme boards of Wang et al. (2025a), 3.8M training boards and a test pool of 423k, as 81 digit tokens with the clue cells fixed. We mark the clue cells with an embedding added to the input. This changed little for the masked and binary kernels but was necessary for uniform diffusion to train at all, since a clue is otherwise indistinguishable from a uniformly corrupted cell. We partition the test pool into validation and test sets of about 200k and

Table 9: Selected sampler settings on text. The confidence coordinate (coord.) is the cavity (c) or the denoiser (d).
<table><tr><td>Kernel</td><td>Setting</td><td>Text8</td><td>TinyStories</td><td>OpenWebText</td></tr><tr><td rowspan="8">Binary symmetric</td><td> $T$ </td><td>1</td><td>0.9</td><td>0.9</td></tr><tr><td> $\mathrm { t o p } { - } p$ </td><td>1</td><td>1</td><td>1</td></tr><tr><td>spacing</td><td>log</td><td>cosine</td><td>log</td></tr><tr><td>η</td><td>1</td><td>0.5</td><td>0.8</td></tr><tr><td>coord.</td><td>c</td><td>d</td><td>c</td></tr><tr><td>γ</td><td>0</td><td>10</td><td>0</td></tr><tr><td>λ</td><td>1</td><td>0</td><td>1</td></tr><tr><td> $\beta$ </td><td>0</td><td>0</td><td>0</td></tr><tr><td rowspan="8">Uniform</td><td> $T$ </td><td>0.9</td><td>1</td><td></td></tr><tr><td> $\mathrm { t o p } { - } p$ </td><td>1</td><td>1</td><td>0.8</td></tr><tr><td> $\operatorname { s p a c i n g }$ </td><td>cosine</td><td>cosine</td><td>0.9</td></tr><tr><td>η</td><td>0.2</td><td>0.2</td><td>log 0.1</td></tr><tr><td> $\operatorname { c o o r d . }$ </td><td>C</td><td>C</td><td>c</td></tr><tr><td>γ</td><td>0</td><td>1</td><td>0</td></tr><tr><td>λ</td><td>0</td><td>0</td><td>0</td></tr><tr><td> $\beta$ </td><td>0</td><td>0</td><td>0</td></tr><tr><td rowspan="8">Binary absorbing</td><td> $T$ </td><td>0.9</td><td></td><td></td></tr><tr><td> $\mathrm { t o p } { - } p$ </td><td>1</td><td>1 1</td><td>0.9</td></tr><tr><td>spacing</td><td>log</td><td>log</td><td>1</td></tr><tr><td> $\eta$ </td><td>0.8</td><td>0.8</td><td> $\log$  1</td></tr><tr><td>coord.</td><td>C</td><td>d</td><td>C</td></tr><tr><td>γ</td><td>0</td><td>0</td><td>0</td></tr><tr><td>λ</td><td>1</td><td>1</td><td>1</td></tr><tr><td> $\beta$ </td><td>0</td><td>0</td><td>0</td></tr><tr><td rowspan="8">Masked</td><td> $T$ </td><td></td><td></td><td></td></tr><tr><td> $\mathrm { t o p } { - } p$ </td><td>0.9 1</td><td>0.9 1</td><td>0.9 0.9</td></tr><tr><td>spacing</td><td>cosine</td><td>cosine</td><td>cosine</td></tr><tr><td>η</td><td>0.05</td><td>0.5</td><td>0.01</td></tr><tr><td>coord.</td><td>C</td><td>c</td><td>c</td></tr><tr><td>γ</td><td>3</td><td>1</td><td>0</td></tr><tr><td>λ</td><td></td><td>0</td><td>0</td></tr><tr><td>β</td><td>0 0</td><td>0</td><td>0</td></tr></table>

223k boards that are orbit-disjoint under the augmentation symmetries (digit relabeling, board flips, and rotations), and the hard subset of the test set keeps the boards with at most 22 clues (4.1%).

Boxoban. For Boxoban, we use the $1 0 \times 1 0$ levels of Guez et al. (2018) with the $\mathbf { A } ^ { \star }$ solutions of Garriga-Alonso et al. (2024) and keep the levels whose solution replays to a solved board. Each sequence is the flattened board followed by 128 action tokens, the solution padded with STAY, and we score the whole action region. Because the official test pool holds only 1,000 levels, we partition the 92k solved levels of the official validation pool once into validation and test sets of 25k levels each and fold the remaining 42k into the 936k training levels, and the hard subset keeps the boards whose solution needs at least 40 moves (19.8%).

Graph coloring. Graph coloring instances are random graphs with a planted four-coloring: 64 vertices in four balanced color classes, with each cross-class edge present independently with probability 0.2, so the mean degree is 9.6, just above the Kesten–Stigum threshold $( k - \mathrm { \dot { 1 } } ) ^ { 2 } = \mathrm { \dot { 9 } }$ for $k = 4$ colors, below which the planted coloring is undetectable. Any proper coloring counts as a solution, not only the planted one. We pass the graph’s adjacency matrix to the model as an attention mask and generate instances on the fly from separate seeded streams for training, validation (2,048 instances), and test (100k). The hard subset consists of 4,096 graphs of 640 vertices at the same mean degree, with the edge probability scaled to 0.02.

During training, we augment Sudoku with the symmetries above and Boxoban with joint board and action rotations and reflections, while validation and test examples are unaugmented. Figure 5 gives the solve rate by difficulty bin and by graph size.

Table 10: Solve rate (%) at the reference NFE for each stage of the tuning procedure evaluated on the test splits and the hard subsets of Table 1, with every sampler selected on the full validation splits, except in the graph-coloring hard column, which selects on the large validation graphs. For masked diffusion, the last row adds the extra ordering and re-corruption settings that replace candidate confidence.
<table><tr><td></td><td colspan="2">Binary symmetric</td><td colspan="2">Uniform</td><td colspan="2">Binary absorbing</td><td colspan="2">Masked</td></tr><tr><td>Sampler</td><td>All</td><td>Hard</td><td>All</td><td>Hard</td><td>All</td><td>Hard</td><td>All</td><td>Hard</td></tr><tr><td>Sudoku</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ancestral</td><td>76.41 (0.33)</td><td>59.22 (0.34)</td><td>76.66 (0.49)</td><td>60.37 (0.42)</td><td>71.90 (0.24)</td><td>52.25 (0.34)</td><td>70.31 (0.29)</td><td>50.75 (0.48)</td></tr><tr><td>Re-corrupt</td><td>94.54 (0.18)</td><td>87.78 (0.40)</td><td>94.77 (0.31)</td><td>88.27 (0.64)</td><td>86.03 (0.24)</td><td>71.75 (0.54)</td><td>80.84 (0.09)</td><td>64.39 (0.21)</td></tr><tr><td>Ψ-sampler</td><td>94.54 (0.18)</td><td>87.78 (0.40)</td><td>94.77 (0.31)</td><td>88.40 (0.68)</td><td>86.03 (0.24)</td><td>71.89 (0.49)</td><td>80.49 (0.13)</td><td>63.95 (0.24)</td></tr><tr><td>+ position conf.</td><td>94.37 (0.18)</td><td>87.47 (0.21)</td><td>95.48 (0.22)</td><td>89.90 (0.48)</td><td>91.23 (0.11)</td><td>81.24 (0.21)</td><td>88.19 (0.19)</td><td>76.56 (0.29)</td></tr><tr><td>+ candidate conf.</td><td>96.27 (0.21)</td><td>92.66 (0.39)</td><td>95.47 (0.21)</td><td>89.81 (0.47)</td><td>91.24 (0.11)</td><td>81.37 (0.20)</td><td>87.96 (0.16)</td><td>76.00 (0.10)</td></tr><tr><td>Boxoban</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ancestral</td><td>83.80 (0.37)</td><td>64.17 (0.57)</td><td>80.00 (0.81)</td><td>55.33 (1.50)</td><td>71.75 (0.91)</td><td>43.50 (1.59)</td><td>76.39 (0.37)</td><td>50.71 (0.58)</td></tr><tr><td>Re-corrupt</td><td>93.53 (0.39)</td><td>81.89 (0.85)</td><td>91.68 (0.63)</td><td>77.35 (1.30)</td><td>84.69 (0.66)</td><td>66.14 (1.45)</td><td>90.62 (0.37)</td><td>75.87 (0.93)</td></tr><tr><td>Ψ-sampler</td><td>93.53 (0.39)</td><td>81.89 (0.85)</td><td>91.68 (0.63)</td><td>77.35 (1.30)</td><td>84.66 (0.65)</td><td>66.01 (1.38)</td><td>90.66 (0.34)</td><td>75.94 (0.88)</td></tr><tr><td>+ position conf.</td><td>94.38 (0.33)</td><td>83.60 (0.78)</td><td>92.34 (0.61)</td><td>78.39 (1.36)</td><td>87.28 (0.80)</td><td>68.25 (1.65)</td><td>91.24 (0.31)</td><td>76.60 (0.70)</td></tr><tr><td>+ candidate conf.</td><td>94.29 (0.30)</td><td>84.14 (0.61)</td><td>92.38 (0.64)</td><td>78.57 (1.42)</td><td>87.30 (0.81)</td><td>68.25 (1.65)</td><td>91.26 (0.31)</td><td>76.60 (0.70)</td></tr><tr><td>Graph coloring</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ancestral</td><td>97.47 (0.05)</td><td>4.70 (0.24)</td><td>97.33 (0.03)</td><td>4.98 (0.25)</td><td>83.17 (0.13)</td><td>0.00 (0.00)</td><td>63.00 (0.54)</td><td>0.00 (0.00)</td></tr><tr><td>Re-corrupt</td><td>99.38 (0.01)</td><td>55.00 (1.01)</td><td>98.94 (0.01)</td><td>48.16 (0.92)</td><td>88.54 (0.33)</td><td>0.58 (0.09)</td><td>86.95 (0.60)</td><td>0.05 (0.03)</td></tr><tr><td>Ψ-sampler</td><td>99.35 (0.00)</td><td>55.10 (1.00)</td><td>98.94 (0.01)</td><td>48.24 (0.94)</td><td>90.64 (0.19)</td><td>0.58 (0.09)</td><td>86.92 (0.56)</td><td>0.05 (0.03)</td></tr><tr><td>+ position conf.</td><td>99.35 (0.00)</td><td>64.49 (1.20)</td><td>99.00 (0.01)</td><td>61.41 (0.34)</td><td>98.83 (0.02)</td><td>19.96 (0.97)</td><td>97.04 (0.03)</td><td>4.94 (0.37)</td></tr><tr><td>+ candidate conf.</td><td>99.35 (0.00)</td><td>64.13 (1.10)</td><td>99.01 (0.02)</td><td>61.41 (0.34)</td><td>98.83 (0.02)</td><td>54.77 (0.39)</td><td>97.44 (0.04)</td><td>4.85 (0.33)</td></tr></table>

![](images/cac9fd254364e235880f460ae8aba2653334a0a35b4c54954dd7c486ddffeac7.jpg)  
Figure 5: Solve rate by difficulty at the reference NFE, harder to the right: Sudoku by the number of clues, Boxoban by the length of the reference solution, and graph coloring by graph size at a fixed mean degree.

## F.6 LANGUAGE MODELING

For all text models, we use the diffusion transformer of Sahoo et al. (2024) with time conditioning, with about 92M non-embedding parameters on Text8 and TinyStories and 214M on OpenWebText, and train every run on 131B tokens. Table 4 lists the remaining training settings.

Text8. Text8 (Mahoney, 2011) is the first $1 0 ^ { 8 }$ characters of a cleaned English Wikipedia dump, over 27 characters, in sequences of $L = 2 5 6$ . We use the standard split into the first 90M characters for training and 5M each for validation and test.

TinyStories. TinyStories (Eldan & Li, 2023) is a corpus of short synthetic stories in simple English. We tokenize it with our own byte-pair-encoding (BPE) tokenizer of V = 1024 tokens, trained on the corpus, and hold one story per sequence of $\bar { L } = 5 1 2 \colon$ the story tokens, an EOS token, and PAD tokens up to L. We keep the train and validation splits of the released corpus and drop the 3.9% of training and 3.6% of validation stories that do not fit. This leaves 2,611,361 training stories and 26,637 validation stories, which we split evenly into validation and test sets. During training, we corrupt and attend to padding like any other token but never score it, so the model does not learn where a story ends. We therefore sample at a known length: each sample takes the padding positions of a validation story, and these positions follow the forward bridge $q _ { s | t , 0 } \big ( \cdot \mid Z _ { t } ^ { i } , \mathtt { P A D } \big )$ from the prior down to $t _ { 1 }$ instead of being shown clean, so the model sees them as it did in training.

OpenWebText. OpenWebText (Gokaslan & Cohen, 2019) is an open reproduction of the WebText corpus. We tokenize it with our own BPE tokenizer of V = 1024 tokens, trained on the corpus, and pack documents into sequences of L = 1024. We wrap each document in BOS and EOS tokens, follow it with a uniformly random number of 0 to 127 PAD tokens, and cut the resulting stream into consecutive sequences of L tokens, so a sequence can hold several documents or part of one, and we do not mask attention between documents. As for TinyStories, we corrupt and attend to padding but never score it, and we exclude it from the likelihoods and the decoded text of App. F.3. The corpus has no official split, so we hold out a random 100k documents before packing and split them evenly into validation and test sets, which keeps the three sets document-disjoint. Like Text8, we sample it unconditionally.

Table 11 gives, at the validation-selected setting of each kernel, the pooled entropy, MAUVE, and GenPPL, together with the held-out NELBO of App. F.3, which does not depend on the sampler.

Table 11: Entropy H (nats), MAUVE, GenPPL, and NELBO (bits per character) at the validationselected setting. Parentheses give the standard error over the two generation seeds, and for NELBO the Monte Carlo standard error. The reference entropies are 7.308 (Text8), 5.811 (TinyStories), and 7.535 (OpenWebText).
<table><tr><td>Dataset</td><td>Kernel</td><td>H</td><td>MAUVE</td><td>GenPPL</td><td>NELBO</td></tr><tr><td>Text8</td><td>Binary symmetric</td><td>6.805 (0.004)</td><td>0.913 (0.001)</td><td>186 (3)</td><td>1.454 (0.004)</td></tr><tr><td></td><td>Uniform</td><td>7.244 (0.008)</td><td>0.896 (0.006)</td><td>492 (1)</td><td>1.428 (0.003)</td></tr><tr><td></td><td>Binary absorbing</td><td>6.873 (0.003)</td><td>0.821 (0.000)</td><td>404 (7)</td><td>1.476 (0.004)</td></tr><tr><td></td><td>Masked</td><td>6.650 (0.010)</td><td>0.854 (0.008)</td><td>272 (3)</td><td>1.384 (0.003)</td></tr><tr><td>TinyStories</td><td>Binary symmetric</td><td>5.676 (0.005)</td><td>0.882 (0.006)</td><td>9.4 (0.1)</td><td>0.658 (0.002)</td></tr><tr><td></td><td>Uniform</td><td>5.682 (0.004)</td><td>0.904 (0.001)</td><td>9.2 (0.0)</td><td>0.493 (0.001)</td></tr><tr><td></td><td>Binary absorbing</td><td>5.800 (0.002)</td><td>0.585 (0.011)</td><td>15.2 (0.0)</td><td>0.684 (0.002)</td></tr><tr><td></td><td>Masked</td><td>5.751 (0.005)</td><td>0.850 (0.009)</td><td>8.0 (0.0)</td><td>0.452 (0.001)</td></tr><tr><td>OpenWebText</td><td>Binary symmetric</td><td>7.047 (0.005)</td><td>0.840 (0.001)</td><td>31.3 (0.1)</td><td>1.490 (0.003)</td></tr><tr><td></td><td>Uniform</td><td>7.020 (0.002)</td><td>0.822 (0.004)</td><td>47.4 (0.0)</td><td>1.236 (0.002)</td></tr><tr><td></td><td>Binary absorbing</td><td>7.168 (0.002)</td><td>0.747 (0.004)</td><td>54.6 (0.3)</td><td>1.499 (0.002)</td></tr><tr><td></td><td>Masked</td><td>7.250 (0.006)</td><td>0.871 (0.013)</td><td>51.0 (0.6)</td><td>1.179 (0.002)</td></tr></table>

Figure 6 gives GenPPL at the points of Figure 4.  
![](images/830e6e2f25c18d9e9b73032433ce42149211c9f79850af84bf3675df414f4ac1.jpg)  
Figure 6: GPT-2 Large GenPPL against token entropy at the points of Figure 4.

Table 2 reports each kernel at the setting where its own validation MAUVE is highest, while Table 12 reports every kernel of a corpus at one entropy, chosen where their interpolated MAUVE is jointly highest. On Text8 the kernels peak at different entropies, but the ranking does not change. On TinyStories, uniform diffusion scores lower at the common entropy, and on OpenWebText both uniform and binary symmetric diffusion score lower while masked diffusion stays best.

Table 12: Common-entropy operating point: the measured sample set closest to the entropy at which the corpus’s kernels are jointly best (Text8 6.80, TinyStories 5.75, OpenWebText 7.25 nats). Parentheses give the standard error over the two generation seeds.
<table><tr><td>Dataset</td><td>Kernel</td><td>H</td><td>MAUVE</td><td>GenPPL</td></tr><tr><td>Text8</td><td>Binary symmetric</td><td>6.805 (0.004)</td><td>0.913 (0.001)</td><td>186 (3)</td></tr><tr><td></td><td>Uniform</td><td>6.924 (0.003)</td><td>0.891 (0.006)</td><td>358 (1)</td></tr><tr><td></td><td>Binary absorbing</td><td>6.803 (0.011)</td><td>0.821 (0.007)</td><td>329 (10)</td></tr><tr><td></td><td>Masked</td><td>6.650 (0.010)</td><td>0.854 (0.008)</td><td>272 (3)</td></tr><tr><td>TinyStories</td><td>Binary symmetric</td><td>5.676 (0.005)</td><td>0.882 (0.006)</td><td>9.4 (0.1)</td></tr><tr><td></td><td>Uniform</td><td>5.756 (0.007)</td><td>0.877 (0.001)</td><td>12.3 (0.0)</td></tr><tr><td></td><td>Binary absorbing</td><td>5.800 (0.002)</td><td>0.585 (0.011)</td><td>15.2 (0.0)</td></tr><tr><td></td><td>Masked</td><td>5.751 (0.005)</td><td>0.850 (0.009)</td><td>8.0 (0.0)</td></tr><tr><td>OpenWebText</td><td>Binary symmetric</td><td>7.288 (0.013)</td><td>0.800 (0.006)</td><td>31.8 (0.5)</td></tr><tr><td></td><td>Uniform</td><td>7.287 (0.010)</td><td>0.808 (0.002)</td><td>96.0 (0.3)</td></tr><tr><td></td><td>Binary absorbing</td><td>7.168 (0.002)</td><td>0.747 (0.004)</td><td>54.6 (0.3)</td></tr><tr><td></td><td>Masked</td><td>7.250 (0.006)</td><td>0.871 (0.013)</td><td>51.0 (0.6)</td></tr></table>

## G SUDOKU SOLVE TRAJECTORIES

We show one sample per kernel on the first Sudoku test board in evaluation order, which has 24 clues, in Figure 7. We generate the samples with the seed-0 models of Table 1, using ancestral sampling at temperature T = 1 and 256 NFE. All four kernels solve this board.

![](images/fda224466adabdc540ddf3c56cd28e69622a2a78ed9aedcf2fd85adb4e9da8d3.jpg)  
Figure 7: Candidate sets of the first Sudoku test board from t = 0.9 to the generated sample at t = 0, with digit d of a set at keypad position d. Bottom: the candidate set of the outlined cell at every step, with the solution’s digit on a light gray band.

## H GENERATED SAMPLES

We show two samples per kernel from the sample sets of Table 2, drawn uniformly at random from the 2,000 of the first generation seed.

## H.1 TEXT8

## Binary symmetric diffusion.

ding enemies somewhat convinced by recruiting the other to plan the new world edition of the book of jonah depends on the most wonderful time luke says jonah sleeps for fight jonathan goes for life and takes a holy life in section three three this version

e existence of the universe is held by many people according to and without personal concern about the future of the universe the aim of the family is a creation of a number of possible worlds including unique events shakti chandra s state gandhbar posits

## Uniform diffusion.

p down as the leader of lady karel shartlov the russian west in one nine nine three and ended soon in gorbachev s farewell to the boy during the dust early spring of one nine nine nine after replacing the chaotic labor strip in one nine nine nine kasparov

shield which was designed by the backcracker see john taylor blind eyed clecks neckble os with the axecoat pins set outside the neckble clerk stand by automatic guards the first background of which was to be extended to the blind sweep strip size mostly a

## Binary absorbing diffusion.

one nine six zero s bill oloyd orca yielded the fairly golden fate setting for a soft lonely appearance with artists in the magazine rogue that good even lee hughes continued the living world bennett s back work following both work that pore brad fox and u

its opposite to the clearing sky due to the dumping of a gase found in exceptionally outstanding appendorenses which combines with the atmosphere such as rydberg thermoboat megasi the temples of samuel m sick and manto benzo by martota sick rock george ler

## Masked diffusion.

rustor that so works usually in any heat production applications but the same application as at closed fuel internal combustion is also less expensive in reducing fuels usually from flowering applications the result is considered by some to be quite genera

sical elements of nature from traditional christian experience two in their nature and helped indicate the effect of so falled jesus came to be found there and like christ with his unmentioned child and the father with the father and divine breat with you

## H.2 TINYSTORIES

## Binary symmetric diffusion.

Once upon a time, there was a yellow bird. The bird lived in a big tree. The bird had a rag on a high branch. The rag was full of pretty leaves and leaves. The bird liked to play with the rag and the leaves. One day, the bird saw a big, red ball. The bird wanted to play with the ball. But the ball was too big for the bird. The bird was sad and did not want to play with the ball anymore. The bird tried to be brave. It flew down from the tree and got the ball. The bird put the ball on the rag and started to play. Soon, other birds came to play with the yellow bird. They all played together and had fun. The bird was not sad anymore. The bird was happy and made new friends.

Once upon a time, there was a mommy and a little girl. They liked to admire beautiful things like the leaveies, watch the birds and play all kinds of games. Today, they decided to create a special use of nature to keep their attention. The mommy pulled out a special bottle from the table. She gave the little girl a glass with water. It was gentle and gentle, and the little girl watched carefully. The little girl looked closely. She watched the birds soar high in the sky and in different ways. It seemed like it was completely so beautiful to fly. The mommy and the little girl watched in amazement. The little girl smiled, and felt so proud. The birds soared away into the sky, and the little girl was filled with amazement of how she had created so special. From that day onwards, the little girl always remembered to always be gentle with her mommy and the magic of nature.

## Uniform diffusion.

Once upon a time, there was a little girl named Mia. She lived in a small house with her mom, dad, and dog, Poppy. Mia was very excited to play with Poppy. One day, Mia found a piece of paper. She wanted to fold it and make a fun bridge. So, Mia showed the paper to Poppy and said, “Poppy, let’s make a bridge!” Poppy smiled and wagged her tail. Mia and Poppy folded the paper into many shapes like animals and stars. They made a beautiful flower bridge with paper. Mia and Poppy were so excited about their flower bridge. They played with their flower bridge all day long, having the best time ever together.

Once upon a time, there was a little girl named Sue. Sue loved to play outside with her friends. One day, Sue and her friends were playing in the mud. They were having a great time, but their clothes and feet got dirty. The clothes were dull and not very nice. “I want to change my game,” said Sue. She went inside her house and got her wet clothes. She left them in the sun to dry them off. But the clothes were still dirty and Sue left the wet clothes outside. The sun was hot and the clothes became wet and dirty. Sue’s mom came to the room and saw the dirty clothes. “You have to change into clean clothes now,” said her mom. Sue was sad because she wanted her clothes to be nice. She went back outside to play with her friends and use clean clothes. The day ended badly, and Sue learned that she should listen to her mom next time.

## Binary absorbing diffusion.

Once upon a time, there was a grumpy cat named Tom. He lived in a small house with his friend, a little girl named Sue. They liked to play and eat chores. They were very happy together. One day, Sue found a big chain in the yard. She picked it up and showed it to Tom. “Look, Tom! I found a chain!” Sue said. Tom looked at the chain and saw a big door on the other side. It was like a secret door. He liked the chain because it was all by himself! Sue saw Tom looking at the big chain and asked, “What’s wrong, Tom?” Tom said, “I don’t know. It lost my chair.” Sue looked at the chain and saw that it was gone. She said, “Don’t worry, Tom. We will find your chain and eat it together!” So, Tom and Sue looked everywhere for the chair. Finally, they found the chain under the big tree. Tom was so happy! He hugged Sue and said, “Thank you, Sue! You are the best friend ever!” From that day on, Tom wasn’t grumpy anymore. Sue and Tom played together every day, and they always had lots of fun.

Once upon a time, there was a little dog named Spot. Spot liked to play and run in the park. One day, Spot saw a big tree. The tree had a tie all around it. Spot was very happy and so big. Spot went up to the tree and asked, “Can I play with you, tree?” The tree nodded and said, “Yes, Spot, you can play today.” Spot was very excited and started to run and jump. After some time, Spot was tired. He sat under the tree and licked his fur. Spot felt very fine and happy. Spot and the tree became good friends. And they played together in the park every day.

## Masked diffusion.

Once upon a time, in a small nation, there lived a cute cat named Lily. Lily had a best friend, a dog named Max. Lily and Max were always happy and liked to play together. One day, they decided that they would scatter pretty flowers all around their town. Lily said to Max, “Let’s make our nation beautiful!” Max thought it was a great idea. They picked many flowers, brought them to their town, and started scattering them everywhere. Soon, the nation was full of colorful leaves, flowers, and big trees. Lily and Max were very proud of what they did. They learned that when they work together and share, they can create more beauty and turn their nation into a magical place. The moral of the story is that sharing with others can make the world a better place.

Once upon a time, there was a little red car. This car loved to drive fast. One day, the car saw a big, distant hill. The car thought, “I want to go to the top of that hill.” The car tried to go up the hill, but it was too hard. The car was sad. It could not drive up the hill. The car said, “I am sad. I want to go to the top of the hill.” Then, the car saw a big truck. The truck said, “Why are you sad, little car?” The car said, “I can’t go up the big hill.” The truck wanted to help. It pushed the car to the top of the hill. The car was very happy. Now the car could drive down the hill. The car went fast and had fun. The car and the truck became good friends. They drove every day. They had lots of fun together. The end.

## H.3 OPENWEBTEXT

## Binary symmetric diffusion.

neighborhood neighborhoods. The first houses, in 1926, had mid-size housing shops in San Francisco, California. As San Francisco’s population grew, new houses were built. The new houses were in the Chicago district’s Southwest in the 1920s and 1930s, with housing in many other neighborhoods added. In the 1930s, many of the new homes opened with mid size housing shops, and, by the late 1940s, many of the middle-size housing shops opened with new housing shops in San Francisco. Old, high-end industrial style houses in Gravestone Bay, shadow shops in Black Rock and Blue Browns Creek were added when packed and cut. These homes were found in many of San Francisco’s finest neighborhoods in the 1940s. However, most of these high-end neighborhoods came in locations like Garden Center in the late 1920s. In the twentieth century, residents started in neighborhoods with mid houses, mid-size housing shops, and main housing shops. Some middle-size neighborhoods had mid-size housing shops. Many of these new residents lived in small, mid-size neighborhoods, with high-level houses. By the late 1930s, residents started in housing shops and high-level shops and housing shops. Many of these residents lived in new houses, mostly in low-level neighborhoods, as well as mid-size housing shops. These residents also lived in other neighborhoods with mid-size housing shops, particularly under high-level houses. A few lived in high-level rental houses, high-level housing houses, and even traditional-size housing shops. Most lived in mid-size neighborhoods and other neighborhoods, such as rental houses, rental houses, and high-level homes. Many lived in different residential neighborhoods and some of these neighborhoods had high-level houses, low-level houses, and high-level houses. Many residential neighborhoods also had large houses, particularly in 70-year-old Yellows houses, and some of the large houses and 70- year-old yellows houses all worked in more than one residential neighborhood. Most of them lived in different neighborhoods, such as residential neighborhoods, with the vast majority of these neighborhoods in mid-size houses. Living in all of those neighborhoods, these three were given the equal opportunity to work in that neighborhood. Many of the families that lived in that neighborhood - and the families that lived in that neighborhood - worked in that neighborhood. Many of them worked in that neighborhood, and these ad

2004, the Washington Post reported that the Democratic Party, among Barack Obama’s closest allies, is considered by some to be one of the most viable candidates in history. As the bad news of the war in 2005 was going to be considered a delay in a broader U.S. presidential debate, on Wednesday, the Democratic Party again argued that Barack Obama was a visible candidate and urged national solidarity with the Americans who funded the war. Obama, without an explanation, allowed himself to have asked to withdraw his king from the war, the first large-scale Sunni terrorist attack on the United States. In February 2002, he declared Afghanistan the country of the past. A few months later, when Obama was re-elected by George W. Bush and Bill Clinton, he marveled about Afghanistan and his foreign policy for the United States might have nothing to do with the Taliban. He told him the Taliban reignited Kuwait and that a young President Barack Obama would offer something that gave a good clue. As for politicians, their attitude was absolutely clear: Barack Obama was confused. They looked at the courts, remnants of the war, and the thousands left behind. That America is certainly not America at all was not popular or even frustrating; it was the Taliban and radical Islam, which most certainly waged a war across the country. Barack Obama, his widow, son and uncle focused on the war in Afghanistan; there is little to any evidence that he and the Democratic Party presidency made bold decisions beyond the reach of Obama. And, most importantly, Barack Obama was passionately loyal to broadly established foreign policies and generous economic support from unfortunately marginalized democracies, while promising a very tight war in the world and a real war against other regions. And America’s foreign policy was balanced by its political hegemony. Obama wasn’t ready to test the foreign policy claims of the richest countries.

Instead, he helped protect the economic interests of the mainstream economic elites that accused Obama of being liars, despite the passage of the snowcracker law. Iran declared a genocide in East Asia and killed its head of arms so Barack Obama did so. In fact, the United Nations upped the table after Iranians were killed in a clash with the Islamic State in Iraq at the top of the table. Obama was just as keen as the pundits about cooking it up. When it came to warfare, Washington Post reporter Sean Hannity’s May 2011 message was like this: In an interview, roughly the one time he was present on the war – the African drone, Hannity said: ’If you go to Arkansas, you

## Uniform diffusion.

you need guidelines to address problems that many dependent firms still have, as do most other commercial units. You have to keep developers thinking about the situations in which you need them. Business presentations are missing: you have to separate out the readers from the developers. They need help. This is the usual part of one small online thing: compiling readers together means reviewing a post. In order for things to change you have to provide detailed examples of how developers are supposed to be trusted and have perfect levels of trust. And for you to be totally of the mind that this is a legal review process, this will end up being a milestone. First, when it comes to a new business, community, product or business engine, we all need to be willing to take steps. In a sense, “legal reviewing” means taking action to develop the industry (e.g. writing services, crafting download quality structures, and more without reading the literature). Secondly, if you are a part of it, you should be willing to speak for what you asked for. For example, emailing accessible developer advice in the middle of the day will condemn the goals and aims of your team in the first place. It’s the most academic thing to go along with it however you do it. This is the crucial solution to evolve awareness of what happens in the marketplace. Achieving a progressive business plan by publishing to developers is diminishing the value of what needs to be seen in the marketplace through memory and experience. Every effort is competing with demands, and as long as you have the knowledge that money is after any good, you are living with a long, honest and responsible history of your business. Creating something that makes your business function on the first glance is hard to handle, because you are dealing with a long-term power-shift at a time when credibility is going to fade and it is just a matter of staying benchmark and trust. That’s not to say that there aren’t global shifts towards new values. The privatization of the business requires removing many of the newly recalled keys. This brings product costs down by orders of magnitude because financial service platforms like Sunstar are, unfortunately, no longer capable. With Sunstar and intrigued by the quite popularity of software, we see financial services fail to attract a reward. They are always being used with new-recalled keys. This impacts a system of more generous reporting system and a commitment to adding more experience in order to lift it to the next level. This is a serious problem for us in the world of the product. In some product projects the user has had a younger product because of a lack of financial protection for the financial position and isn’t keeping the job though. Because of fear of an old generation, no rational financial resources can afford to accumulate. Because great projects can re-energy capital, they must be protected by a large corpor

ting to the end... “I know this group of winners very well. They have shown special interest in how to win.” They were very emotionally convincing. “I don’t want to die.” (That’s one of the pictures he had written for me. I’m in the full picture.) It wasn’t the best time, so I gather and think, “Want to breathe out here.” Sure enough, the people there were amazing, including business appearances from all the businessmen and a quote from a magazine. “I hope I can learn more,” he says. “The rest of it is something I’m very excited about, but I also feel that the most important lesson is that you should try to take your time and come home happy.” “You don’t worry though,” he says. “You pay yourself.” “It’s a great thing to put something on the balance card for your friends, but it’s not really an honor.” “Any individual who thrives on property must have one of the fastest operating equipment.” “There are no prizes,” he says. “You play hard, and this is no different. I think they need to have some faith in winning. Unfortunately they have not enough.” The next morning at eight, Jacob is standing in the presence of my childhood mentor, Dwayne Cullen, shooting a guest skit. He’s out of the club as soon as I get to it. “I mean, I’m not starving, I’m working, and I’ve got a lot of time. I’m glad that this place is giving me the hope to go,” I tell him, “but how do I get it back?” “I’m glad that this is my place,” he says. It’s where I want to live.” He just graduated from New York County College. I show my friend he has to go. But then he has to stop him. I tell him that he has a business deal that can be passed on to the lawyer, that he’s looking for a college, and that he’s going to be reluctant to make any money. “Don’t worry about anything. I’m not sure who will pay me,” he says.

“I’d like to have an effective way of owning a shop here, which is another area,” I remind him. “I’m not pro-gay. I’m doing what I can to help. Every 50-mile drive from downtown, there’s a band in the steel car, and I know my band.” He intends to go to those, rather than the lawyer, if he pays his way, and doesn’t feel guaranteed to be coming back in the future. “I hate to say that I’m a businessman, but that I’m an economist. That’s a lot of work,” he says. “The interesting part is that I also have access to schools. There are a lot of schools that are useful. This is a very powerful position for young kids. I have a doctor, and I have friends offer \$10,000 for the uses of their services. They are very prominent to me. They have my best friends as well. You go to something like 6 a.m. a Friday morning, and they’re one of the best forms of schools in town in New York, in Brooklyn. And you know what, guess that was just a joke, at one point. Now I can get a slice of my money. It’s a b

## Binary absorbing diffusion.

negative conclusions about the risks of climate change or conflicting conclusions about climate change about the relationship between the causes of climate change and rational conclusions about particular risk factors that measure the effect.” Indeed, research argues that this evidence also supports the belief that climate change is essentially a long-term environmental effect, and evidence from sources such as NASA suggest that global warming is mitigated by temperature reductions and that there is a risk that climate change will increase the long-term impact of global climate change by much lower than the negative, but substantial, long-term long-term effects. Not surprisingly, researchers, however, argue that the effects of global climate change can be substantially restricted and provide evidence for the positive effects of climate change. The research was published in Nature. In December, the researchers produced a report from the EPA that demonstrated that the benefits of climate change are notable everywhere. The study showed that the effects often occur in many areas... it is a matter of ideology that climate change is widespread. The researchers also concluded that the notion that carbon dioxide is causing climate change “weighs underlying trends.” “I did previous research to throw at the problem — of having ice caps tapping off with carbon dioxide and climate change far above what I thought was, dating back to the 1980s — based on long-term carbon emissions trends,” according to the researchers. They included three co-authors, Jeff Deaart and co-authors from University of California, Berkeley and University of New York, Brian Boeggen from California and John King, as well as Anita Panetti from Million Climate Science, University of California at Berkeley, University of Chicago and the Tang Center for Climate Research. Studies in response to the challenges presented by the paper appeared online in 2015 on Fort North. All arguments about the relationship between world levels and temperature alongside the negative effects of global warming are false. The paper was published in the journal Climate Change in January 2015. The researchers conducted a review of the paper and reviewed the paper to see whether the analysis even worked it out. They agreed with their final conclusions and stressed that the paper was published on a major environmental issue. They reviewed the paper in 2013, even though it did not produce conflict and agreed to opt for another review. Lab data The research done by Monica Mroinsky and Chor Camillo was inspired by UCS research that seeks to establish the argument that global climate change is affecting long-ter

ironmental impacts, including the drought and environmental impacts, that have apparent state and local impacts on public health. Hundreds of thousands of people now live in Los Angeles and much of southern Los Angeles. Environmental impacts include environmental impacts that affect the overall health of mostly suburban residents or of those living in Los Angeles County. Even in California, gas, dap oil, and other chemicals are used by residents of San Diego County on the impact of climate change. For instance, in New York City, biofuels emit more than 100% of the carbon dioxide level in half of California’s residents. Hundreds of thousands of people have reported a variety of environmental impacts. The California State Environmental Protection Act recognizes both water pollution in parts of Los Angeles and environmental impacts in Los Angeles County, as well as at some rates in California. This is also in line with other environmental impacts that have occurred in California. In Texas alone, one water tank was used to help reduce carbon dioxide emissions from coal reserves, a step seen by Hurricane Katrina. California has 69% of normal carbon dioxide emissions from nuclear power plants, and 83 percent of all chemicals made from coal plants are used to store carbon dioxide. More than a third is from gas, and about a third is from New York. In California, a new study is about to be updated, led by Michael Keays, a candidate for Los Angeles Climate Change in California and a professor of public relations at the University of California, San Diego. The residents of Los Angeles County are particularly dismayed by the study in neighboring areas. In Los Angeles, former Assistant U.S. Attorney Michael Brown of the Los Angeles County Department of

Public Health Services conducted a new study of environmental impacts in Los Angeles. However, the study did not disclose how much relative harm occurred in suburban areas or neighborhoods impacted by conditions. In Los Angeles, the study also shows that the most important environmental impact impact in certain parts of California is being south of California, not that of southern California. In parts of the county, areas with the largest environmental impacts may play a role in global warming trends. Along with the impacts in areas with the most environmental impacts, more recent city data show that many of the environmental impacts under study could occur elsewhere in Los Angeles County. As a result, the environmental impacts are expected to escalate more widely in parts of California, stretching over mr

## Masked diffusion.

vailable more resources in the country, by giving them to the environment. Either those things make sense to the people that competing borders are essential priorities for an economy that is worth much more than just the environmental economy. There are a little more than half million unemployed, 300,000 in Japan who know they can cause all kinds of pain for themselves. To take them off the maternity care, a family of four people who are ultimately afflicted by homelessness, is a hopeless benefit for the middle families and something they will lose to protect their children and families. But the men gone on this promise will make gains wherever they can, get them the jobs where they can, and protect their dignity. Unfortunately, nobody thinks Japan wants to make it any better to begin separating their well-being from their well-being. These people have lost any interest in being exactly hybrid cultures. Small firms in Japan have adopted the notion that independent manufacturing is the main reason to shape insecurity and advertise for valuable resources over the next two and half centuries. Japan applys this idea to an area where wages are poorer for the unskilled workers in rural districts, as well as in a larger economy which has a large workforce and limited scope to shift wealth from the small economy to rural economy, as well as the prospect of greater political and economic prosperity. Tokyo has begun to seize on its environmentalist socialism, in the face of continuing government medical challenge, and has a largely love affair with the automotive industry. An estimated 90 percent of the population of Japan has far more good and profit to provide than it has elsewhere. Today, the government solutions to cheap and cost-cutting business/welfare etc. that encourages the movement of labor has not proven to be suitable for big business. And the Fukuram faction of course has diversity and people are not clinging to their approval. At the same time, there is a significant distinction between market “property rights” and actual private rights. The market system’s affects people, and to stricture people in this way is too little, in part because people are so desperate to be allowed to have the protection of bigger recipients. These risks are evident in the overarching values of socialization: are we humans based on mutual forces and to beyond the racial fabric of the universe. Ministers are encouraged that there seems nothing left. Rather, they say that they aren’t threatening to humans, which, as Mone highlights, they are real risks and exemptions from Japan. Its permanent “cultural order” will change the way people live. They argue that they prefer to benefit to themselves, they are increasingly concerned that those who did not have rights to what had been ’personal and taxpayers’ land’ will be judged

visit. Frankenstein Night Races The story of the Boxhorns: A.J., who’s on a hard run to return to a normal life. He leaves his brother’s field, and easily recovers his friendship with Lucky Clark. Frankenstein Night is a movie about A.J. looking for a living as a sophomore. A.J. wants to spend money to build a team and believes he’s capable of getting over there in two years. A.J. decides to return to Thailand and finish the season. In the beginning, he had a crazy time, but he doesn’t want to come back. Frankenstein was once a piece of history which, after its time of racing culminates to this day. August Perriles, the best runner from 1984-1988. Another genius TBP-man. Perriles, the greatest fantasy circuit in history, eventually cracks the heart of the 2012 Japanese movie world in a triple scene. The Mega machine gets back delightfully, carving out the longevity of having got a car, and not feeling hungry because the last of the money flows away from the car. He gets busy being silly, with a car that once had his money. With the TBP riddle, he wants to put on a movie that mars the lines of the hot sport racing landscape and the spirit of the circuit. 9. JFR (1984) (worked by Don Beck) Richard Duffy: Distance, 1,727.35 km Weight: 415 lb in length, two cranes, legs: 60,032. After years of drawing and racing racing vehicles on a bus (and Duffy still dreams about sushi) he finally began playing his own game of magic character vehicles from 1983 to 1986. The game expired in the mid 80s and has a unique timeline of 1987. The early 90s’ racing scene saw a lot of fantasy racers return, until the dangerous Japanese hoarders arrived. Jay-Hirst : Height: 4.21 mm Weight: 110 lb in length. A movie about a hoarder built over 100 miles. 10. Stadio Rueville: Miles 1985 to 2000. Before 1984 non-Japanese circuit. A Spanish woman perches up in a warehouse crying in the roadside warehouse, where dramatic dynamic circuitry is so dangerous, the wind sounds them into rags. JFR drops a mythical mountain of creatures watching the bus. JFR decides to begin the one-shot. 1985 Halle: Distance, 29.7 km Weight: L, bike ride: 1.8 meters. Saga is quite something, for what it’s worth, repetitively fun. Jeffrey Carwich, along with Samuel Zeiss, dabbles in being the world’s greatest