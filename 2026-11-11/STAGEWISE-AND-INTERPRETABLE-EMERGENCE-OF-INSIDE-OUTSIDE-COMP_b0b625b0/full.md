# STAGEWISE AND INTERPRETABLE EMERGENCE OF INSIDE–OUTSIDE COMPUTATION IN TRANSFORMER-BASED LANGUAGE MODELS

Jack T. Parley

Institute of Physics

Ecole Polytechnique F <sup>´</sup> ed´ erale de Lausanne (EPFL) ´

Lausanne, Switzerland

## ABSTRACT

Variability in syntactic structure and parsing ambiguity are fundamental aspects of natural language which modern LLMs have to learn to deal with proficiently. Previous “post-mortem” studies have shown that fully trained Transformers indeed approximate canonical parsing algorithms such as inside-outside, but little is established theoretically about how this is actually learned. Here, we make a first step towards addressing this by studying the learning dynamics in synthetic context-free grammars with variable parse tree topology. The stagewise learning curves are fully interpretable, in the sense that all of the loss stages a) can be computed analytically, and b) can be clearly associated to the representation of a hidden latent tree variable, which allows to lift ambiguity and utilize information from wider contexts. By probing representations, we visualize the emergence of inside-outside computation across depth and during training: while during early stages local (bigram/trigram) information is used and involves only a few initial layers, gradually the Transformer develops an encoder-decoder structure where the most abstract latent information is encoded in the middle, and the latter half of the network is used to progressively refine the final token prediction. This organization of computation is reminiscent of findings in real-world LLMs, and importantly applies here to a synthetic model which allows full analytical control of the data statistics, thus opening the door to theoretical progress.

## 1 INTRODUCTION

Natural language is syntactically ambiguous. Grammars have evolved to optimize efficient communication while balancing complexity and ambiguity (Hahn et al., 2020; Gibson et al., 2019), but as a result many commonplace linguistic constructions are inherently problematic. A classic example in English is prepositional phrase attachment: the phrase “I saw him with the binoculars”, has two equally valid interpretations, each of which corresponds to a distinct parse tree. This is an example of unresolvable ambiguity, at least at the sentence level. For an LLM predicting the next token, there are plenty of examples where the sentence may be locally problematic, but where this can be resolved given sufficient grammar knowledge. For example, in the simple example of Subject-Verb agreement sketched in Fig. 1a, attending to any of the closest nouns leads to a wrong prediction; if the entire sentence is parsed correctly, however, the subject is identified and the spurious local cues cease to be problematic.

There is a vast amount of work, which we will not review here, inspired by the question of how modern LLMs deal with aspects of syntax. Many works have emphasized how Transformers struggle with syntactic generalization (Yao & Koller, 2022), while others have shown the rich representation of syntax by probing activations in state-of-the art models (Manning et al., 2020). Studying such questions cleanly “in the wild”, however, is notoriously difficult. Although some successes have been reported in identifying specific syntax-related circuits (Ferrando & Costa-jussa, 2024), recov-\`

![](images/f408e7cac302cc7eba5dff07a3ed487e1ad6bff03bc547f36fa396c49245df92.jpg)  
(a) Inspiration: natural language syntax

![](images/aa9308c1cb5a8073589eee932f101448ba2b5511089d2bfb70f30e15066116cf.jpg)  
(b) Inside-outside computation in the VT-RHM.

![](images/c856e8bceba0400738162dae6d66369cff332934f2206ea8b455edba030dfb40.jpg)  
(c) Stagewise learning curves of GPT-2 style Transformers.

Figure 1: Inside–outside computation and stagewise learning. (a) Example illustrating the importance of hierarchical structure in natural language, where attending to local cues would lead to an erroneous completion. (b) Illustration of the inside-outside algorithm at work in a depth $L = 3$ VT-RHM. While building the hidden latent structure (inside pass), due to local ambiguity spurious nonterminals are found, which would lead to erroneous predictions. This has to be “cleaned $\boldsymbol { \mathbf { u } \mathbf { p } } ^ { \prime \prime }$ by an outside pass: in the case shown, the alternative tree topology (in red) is found to be spurious due to its grammatical incompatibility with the first half of the sentence (the top rule cannot be completed). Mathematically, the algorithm proceeds by building up inside tensors $M ^ { ( l ) }$ , then progressively integrating this information into outside tensors $O ^ { ( l ) }$ (see text). The case sketched corresponds to Stage 3, as it uses all layers of rules. (c) Test cross-entropy loss on the ternary-tree sentence completion task, for GPT-2 style Transformers trained on fixed-tree or varying-tree training sets (see Fig. 2). Horizontal plateaus indicate the theoretical loss values studied in Sec. 5. The stars refer to the checkpoints used for the probing measurements, chosen to be representative of each learning stage (their precise location within each stage does not affect results qualitatively).

ering and interpreting such circuits more generally remains challenging, often requiring manual intervention and human judgment (Ameisen et al., 2025; Lindsey et al., 2025).

Mathematically, if we model language with a probabilistic-context-free-grammar (PCFG), the question can be phrased as whether and how LLMs learn canonical parsing algorithms to deal with syntax. PCFGs, originating in formal language theory (Chomsky, 1957), simply define a set of grammar rules with a corresponding probability. In computational linguistics, the associated canonical algorithm for dealing with syntactic ambiguity is the inside-outside algorithm (I-O) (Baker, 1979). From an inference perspective, this constitutes the Bayes-optimal algorithm to perform masked token prediction given data generated from a PCFG (Zhao et al., 2023).

Some previous works have attempted to ask precisely the question of whether and how inside-outside is implemented in Transformer (Vaswani et al., 2017)-based language models, mainly (Allen-Zhu & Li, 2025; Zhao et al., 2023). (Allen-Zhu & Li, 2025), in particular, performed “post-mortem” interpretability analysis on the attention patterns of the language model, showing that it mimics dynamic programming (i.e. inside-outside), but sheds little light on the learning dynamics or the organization of latent information and outside computation across depth. In a recent paper, (Zhou et al., 2026) does study latent structure across depth, but the experiments were limited to (next-) token prediction on a fixed-tree topology. Finally, and perhaps most importantly, although all these works consider some form of synthetic grammar none provide any analytical/quantitative control over the generative process, i.e. the data statistics induced by the grammar or associated language.

As an initial step towards filling this gap, we study here the learning dynamics of a family of synthetic PCFGs introduced in (Parley et al., 2026). The key ingredients of this model, which is an extension of the model of (Cagnetta et al., 2024) to varying-tree topologies (thus allowing for syntactic ambiguity), are twofold: it is layered, and the grammar is built by sampling rules uniformly without replacement from a finite pool of nonterminal symbols. These two ingredients are key to opening the door for interpretability: the randomness of the rules allows for analytical calculations, while, together with the layered structure, a hierarchy of correlation strengths is built into the grammar, in turn inducing a stagewise interpretable learning of the token-prediction task.

Building on this interpretable stagewise learning, we will show that with these very simple ingredients, our model of data is rich enough to recover some salient features of the way language computation has been found to be organized in LLMs. While during the early stages of training purely local statistics (bigram/trigram) is used and only the initial layers of the Transformer are active, gradually the Transformer develops an encoder-decoder structure where the deepest latent information is encoded in the middle, and the latter half of the network is used to progressively refine the final prediction. This echoes the general picture which has emerged empirically regarding the functional differentiation of layers in Transformer language models: early layers tend to encode local and surface-level information, while intermediate and deeper layers progressively integrate more abstract, contextual, and hierarchical representations (Tenney et al., 2019; Jawahar et al., 2019; Kervadec et al., 2026).

Although inside-outside computation clearly does not account for all of natural language processing, what we wish to emphasize is that we recover this rich behaviour in a synthetic model which allows for complete analytical/quantitative control over all statistics, and in particular allows to compute all correlations relevant for learning. This was done in (Parley et al., 2026) already in the context of supervised learning. We will thus argue that our data model can potentially open the door to building quantitative theories of many interesting phenomena related to syntax and parsing in LLMs.

## 2 VARYING-TREE RHM

We give here only a brief and informal presentation of our family of synthetic grammars, which was presented formally and in great detail in (Parley et al., 2026). The Varying-tree Random Hierarchy Model (VT-RHM), defines an ensemble of layered PCFGs of depth L. To build a single grammar instance G, we introduce as

$$
\mathcal { N } ^ { ( 0 ) } , \mathcal { N } ^ { ( 1 ) } , \ldots , \mathcal { N } ^ { ( L ) }\tag{1}
$$

the sets of symbols at successive levels of the hierarchy, with $\mathcal { N } ^ { ( L ) } \equiv \Sigma$ the terminal vocabulary. Throughout, we take

$$
\begin{array} { r } { \left| \mathcal { N } ^ { ( \ell ) } \right| = v , \qquad \ell = 0 , \dots , L , } \end{array}\tag{2}
$$

so that every level contains v possible symbols.

For each level $\ell = 1 , \ldots , L$ and each parent symbol $z \in \mathcal { N } ^ { ( \ell - 1 ) }$ , the grammar contains $m _ { 2 }$ binary rules

$$
z  ( a , b ) , \qquad a , b \in \mathcal { N } ^ { ( \ell ) } ,\tag{3}
$$

and $m _ { 3 }$ ternary rules

$$
z  ( a , b , c ) , \qquad a , b , c \in \mathcal { N } ^ { ( \ell ) } .\tag{4}
$$

For every parent z, the grammatical pairs $( a , b )$ and triples $( a , b , c )$ are sampled uniformly without replacement from the $v ^ { 2 }$ and $v ^ { 3 }$ possible combinations, respectively.

We stress that, conceptually, we are explaining how a single grammar instance is sampled; once the grammar is given, one then can sample sentences (strings of tokens) from the grammar by expanding each nonterminal symbol according to one of its associated rules. This process starts by selecting (with uniform probability $1 / v )$ a root symbol from $\mathcal { N } ^ { ( 0 ) }$ , and expanding across $L$ layers until a string of terminal symbols is produced. For simplicity, we consider these rule probabilities to be uniform, i.e. each binary rule has a probability $p _ { 2 } / m _ { 2 }$ , and each ternary ternary rule $p _ { 3 } / m _ { 3 }$ , with $p _ { 2 } + p _ { 3 } = 1$ so that the PCFG is normalized. $\{ p _ { 2 } , p _ { 3 } \}$ , therefore, set the overall probability of binary vs ternary branchings. In general, the sentence length d will fluctuate in the integer interval $[ 2 ^ { L } , \dots 3 ^ { L } ]$

The final important parameter to define is $f ,$ which we use to parametrize the number of rules following $m _ { 2 } = f v$ and $m _ { 3 } = f v ^ { 2 }$ . Importantly, $f$ will also control the level of local ambiguity of a grammar instance. To see this, notice that $f$ corresponds to the average fraction of grammatical binary/ternary tuples (i.e., that can be derived by some parent symbol) with respect to the total set of $\boldsymbol { v } ^ { 2 } , \boldsymbol { v } ^ { \mathrm { 3 } }$ distinct tuples. Therefore, $\mathrm { i f } \left( a , b , c \right)$ is chosen as a ternary rule, both $( a , b )$ and $( b , c )$ can also be present in the grammar with probability f. If, in the final sequence, they are by chance followed or preceded by the appropriate symbol $( c \mathrm { o r } a ) , ( a , b , c )$ will appear fortuitously, causing ambiguity vis-a-vis the genuine occurrence of\` $( a , b , c )$ as a ternary rule.

## 2.1 CURSE OF DIMENSIONALITY

Having defined a grammar instance ${ \mathcal { G } } ,$ the associated language is the set of all terminal strings x such that $P _ { \mathcal { G } } ( \mathbf { x } ) > 0$ , where this probability is obtained by summing over all possible derivations of $\mathbf { x } ,$ each weighted by the product of the probabilities of the production rules on the tree. The curse of dimensionality (Bishop, 2006) is illustrated here by the fact that the size of this language grows exponentially in the sentence length, hence doubly exponentially with depth $L .$

To get a sense of numbers, for the grammar parameters considered in the experiments below $( L =$ $3 , \ v \ = \ 8 , \ m _ { 3 } \ = \ 1 6 )$ , just considering the tree topology with only ternary branchings yields a total number of sentences $v m _ { 3 } ^ { 1 3 } \ \sim \ 1 0 ^ { 1 6 }$ , orders of magnitude beyond any training set size we can realistically provide to the Transformer. Learning a self-supervised task such as predicting the last token (see below), therefore, will only succeed in a sample-efficient manner if compositional generalization occurs, i.e. if the Transformer exploits the hidden latent structure of the grammar. Besides the number of generated sentences, the number of distinct tree topologies also grows superexponentially in depth (Parley et al., 2026): for $L = 3$ , this number is already $N _ { t } = 2 5 6$

## 3 LEARNING TASK: SENTENCE COMPLETION (LAST-TOKEN PREDICTION)

To study the acquisition of inside-outside, we will consider a self-supervised sentence completion (last-token prediction) task. That is, given a sentence prefix $x _ { 1 } , \ldots x _ { d - 1 } ,$ predict $x _ { d }$ . No information is given regarding the hidden latent variables. Last-token prediction can be seen as a special sub-case of a more general next-token prediction task; we focus on it here due to its increased interpretability properties.

## 3.1 INSIDE-OUTSIDE ALGORITHM

Given a PCFG, the inside-outside (I-O) algorithm (Baker, 1979) is Bayes-optimal for masked-token prediction (Zhao et al., 2023). We provide the full details of the tensorial implementation of I-O for the VT-RHM in App. B, sketching here only the main idea of the algorithm. Schematically, for a given sentence the inside pass recursively builds up probabilities of substrings conditioned on an “anchored” nonterminal variable. The substring is defined by its span, with start position $i = 1 , \ldots d$ and span length λ. The corresponding inside tensor thus contains the inside probabilities

$$
M _ { i , \lambda } ^ { ( \ell ) } [ z ] = P ( x _ { i } , \dots , x _ { i + \lambda - 1 } | z ) ,\tag{5}
$$

that is, the probability that the observed substring of length λ starting at position i is generated by the latent symbol $z \in \mathcal { N } ^ { ( l ) }$ . Note that the inside tensors $\mathsf { \bar { \{ M ^ { ( l ) } \} } }$ inherit the layered structure of the

grammar, and are built bottom up starting from $l = L$ (token level), see also the sketch <sup>1</sup> in Fig. 1b. In our context of last-token prediction, we initialize the token-level tensor as $M _ { i , 1 } ^ { ( L ) } [ z ] = \delta _ { x _ { i } , z }$ for $i = 1 , \ldots d - 1$ , and uniform $M _ { d , 1 } ^ { ( L ) } [ z ] = 1 / v ,$ ∀z for the final masked position.

To obtain the Bayes-optimal prediction of the masked token, one then performs an outside pass, computing (from root to leaves) quantities of the form

$$
O _ { i , \lambda } ^ { ( \ell ) } [ z ] = P \left( \pmb { x } _ { \backslash ( i , \lambda ) } , z \right) ,\tag{6}
$$

The notation $\scriptstyle { \pmb { x } } _ { \backslash ( i , \lambda ) }$ indicates the surrounding “context” of the span $( i , \lambda )$ , i.e. the rest of the sentence excluding $x _ { i } , \ldots , x _ { i + \lambda - 1 }$ . The outside tensor thus aggregates information from surrounding context allowing to refine (and correct) what was found in the inside pass. Ultimately, for last-token prediction one is interested in the quantity $O _ { d , 1 } ^ { ( L ) } [ z ]$ , from which one can derive the distribution <sup>2</sup>

$$
P ( X _ { d } = z \mid x _ { 1 } , \ldots , x _ { d - 1 } ) = \frac { { \cal O } _ { d , 1 } ^ { ( L ) } [ z ] } { \sum _ { z ^ { \prime } \in { \cal N } ^ { ( L ) } } { \cal O } _ { d , 1 } ^ { ( L ) } [ z ^ { \prime } ] } ,\tag{7}
$$

With this, one obtains the Bayes-optimal last-token predictor. To characterize our datasets, we average over sentences and consider the conditional last-token entropy of the distribution induced by a given grammar instance $\mathcal { G } _ { : }$

$$
{ \mathcal { L } } _ { \mathrm { o p t } } = - \mathbb { E } _ { \mathbf { X } \sim P _ { \mathcal { G } } } \left[ \ln P _ { \mathcal { G } } ( X _ { d } \mid X _ { 1 } , \ldots , X _ { d - 1 } ) \right] = H _ { \mathcal { G } } ( X _ { d } \mid X _ { 1 } , \ldots , X _ { d - 1 } ) ,\tag{8}
$$

which constitutes the information-theoretic lower bound on the cross-entropy loss of any last-token predictor.

We finally note that in (8) we have dropped the dependence of $\mathcal { L } _ { \mathrm { o p t } }$ on the grammar instance. This is because our data model displays self-averaging (Parley et al., 2026), so that quantities such as $\mathcal { L } _ { \mathrm { o p t } }$ measured on different grammar instances concentrate around their mean over grammar realizations $\mathbb { E } _ { \mathcal G } \left[ \mathcal { L } _ { \mathrm { o p t } } \right]$ , even for moderate values of vocabulary size such as $v \ = \ 8$ considered below. This property, which is the same that allows to compute typical thermodynamic properties of disordered systems in physics (Nishimori, 2001), will be crucial also here as it will allow us to understand any single grammar instance in terms of a typical (ensemble-averaged) grammar instance.

## 3.2 OUR SETUP

As sketched in Fig. 2, we consider the following training/evaluation setup. Our Transformers can be trained on either sentences with arbitrary tree topology from the VT-RHM with $\scriptstyle p _ { 2 } = p _ { 3 } = 1 / 2 .$ i.e. equally probable binary/ternary branchings, or sentences derived only from the strictly ternary fixed-tree topology $( p _ { 3 } ~ = ~ 1 )$ . In both cases, we test for sentence completion on a disjoint set of sentences with ternary fixed-tree topology. In compact notation, therefore,

$$
\mathbf { X } _ { \mathrm { t r a i n } } \sim \{ \mathcal { P } _ { \mathrm { V T } } , \mathcal { P } _ { \mathrm { t e r } } \} , \quad \mathbf { X } _ { \mathrm { t e s t } } \sim \mathcal { P } _ { \mathrm { t e r } }\tag{9}
$$

where $\mathcal { P } _ { \mathrm { V T } }$ and $\mathcal { P } _ { \mathrm { t e r } }$ correspond respectively to the sentence distribution induced by setting $( p _ { 2 } =$ $p _ { 3 } = 1 / 2 )$ and $( p _ { 2 } = 0 , p _ { 3 } = 1 )$ for a fixed grammar instance. The rationale for evaluating on the ternary test set $\mathrm { i s , }$ , besides benchmarking, the following. Considering the varying-tree dataset as a task mixture, where each task is sentence completion on a different tree topology, the ternary tree is the last and most difficult to generalize $^ { \mathrm { t o , } }$ , hence constituting the bottleneck for complete learning. This has two reasons: firstly, simply because of the model definition, the number of ternary rules m<sub>3</sub> per symbol is significantly larger than the number of binary rules $m _ { 2 }$ , so that the associated correlations for learning are weaker (as shown for classification in (Parley et al., 2026)) and the Transformer thus requires more data to acquire these rules. Secondly, in the varying-tree dataset the ternary topology is in fact very rare, with probability $p _ { 3 } ^ { 1 3 } = 2 ^ { - 1 3 } \approx 1 0 ^ { - 4 }$ . We are thus evaluating genuine structural generalization to a practically unseen topology or circuit.

![](images/66ffda3d4e90fda5f00f015236dcaebab4dadfa660a5e63df4e41e98b411bdf1.jpg)  
Figure 2: Sketch illustrating our two training protocols. In both cases, we consider a self-supervised sentence completion task, i.e. of predicting $x _ { d }$ given the prefix $x _ { 1 } , \ldots . x _ { d - 1 }$ . In varying-tree training (top), the sentences can have arbitrary tree topology, and are generated using $\scriptstyle p _ { 2 } = p _ { 3 } = 1 / 2 ,$ , i.e. equally probable branching ratios. In fixed-tree training, instead (bottom), sentences are derived according to the fixed (ternary) tree topology. For evaluation, we consider in both cases sentence completion on the ternary tree, i.e. the bottom case.

## 4 EXPERIMENT DETAILS

We employ GPT-2-style Transformers (Radford et al., 2019), consisting of D stacked blocks, each comprising causal (i.e. “decoder-only”) multi-head self-attention and a feedforward multilayer perceptron (MLP), with pre-layer normalization and residual connections. The MLPs use ReLU activations. We refer to each such block as a “layer” of the Transformer throughout this work, so that a Transformer of depth D consists of D such layers.

All our experiments below are performed for a VT-RHM grammar instance with $L = 3 ,$ , and grammar parameters $v = 8 , f = 1 / 4$ . To accommodate all sentence lengths, we fix the context window to $\bar { T ^ { } } = 2 6 = 3 ^ { L } - 1$ . For the ternary topology, the prefix $x _ { 1 } , \ldots . x _ { d - 1 }$ fills up the entire window, while for shorter sentences we simply pad with zeros to the left. We consider a fixed batch size $B = 1 2 8$ , and online learning using Adam (Kingma & Ba, 2015), optimizing over learning rate and weight decay. We further make standard parameter choices for embedding dimension, number of heads and hidden size of the MLP: $d _ { \mathrm { e m b } } = 5 1 2$ (256 for $D = 1 2 ) , n _ { h } = 1 6$ and $d _ { m l p } = 4 \times d _ { \mathrm { e m b } }$ We consider learnable positional embeddings. The last-token prediction is read off from the final position at the last layer, and other positions are not trained on (as they would be for next-token prediction).

## 5 LOSS STAGES

In Fig. 1c, we show the learning curves for our GPT-2 style Transformers evaluated on the ternary sentence completion task. Strikingly, the last-token cross-entropy test loss shows a clear stagewise behaviour. The key insight in (Cagnetta et al., 2024) was to show that such stages can be understood by considering the relative strength of correlation between the last token and preceding segments of the sentence, which can be used to build representations of the hidden latent tree variables. In (Cagnetta et al., 2024), these insights were developed into a theory of sample complexity (i.e. number of data required) of each learning stage. Here, we will leave open a precise mathematical characterization of sample complexity, but build on these ideas to interpret the varying-tree learning curves: in Sec. 6, we will confirm by probing that each stage indeed can be associated to the emergence of the corresponding hidden variable representation.

Stages 1a and 1b: from bigram/trigram confusion to the fixed-tree trigram. In the initial learning stage, the Transformer exploits trigram statistics to predict the last token. For the fixed-tree $( p _ { 2 } = 0 , p _ { 3 } = 1 )$ , the corresponding loss can be estimated by counting the number of compatible last-token candidates, following (Cagnetta & Wyart, 2024). For the varying tree, however, the final trigram $( a b c )$ can admit two distinct derivations: (i) directly through a ternary rule $( z  a b c )$ , or (ii) through a binary rule $( z ^ { \prime } \to b c )$ , preceded by a subtree whose expansion ends with a. When $p _ { 2 } = p _ { 3 } = 1 / 2$ and $f _ { 2 } = f _ { 3 } = f$ , these two configurations contribute comparable probabilities, leading to a higher trigram loss.

To estimate this loss, we account for the number of compatible binary and ternary rules, as well as their possible overlap. Denoting by $N _ { T }$ and $N _ { B }$ the respective numbers of compatible last-token candidates, and by $\mathbb { E } _ { \mathcal { G } } [ \cdot ]$ the expectation over grammar realizations, we have $\mathbb { E } _ { \mathcal { G } } [ \dot { N } _ { T } ] = \mathbb { E } _ { \mathcal { G } } [ N _ { B } ] =$ $v f .$ . In the limit $v f \gg 1$ , these counts concentrate around their mean, yielding the following ensemble-averaged losses, defined as $\mathcal { L } _ { t r i g } = \mathbb { E } _ { \mathcal { G } } [ H _ { \mathcal { G } } ( x _ { d } \mid x _ { d - 2 } , x _ { d - 1 } ) ]$ (see Appendix C):

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { t r i g } ^ { F T } \simeq \ln ( v f ) , } \\ { \displaystyle \mathcal { L } _ { t r i g } ^ { V T } \simeq \ln ( v f ) + ( 1 - f ) \ln 2 . } \end{array}\tag{10}
$$

Thus, the excess loss due to the ambiguity between binary and ternary derivations is approximately $( 1 - f )$ ln 2. As shown in Fig. 1c, for varying-tree training the Transformer loss in the initial stage indeed agrees well with our analytical prediction $^ 3$ for $\breve { \mathcal { L } } _ { t r i g } ^ { V T }$ : we refer to this as bigram-trigram confusion, and it is due to the fact that the Transformer cannot locally distinguish between the genuine ternary rules, where one should apply trigram statistics, and their fortuitous occurrence as spurious symbol followed by binary rule, where it is optimal to use the bigram.

To progress from the varying-tree to the fixed-tree trigram, the Transformer must learn to distinguish genuine ternary suffixes from fortuitous binary configurations. In particular, if we condition on the sentence ending by two consecutive grammatical triplets, then, neglecting rare $\mathcal { O } ( f ^ { 3 } )$ events, the final triplet must arise from a genuine ternary production. Conditioned on this information, the fixed-tree loss $\mathcal { L } _ { t r i g } ^ { F T }$ becomes attainable. We test this hypothesis through probing in Sec. 6. We show there that there is a transient regime where the penultimate triplet is recognized (i.e. its latent is represented), but not the third-last one: this allows the Transformer to correctly apply the trigram rule on ternary sentences.

Ensuing stages: inside-outside on the depth $\tilde { L } = 2 , \dots L$ subtree. Beyond Stages 1a and $^ { \mathrm { 1 b , } }$ the learning curves reflect the stagewise learning of the rules. For last-token prediction on the varyingtree, we can (numerically) obtain the Bayes-optimal prediction given knowledge of the grammar rules up to depth $\tilde { L } = 2 , \dots L$ , by running a “depth-restricted” inside-outside algorithm. This is defined precisely in App. B, and yields (for $\tilde { L } = 2 )$ the dotted horizontal line shown in Fig. 1c.

One may notice that this value is practically indistinguishable from its fixed-tree counterpart, which can be computed analytically following the iterative argument in (Cagnetta et al., 2024) (also recalled in App. A). The intuition behind the argument is that each new layer of grammar rules can be used to constrain the number of compatible parent candidates. In App. A, where we consider the full range of $f \in [ 0 , 1 ]$ , we show that the fixed and varying-tree loss values coincide up to higher-order corrections over a wide range (roughly up to $f \approx 3 / 8 )$ . This is the regime of $f$ where, although there is local ambiguity, if the grammar rules are known the topology of the sub-tree is uniquely determined (no global ambiguity), as discussed in (Parley et al., 2026). The intuition is that, once the topology is uniquely determined, the last-token entropy is simply set by $f ,$ which we have chosen to be equal for both rule types <sup>4</sup>. We stress that, although the loss values practically coincide numerically, and we denote both of them by the same symbol $\bar { \mathcal { L } } _ { \tilde { L } }$ , attaining this value in the varyingtree case still requires a significantly more complex algorithm (inside-outside) to fix the correct tree topology, as we will reveal below by probing (Fig. 4).

## 6 INTERPRETABILITY ANALYSIS

To open the door to interpretability, we now perform linear probing experiments on the residualstream activations of the Transformer. We will first briefly introduce our probing method, and then present its two applications. On the one hand, it will allow us to test our understanding of the learning stages as arising from the representation of deeper and deeper (or more and more abstract) hidden grammatical structure. On the other hand, it will allow us to visualize cleanly the inside outside organization of computation across depth, which we will do by tracking the improvement of the last-token probe loss.

![](images/f89956c1fc9084af15c9122f587d482ccf9d1dfee0c9f5770b81c1e151286d7d.jpg)

![](images/4a89d29b8f3733c546945567a92c9910caffe0b71d053d7f8183e0060285999d.jpg)

![](images/0e4116cc7f31ccdc308e8100cc2a4aaa6fbe521a117d36d26d299d0d56c90344.jpg)  
Figure 3: Stagewise emegence of hidden variable (nonterminal) representations Top left: sketch of the ternary tree topology, indicating the location of the three hidden latent variables which we focus on for interpretability. Our testable hypothesis is that each successive learning stage of the Transformer will be enabled by linearly representing (in its activations) the sentence ground-truth identity of the corresponding variable, allowing to exploit longer contexts. At each stage, we plot the test accuracy of the trained linear probe for each latent variable (see text), finding a clean correspondence between learning stage and linear representation of the required latent. (c) These probe accuracies saturate already at the middle layer of the networks. We show this in particular for the deepest latent, which is required for the final Stage 3, in Transformers of depth $D \stackrel { - } { = } 6$ and $D = 1 2$ showing that it is linearly represented beyond the middle layer in both cases. We refer to App. D for the accuracy plots across depth of the other two latents.

Linear probing methodology. Consider a Transformer with D layers and embedding dimension $d _ { \mathrm { e m b } }$ . Given an input prefix $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { d - 1 } )$ , we denote by

$$
h _ { i } ^ { ( k , t ) } ( \mathbf { x } ) \in \mathbb { R } ^ { d _ { \mathrm { e m b } } }\tag{11}
$$

the residual-stream representation at position $i ,$ after layer $k ,$ for the model obtained at training step t. We use $k = 0$ to denote the initial embedding representation, and $k = 1 , \ldots , D$ for the outputs of the successive Transformer layers.

![](images/7ab619f1aeb2730f525fc573ead5860a702c7f93ce87ebef3a06bfcad27fb4bf.jpg)  
Figure 4: Visualization of inside-outside structure and contrast to fixed-tree training. Left: test cross-entropy loss of the last-token probe across depth for fixed-tree training. We show for simplicity the matched-depth case $D = L = 3 .$ , where, strikingly, the Bayes-optimal loss for the k-th stage can already be recovered after the k-th layer of the Transformer. Right: under varying-tree training, the Transformer organizes itself in a radically different way, as expected from the required insideoutside computation. Our controlled synthetic grammar allows to cleanly visualize this structure: vertical dashed lines indicate the inside-outside boundary at the middle of the network, referring to the fact that the inside information of the most abstract latent has already been encoded (Fig. 3). The outside pass is implemented as a progressive refinement of the token prediction across the second half of the network, with the later stage (3) requiring more layers to converge to the final Bayesoptimal value.

For a categorical target variable $Z ,$ taking values in $\{ 1 , \ldots , v \}$ , we train a separate linear probe at each residual-stream depth k and training step t. The probe defines a probability distribution

$$
q _ { Z } ^ { ( k , t ) } ( z \mid \mathbf { x } ) = \mathrm { s o f t m a x } \left( W _ { Z } ^ { ( k , t ) } h _ { i } ^ { ( k , t ) } ( \mathbf { x } ) + b _ { Z } ^ { ( k , t ) } \right) _ { z } ,\tag{12}
$$

where $W _ { Z } ^ { ( k , t ) } \in \mathbb { R } ^ { v \times d _ { \mathrm { e m b } } }$ and $b _ { Z } ^ { ( k , t ) } \in \mathbb { R } ^ { v }$ are the probe parameters. The token position i is chosen according to the target variable, as specified below.

We evaluate the probes on an independent test set, measuring both the classification accuracy

$$
\mathcal { A } _ { Z } ^ { ( k , t ) } = \mathbb { E } _ { \mathbf { X } , Z } \left[ \mathbb { 1 } \left\{ \arg \operatorname* { m a x } _ { z } q _ { Z } ^ { ( k , t ) } ( z \mid \mathbf { X } ) = Z \right\} \right]\tag{13}
$$

and, to facilitate comparison with our training curves, the normalized cross-entropy loss

$$
\mathcal { L } _ { Z } ^ { ( k , t ) } = - \frac { 1 } { \ln v } \mathbb { E } _ { \mathbf { X } , Z } \left[ \ln q _ { Z } ^ { ( k , t ) } ( Z \mid \mathbf { X } ) \right] .\tag{14}
$$

Here, $\mathbb { E } _ { \mathbf { X } , Z }$ denotes the expectation over the joint distribution of test sentences and their associated ground-truth latent labels, for a fixed grammar realization. The normalization is chosen so that uniform predictions correspond to $\mathcal { L } _ { Z } ^ { ( k , t ) } = 1$

The probes are multinomial logistic classifiers, trained on standardized residual-stream activations. Separate probes are fit independently at different depths and training stages, and regularization is performed in the standard way by controlling for comparable performances on the probe training and test sets.

Probing for hidden latent tree variables (nonterminals). We first investigate whether the Transformer representations encode information about the hidden latent variables of the derivation (parse tree). Let $\mu _ { j } ^ { ( \ell ) }$ denote the latent symbol associated with node $j$ at grammatical depth $\ell ,$ using the root-based indexing convention introduced in Sec. 2. We focus in particular on three specific latent variables on the ternary tree, whose encoding is essential at each ensuing stage of learning. Namely

$$
\{ \mu _ { 8 } ^ { ( 2 ) } , \mu _ { 7 } ^ { ( 2 ) } , \mu _ { 1 } ^ { ( 1 ) } \}\tag{15}
$$

which are sketched in Fig. 3. Building on the observations of (Zhou et al., 2026; Allen-Zhu & Li, 2025), we assume that, due to the causal masking of attention, the relevant position to test for encoding of the variable is at the span boundary position, that is $i = 2 4 , 2 1$ and 9 for the three latents respectively.

As representatives of each stage, we consider Transformer model checkpoints at $t \_ =$ 1024, 16384, 262144 and $2 0 9 7 1 5 2 .$ , indicated by the stars in Fig. 1c. We then create training sets made up of $\dot { P } _ { \mathrm { t r a i n } } = 2 ^ { 1 5 }$ residual-stream activations from $P _ { \mathrm { t r a i n } }$ ternary sentences $\mathbf { X } \sim \mathcal { P } _ { \mathrm { t e r } } ,$ , and train the probes on predicting the target latent symbol for each sentence.

Our measurements yield the whole profile across depth of the probing accuracy for each latent at each stage: we refer to App. D for the full plots. In Fig. 3, we present the main results. Firstly, we test the correspondence between each learning stage and representation of the required latent, finding a clean agreement (top right). Indeed, for Stage 2 and 3, corresponding to inside-outside on sub-trees of depth 2 and 3, the inside information of level 2 and 1 latent variables respectively is a necessary condition. We find also clearly that, dynamically, the middle level-2 latent $\mu _ { 8 } ^ { ( 2 ) }$ (in purple) appears earlier, explaining the existence of the intermediate Stage 1b where the lower, conditioned (fixed-tree) trigram loss is achieved <sup>5</sup>.

Tracking last-token performance. In the above, we have probed the representation of latent grammatical variables, which are built during the inside pass (recall Fig. 1b). To study the outside pass, we will investigate how this inside information is integrated into the last-position representation which will ultimately carry the token prediction. We therefore fix $i = 3 ^ { L } - 1 = 2 6$ , and train linear probes on the correct target $x _ { d }$ for each prefix.

Mirroring the two training protocols, for the varying-tree checkpoints we create training sets made up of $P _ { \mathrm { t r a i n } } = 2 ^ { 1 5 }$ activations to sentences $\mathbf { X } \sim \mathcal { P } _ { \mathrm { V T } }$ , while for the fixed-tree checkpoints (extracted equivalently at each stage) we consider $\mathbf { X } \sim { \mathcal { P } } _ { \mathrm { t e r } } .$ . As in the experiments, we test the probes on held-out data from $\mathbf { X } \sim \mathcal { P } _ { \mathrm { t e r } } ,$ , obtaining the probe cross-entropy loss profile $\mathcal { L } _ { x _ { d } } ^ { ( k , t ) }$ . For $k = D$ , one expects to recover in each case roughly the Bayes-optimal value of the stage. For $k < D$ , instead, what we are doing is equivalent to “depth pruning”: given Transformer activations, we are retraining a new linear+softmax (LSM) head on the last position of a pre-terminal layer, effectively chopping off the later layers.

We show the $\mathcal { L } _ { x _ { d } } ^ { ( k , t ) }$ profiles in Fig. 4, contrasting fixed-tree and varying-tree training. In the first case, where we can simply match the Transformer depth to the depth of the grammar, we find a very simple structure where the Stage-k Bayes-optimal prediction is already linearly decodable after the $k { \mathrm { - t h } }$ layer. Under varying-tree training, instead, we recover the expected inside-outside structure, which we can visualize cleanly thanks to our controlled synthetic grammar. We do so again for transformers of depth $D = 6$ and 12. Namely, given the results from Fig. 3 we can draw an inside-outside boundary (vertical dashed line) at the middle layer, where we have found the most abstract latent information to be encoded. We see then that the second half of the network uses the inside information to progressively refine the last-token prediction. As expected from the layerwise structure of the algorithm (see Fig. 1b), we find also that the final Stage 3 requires more layers than Stage 2 to converge to the Bayes-optimal value. At early stages of training instead, where the purely local (trigram) statistics are used, we find indeed that only a few initial transformer layers are involved, with the profile rapidly saturating at the final loss value.

Discussion: the question of minimal Transformer depth. A natural question Fig. 4 may raise is why we have not considered the matched-depth setting $\bar { D } = L$ also in the right panel. Although we do not provide complete evidence for a rigorous theoretical statement, in practice we have found that even after significantly increasing the width $d _ { \mathrm { e m b } }$ , training over a long horizon $( t \sim 1 0 ^ { 7 } ~ \mathrm { s t e p s } )$ , and sweeping over Adam hyperparameters, we have not been able to converge to the final stage Bayes optimal loss using $D = 3$ Transformers (see figure in App. D). One can therefore conjecture that

D = 4 = L+1 is the minimal depth for successful varying-tree training. On the other hand, we have found that increasing depth further up to $D = 6 = 2 L$ leads to better and more stable training, while comparing D = 6 and D = 12 the training curves are practically identical suggesting a saturation of the depth advantage beyond D = 6.

## 7 CONCLUSIONS AND OPEN DIRECTIONS

How do LLMs build up hierarchical representations of language, and end up mimicking highly non-trivial parsing algorithms from computational linguistics? Inspired by this question, we have analyzed the learning of inside-outside computation by GPT-2 style Transformers on a synthetic, controlled grammar, the VT-RHM (Parley et al., 2026). We developed an understanding of the loss stages, and contrasted fixed-tree and varying-tree training protocols, highlighting the stark distinctions in learned computational circuits. Our main interpretability result, captured in Fig. 4, regards organization of computation across depth: we can cleanly show that, while during early stages of training (bigram/trigram) only initial layers are involved, the Transformer later develops a clear encoder-decoder structure, with the middle layers encoding the most abstract information and the latter half of the network progressively refining the token prediction. This general behavior echoes findings in real-world language models (Tenney et al., 2019; Jawahar et al., 2019; Kervadec et al., 2026), which is remarkable, especially given the fact that the VT-RHM is built from very simple ingredients and allows for analytical understanding of all relevant statistics (Parley et al., 2026). We argue that this opens the door to using the VT-RHM for building quantitative descriptions of various phenomena, which we include in the following open directions:

Learning algorithm and sample complexity of token prediction. For the supervised root classification problem (Parley et al., 2026), it was shown that, at least in the regime of large vocabulary size and number of rules, the rules of the VT-RHM are provably learned by a layerwise algorithm based on clustering by correlation, with an associated sample complexity scale (number of data). Extend ing this to last or next-token prediction, it will be important to understand whether a similar formal learning result can be established, and whether it requires thinking beyond a strictly layerwise algorithm. Furthermore, it will be interesting to characterize precisely which latent-token correlations must be used to build the outside pass (decoder) in a sample-efficient manner.

Representation geometry. Although we have tested the expected correspondence between latent representations and learning stages, we have not studied how the Transformer actually encodes the relevant information. Our controlled grammar may give insight into the origin of syntactic encoding in LLMs: in (Diego-Simon et al., 2024), it was found that syntactic relations are encoded by the´ relative direction between embeddings, but it is not understood how such representations are learned.

Thresholds to syntactic generalization. In Fig. 4, we have only considered the two “extreme” forms of training, i.e. fixed-tree vs the full varying-tree dataset. A natural question that arises is how the Transformer computation evolves between the two: how rich does the tree mixture need to be in order to impose encoder-decoder structure? This may also shed quantitative insight into phenomena such as Grokking of hierarchical (i.e. tree-like) structure in Transformers (Murty et al., 2023).

We hope our work encourages to community to build on this data model and extensions thereof. All relevant code will be made publically available on the Github page https://github.com/ jackparley/interpretable\_inside\_outside.

## ACKNOWLEDGEMENTS

The author thanks Matthieu Wyart and Francesco Cagnetta for important discussions during the early phases of this project. We further acknowledge F.C. as the author of the Github repository which was built upon in order to study the varying-tree model. The author acknowledges funding from the PCSL lab (Prof. M. Wyart) at EPFL, where this research was performed, and the SCITAS infrastructure which was used in order to perform the experiments (mainly H100 GPUs).

## REFERENCES

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of Language Models: Part 1, learning hierarchical language structures. Transactions on Machine Learning Research, 2025. URL https: //openreview.net/forum?id=mPQKyzkA1K.

Emmanuel Ameisen, Jack Lindsey, Adam Pearce, Wes Gurnee, Nicholas L. Turner, Brian Chen, Craig Citro, et al. Circuit tracing: Revealing computational graphs in language models. Transformer Circuits Thread, 2025. URL https://transformer-circuits.pub/2025/ attribution-graphs/methods.html.

J. K. Baker. Trainable grammars for speech recognition. The Journal of the Acoustical Society of America, 65:S132, 1979. ISSN 0001-4966. doi: 10.1121/1.2017061. URL https://doi. org/10.1121/1.2017061.

Christopher M. Bishop. Pattern Recognition and Machine Learning. Information Science and Statistics. Springer, 2006.

Francesco Cagnetta and Matthieu Wyart. Towards a theory of how the structure of language is acquired by deep neural networks. In Advances in Neural Information Processing Systems 37 (NeurIPS 2024), 2024. URL https: //proceedings.neurips.cc/paper\_files/paper/2024/hash/ 9740da1c07c7b451af14e11523f95271-Abstract-Conference.html.

Francesco Cagnetta, Leonardo Petrini, Umberto M. Tomasini, Alessandro Favero, and Matthieu Wyart. How deep neural networks learn compositional data: The random hierarchy model. Physical Review X, 14(3):031001, 2024.

Noem Chomsky. Syntactic structures. Syntactic structures. Mouton, Oxford, England, 1957. Pages: 116.

Pablo J. Diego-Simon, St ´ ephane D’Ascoli, Emmanuel Chemla, Yair Lakretz, and Jean-R ´ emi King.´ A polar coordinate system represents syntax in large language models. Advances in Neural Information Processing Systems, 37:105375–105396, 2024.

Javier Ferrando and Marta R. Costa-jussa. On the similarity of circuits across languages: A case\` study on the subject-verb agreement task. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 10115–10125. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-emnlp.591.

Edward Gibson, Richard Futrell, Steven T. Piantadosi, Isabelle Dautriche, Kyle Mahowald, Leon Bergen, and Roger Levy. How efficiency shapes human language. Trends in Cognitive Sciences, 23(5):389–407, 2019. doi: 10.1016/j.tics.2019.02.003.

Joshua Goodman. Semiring parsing. In Proceedings of the 37th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 1–8. Association for Computational Linguistics, 1999.

Michael Hahn, Dan Jurafsky, and Richard Futrell. Universals of word order reflect optimization of grammars for efficient communication. Proceedings of the National Academy of Sciences, 117 (5):2347–2353, 2020.

Ganesh Jawahar, Benoˆıt Sagot, and Djame Seddah. What does BERT learn about the structure of´ language? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 3651–3657. Association for Computational Linguistics, 2019. doi: 10.18653/v1/ P19-1356. URL https://aclanthology.org/P19-1356/.

Corentin Kervadec, Iuliia Lysova, Iuri Macocco, Marco Baroni, and Gemma Boleda. Tracing computation density in LLMs. In Proceedings ofthe 2026 Conference on Empirical Methods in Natural Language Processing, pp. 37124–37144, Budapest, Hungary, 2026. Association for Computational Linguistics. URL https://aclanthology.org/2026.emnlp-main.1633/.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. URL https://arxiv.org/abs/1412. 6980.

Jack Lindsey, Wes Gurnee, Emmanuel Ameisen, Brian Chen, Adam Pearce, Nicholas L. Turner, Craig Citro, David Abrahams, Shan Carter, Basil Hosmer, Jonathan Marcus, Michael Sklar, Adly Templeton, Trenton Bricken, Callum McDougall, Hoagy Cunningham, Thomas Henighan, Adam Jermyn, Andy Jones, Andrew Persic, Zhenyi Qi, T. Ben Thompson, Sam Zimmerman, Kelley Rivoire, Thomas Conerly, Chris Olah, and Joshua Batson. On the biology of a large language model. Transformer Circuits Thread, 2025. URL https://transformer-circuits. pub/2025/attribution-graphs/biology.html.

Christopher D. Manning, Kevin Clark, John Hewitt, Urvashi Khandelwal, and Omer Levy. Emergent linguistic structure in artificial neural networks trained by self-supervision. Proceedings of the National Academy ofSciences, 117(48):30046–30054, 2020. doi: 10.1073/pnas.1907367117.

Shikhar Murty, Pratyusha Sharma, Jacob Andreas, and Christopher D. Manning. Grokking of hierarchical structure in vanilla transformers. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). Association for Computational Linguistics, 2023.

Hidetoshi Nishimori. Statistical Physics of Spin Glasses and Information Processing: An Introduc tion. Oxford University Press, 2001.

Jack T. Parley, Francesco Cagnetta, and Matthieu Wyart. Deep networks learn to parse uniformdepth context-free languages from local statistics. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Technical report, OpenAI, 2019. URL https://cdn.openai.com/better-language-models/language\_ models\_are\_unsupervised\_multitask\_learners.pdf.

I. Tenney, D. Das, and E. Pavlick. BERT rediscovers the classical NLP pipeline. In Anna Korhonen, David Traum, and Llu´ıs Marquez (eds.),\` Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics, pp. 4593–4601, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1452. URL https://aclanthology. org/P19-1452.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

Yuekun Yao and Alexander Koller. Structural generalization is hard for sequence-to-sequence models. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pp. 5048–5062. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022. emnlp-main.337.

Daniel H. Younger. Recognition and parsing of context-free languages in time n<sup>3</sup>. Information and Control, 10(2):189–208, 1967.

Haoyu Zhao, Abhishek Panigrahi, Rong Ge, and Sanjeev Arora. Do transformers parse while predicting the masked word? In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 16513– 16542, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. emnlp-main.1029. URL https://aclanthology.org/2023.emnlp-main.1029/.

Zhiling Zhou, Tianhao Wang, and Zhuoran Yang. How transformers represent hierarchies: A localto-global mechanism. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pp. 165889–165934. PMLR, 2026. URL https://proceedings.mlr.press/v306/zhou26at.html.

## A LAST-TOKEN CONDITIONAL ENTROPY ON THE VARYING-TREE

In this appendix, we show detailed numerical evidence for the claim in the main text, namely that the fixed-tree and varying-tree ensemble-averaged last-token conditional entropy

$$
\mathbb { E } _ { \mathcal { G } } [ H _ { \mathcal { G } } ( X _ { d } \mid X _ { 1 } , \dots , X _ { d - 1 } ) ]\tag{16}
$$

are practically indistinguishable over a wide range of $f ,$ up to around $f { \approx } 3 / 8$

Before turning to numerics, we recall how the fixed-tree expected losses can be determined following the iterative argument in (Cagnetta & Wyart, 2024). Referring to $N _ { \tilde { L } }$ as the number of last-token candidates under a sub-tree of depth ${ \tilde { L } } ,$ we have that <sup>6</sup>

$$
\mathbb { E } [ N _ { \tilde { L } } ] = 1 + ( v - 1 ) \frac { m _ { 3 } \mathbb { E } [ N _ { \tilde { L } - 1 } ] - 1 } { v ^ { 3 } - 1 } , \qquad \tilde { L } \geq 2 .\tag{17}
$$

The corresponding loss is then approximated by $\mathcal { L } _ { \tilde { r } . } ^ { F T }$ ≈ ln $\mathbb { E } [ N _ { \tilde { L } } ]$ , and can be computed iteratively starting from the fixed-tree trigram loss $\mathcal { L } _ { t r i g } ^ { F T } \equiv \dot { \mathcal { L } _ { 1 } ^ { F T } }$ introduced in the main text, which we recall is approximated as $\mathcal { L } _ { 1 } ^ { F T } \simeq \ln ( v f )$ for large $v f$

In Fig. 5, we show, for fixed $v = 1 6$ and varying $f ,$ the analytical curves computed from the above recursion (17). We then sample VT-RHM grammars of the same depth $\tilde { L } = L ,$ , to test whether the last-token conditional entropy is well captured by the analytical curves. This is done by sampling many sentences x from each grammar, running inside-outside (see App. B), then averaging over both sentences and grammar instances <sup>7</sup>. We find that indeed there is good agreement, up to around $f \approx 3 / 8$ This was the same range of $f ,$ where, in (Parley et al., 2026) (see Fig. 2 there), it was shown that the grammars are globally unambiguous, i.e. that the probability of a sentence being derived by two different tree topologies is negligible. We leave open a precise analytical characterization of the varying-tree last-token entropies over the full range of $f \in [ 0 , 1 ] .$ , which may be possible by considering the additional terms due to multiple tree topologies per sentence.

## B TENSOR FORMULATION OF INSIDE-OUTSIDE

Here, we provide the detailed equations for the inside-outside algorithm, which for our layered grammars we can formulate in a tensorized fashion: this is crucial for their parallelized implementation (available on Github), which becomes computationally crucial already for grammars of depth $L \geq 3$ . Following the notation in (Parley et al., 2026), we denote the level-ℓ rule tensors by $R ^ { \left( \ell \right) }$ Their entries are

$$
R _ { z \to ( a , b ) } ^ { ( \ell ) } = \left\{ { p _ { 2 } / m _ { 2 } } , \begin{array} { l l } { { \mathrm { i f } z \to ( a , b ) \mathrm { i s ~ g r a m m a t i c a l , } } } \\ { { \mathrm { o t h e r w i s e , } } } \end{array} \right.\tag{18}
$$

for binary productions, and

$$
R _ { z  ( a , b , c ) } ^ { ( \ell ) } = \{ { p _ { 3 } } / { m _ { 3 } } , \begin{array} { l l } { { \mathrm { i f } z  ( a , b , c ) \mathrm { i s \ g r a m m a t i c a l , } } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e , } } } \end{array}\tag{19}
$$

for ternary productions. Here $p _ { 2 }$ and $p _ { 3 }$ are the probabilities of choosing a binary or ternary expansion at a node, respectively, with $p _ { 2 } + p _ { 3 } = 1$ . Conditional on the branching factor, the corresponding grammatical rules are therefore equiprobable. In a layered PCFG, given the rule tensors $\bar { R } ^ { ( \ell ) }$ , the rule-based parser computes the level-(ℓ − 1) chart/tensor $M ^ { ( \ell - 1 ) }$ following the recursion

$$
\begin{array} { l } { { { \cal M } _ { i , \lambda } ^ { ( \ell - 1 ) } [ z ] = \displaystyle \sum _ { \lambda ^ { \prime } , a , b } { \cal R } _ { z \to ( a , b ) } ^ { ( \ell ) } \times { \cal M } _ { i , \lambda ^ { \prime } } ^ { ( \ell ) } [ a ] \times { \cal M } _ { i + \lambda ^ { \prime } , \lambda - \lambda ^ { \prime } } ^ { ( \ell ) } [ b ] } } \\ { { ~ + \displaystyle \sum _ { \lambda ^ { \prime } , \lambda ^ { \prime \prime } , a , b , c } { \cal R } _ { z \to ( a , b , c ) } ^ { ( \ell ) } \times } } \\ { { { \cal M } _ { i , \lambda ^ { \prime } } ^ { ( \ell ) } [ a ] \times { \cal M } _ { i + \lambda ^ { \prime } , \lambda ^ { \prime \prime } } ^ { ( \ell ) } [ b ] \times { \cal M } _ { i + \lambda ^ { \prime } + \lambda ^ { \prime \prime } , \lambda - \lambda ^ { \prime } - \lambda ^ { \prime \prime } } ^ { ( \ell ) } [ c ] . } } \end{array}\tag{20}
$$

![](images/d1a841e73097cf08911ccfc118ddb10ea642baf044ed646539abcba833a4fba6.jpg)  
Figure 5: Comparison of the theoretical curves (full lines) obtained from the fixed-tree recursion (17), with the expected last-token conditional entropies computed numerically using inside-outside on varying-tree grammar datasets. We consider $v = 1 6 ,$ vary $f \in [ 0 , 1 ]$ , and grammars of different depth L. Up to roughly $f \approx 3 / 8$ , there is good agreement: this corresponds to the regime of no global ambiguity (i.e. single topology per sentence) discussed in (Parley et al., 2026).

For each prefix $x _ { 1 } , \ldots . x _ { d - 1 }$ , the chart is initialized at the L-th (terminal) level as $M _ { i , 1 } ^ { ( L ) } [ z ] = \delta _ { x _ { i } , i }$ for $i = 1 , \dots d - 1$ , and $M _ { d , 1 } ^ { ( L ) } [ z ] = 1 / v$ , ∀z at the last position. Above, we have expressed the parser in its sum-product version (inside algorithm), so that the chart stores inside probabilities $M _ { i , \lambda } ^ { ( \ell ) } [ z ] = P r ( x _ { i } , \ldots , x _ { i + \lambda - 1 } \mid z )$ for $z \in \mathcal { N } ^ { ( \ell ) }$ . Alternative choices of operation recover other standard parsing algorithms. For example, replacing $( + , \times )$ by $( \vee , \wedge )$ (Boolean parsing) recovers recognition/CYK (Younger, 1967) (all such algorithms can be unified under the semiring parser formulation (Goodman, 1999)).

The inside algorithm computes, for each symbol and span, the probability of generating the substring inside that span. For last-token prediction, we instead need the complementary quantity: the probability of generating the observed context outside a given span. This is computed by the corresponding outside pass.

The outside chart O is defined so that

$$
O _ { i , \lambda } ^ { ( \ell ) } [ z ]\tag{21}
$$

stores the total weight of all derivations of the observed sentence outside the span (i, λ), conditional on this span being generated by the symbol $z \in \mathcal { N } ^ { ( \ell ) }$

The outside pass is, in general, initialized at the root level (see note below on marginalizing over the root symbol) by

$$
O _ { 1 , d } ^ { ( 0 ) } [ y ] = p ( y ) , \qquad y \in { \mathcal { N } } ^ { ( 0 ) } ,\tag{22}
$$

It is then computed recursively from level $\ell - 1$ to level ℓ by distributing the outside weight of each parent span to its children.

For a binary rule $z  ( a , b )$ , each decomposition of a span (i, λ) into $( i , \lambda ^ { \prime } )$ and $( i + \lambda ^ { \prime } , \lambda - \lambda ^ { \prime } )$ contributes

$$
\begin{array} { c } { { { \cal O } _ { i , \lambda ^ { \prime } } ^ { ( \ell ) } [ a ] + = { \cal O } _ { i , \lambda } ^ { ( \ell - 1 ) } [ z ] { \cal R } _ { z  ( a , b ) } ^ { ( \ell ) } { \cal M } _ { i + \lambda ^ { \prime } , \lambda - \lambda ^ { \prime } } ^ { ( \ell ) } [ b ] , } } \\ { { { \cal O } _ { i + \lambda ^ { \prime } , \lambda - \lambda ^ { \prime } } ^ { ( \ell ) } [ b ] + = { \cal O } _ { i , \lambda } ^ { ( \ell - 1 ) } [ z ] { \cal R } _ { z  ( a , b ) } ^ { ( \ell ) } { \cal M } _ { i , \lambda ^ { \prime } } ^ { ( \ell ) } [ a ] . } } \end{array}\tag{23}
$$

Similarly, for a ternary rule $z  ( a , b , c )$ , each decomposition into spans $( i , \lambda ^ { \prime } ) , ( i + \lambda ^ { \prime } , \lambda ^ { \prime \prime } )$ , and $( i + \lambda ^ { \prime } \dot { + } \lambda ^ { \prime \prime } , \lambda - \lambda ^ { \prime } \dot { - } \lambda ^ { \prime \prime } )$ contributes

$$
\begin{array} { c } { { { \cal O } _ { i , \lambda ^ { \prime } } ^ { ( \ell ) } [ a ] ~ { + } = { \cal O } _ { i , \lambda } ^ { ( \ell - 1 ) } [ z ] { \cal R } _ { z  ( a , b , c ) } ^ { ( \ell ) } { \cal M } _ { i + \lambda ^ { \prime } , \lambda ^ { \prime \prime } } ^ { ( \ell ) } [ b ] { \cal M } _ { i + \lambda ^ { \prime } + \lambda ^ { \prime \prime } , \lambda - \lambda ^ { \prime } - \lambda ^ { \prime \prime } } ^ { ( \ell ) } [ c ] , } } \\ { { { \cal O } _ { i + \lambda ^ { \prime } , \lambda ^ { \prime \prime } } ^ { ( \ell ) } [ b ] ~ { + } = { \cal O } _ { i , \lambda } ^ { ( \ell - 1 ) } [ z ] { \cal R } _ { z  ( a , b , c ) } ^ { ( \ell ) } { \cal M } _ { i , \lambda ^ { \prime } } ^ { ( \ell ) } [ a ] { \cal M } _ { i + \lambda ^ { \prime } + \lambda ^ { \prime \prime } , \lambda - \lambda ^ { \prime } - \lambda ^ { \prime \prime } } ^ { ( \ell ) } [ c ] , } } \\ { { { \cal O } _ { i + \lambda ^ { \prime } + \lambda ^ { \prime \prime } , \lambda - \lambda ^ { \prime } - \lambda ^ { \prime \prime } } ^ { ( \ell ) } [ c ] ~ { + } = { \cal O } _ { i , \lambda } ^ { ( \ell - 1 ) } [ z ] { \cal R } _ { z  ( a , b , c ) } ^ { ( \ell ) } { \cal M } _ { i , \lambda ^ { \prime } } ^ { ( \ell ) } [ a ] { \cal M } _ { i + \lambda ^ { \prime } , \lambda ^ { \prime \prime } } ^ { ( \ell ) } [ b ] . } } \end{array}\tag{24}
$$

The outside chart can be used to compute posterior marginals over any span. Indeed, for any span (i, λ) and symbol $z \in \mathcal { N } ^ { ( \ell ) }$

$$
P r ( z \Rightarrow x _ { i } , . . . , x _ { i + \lambda - 1 } \mid \mathbf { x } ) = \frac { O _ { i , \lambda } ^ { ( \ell ) } [ z ] M _ { i , \lambda } ^ { ( \ell ) } [ z ] } { P r ( \mathbf { x } ) } ,\tag{25}
$$

where

$$
P r ( \mathbf { x } ) = \sum _ { y \in \mathcal { N } ^ { ( 0 ) } } p ( y ) M _ { 1 , d } ^ { ( 0 ) } [ y ] .\tag{26}
$$

For last-token prediction, the context $( x _ { 1 } , \ldots , x _ { d - 1 } )$ is observed and the final token $x _ { d }$ is masked. After running the outside pass, the outside values

$$
O _ { d , 1 } ^ { ( L ) } [ z ] , \qquad z \in { \mathcal { N } } ^ { ( L ) } ,\tag{27}
$$

give the total probability of generating the observed context under the constraint that the final terminal is z. Therefore the Bayes-optimal last-token predictor is

$$
\hat { x } _ { d } = \arg \operatorname* { m a x } _ { z \in \mathcal { N } ^ { ( L ) } } O _ { d , 1 } ^ { ( L ) } [ z ] ,\tag{28}
$$

and the full predictive distribution is

$$
P r ( X _ { d } = z \mid x _ { 1 } , \ldots , x _ { d - 1 } ) = \frac { O _ { d , 1 } ^ { ( L ) } [ z ] } { \sum _ { z ^ { \prime } \in \mathcal { N } ^ { ( L ) } } O _ { d , 1 } ^ { ( L ) } [ z ^ { \prime } ] } .\tag{29}
$$

## B.1 MARGINALIZING OVER THE ROOT SYMBOL

For the last-token prediction task, the root class variable can be marginalized out explicitly. Defining the effective root production probabilities

$$
\overline { { R } } _ { a , b } ^ { ( 1 ) } = \sum _ { y } p ( y ) R _ { y  ( a , b ) } ^ { ( 1 ) } , \qquad \overline { { R } } _ { a , b , c } ^ { ( 1 ) } = \sum _ { y } p ( y ) R _ { y  ( a , b , c ) } ^ { ( 1 ) } ,\tag{30}
$$

the original grammar is equivalent, at the level of the marginal distribution over terminal sequences, to a grammar with a single root symbol and production probabilities $\overline { { R } } ^ { ( 1 ) }$

Defining the effective root production probabilities

$$
\overline { { R } } _ { a , b } ^ { ( 1 ) } = \sum _ { y } p ( y ) R _ { y  ( a , b ) } ^ { ( 1 ) } , \qquad \overline { { R } } _ { a , b , c } ^ { ( 1 ) } = \sum _ { y } p ( y ) R _ { y  ( a , b , c ) } ^ { ( 1 ) } ,\tag{31}
$$

the inside pass need only be computed up to level $\ell = 1$ . The outside pass can then be initialized directly at this level, without explicitly constructing the level-0 inside chart or retaining the root class labels. Furthermore, since we are only interested in the last token, we need only consider spans ending at position d, for which

$$
\begin{array} { r l } {  { \boxed { O _ { i , d - i + 1 } ^ { ( 1 ) } [ z ] = \sum _ { a } \overline { { R } } _ { a , z } ^ { ( 1 ) } M _ { 1 , i - 1 } ^ { ( 1 ) } [ a ] } } } \\ & { + \sum _ { j = 1 } ^ { i - 2 } \sum _ { a , b } \overline { { R } } _ { a , b , z } ^ { ( 1 ) } M _ { 1 , j } ^ { ( 1 ) } [ a ] M _ { j + 1 , i - j - 1 } ^ { ( 1 ) } [ b ] . } \end{array}\tag{32}
$$

When predicting only the final token, the outside recursion can be restricted to spans ending at the final position, since the final token necessarily belongs to the rightmost child at every branching. This yields exactly the same Bayes-optimal predictive distribution as the full inside–outside algorithm.

![](images/845e0acda6bf0bfbecb54a8a93c2e10b7461f7c2c90e76fd4d9ff15dbdd7c33d.jpg)  
Figure 6: Numerical validation of the large-vf theoretical prediction for the varying-tree trigram loss. Sampling grammars for various combinations of $( v , f )$ , we numerically estimate $H ( x _ { d } \mid$ $x _ { d - 2 } , x _ { d - 1 } )$ by averaging over sentences. Different grammar realizations (faint curves) concentrate around the mean due to self-averaging. We compare the numerical results with the theoretical prediction $\mathcal { L } _ { t r i g } ^ { V T } \simeq \ln ( v f ) + ( 1 - f )$ ln 2, normalized by ln v. This large $v f$ approximation is already very accurate for $v = 8 , f = 1 / 4$

## B.2 DEPTH-RESTRICTED INSIDE-OUTSIDE

We also consider a depth-restricted version of the inside–outside algorithm. Let $\widetilde { L } < L$ denote a cutoff depth. Up to depth ${ \widetilde { \cal L } } ,$ , inference uses the rule tensors of the particular realized grammar. At deeper levels, the realization-specific rule tensors are replaced by their uniform average R over rule sampling, that is $p _ { 2 } / v ^ { 2 }$ across all binary tuples and $p _ { 3 } / v ^ { 3 }$ across all ternary tuples. Note that, although such rules are almost structureless, this still implies an implicit prior on the fact that there are binary and ternary rules, and they have equal overall probability weight $p _ { 2 } = p _ { 3 }$ . We expect the effect of this to be negligible, and we have checked (see e.g. dotted line in Fig. 1c) that numerically the approach reproduces well the expected last-token entropies.

More precisely, if $R ^ { ( \ell ) }$ denotes the rule tensor connecting levels $\ell - 1$ and $\ell ,$ we define

$$
\widetilde { R } _ { \widetilde { L } } ^ { ( \ell ) } = \left\{ { R } ^ { ( \ell ) } , \quad \ell > L - \widetilde { L } , \atop \overline { { { R } } } \quad  \right.\tag{33}
$$

The depth- $\tilde { \cdot } \tilde { L }$ inside–outside algorithm is then obtained by running the usual inside and outside recursions after replacing $R ^ { ( \ell ) }$ by $\widetilde { R } _ { \widetilde { L } } ^ { ( \ell ) }$ at every level. The case $\widetilde { L } = L$ recovers the full inside–outside algorithm.

## C TRIGRAM LOSS IN THE VARYING-TREE GRAMMAR

We derive the large-vf prediction for the conditional entropy $H ( x _ { d } \mid x _ { d - 2 } , x _ { d - 1 } )$ , corresponding to the initial trigram-learning stage. We consider a grammar with v symbols per layer, $m _ { 2 } = f v$ binary rules and $\overline { { m _ { 3 } } } \ : = \ : f v ^ { 2 }$ ternary rules per parent, and $p _ { 2 } = p _ { 3 } = 1 / 2$ . The right-hand sides of each rule type are sampled uniformly without replacement across all parents. Thus, there are $v m _ { 2 } = f v ^ { 2 }$ distinct binary right-hand sides and $v m _ { 3 } = f v ^ { 3 }$ distinct ternary ones.

Compatible derivations. Consider a final trigram (abc). We distinguish two possible local configurations: a ternary production $z  a b c ,$ and a binary production $z ^ { \prime } $ bc preceded by a subtree ending with a. We approximate the two configurations as contributing equally weighted compatible derivations, after marginalizing over the preceding tree.

Define the indicators

$$
T _ { a b c } = \mathbb { 1 } \{ a b c \mathrm { i s ~ a ~ t e r n a r y ~ R H S } \} , \qquad B _ { b c } = \mathbb { 1 } \{ b c \mathrm { i s ~ a ~ b i n a r y ~ R H S } \} .\tag{34}
$$

Under the equal-weight approximation, the unnormalized probability of the trigram is

$$
W _ { a b c } = T _ { a b c } + B _ { b c } .\tag{35}
$$

In particular, a token compatible with both configurations receives twice the weight of one compatible with only one.

For a fixed context (ab), define

$$
N _ { T } ( a b ) = \sum _ { c } T _ { a b c } , \qquad N _ { B } ( b ) = \sum _ { c } B _ { b c } ,\tag{36}
$$

and

$$
Z _ { a b } = \sum _ { c } W _ { a b c } = N _ { T } ( a b ) + N _ { B } ( b ) .\tag{37}
$$

The conditional distribution of the final token is then approximated by

$$
P ( c \mid a , b , \mathcal { G } ) \simeq \frac { W _ { a b c } } { Z _ { a b } } ,\tag{38}
$$

for contexts with $Z _ { a b } > 0$

Conditional entropy and overlap correction. For a given context (ab), the conditional entropy reads

$$
\begin{array} { l } { { \displaystyle H _ { a b } = - \sum _ { c } \frac { W _ { a b c } } { Z _ { a b } } \ln \frac { W _ { a b c } } { Z _ { a b } } } } \\ { ~ = \ln Z _ { a b } - \frac { 1 } { Z _ { a b } } \sum _ { c } W _ { a b c } \ln W _ { a b c } . }  \end{array}\tag{39}
$$

Since $T _ { a b c } , B _ { b c } \in \{ 0 , 1 \}$ , the weights $W _ { a b c }$ take values in {0, 1, 2}. Consequently,

$$
W _ { a b c } \ln W _ { a b c } = ( T _ { a b c } B _ { b c } ) 2 \ln 2 .\tag{40}
$$

given that $W _ { a b c }$ ln $W _ { a b c }$ is only non-zero when both indicators equal 1 (we use the standard convention 0 ln 0 = 0). Defining the number of overlapping candidates as

$$
D _ { a b } = \sum _ { c } T _ { a b c } B _ { b c } ,\tag{41}
$$

Eq. (39) becomes

$$
H _ { a b } = \ln Z _ { a b } - \frac { 2 D _ { a b } } { Z _ { a b } } \ln 2 .\tag{42}
$$

To obtain the total trigram loss, we must average over contexts weighted by their probability of occurrence. Since each allowed ternary RHS contributes once to $\sum _ { a b c } ^ { - } W _ { a b c }$ , while each allowed binary RHS contributes for all v possible values of a, we have

$$
\sum _ { a , b } Z _ { a b } = v m _ { 3 } + v ( v m _ { 2 } ) = 2 f v ^ { 3 } .\tag{43}
$$

Thus,

$$
P ( a , b \mid { \mathcal { G } } ) \simeq { \frac { Z _ { a b } } { 2 f v ^ { 3 } } } ,\tag{44}
$$

and the trigram entropy for a fixed grammar is

$$
\begin{array} { l } { { \displaystyle { \mathcal L } _ { t r i g } ^ { V T } ( { \mathcal G } ) = \sum _ { a , b } P ( a , b \mid { \mathcal G } ) H _ { a b } } } \\ { ~ } \\ { { \displaystyle ~ \simeq \frac { 1 } { 2 f v ^ { 3 } } \sum _ { a , b } [ Z _ { a b } \ln Z _ { a b } - 2 D _ { a b } \ln 2 ] } . } \end{array}\tag{45}
$$

We now average over grammar realizations. By symmetry, the distribution of $Z _ { a b }$ is the same for every context, so that

$$
\mathbb { E } _ { \mathcal { G } } \left[ \sum _ { a , b } Z _ { a b } \ln Z _ { a b } \right] = v ^ { 2 } \mathbb { E } [ Z \ln Z ] ,\tag{46}
$$

where $Z = N _ { T } + N _ { B }$ denotes the count for an arbitrary fixed context.

Furthermore, binary and ternary rule sets are sampled independently, with each possible RHS having marginal inclusion probability f. Hence,

$$
\mathbb { E } _ { \mathcal { G } } [ D _ { a b } ] = \sum _ { c } \mathbb { E } [ T _ { a b c } B _ { b c } ] = v f ^ { 2 } .\tag{47}
$$

Substituting into Eq. (45), we obtain

$$
\boxed { \mathcal { L } _ { t r i g } ^ { V T } \simeq \frac { \mathbb { E } [ Z \ln Z ] } { 2 v f } - f \ln 2 . }\tag{48}
$$

The second term accounts for the overlap between binary and ternary derivations: two distinct derivations may produce the same last token, thereby reducing the entropy relative to treating them as distinct outcomes.

Large-vf approximation. The numbers of compatible ternary and binary continuations are respectively $\begin{array} { r } { \bar { N _ { T } } ( a b ) = \sum _ { c } T _ { a b c } } \end{array}$ and $\begin{array} { r } { N _ { B } ( b ) = \sum _ { c } \mathsf { \bar { B } } _ { b c } , } \end{array}$ . Since each possible right-hand side is selected with probability f, their expectations over grammar realizations are

$$
\mathbb { E } [ N _ { T } ] = \mathbb { E } [ N _ { B } ] = v f .\tag{49}
$$

For $v f \gg 1$ , these counts concentrate around their mean, so that $Z = N _ { T } + N _ { B } \simeq 2 v f$ . Substituting into Eq. (48) gives

$$
\begin{array} { r } { \left| \mathcal { L } _ { t r i g } ^ { V T } \simeq \ln ( 2 v f ) - f \ln 2 = \ln ( v f ) + ( 1 - f ) \ln 2 . \right| } \end{array}\tag{50}
$$

For the fixed tree, only ternary derivations contribute, so that $W _ { a b c } = T _ { a b c }$ and $Z _ { a b } = N _ { T } ( a b )$ . All compatible last tokens therefore have equal probability. Repeating the context-weighted average, we obtain

$$
\mathcal { L } _ { t r i g } ^ { F T } = \frac { \mathbb { E } [ N _ { T } \ln { N _ { T } } ] } { v f } \simeq \ln ( v f ) ,\tag{51}
$$

where the final approximation again follows from the concentration of $N _ { T }$ around its mean. Consequently, the excess varying-tree trigram loss approaches

$$
\begin{array} { r } { \boxed { \mathcal { L } _ { t r i g } ^ { V T } - \mathcal { L } _ { t r i g } ^ { F T } \simeq ( 1 - f ) \ln 2 . } } \end{array}\tag{52}
$$

In Fig. 6, we compare the large-vf prediction with the trigram conditional entropies measured numerically by building varying-tree grammar instances and averaging over sentence endings. The agreement is already very good for $v \geq 8 ,$ , confirming that the large $v f$ prediction already captures well the initial varying-tree trigram loss in the training curves.

## D SUPPLEMENTARY FIGURES

We include here some supplementary empirical data which were alluded to in the main text. Firstly, in Fig. 7 we show the training curves for $D = 3 \mathrm { \ G P T { - } } 2$ style transformers in the varying-tree setting. As mentioned in the main text, even increasing significantly $d _ { \mathrm { e m b } } .$ , so that the parameter count matches the $D = 6$ model (this count grows quadratically with the embedding dimension), and training over a long window (up to $t \approx 1 0 ^ { 7 } )$ , we have not converged to the final Bayes-optimal loss.

Finally, in Figs. 8 and 9, we show the full held-out accuracy $\mathcal { A } _ { Z } ^ { ( k , t ) }$ profile across depth for all latents at all stages. These data are used for Fig. 3 (top right); specifically, we extract the accuracy value for each latent at each stage from the middle layer, where it has already saturated (any layer beyond the middle one would give the same result). We do this for $D = 1 2$ , although the $D = 6$ data show the same stagewise behaviour.

![](images/40fbd63ba87ee001c973b94c8013bd90e817618e3056de9f3b8fad5d7d3c85ef.jpg)  
Figure 7: Evaluation loss on the ternary sentence completion task, for depth $D = 3$ Transformers. To have roughly the same number of parameters as $( D = 6 , d _ { \mathrm { e m b } } = 5 1 2 )$ , we increase the embedding dimension to $d _ { \mathrm { e m b } } = 7 6 8$ . Sweeping over Adam hyperparameter settings (learning rate $\eta ,$ weight decay λ), we have not managed to achieve full convergence to the final Bayes-optimal loss. For reference, we include the $D = 6$ varying-tree training curve as well as the $D = 3$ fixed-tree training curve, which converges excellently.

(a) $\mu _ { 8 } ^ { ( 2 ) }$  
![](images/f601c8084dcd7bb2764af82238ab71960f07e8f21caf7bec7b06908df0005498.jpg)

(b) $\mu _ { 7 } ^ { ( 2 ) }$  
![](images/476f0fb5eab72ebc573cdb60ea1a08e4724744432706e7a9b597a226165fd1d8.jpg)

(c) $\mu _ { 1 } ^ { ( 1 ) }$  
![](images/ff2e3b87b6ec854881a809f235b83c6f6edb82b1f49277d87826245e15196e1d.jpg)  
Figure 8: Held-out probe accuracy $\mathcal { A } _ { Z } ^ { ( k , t ) }$ across layers for a $D = 6$ Transformer, for the three latents studied in the main text, at the 4 stage checkpoints.

(a) $\mu _ { 8 } ^ { ( 2 ) }$  
![](images/00ff4ae1b0e85e62e2840f7b18faad922d2b859957b918a877b6f478521de423.jpg)

![](images/93fd5617ab1c9ab6a132886e31d1558a711547c0b5b405c38d089621e7d74d85.jpg)

(c) $\mu _ { 1 } ^ { ( 1 ) }$  
![](images/ff9a41595632eeccede073466cb9018a339f8b1b6bed6864190ce5136b96fa21.jpg)  
Figure 9: Same as Fig. 8, but for a $D = 1 2$ Transformer.