# SEMIOTIC RELATIONS AND PROOF METHODS: A CROSS-GENRE STUDY OF ARGUMENT STRUCTURE WITH LARGE LANGUAGE MODELS

Edirlei Soares de Lima   
Academy for AI, Games and Media   
Breda University of Applied Sciences Breda, The Netherlands soaresdelima.e@buas.nl   
Marco A. Casanova   
Department of Informatics   
PUC-Rio   
Rio de Janeiro, Brazil   
casanova@inf.puc-rio.br   
Antonio L. Furtado   
Department of Informatics   
PUC-Rio   
Rio de Janeiro, Brazi   
furtado@inf.puc-rio.br

## ABSTRACT

When a direct proof of a statement S seems hard or even impossible to obtain, there may exist another statement (or set of statements) S<sup>∗</sup>, somehow related to S, on the basis of which S can be proved. In order to investigate what options can be used to move from S to S<sup>∗</sup>, four kinds of semiotic relations inspired by the four master tropes of semiotic research are briefly reviewed. Specifically, our syntagmatic, paradigmatic, antithetic and meronymic relations correspond, respectively, to metonymy, metaphor, irony and synecdoche. It is suggested that these four semiotic relations determine the options to move from S to S<sup>∗</sup>, leading to proof by inference, proof by analogy, proof by contradiction, and proof by case analysis. To examine how the four relations are actually used across different kind of argument, we complement the framework with an empirical study. We turn the four relations into explicit operational definitions and apply them to a cross-genre corpus of mathematical, legal, and everyday argument using a panel of large language models. We find that the relations are used very unevenly across genres: mathematical proofs draw on all four, whereas legal and everyday reasoning rely almost entirely on inference.

Keywords Computational Argumentation · Argument Structure · Large Language Models · Cross-Genre Analysis · Semiotic Relations · Proof Methods

## 1 Introduction

The objective of this paper is to investigate what options one has to move from a statement S that does not seem to be directly provable to some related statement (or set of statements) S<sup>∗</sup>, which could be shown to be true and to imply that S itself must be true. We suggest that there are four ways to move from S to S<sup>∗</sup>, enabled by what we have categorized as semiotic relations. These relations have been drawn from the so-calledfour master tropes, a topic of major interest in the area of semiotic research [5]. It is no coincidence that ‘trope’ comes from the Greek ‘τροπος’ from ‘τρεπειν’, ‘to turn’, akin to the notion of moving that underlies the present discussion.

Our four semiotic relations, together with their intuitive meaning, associated logical connectives, and corresponding tropes are shown in Table 1.

Table 1: Semiotic relations and their corresponding connectives and tropes.
<table><tr><td>Relation</td><td>Meaning</td><td>Connective</td><td>Trope</td></tr><tr><td>Syntagmatic</td><td>Contiguity, sequence</td><td>and</td><td>Metonymy</td></tr><tr><td>Paradigmatic</td><td>Similarity, alternatives</td><td>or</td><td>Metaphor</td></tr><tr><td>Antithetic</td><td>Opposition, negation</td><td>not</td><td>Irony</td></tr><tr><td>Meronymic</td><td>Hierarchy, details</td><td>part-of</td><td>Synecdoche</td></tr></table>

These four tropes were characterized as fundamental, among the numerous rhetorical tropes popular in Greco-Roman antiquity [47], first in the $\mathbf { X V I ^ { \mathrm { t h } } }$ century [48] and again in the $\mathrm { X V I I I ^ { t h } }$ century [55]. In modern times they were revived in a seminal study [2]. Their universality has been repeatedly emphasized, with the indication that they may constitute “a system, indeed the system, by which the mind comes to grasp the world conceptually in language” [8]. Applications to several topics have been reported, for instance to worldviews and ideologies [62] and, in our own work, to digital interactive composition of story-plots [9, 10, 11, 13, 23].

With respect to the names we assigned to the proposed semiotic relations, the terms ‘syntagmatic’ and ‘paradigmatic’ correspond to the two linguistic axes of de Saussure [17]. The term ‘antithetic’ reflects the fact that, according to Burke [2], the perspective induced by the irony trope is associated with dialectic, which features antithesis as a key concept expressing negation. Finally, in Winston et al. [64], wherein six types of part-of links are distinguished, one reads: “We will refer to relationships that can be expressed with the term ‘part’ in the above frames as ‘meronymic’ relations after the Greek ‘meros’ for part”.

Informally speaking, the preferred strategy to apply when S is not directly provable is to look for other statements, in the same domain, from which S could be deduced. If no clues are offered by the original domain, one may try to locate an analogue to $S$ in another domain, which may be more amenable to a successful treatment. Especially when $S$ is an assertion that something cannot hold, an often convenient option is to assume the contrary and then show that the assumption leads to an inconsistency. Finally, if a general proof of S is unfeasible, one may break down the problem into an exhaustive list of cases, to be handled separately one by one. The main thrust of this paper is that these four options to prove a statement $S$ in connection with a statement (or set of statements) $S ^ { * }$ – namely proof by inference, proof by analogy, proof by contradiction, and proof by case analysis – are determined by the four semiotic relations mentioned before.

The framework is illustrated in Section 2 through a small set of carefully chosen examples. Whether the four relations it proposes describe argumentation as it actually occurs, once they are applied beyond such selected cases to large and varied collections of real arguments, is a question those examples cannot settle. The present paper takes it up empirically. We convert the four relations into explicit operational definitions, together with a residual category for arguments that none of them capture, and apply the resulting scheme to a cross-genre corpus of more than a thousand real arguments, drawn from established sources of mathematical, legal, and everyday reasoning, using a panel of large language models from independent families. Our aim is to characterize how the four relations are used across these genres, how consistently they can be applied, and how often an argument rests on a single relation or on several acting together, taking their grounding in the master tropes as given.

The paper is organized as follows. Section 2 explores the application of the four proof methods, relying on examples to illustrate the connection of each method with the respective enabling semiotic relation. Sections 3 and 4 discuss a few problems arising from the complementary processes of finding a proof and expressing it convincingly. Section 5 reports the empirical study, in which the four relations are operationalized and applied across genres to characterize how they are used. Section 7 contains concluding remarks.

## 2 Applying the Semiotic Relations

Certain statements are obviously true by definition, or are verifiable through a simple inspection. Direct proof that something exists merely requires exhibiting an instance, even though some work may be required to construct it, as with the statement that there exist irrational numbers a and b such that $a ^ { b }$ is rational – which is usually evidenced by producing some series of equalities (which, curiously, can only be checked symbolically since the first two cannot be computed over the domain $\bar { \mathbb { Q } }$ of rational numbers):

$$
a = { \sqrt { 2 } } , \qquad b = \log _ { 2 } 9 , \qquad a ^ { b } = 3
$$

but it often happens that no such direct proof is feasible.

To prove a statement $S$ in such circumstances, we can move to some other statement (or set of statements) $S ^ { * }$ , which must be preliminarily shown to be linked to $S$ by a semiotic relation, and then try, recursively, to prove $S ^ { * }$ . There are (at least) four such “moves”, each of them corresponding to one of the rhetorical master tropes.

We say that a syntagmatic relation holds between S and $S ^ { * }$ if S is a logical consequence of $S ^ { * }$ . The associated trope is metonymy. The resulting method is proof by inference.

A paradigmatic relation holds between $S$ and $S ^ { * }$ if after suitable mappings the relevant features of $S$ can be converted into features of $S ^ { * }$ . The associated trope is metaphor. The resulting method is proofby analogy.

An antithetic relation holds between $S$ and $S ^ { * } \operatorname { i f } S ^ { * }$ could be shown to be inconsistent $\mathrm { i f } \sim S$ were true. The associated trope is irony. The resulting method is proofby contradiction (also called reductio ad absurdum).

A meronymic relation holds between $S$ and $S ^ { * } \operatorname { i f } S ^ { * }$ is a set of statements into which S can be decomposed exhaustively.   
The associated trope is synecdoche. The resulting method is proof by case analysis.

These proof methods are based on what might be called meta-theorems, expressed below in a semiformal style (for a more rigorous treatment, see [18]), in terms of theories (denoted by Γ) and sentences (denoted by ϕ and γ):

proofby inference   
if $\cdot \Gamma \vdash ( \gamma  \phi )$ and $\Gamma \vdash \gamma ,$   
then $\Gamma \vdash \phi .$   
proofby analogy   
let $\dot { \pi } [ \Gamma ] = \ddot { \Gamma ^ { \prime } }$ be a faithful interpretation;   
i $\mathrm { \Phi } \mathrm { T } ^ { \prime } \vdash \mathrm { \Phi } ^ { \prime }$ where $\phi ^ { \prime } = \phi ^ { \pi }$   
then $\Gamma \vdash \phi$   
(noting thatfaithful means $\gamma \in \Gamma \Leftrightarrow \gamma ^ { \pi } \in \Gamma ^ { \prime } )$   
proofby contradiction   
if $( \mathrm { \dot { T } } ; \lnot \phi )$ is inconsistent,   
then $\Gamma \vdash \phi$   
(noting that inconsistent means there is some $\gamma$ such that $( \Gamma ; \lnot \phi ) \vdash \gamma$ and $( \Gamma ; \lnot \phi ) \vdash \lnot \gamma )$   
proof by case analysis   
let $\mu ( \phi ) = \{ \phi _ { 1 } , \phi _ { 2 } , . . . , \phi _ { n } \}$ be an exhaustive decomposition;   
if $\mathrm { \ddot { T } } \vdash \dot { \phi } _ { i }$ for all $1 \leq i \leq n ,$   
then $\Gamma \vdash \phi$   
(noting that exhaustive means $\left\{ \phi _ { 1 } , \phi _ { 2 } , \ldots , \phi _ { n } \right\}$ tautologically implies $\phi )$

## 2.1 Proof by Inference

Example 1: “Socrates is mortal”. This time-honoured example recognizes that mortality is a human condition, as expressed by the rule: $\forall x ( h u m a n ( x )  m o r t a l ( x ) )$ ). Since Socrates is known as a human being, the rule applies and the statement follows as a consequence.

Example 2: “Harry, who was born in Bermuda, is a British subject”. Stephen Toulmin has argued convincingly that the conventional syllogism structure must be expanded to deal with reasoning in the domain of justice [52]. So it is not enough to consider what he calls the data (Harry was born in Bermuda), the claim (Harry is a British subject) and the warrant (a man born in Bermuda is a British subject). To these three elements he adds a modality or, to use his own terms, a qualifier (presumably), given that the rule admits exceptions that constitute a possible rebuttal (unless both his parents were aliens, or he has become an American citizen, $\mathrm { o r \ldots \rangle }$ . But, even more characteristic of legal argument, is the warrant (statutes and other legal provisions); indeed the judicial system is governed by positive law (as opposed to natural law), which must have been officially established, and which may differ for different countries (e.g. notice among the exceptions the prevalence of ius sanguinis over ius soli, in contrast to Brazilian norms). Toulmin’s scheme can best be comprehended under the form of a diagram, as shown in Figure 1.

![](images/b0629edf04fe885a3e351db5d7a86f9818fb09343383f5355dabcb06bea79e3e.jpg)  
Figure 1: Toulmin’s argument scheme [52].

To Toulmin’s remarks one must add that the existence of what he calls the ‘data’ may not be recognized in justice if not officially registered as well (e.g. via a birth certificate). In database terminology this corresponds to the ‘closed world assumption’ [4]. Also recall the assumption in criminal law that a defendant is judged ‘not guilty’ (thus avoiding the term ‘innocent’) unless proved responsible for the alleged offence, which in turn must have been exactly specified by a previous law (nullum crimen sine prævia lege pœnale). All these considerations bring to mind the principle of ‘negation as finite failure’, also explained in $[ 4 ] - \sim \bar { S }$ holds whenever S neither resides in the database nor can be derived from the stored data and the rules that have been explicitly defined.

## 2.2 Proof by Analogy

Example 3: “There can be no efficient algorithm to determine the minimum number of schedules for tests of a group of students, so that no student will miss a test because its schedule coincides with that of some other course in which the student is enrolled”. Establishing non-conflicting schedules has an analogue in graph theory, if courses are mapped into nodes, and the fact that two courses $c _ { 1 }$ and $c _ { 2 }$ have one or more students in common is mapped into an edge connecting the nodes labelled $c _ { 1 }$ and $c _ { 2 } .$ . Then the original problem is converted into the problem of finding the chromatic number of a graph, which has been shown to be NP-complete (and hence of intractable computational complexity) [28].

Example 4: “A Buddhist monk begins at dawn one day walking up a mountain, reaches the top at sunset, meditates at the top overnight until, at dawn, he begins to walk back to the foot of the mountain, which he reaches at sunset. Make no assumptions about his starting or stopping or about his pace during the trips. Is there a place on the path which the monk occupies at the same hour of the day on the two trips?” The solution given in [21] involves a close analogue for which, rather surprisingly, no mathematical treatment is required, and in fact the answer is immediately evident The mappings involve blending the scene of the monk climbing with that of his return. The action takes place in a single day, with the monk and his double walking in opposite directions – and so inevitably meeting himself at some intermediate place.

## 2.3 Proof by Contradiction

Example 5: “There exists an infinity of prime numbers”. Assume, on the contrary, that the primes form a finite set $L = \{ p _ { 1 } , p _ { 2 } , . . . , p _ { n } \}$ . The proof dates from ancient times [19]. Taking all the primes in $L ,$ one can obtain: $P = p _ { 1 } \times p _ { 2 } \times \cdot \cdot \cdot \times p _ { n } + 1$ . The number P calculated in this way is either a new prime, in which case we already have a contradiction, or a multiple decomposable into prime factors: $P = q _ { 1 } \times q _ { 2 } \times \cdot \cdot \cdot \times q _ { m }$ . But the $q _ { i }$ should be different from the prime numbers in $L ,$ since $P$ is not divisible by any of them (the division would always yield 1 as remainder). So the $q _ { i }$ would be new primes, again contradicting the ${ \sim } S$ assumption.

## 2.4 Proof by Case Analysis

Example 6: “The absolute value of the sum of two non-zero numbers is less than or equal to the sum of their absolute values”. In formal notation: $| a + b | \leq | a | + | b |$ . There seem to be four cases, which can be easily treated by elementary arithmetic:

case 1. if a and b are positive, the left side is equal to the right side;

case 2. if a is positive and b negative, the left side is less than the right side;

case 3. if a is negative and b positive, the left side is less than the right side;

case 4. if both a and b are negative, the left side is equal to the right side.

Actually the four cases are reducible to three, by collapsing cases 2 and 3 in view of the commutative property of addition.

Example 7: “The sum of all natural numbers from 0 to n is equal to $n \times ( n + 1 ) / 2 ^ { , , }$ . To show case by case that this holds for any value of n would lead to an infinite process. Fortunately, thanks to a technique known as finite induction, the problem can be reduced to the following cases:

case 1. for $n = 0 .$ , the result of computing the formula is 0, which is obviously correct;

case 2. assume that for $n = i$ the formula works correctly, i.e.: $0 + 1 + \cdot \cdot \cdot + i = i \times ( i + 1 ) / 2 ;$

case 3. for $n = i + 1$ , it must be shown that the formula yields $( i + 1 ) \times ( ( i + 1 ) + 1 ) / 2$ . This last case, called the induction step, can be established by using the assumption for $n = i$ and then performing a series of simple algebraic transformations: $( 0 + 1 + \cdot \cdot \cdot + i ) + ( i + 1 ) = i \times ( i + 1 ) / 2 + ( i + 1 ) =$

$$
( i \times ( i + 1 ) \bar { + } 2 \times ( i + 1 ) ) / 2 = ( i + 1 ) \times ( i + 2 ) / 2 = ( i + 1 ) \times ( ( i + 1 ) + 1 ) / 2 .
$$

Example 8: “Four colours are enough to colour a geographical map so that no two adjacent political units have the same colour”. This is the so-called four colours conjecture, which defeated the attempts of many researchers for a long time, until being finally established as a proven theorem by two researchers working together in 1976 [1]. They first managed to identify an exhaustive list of cases, corresponding to 1936 “irreducible configurations”. To handle such an overwhelming number of cases, they were forced to appeal to computer support. Subsequent efforts were made to reduce this number, but to our knowledge it still remains quite large.

## 3 Finding a Proof

Finding the proof of a statement $S$ and expressing the proof are more often than not two sharply different processes. In particular, to find a proof by inference, a person must start in a backward direction by applying a reasoning strategy called abduction [39]. Its purpose is to search for some hypothesis $S ^ { * }$ that may be used next to justify S. To perform abduction, one assumes that S holds and then looks for some existing rule of the form $S ^ { * } \to S$ relating S and $S ^ { * }$ . In a sense, abduction involves traversing the rule in a right-to-left direction, inversely therefore to how we handle deduction, on which the process of expressing a proof by inference is based. Recall that medical doctors rely on abduction while they try to trace back the observed symptoms to diseases that may have caused them, and that differential diagnosi becomes necessary if more than one disease is hypothesized.

Indeed, the reputed mathematician George Polya confirmed this primary opening role of abductive reasoning, when he asserted that guessing should precede proving [45]: “Finished mathematics presented in a finished form appears as purely demonstrative, consisting of proofs only. Yet mathematics in the making resembles any other human knowledge in the making. You have to guess a mathematical theorem before you prove it; you have to guess the idea of the proof before you carry out the details”. On the other hand, an assertion of the Indian mathematician Srinivasa Ramanujan [49]: “Sir, an equation has no meaning for me unless it expresses a thought of GOD”, attributes his guesses to a sort of inspiration, which suggests that creative abduction may result from lucky intuition rather than systematic reasoning.

The rules themselves should have been formulated beforehand, typically by induction, i.e. by observing that S occurs whenever $S ^ { * }$ does, and that this can be attributed to logical implication or at the very least to probabilistic evidence, rather than to fortuitous coincidence (the post hoc ergo propter hoc fallacy). After the advent of computers, data mining runs [25] (involving statistical correlation and several other techniques) began to be routinely performed over large data repositories to discover such useful rules.

For proof by analogy, the preliminary search is even trickier. One must be able to look for analogues in domains other than that of the statement on hand, and abstract the essentials from knowledge expressed in a widely distinct formalism. Perhaps the required competence hinges on access to a repertoire of well-structured and well-indexed mental forms, either characterized as ideas [41], or archetypes [27], or basic metaphors [33], or scripts [50], etc. Whether they are inborn or acquired is the topic of endless debate.

Children are encouraged very early in school to answer analogy questions in the form “A is to B as C is to what?”. Indeed proportionality is a helpful criterion to formulate the mappings between the features of the original statement and the candidate analogue. A modern discipline, case-based reasoning [30], attempts to automate the search for analogues, ideally working on some rich computer-accessible library. One technique to construct such libraries involves extracting patterns from the observed detailed descriptions through most specific generalization [7, 22].

For proof by contradiction, determining $S ^ { * }$ can sometimes be almost immediate. In Example 5, in opposition to the notion of an infinity of numbers with the property of being prime, one promptly perceives, without leaving the original domain, that a contrary notion is that the existing primes form a finite set – and from that follows the idea of using the members of this set to construct the statement that will lead to a contradiction. But other problems are not so simple. We shall look at the famous Fermat’s Last Theorem (proposed in 1637, just before his death), which was expressed by a simple algebraic equation, but was proved by contradiction much later [63], using a rather advanced geometry result about the modularity of elliptic curves. So it combines analogy with contradiction (plus long series of inferences) and, on top of all that, it illustrates how crucial it is to restrict the cases to be covered in a proof by case analysis to precisely what is required to prove the statement – it became eventually clear that it suffices to consider semistable elliptic curves. Once again as in Example 8, we shall only provide a very brief and very informal note, since a rigorous account would require mathematics well beyond the scope of this paper.

Example 9: “No three positive integers $a , b ,$ and c can satisfy the equation $a ^ { n } + b ^ { n } = c ^ { n }$ for any integer value of n greater than two”. Thanks to the effort of a number of researchers from 1637 to 1995 (when Wiles’s paper was published), it was proved, case by case, that any solution to this deceptively simple equation could be used to generate a non-modular semistable elliptic curve, whereas it was also proved that all such elliptic curves had to be modular – a contradiction that implies that there can be no solutions to the equation, thus finally transforming the conjecture into a theorem.

## 4 Expressing a Proof

Let us now turn to the second process mentioned at the beginning of the previous section, namely, having succeeded in proving that a sentence is true by a judicious application of methods such as those exemplified in Section 2, how to suitably express the demonstration to other people. To realize what is involved it is convenient to view this as a communication process, requiring our attention to at least the six items contained in the diagram shown in Figure 2.

![](images/54ef9e40e26e853fabefd21b8cbd7ee0b6e6d8705d038cf68d37fb8235b09e86.jpg)  
Figure 2: Jakobson’s model of the communication process [26].

In words: the researcher (sender) who devised the proof formulates the demonstration (message) in some formal or informal language (code) and passes it through some medium (channel) to an interested person (receiver) who should be able to understand it. The cultural environment prevailing at a given place and time (context) imposes conditions that may exert a favourable or unfavourable influence on the outcome of the process.

Of course the sender must make sure that the proof is correct with respect to both contents and form, but the effort can ultimately succeed only if the receiver can decode the demonstration to the point of actually learning it and taking maximum advantage of the new knowledge thus acquired. The choice of a formalism is sometimes crucial to this end. For instance, the use of finite induction for Example 7 above is considered by Chateaubriand [6] as inappropriate for teaching beginning students. Even if the algebraic manipulations can be followed by them, the stepwise argument would not “relate meaningfully” to the students, whereas a more effective presentation relying on a pictorial sketch would have a better chance of eliciting a reaction of “dawning understanding”. Lakoff and Núñez [34] provide several other similarly intuitive explanations using basic metaphors, for instance to show how complex numbers can be clearly understood by blending arithmetic and geometric notions.

More generally, the correct connection from sender to message in Jakobson’s communication scheme is just one prerequisite of the process. It corresponds to the adequacy of the signifier to the signified, in Saussure’s terms [17]. But communication must reach its final destination, the receiver, bringing to mind the three-element view – object, representamen, interpretant – advocated by Peirce [39]. It is through this path that the human intellect can, although incompletely and imperfectly, grasp a glimpse of the real.

Perversely, even if something is understood correctly, a naive receiver may draw one or more wrong conclusions (cf. the notion of misconstruals in Webber [59]) from it. It is a well known fact that statistical reports, though in themselves possibly correct, are very often misinterpreted out of inexperience or bad faith. But let us examine two kinds of wrong conclusions that may result from an undue application of a seemingly universal principle to a true statement: “if a statement S involving a is true and a = b, the substitution of b for a yields a statement that is also true”.

First, take the true statement “the expression 3 + 1 + 2 contains three terms”, and note that 3 + 1 + 2 = 5 + 1. By substitution, “the expression 5 + 1 contains three terms” should be true, but it is patently false. Clearly the substitution could not have been done, since this particular statement is an argument de dicto, whereas the value equality is a de re consideration. Or we might say, perhaps, that the statement referred to a signifier and the comparison to a signified, in Saussure’s terminology.

The second case is a little less trivial. Suppose the statement “Gottlob believes that Venus is a planet” is true, and consider the relatively well-known equality Venus = Evening Star. The substitution, giving “Gottlob believes that the Evening Star is a planet” is not necessarily true, however. Even if Gottlob is aware of the equality, he may have never taken the trouble to perform the substitution, and therefore the maximum that we could say in this case, introducing a modality, is that “Gottlob possibly believes that the Evening Star is a planet”. The full-fledged substitution would only be warranted if both the equality and the substitution took place in Gottlob’s head, i.e. at the level of Peirce’s interpretant.

## 5 An Empirical Evaluation of the Semiotic Relations

Sections 2 through 4 have developed the four relations and the two processes that surround any proof, its discovery and its communication. We now return to the relations themselves and ask a question that the illustrative examples of Section 2 cannot settle: whether they describe reasoning as it actually occurs across many arguments and different domains. To address this empirically, we provide operational definitions for the four relations and use a panel of large language models to apply them to a corpus of 1,126 arguments drawn from mathematics, law, and everyday reasoning.

Applying the relations at this scale enables an examination of questions left open by the examples in Section 2. Specifically, we investigate the extent to which the four relations cover real argumentation, the uniformity of their usage, and the consistency with which a single argument is assigned a relation. We further determine whether the relations function as mutually exclusive alternatives or in combination, and, where the data permit, whether the labels reflect the structure of the argument rather than its specific lexical choices. The remainder of this section addresses these questions in turn.

## 5.1 Methodology

## 5.1.1 Operational Definitions of the Relations

The evaluation begins by converting the four relations into operational definitions applicable to any argument. Each definition opens with the wording from Section 2 and adds criteria specifying the classes of argument it covers. The four relation definitions below are inserted verbatim into the prompt supplied to the language models that perform the annotation (Section 5.1.3), and are applied exactly as shown:

• SYN (syntagmatic; inference). S is a logical consequence of S<sup>∗</sup>. The reasoner locates a rule or prior result and applies it to reach S. Includes syllogism, rule application, chained deduction, and defeasible warrant-based argument.

• PAR (paradigmatic; analogy). After suitable mappings, the relevant features of S are converted into features of $S ^ { * }$ . The reasoner solves the mapped problem instead of the original one. The analogue often lies in another domain, as when a scheduling problem is mapped onto graph colouring, but it need not: mapping one case onto a structurally similar case within the same domain is still PAR, as when a problem is solved by blending a situation with a parallel version of itself. What matters is that a correspondence is drawn between two cases and the argument runs through it. Includes reduction, structural analogy, argument from a parallel case, and conceptual blending.

• ANT (antithetic; contradiction). S<sup>∗</sup> is shown to be inconsistent under the assumption of ∼ S. Includes reductio ad absurdum and any argument whose force comes from the untenability of the denial. Concession is not ANT. Granting a point and then arguing past it (“Of course X, but Y”; “Admittedly X; nevertheless $\mathbf { Y } ^ { \prime \prime } )$ is a rhetorical move, not an argument from inconsistency; label such passages by whatever establishes the main claim. Nor is mere contrast between two things ANT. The test is whether denying the claim is shown to lead to something untenable.

• MER (meronymic; case analysis). S<sup>∗</sup> is a set of statements into which S decomposes exhaustively, each handled separately. Includes case splits, exhaustion, and finite induction (base case + induction step).

The scheme also includes a residual category, NONE, designated for cases that satisfy none of the four relations or advance no argument, such as unsupported assertions or factual recitations without inference. This category records the absence of a relation rather than annotator uncertainty, and arguments that fit one relation but are difficult to place are assigned their best-fitting relation. Each argument is assigned a single dominant relation, defined as the one upon which the argument depends such that its removal would leave the claim unestablished, alongside any number of subordinate relations that perform genuine supporting work. The complete annotation prompt, including the verbatim wording for NONE and the model instructions, is presented in Appendix A.

## 5.1.2 Corpus

The evaluation is conducted on a corpus covering the three genres of reasoning illustrated in Section 2, namely mathematical proofs, legal arguments, and everyday reasoning. The corpus contains 1,126 items, each between 50 and

300 words in length, drawn from three established sources and deduplicated. Its composition is summarized in Table 2, with the sources described below.

Table 2: Composition of the corpus.
<table><tr><td>Genre</td><td>Source</td><td>Unit</td><td>N</td></tr><tr><td>Mathematics</td><td>NaturalProofs (ProofWiki) [60]</td><td>Theorem-proof pairs</td><td>627</td></tr><tr><td>Legal</td><td>ECHR corpus [24]</td><td>Argument spans</td><td>400</td></tr><tr><td>Everyday</td><td>Microtext Corpus [40]</td><td>Single arguments</td><td>99</td></tr><tr><td colspan="3">Total</td><td>1,126</td></tr></table>

The mathematical items comprise complete theorem-and-proof pairs drawn from a ProofWiki-derived collection [60], with formatting markup removed. The everyday items consist of short single-argument texts on policy questions from the Argumentative Microtext Corpus [40]. The legal items are argument spans from European Court of Human Rights decisions [24], which also include expert annotations related to the argument type, a label that is withheld from the models and used only for the external validation presented in Section 5.2.5.

The distinct argumentative structures of the three genres limit cross-genre comparison, as mathematical and everyday items constitute complete argumentative units while legal items function as components of a larger judgment. A legal item may state a rule, report a party’s submission, or reach a conclusion supported by an adjacent part of the decision. Consequently, results are reported separately by genre rather than pooled. Since none of the three sources is a random sample of its genre, the reported distributions characterize these specific corpora rather than the genres in general.

The corpus, annotations, and analysis code are publicly available at https://doi.org/10.5281/zenodo.22691375.

## 5.1.3 Annotation Procedure

A panel of large language models applies the relations to the corpus, making labelling consistency measurable and ensuring that the models follow the operational definitions in Section 5.1.1 rather than relying on prior knowledge. This subsection details the panel, the labelling protocol, and the blinding conditions.

The panel comprises three large language models from independent families, as listed in Table 3, a design choice that reduces the risk that labels reflect the bias of a single type of model rather than the structure of the arguments. All three are open-weight models from unrelated developers, and the panel is intended to be diverse rather than matched in capability, since the study measures how consistently the scheme is applied rather than the accuracy of any one model. The models ran locally on a vLLM inference server,<sup>1</sup> with an OpenAI-compatible API, and were queried under identical settings.

Table 3: The large language models comprising the annotation panel.
<table><tr><td>Model</td><td>Developer</td><td>Parameters</td></tr><tr><td>Qwen3.8-27B</td><td>Alibaba</td><td>27B</td></tr><tr><td>Gemma-4-31B</td><td>Google</td><td>31B</td></tr><tr><td>GLM-5.2-753B</td><td>Z.ai</td><td>753B</td></tr></table>

Each model labelled every item three times at a sampling temperature of 0.7, producing nine independent judgements per item and 10,134 labels in total (1,126 items × 3 models × 3 runs). This repeated sampling at a non-zero temperature, rather than a single deterministic pass, allows the measurement of self-consistency, such that an item labelled identically across all three runs provides stronger evidence than one whose label varies. For each item, the model was first required to state, in free text, the claim being established (S), the support on which it rests (S ), and how the two are connected, before recording the dominant relation, any subordinate relations, and a confidence rating on a three-point scale (1 = guessing, 2 = plausible, 3 = clear). Eliciting this statement prior to the label separates the identification of the argument from the assignment of a relation, allowing the two to be examined independently. The complete annotation prompt is provided in Appendix A.

The models were blinded to all information beyond the text of each item, which was presented in isolation with no indication of its genre, source, or selection procedure. The instruction wrapper named neither the originating work, nor the field of semiotics, nor the theorists associated with the four tropes. Furthermore, the four relation definitions were presented in a randomized order on each call, with NONE fixed in the final position, ensuring that the ordering of the options could not influence the labels. These measures concern only what the models see at inference time. Since the three corpus sources are publicly available, the item texts were most likely part of the models’ pretraining data. This, however, does not contaminate the results in the way it would a benchmark, since the task provides no gold labels to recover and the scheme applied here is introduced in this paper. Prior exposure could at most shape how an item is read, and it is the consistency of that reading that Section 5.2.3 reports.

## 5.2 Results

The following subsections characterize the scheme’s behaviour when applied to the corpus, covering the distribution of relations across genres, the manner in which those relations combine, the reliability with which they are applied, whether the prevalence of inference reflects genuine judgements rather than a default, and a bounded external test of validity.

## 5.2.1 Distribution Across Genres

The first result of the study to be analysed is the distribution of labels across the four relations, computed separately for each genre. Table 4 reports the share of items, for each genre and category, whose label, defined as the most frequent of the nine judgements, falls within that category.

The distribution exhibits two patterns. First, the four relations cover argumentation broadly, with almost every item outside the legal genre belonging to one of them and the residual NONE category remaining small. Second, the relations are used very unevenly across genres. Mathematics draws on all four, where inference and case analysis account for most items (SYN 56.5%, 95% CI [52.6, 60.3]; MER 32.1%, [28.5, 35.8]) and contradiction occurs at a non-trivial rate (ANT 10.2%, [8.1, 12.8]). Legal and everyday reasoning, in contrast, concentrate heavily on inference (SYN 83.5% and 93.9%, respectively), with the other three relations nearly absent. The scheme thus covers reasoning broadly, though it distinguishes among the relations much more clearly within mathematics than in the other two genres.

Table 4: Distribution of the modal label by genre, given as the count of items with the corresponding percentage of the n items in each genre in parentheses.
<table><tr><td rowspan="2">Genre</td><td rowspan="2">n</td><td colspan="5">Modal Label</td></tr><tr><td>SYN</td><td>PAR</td><td>ANT</td><td>MER</td><td>NONE</td></tr><tr><td>Mathematics</td><td>627</td><td>354 (56.5%)</td><td>5 (0.8%)</td><td>64 (10.2%)</td><td>201 (32.1%)</td><td>3 (0.5%)</td></tr><tr><td>Legal</td><td>400</td><td>334 (83.5%)</td><td>18 (4.5%)</td><td>0 (0.0%)</td><td>5 (1.2%)</td><td>43 (10.8%)</td></tr><tr><td>Everyday</td><td>99</td><td>93 (93.9%)</td><td>5 (5.1%)</td><td>1 (1.0%)</td><td>0 (0.0%)</td><td>0 (0.0%)</td></tr></table>

The share of items labelled NONE differs significantly across genres. For genres composed of complete argumentative units, this share is minimal (mathematics 0.5%, everyday 0.0%), whereas the legal genre exhibits a higher rate (10.8%, 95% CI [8.1, 14.2]). This difference reflects the structure of the corpus described in Section 5.1.2 rather than a failure of the four relations to cover legal reasoning, as a legal item that reports a submission or states a rule without drawing a conclusion exhibits none of the four relations and is consequently labelled NONE.

## 5.2.2 Combination and the Internal Structure of the Scheme

Beyond the marginal distribution, the labels reveal how the four relations interact. The share of items with a subordinate relation varies by genre, comprising 36.0% in mathematics, 6.1% in everyday, and 2.2% in legal, based on a majority of votes. These numbers indicate that the relations co-occur rather than acting as exclusive alternatives, most often in mathematical proofs.

Broken down by the dominant relation, the combinations exhibit a clear asymmetry (Table 5). Inference is nearly always self-sufficient, with only 5.9% of items whose dominant relation is SYN carrying a subordinate relation. Contradiction and case analysis operate in the opposite direction, incorporating a subordinate relation in 80.0% and 68.6% of cases, respectively. In most of these instances, the subordinate relation is inference. Given that contradiction and case analysis appear almost exclusively in mathematics, this pattern is largely a feature of mathematical proofs. Arguments based on contradiction or exhaustive case splits typically perform inferential steps within the argument, whereas inference, when dominant, usually stands alone.

Table 5: Combination structure by dominant relation, over the items whose modal label is one of the four relations. “With subordinate” gives the number and percentage of such items that carry at least one subordinate relation on a majority-of-votes basis.
<table><tr><td>Dominant</td><td>n</td><td>With subordinate</td><td>Most common subordinate</td></tr><tr><td>SYN</td><td>782</td><td>46 (5.9%)</td><td>MER</td></tr><tr><td>PAR</td><td>27</td><td>3 (11.1%)</td><td>SYN</td></tr><tr><td>ANT</td><td>65</td><td>52 (80.0%)</td><td>SYN</td></tr><tr><td>MER</td><td>204</td><td>140 (68.6%)</td><td>SYN</td></tr></table>

## 5.2.3 Reliability

Since the labels derive from an automated panel, their reliability depends on the consistency with which the scheme is applied, encompassing both within-model stability across repeated runs and agreement between independent models. Table 6 reports three measures of this consistency. Self-consistency, defined as the rate at which a single model returns the same label for an item across three runs, ranges from 81.1% to 89.1% across the three models. Cross-model unanimity, the rate at which all three models agree on their per-model modal label, is 79.2% (95% CI [76.8, 81.5]), while chance-corrected agreement, measured by Krippendorff’s α [31] over all nine judgements per item, is 0.712 overall.

Table 6: Reliability of the annotation, with Krippendorff’s α reported by genre. Self-consistency and cross-model unanimity are corpus-wide.
<table><tr><td>Measure Value</td></tr><tr><td>Self-consistency across runs (range over models) 81.1-89.1%</td></tr><tr><td>Cross-model unanimity (all three agree) 79.2%</td></tr><tr><td>Krippendorff&#x27;s α, overall 0.712</td></tr><tr><td>mathematics 0.748</td></tr><tr><td>everyday 0.582</td></tr><tr><td>legal 0.551</td></tr></table>

The reported per-genre Krippendorff’s α values require careful interpretation. In genres where a single relation dominates, such as inference in the legal and everyday genres, expected chance agreement is high, which depresses α even when raw agreement remains identical. Consequently, relying on α alone would mischaracterize the more concentrated genres as less reliably measured, despite raw agreement being highest in precisely those contexts. The most frequent disagreement occurs between MER and SYN, followed by NONE against SYN and PAR against SYN, a pattern consistent with inference serving as the default with which the marked relations are most easily confused. These measures reflect the consistency of scheme application rather than the correctness of the labels.

## 5.2.4 Confidence in the Inference Label

The distribution results presented in Section 5.2.1 point to the high frequency of inference, a pattern that holds across all genres and covers the large majority of legal and everyday reasoning. A natural objection is that models default to SYN when no other relation stands out, making its prevalence an artefact of the instrument rather than a property of the arguments. This objection leads to a testable prediction. Since a fallback is a choice made under uncertainty, SYN labels should show low confidence. The confidence rating recorded with each label allows us to test this prediction directly.

The confidence ratings contradict this prediction (Table 7). SYN labels exhibit a mean confidence of 2.93 on the three-point scale, with 93.4% assigned the maximum rating. This value is comparable to case analysis (2.94) and contradiction (2.98), and exceeds that of analogy (2.78). Only analogy and the residual category NONE attract lower confidence, indicating that the models place their uncertainty there. Inference is therefore not the low-confidence fallback that a defaulting instrument would produce; the models are as sure of a SYN label as of a label from any of the marked relations.

## 5.2.5 External Validity: a Bounded Discrimination Test

The evaluations presented in the previous sections are fully based on the annotations produced by the large language models, which let us analyse how the relations are distributed and combined, how consistently they are applied, and whether inference was a default option instead of a real inference label. However, none of these measures compares labels against an independent standard, meaning they cannot demonstrate that a label reflects the structure of an argument rather than its surface wording. The legal corpus permits one bounded test of this specific distinction, employing the argument-type annotations from [24] as an expert key that is never presented to the large language models.

Table 7: Confidence by dominant label, computed over all judgements assigned each label. Mean confidence is on the three-point scale (1 = guessing, 2 = plausible, 3 = clear).
<table><tr><td>Label</td><td>Mean confidence</td><td> $\%$  at maximum</td></tr><tr><td>SYN</td><td>2.93</td><td>93.4</td></tr><tr><td>PAR</td><td>2.78</td><td>78.5</td></tr><tr><td>ANT</td><td>2.98</td><td>97.6</td></tr><tr><td>MER</td><td>2.94</td><td>94.1</td></tr><tr><td>NONE</td><td>2.76</td><td>75.7</td></tr></table>

The expert scheme and our labels are not directly comparable, as one records the kind of legal argument while the other records the relation between S and $S ^ { * }$ . This is therefore a discrimination test, particularly because several of Habernal’s argument types produce items that cite many earlier court decisions and appear similar on the surface, even though the underlying arguments differ. The central question is whether our labels can distinguish these types, with the clearest contrast lying between two such categories. In distinguishing a prior case, a cited precedent is shown not to apply, so the argument proceeds through a comparison between cases, a structure our definitions predict as PAR. In prior case law, earlier decisions are cited as authority for a rule that is then applied, which the definitions predict as SYN. Because both types cite prior decisions with comparable frequency, any difference in our labels between them cannot stem from the citations; rather, a higher rate of PAR labels on distinguishing than on prior case law would reflect the argument itself. The test also covers comparative law, which argues from other jurisdictions (predicted PAR), and subsumption, which applies a rule to the facts (predicted SYN). This mapping was fixed in advance, derived from the definitions and established before any labels were inspected.

Distinguishing and comparative law annotations are rare, and the 400 legal items in the main analysis contain too few to estimate a PAR rate. To address this, we add 40 further legal items of rare argument types from the same corpus [24], selected on the expert argument type assigned by the original annotators and thus independently of any model label. Of these, the distinguishing and comparative law items are the ones this test uses. These items are used only for this test and are excluded from the distribution results in Section 5.2.1.

The results indicate that the labels separate the types as predicted (Table 8). PAR labels appear on distinguishing and comparative law at a rate of 40.7% (11 of 27, 95% CI [24.5, 59.3]), compared to 5.1% (4 of 79, [2.0, 12.3]) on prior case law (Newcombe 95% CI [17.9, 54.5] [38]; one-sided Fisher exact $p = 3 . 2 \times 1 0 ^ { - 5 } )$ . The separation is even stronger on distinguishing alone (50.0%, 11 of 22; $p = 3 . 7 \times 1 0 ^ { - 6 } )$ , and the comparative-law items (5 in total) receive no PAR labels, so the effect is due entirely to distinguishing. Subsumption attracts SYN at 78.6% (173 of 220, 95% CI [72.8, 83.5]), well above the 45.5% rate on distinguishing. Since SYN is the dominant label in the corpus, this serves as a sanity check rather than strong evidence.

Table 8: Discrimination-test rates on legal items. Each row gives the rate of one relation within one expert type (or pair of types).
<table><tr><td>Measure</td><td>n</td><td>Rate</td></tr><tr><td>PAR on distinguishing + comparative law</td><td>27</td><td>40.7%</td></tr><tr><td>PAR on distinguishing only</td><td>22</td><td>50.0%</td></tr><tr><td>PAR on prior case law (baseline)</td><td>79</td><td>5.1%</td></tr><tr><td>SYN on subsumption</td><td>220</td><td>78.6%</td></tr></table>

## 6 Related Work

The idea that the search for a proof follows a small number of recurring methods has a long history in the study of mathematical practice. Pólya’s account of heuristics catalogues strategies such as reasoning by analogy, generalization, decomposition into subproblems, and working backwards from the goal [43, 44], several of which correspond closely to the relations examined here. Lakatos, in turn, portrays mathematics as advancing through cycles of proof and refutation, in which a conjecture and its proof are reshaped by counterexamples [32]. While these works describe how proofs are found and revised, the framework presented in this paper makes a different and more specific claim, that the choice among such methods is governed by the same four relations that underlie the master tropes.

The structure of argument more broadly has been formalized in ways that are also relevant to the framework. Toulmin analyses an argument into data, claim, warrant, and further qualifying elements [52], a model that appears in Example 2; Walton’s argumentation schemes catalogue the stereotypical patterns of presumptive reasoning, each paired with a set of critical questions [56, 57]. Such schemes aim at fine-grained coverage of argumentative moves and are assembled from the bottom up, whereas the present framework derives four relations from the master tropes, at a higher level of abstraction. A single one of our relations subsumes many individual schemes, so the two are complementary rather than competing.

The empirical evaluation presented in Section 5, by contrast, connects to the computational study of argument. It is most closely related to work in argument mining, which segments texts into argumentative units and classifies their roles or types. General-purpose schemes typically reduce an argument to premises and a claim linked by relations of support or opposition, whereas domain-specific efforts adopt richer typologies. For example, the corpus of European Court of Human Rights decisions used in our study annotates a fine-grained set of expert argument types [24]. Our relations operate at a different level. Rather than labelling components or cataloguing genre-specific patterns, they characterize the single move that carries an argument from its claim S to the supporting statement S , and they are the same four across every genre.

The relations are also distinct from the discourse relations of frameworks such as Rhetorical Structure Theory [37] and the Penn Discourse Treebank [46], which annotate coherence links between adjacent spans of text, among them contingency, comparison and elaboration. These describe how a text is organized, not how a claim is established, and a passage may realize a discourse relation while making no argumentative move at all. The two kinds of annotation are largely orthogonal, which is why an existing discourse-relation corpus does not provide a ready-made comparison for the scheme we studied in this paper.

Alongside these annotation efforts, another line of work, surveyed by Li et al. [36], has been applying large language models to mathematical reasoning and proof directly, whether by generating natural-language proofs [61], translating them into the languages of formal proof assistants such as Lean [16], or searching for derivations [42]. Most ambitiously, the same techniques have begun to tackle open conjectures that had long resisted proof [29, 53]. Our aim in this work is different. We use models not to find or verify a proof but to label the relation that organizes a proof already given, and our mathematical items are complete natural-language proofs rather than formal derivations [60]. The analogy relation also connects more specifically to an active debate over whether language models reason analogically in a human-like manner [58] or instead lean on surface-level correspondences [35]. That debate concerns the models’ own competence, whereas here analogy is one of the relations the models are asked to recognize in an argument they did not produce.

Because the labels analysed here are produced by large language models, the study also relates to a growing body of work that uses large language models as annotators [12, 15]. Such studies report that these models can match or exceed crowd workers on a range of classification tasks [51], though others caution that agreement among models is not evidence of correctness and that replacing human coders requires explicit evaluation [3], and that model judgements can carry systematic biases of their own [54]. We adopt this instrument for its scale and reproducibility, and throughout we read its agreement as a measure of how consistently the scheme can be applied rather than of whether the labels are correct. More broadly, characterizing a theoretical typology at scale with language models, by operationalizing it as an explicit set of definitions, is a strategy we have applied to other conceptual schemes as well, including Northrop Frye’s theory of fundamental genres [14] and the investigative methods of fictional detectives [12, 15].

## 7 Concluding Remarks

This paper began from a simple observation: when a statement S resists a direct proof, a proof can often be obtained by establishing a related statement S<sup>∗</sup> instead. Around this observation we organized four semiotic relations connecting S to S<sup>∗</sup>, corresponding to the four master tropes and to proof by inference, analogy, contradiction, and case analysis (Section 2); we then considered the two processes that surround any proof, its discovery and its communication (Sections 3 and 4); and we characterized empirically how the four relations are used when applied to argumentation at scale (Section 5). The empirical characterization shows that the relations cover argumentation broadly but are exercised very unevenly. Mathematical proofs draw on all four relations, whereas legal and everyday reasoning rely almost entirely on inference. Their internal structure is asymmetric, with inference largely self-sufficient while contradiction and case analysis typically recruit a subordinate relation, most often inference. A bounded external test in the legal genre, using expert argument-type labels withheld from the models, supports the view that the labels track the structure of an argument rather than its surface vocabulary.

The framework nevertheless remains a simplification, as any account of a complex practice must be. The full complexity of mathematical practice lies well beyond it: theorems such as the four-colour theorem (Example 8) and Fermat’s Last Theorem (Example 9) still come in very lengthy proofs that demand proficiency across several domains, so much so that one is often compelled to accept such results on the authority of a few specialists. Efforts to convey them to a wider audience are only partly successful; Faltings [20], presenting the ideas behind Fermat’s Last Theorem, notes having passed over details judged to be of little interest to the nonspecialist.

Several limitations bound these findings. The most important is that every label analysed in our evaluation experiment is produced by large language models, so the agreement we observe measures how consistently the scheme can be applied, not whether the labels are correct as models trained on overlapping data can agree and still be wrong together. The external test offsets this only in part, since it reaches SYN and PAR in the legal genre alone and speaks to neither MER nor ANT. And because each genre is drawn from a single corpus that is not a random sample, the distributions we report describe these collections rather than their genres in full.

Two directions would address these limitations. A study with human annotators blind to the model labels would yield bias-corrected prevalence and, in particular, the external coverage of MER and ANT that no independent key in the present study reaches. A complementary design that codes the same arguments independently for their structural relation and for their proof method would test the grounding of the relations in the tropes that we have here taken as given, with alignment supporting that grounding and independence showing it to be decorative.

Beyond these questions of method, the flashes of intuition that allow researchers to see how an intractable problem might be solved remain largely resistant to systematic account. A framework such as ours can map the relations through which a proof is organized, but not the moment of insight that finds it. On that, obedient to the lemma inscribed over the entrance of Plato’s Academy (“μηδεὶς ἀγεωμέτρητος εἰσίτω – Let no one ignorant of geometry enter here”), we can only stand modestly at the threshold.

## References

[1] K. Appel and W. Haken. Every planar map is four colorable. Bulletin of the American Mathematical Society, 82 (5):711–712, 1976. doi: 10.1090/S0002-9904-1976-14122-5.

[2] K. Burke. A Grammar of Motives. University of California Press, Berkeley, CA, USA, 1969.

[3] N. Calderon, R. Reichart, and R. Dror. The alternative annotator test for LLM-as-a-judge: How to statistically justify replacing human annotators with LLMs. In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16051–16081, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/ v1/2025.acl-long.782.

[4] M. A. Casanova, F. A. d. C. Giorno, and A. L. Furtado. Programação em Lógica e a Linguagem Prolog. Editora Edgard Blücher, São Paulo, Brazil, 1987.

[5] D. Chandler. Semiotics: The Basics. The Basics. Routledge, London, UK, 1st edition, 2002.

[6] O. Chateaubriand. Logical Forms. Part I: Truth and Description, volume 34 of Coleção CLE. Centro de Lógica, Epistemologia e História da Ciência (CLE), Universidade Estadual de Campinas, Campinas, SP, Brazil, 1st edition, 2001.

[7] A. E. M. Ciarlini, S. D. J. Barbosa, M. A. Casanova, and A. L. Furtado. Event relations in plan-based plot composition. Computers in Entertainment, 7(4):1–37, 2009. doi: 10.1145/1658866.1658874.

[8] J. Culler. The Pursuit ofSigns: Semiotics, Literature, Deconstruction. Routledge and Kegan Paul, London, UK, 1st edition, 1981.

[9] E. S. de Lima, B. Feijó, M. A. Casanova, and A. L. Furtado. Storytelling variants based on semiotic relations. Entertainment Computing, 17:31–44, 2016. doi: 10.1016/j.entcom.2016.08.003.

[10] E. S. de Lima, M. A. Casanova, B. Feijó, and A. L. Furtado. Semiotic structuring in movie narrative generation. In P. Ciancarini, A. Di Iorio, H. Hlavacs, and F. Poggi, editors, Entertainment Computing – ICEC 2023, volume 14455 of Lecture Notes in Computer Science, pages 161–175, Singapore, 2023. Springer Nature Singapore. doi: 10.1007/978-981-99-8248-6\_13.

[11] E. S. de Lima, B. Feijó, M. A. Casanova, and A. L. Furtado. ChatGeppetto – an AI-powered storyteller. In Proceedings of the 22nd Brazilian Symposium on Games and Digital Entertainment (SBGames ’23), ACM International Conference Proceeding Series, pages 28–37, New York, NY, USA, Nov. 2023. Association for Computing Machinery. doi: 10.1145/3631085.3631302.

[12] E. S. de Lima, M. A. Casanova, B. Feijó, and A. L. Furtado. Characterizing the investigative methods of fictional detectives with large language models. arXiv:2505.07601 [cs.CL], 2025. URL https://doi.org/10.48550/ arXiv.2505.07601.

[13] E. S. de Lima, M. M. E. Neggers, B. Feijó, M. A. Casanova, and A. L. Furtado. An AI-powered approach to the semiotic reconstruction of narratives. Entertainment Computing, 52:100810, 2025. doi: 10.1016/j.entcom.2024. 100810.

[14] E. S. de Lima, M. A. Casanova, and A. L. Furtado. Revisiting Northrop Frye’s four myths theory with large language models. arXiv:2602.15678 [cs.CL], 2026. URL https://doi.org/10.48550/arXiv.2602.15678.

[15] E. S. de Lima, M. M. E. Neggers, M. A. Casanova, B. Feijó, and A. L. Furtado. Trait-guided detective story generation in seven classic investigative styles with large language models. Entertainment Computing, 58:101166, 2026. doi: 10.1016/j.entcom.2026.101166.

[16] L. de Moura and S. Ullrich. The Lean 4 theorem prover and programming language. In A. Platzer and G. Sutcliffe, editors, Automated Deduction – CADE 28: 28th International Conference on Automated Deduction, Virtual Event, July 12–15, 2021, Proceedings, volume 12699 of Lecture Notes in Computer Science, pages 625–635, Cham, Switzerland, 2021. Springer. doi: 10.1007/978-3-030-79876-5\_37.

[17] F. de Saussure. Cours de linguistique générale. Grande Bibliothèque Payot. Payot, Paris, France, 1995. Edited by Charles Bally, Albert Sechehaye and Albert Riedlinger.

[18] H. B. Enderton. A Mathematical Introduction to Logic. Academic Press, New York, NY, USA, 1st edition, 1972.

[19] Euclid. The Thirteen Books of the Elements. Dover Publications, New York, NY, USA, 2nd edition, 1956. Translated by Sir Thomas L. Heath.

[20] G. Faltings. The proof of Fermat’s last theorem by R. Taylor and A. Wiles. Notices ofthe American Mathematical Society, 42(7):743–746, 1995. URL https://www.ams.org/notices/199507/faltings.pdf. Translated by Uwe F. Mayer.

[21] G. Fauconnier and M. Turner. Conceptual integration networks. Cognitive Science, 22(2):133–187, 1998. doi: 10.1207/s15516709cog2202\_1.

[22] A. L. Furtado. Analogy by generalization—and the quest of the grail. ACM SIGPLAN Notices, 27(1):105–113, 1992. doi: 10.1145/130722.130741.

[23] A. L. Furtado and A. E. M. Ciarlini. Constructing libraries of typical plans. In K. R. Dittrich, A. Geppert, and M. C. Norrie, editors, Advanced Information Systems Engineering: 13th International Conference, CAiSE 2001, Interlaken, Switzerland, June 4–8, 2001, Proceedings, volume 2068 of Lecture Notes in Computer Science, pages 124–139, Berlin, Heidelberg, Germany, 2001. Springer. doi: 10.1007/3-540-45341-5\_9.

[24] I. Habernal, D. Faber, N. Recchia, S. Bretthauer, I. Gurevych, I. Spiecker genannt Döhmann, and C. Burchard. Mining legal arguments in court decisions. Artificial Intelligence and Law, 32(3):557–594, 2024. doi: 10.1007/ s10506-023-09361-y.

[25] J. Han, M. Kamber, and J. Pei. Data Mining: Concepts and Techniques. The Morgan Kaufmann Series in Data Management Systems. Morgan Kaufmann, Waltham, MA, USA, 3rd edition, 2011. doi: 10.1016/ C2009-0-61819-5.

[26] R. Jakobson. Linguistics and poetics. In S. Rudy, editor, Selected Writings. Volume III: Poetry ofGrammar and Grammar ofPoetry, volume 3, pages 18–51. Mouton, The Hague, Netherlands, 1981.

[27] C. G. Jung. The Archetypes and the Collective Unconscious, volume 9 of The Collected Works of C. G. Jung. Princeton University Press, Princeton, NJ, USA, 2nd edition, 1981.

[28] R. M. Karp. Reducibility among combinatorial problems. In R. E. Miller, J. W. Thatcher, and J. D. Bohlinger, editors, Complexity of Computer Computations, The IBM Research Symposia Series, pages 85–103. Plenum Press, New York, NY, USA, 1972. doi: 10.1007/978-1-4684-2001-2\_9.

[29] Y. Ke, T. Huang, Y. Shu, D. He, J. Gai, and L. Wang. Towards solving the Gilbert–Pollak conjecture via large language models. 2601.22365 [cs.DM], 2026. URL https://doi.org/10.48550/arXiv.2601.22365.

[30] J. L. Kolodner. Case-Based Reasoning. The Morgan Kaufmann Series in Representation and Reasoning. Morgan Kaufmann, San Mateo, CA, USA, 1993.

[31] K. Krippendorff. Content Analysis: An Introduction to Its Methodology. Sage Publications, Thousand Oaks, CA, USA, 2nd edition, 2004.

[32] I. Lakatos. Proofs and Refutations: The Logic of Mathematical Discovery. Cambridge University Press, Cambridge, UK, 1976. Edited by John Worrall and Elie Zahar.

[33] G. Lakoff and M. Johnson. Metaphors We Live By. University of Chicago Press, Chicago, IL, USA, 2003. doi: 10.7208/chicago/9780226470993.001.0001.

[34] G. Lakoff and R. E. Núñez. Where Mathematics Comes From: How the Embodied Mind Brings Mathematics into Being. Basic Books, New York, NY, USA, 1st edition, 2000.

[35] M. Lewis and M. Mitchell. Evaluating the robustness of analogical reasoning in large language models. Transactions on Machine Learning Research, 2025. URL https://openreview.net/forum?id=t5cy5v9wph.

[36] Z. Li, J. Sun, L. Murphy, Q. Su, Z. Li, X. Zhang, K. Yang, and X. Si. A survey on deep learning for theorem proving. In First Conference on Language Modeling (COLM), 2024. URL https://openreview.net/forum? id=zlw6AHwukB.

[37] W. C. Mann and S. A. Thompson. Rhetorical structure theory: Toward a functional theory of text organization. Text: Interdisciplinary Journalfor the Study ofDiscourse, 8(3):243–281, 1988. doi: 10.1515/text.1.1988.8.3.243.

[38] R. G. Newcombe. Interval estimation for the difference between independent proportions: Comparison of eleven methods. Statistics in Medicine, 17(8):873–890, Apr. 1998. doi: 10.1002/(SICI)1097-0258(19980430)17:8<873:: AID-SIM779>3.0.CO;2-I.

[39] C. S. Peirce. The Essential Peirce: Selected Philosophical Writings, Volume 2 (1893–1913), volume 2. Indiana University Press, Bloomington, IN, USA, 1998. Edited by the Peirce Edition Project.

[40] A. Peldszus and M. Stede. An annotated corpus of argumentative microtexts. In D. Mohammed and M. Lewinski,´ editors, Argumentation and Reasoned Action: Proceedings ofthe 1st European Conference on Argumentation, Lisbon 2015, volume 2, pages 801–815, London, UK, 2016. College Publications.

[41] Plato. Cratylus. Parmenides. Greater Hippias. Lesser Hippias. Number 167 in Loeb Classical Library. Harvard University Press, Cambridge, MA, USA, 1926. Translated by Harold North Fowler.

[42] S. Polu and I. Sutskever. Generative language modeling for automated theorem proving. 2009.03393 [cs.LG], 2020. URL https://doi.org/10.48550/arXiv.2009.03393.

[43] G. Pólya. How to Solve It: A New Aspect ofMathematical Method. Princeton University Press, Princeton, NJ, USA, 1st edition, 1945.

[44] G. Pólya. Mathematics and Plausible Reasoning. Volume I: Induction and Analogy in Mathematics, volume 1. Princeton University Press, Princeton, NJ, USA, 1954.

[45] G. Pólya. Mathematics and Plausible Reasoning [Two Volumes in One]. Martino Fine Books, Eastford, CT, USA, 2014.

[46] R. Prasad, N. Dinesh, A. Lee, E. Miltsakaki, L. Robaldo, A. Joshi, and B. L. Webber. The Penn Discourse TreeBank 2.0. In N. Calzolari, K. Choukri, B. Maegaard, J. Mariani, J. Odijk, S. Piperidis, and D. Tapias, editors, Proceedings ofthe Sixth International Conference on Language Resources and Evaluation (LREC’08), pages 2961–2968, Marrakech, Morocco, 2008. European Language Resources Association (ELRA). URL https://aclanthology.org/L08-1093/.

[47] Quintilian. The Orator’s Education, Volume III: Books 6–8. Number 126 in Loeb Classical Library. Harvard University Press, Cambridge, MA, USA, 2001. Edited and translated by Donald A. Russell.

[48] P. Ramus. Arguments in Rhetoric Against Quintilian: Translation and Text ofPeter Ramus’s Rhetoricae Distinctiones in Quintilianum (1549). Landmarks in Rhetoric and Public Address. Southern Illinois University Press, Carbondale, IL, USA, 2010. Edited by James J. Murphy; translated by Carole Newlands.

[49] S. R. Ranganathan. Ramanujan: The Man and the Mathematician, volume 1 of Great Thinkers ofIndia Series. Asia Publishing House, Bombay, India, 1967

[50] R. C. Schank and R. P. Abelson. Scripts, Plans, Goals and Understanding: An Inquiry into Human Knowledge Structures. The Artificial Intelligence Series. Lawrence Erlbaum Associates, Hillsdale, NJ, USA, 1977.

[51] P. Törnberg. ChatGPT-4 outperforms experts and crowd workers in annotating political Twitter messages with zero-shot learning. arXiv:2304.06588 [cs.CL], 2023. URL https://doi.org/10.48550/arXiv.2304.06588.

[52] S. E. Toulmin. The Uses of Argument. Cambridge University Press, Cambridge, UK, updated edition, 2003. doi: 10.1017/CBO9780511840005.

[53] I. Tzachristas, G. Tzachristas, and A. Sui. Open problems solved by LLMs? A survey of verifiable mathematical discovery. In Proceedings of The Big Picture v2: Crafting a Research Narrative, pages 10–21, San Diego, CA, USA, 2026. Association for Computational Linguistics. URL https://aclanthology.org/2026. bigpicture-main.2/.

[54] I. C. E. van Blerck, E. S. de Lima, M. M. E. Neggers, and T. Calders. Unveiling gender bias in LLM-generated hero and heroine narratives. Entertainment Computing, 55:100972, 2025. doi: 10.1016/j.entcom.2025.100972.

[55] G. Vico. The New Science of Giambattista Vico: Unabridged Translation of the Third Edition (1744). Cornell University Press, Ithaca, NY, USA, 1968.

[56] D. N. Walton. Argumentation Schemesfor Presumptive Reasoning. Lawrence Erlbaum Associates, Mahwah, NJ, USA, 1996. doi: 10.4324/9780203811160.

[57] D. N. Walton, C. Reed, and F. Macagno. Argumentation Schemes. Cambridge University Press, Cambridge, UK, 2008. doi: 10.1017/CBO9780511802034.

[58] T. Webb, K. J. Holyoak, and H. Lu. Emergent analogical reasoning in large language models. Nature Human Behaviour, 7(9):1526–1541, 2023. doi: 10.1038/s41562-023-01659-w.

[59] B. L. Webber. Questions, answers and responses: Interacting with knowledge-base systems. In M. L. Brodie and J. Mylopoulos, editors, On Knowledge Base Management Systems: Integrating Artificial Intelligence and Database Technologies, Topics in Information Systems, pages 365–401. Springer-Verlag, New York, NY, USA, 1986. doi: 10.1007/978-1-4612-4980-1\_30.

[60] S. Welleck, J. Liu, R. Le Bras, H. Hajishirzi, Y. Choi, and K. Cho. NaturalProofs: Mathematical theorem proving in natural language. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1 (NeurIPS Datasets and Benchmarks 2021), Round 1, 2021. URL https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ d9d4f495e875a2e075a1a4a6e1b9770f-Abstract-round1.html.

[61] S. Welleck, J. Liu, X. Lu, H. Hajishirzi, and Y. Choi. NaturalProver: Grounded mathematical proof generation with language models. In Advances in Neural Information Processing Systems 35 (NeurIPS 2022). Curran Associates, Inc., 2022. URL https://openreview.net/forum?id=rhdfTOiXBng.

[62] H. White. Metahistory: The Historical Imagination in Nineteenth-Century Europe. Johns Hopkins University Press, Baltimore, MD, USA, 1973.

[63] A. Wiles. Modular elliptic curves and Fermat’s last theorem. Annals ofMathematics, 141(3):443–551, 1995. doi: 10.2307/2118559.

[64] M. E. Winston, R. Chaffin, and D. Herrmann. A taxonomy of part-whole relations. Cognitive Science, 11(4): 417–444, 1987. doi: 10.1207/s15516709cog1104\_2.

## Appendix A The Annotation Prompt

This appendix reproduces the complete system prompt used in the annotation run described in Section 5.1.3, exactly as supplied to the models. The four relation definitions are shown here in a fixed order (SYN, PAR, ANT, MER). At run time they are presented in a randomized order per call, with NONE pinned last, to prevent position in the list from influencing the label. The passage to be classified is supplied in a separate user message consisting of the word Passage: followed by the passage text.

SYSTEM PROMPT:

Classify how a passage of reasoning establishes its claim.

Each passage argues for some claim. Call the claim S, and call whatever the passage rests that claim on S\*. Your job is to identify the relationship between S and S\*. That is, what KIND of move the passage makes to get from its support to its conclusion.

The categories:

\- SYN (syntagmatic; contiguity, sequence; trope: metonymy; method: inference): S is a logical consequence of S\*. The reasoner locates a rule or prior result and applies it to reach S. Includes syllogism, rule application, chained deduction, and defeasible warrant-based argument.

\- PAR (paradigmatic; similarity, alternatives; trope: metaphor; method: analogy): After suitable mappings, the relevant features of S are converted into features of S\*. The reasoner solves the mapped problem instead of the original one. The analogue often lies in another domain, as when a scheduling problem is mapped onto graph colouring, but it need not: mapping one case onto a structurally similar case within the same domain is still PAR, as when a problem is solved by blending a situation with a parallel version of itself. What matters is that a correspondence is drawn between two cases and the argument runs through it. Includes reduction, structural analogy, argument from a parallel case, and conceptual blending.

\- ANT (antithetic; opposition, negation; trope: irony; method: contradiction): S\* is shown to be inconsistent under the assumption of ¬S. Includes reductio ad absurdum and any argument whose force comes from the untenability of the denial. Concession is not ANT. Granting a point and then arguing past it ("Of course X, but Y"; "Admittedly X; nevertheless Y") is a rhetorical move, not an argument from inconsistency; label such passages by whatever establishes the main claim. Nor is mere contrast between two things ANT. The test is whether denying the claim is shown to lead to something untenable.

\- MER (meronymic; hierarchy, details; trope: synecdoche; method: case analysis): S\* is a set of statements into which S decomposes exhaustively, each handled separately. Includes case splits, exhaustion, and finite induction (base case + induction step).

\- NONE: the artifact establishes S by a move none of the four captures, or makes no identifiable argumentative move at all. Do not stretch a category to avoid NONE; its rate is a result, not a failure. But NONE is not a way to record uncertainty. If the passage does make one of the four moves and you are merely unsure which, pick the best fit and set confidence to 1. Reserve NONE for passages where no category applies, for instance a passage that asserts a conclusion without supporting it, or recites facts without drawing an inference. Note that a practical or normative conclusion drawn from stated circumstances is still SYN: the warrant may be implicit and defeasible without ceasing to be an inference.

How to decide:

\- Judge only what is in the passage. Do not use outside knowledge of the topic, and do not reconstruct an argument the text does not actually make.

\- Passages may be excerpts, especially from longer documents. Some will begin or end mid-argument. Classify the move the excerpt itself makes; do not guess at what surrounded it.

\- The dominant category is the one the argument depends on: remove that move and the passage no longer establishes its claim. There is exactly one.

\- Subordinate categories are for moves doing real supporting work. Leave the list empty when there are none. Do not add a category as subordinate merely because the passage contains some faint trace of it; if that field fires on everything it carries no information.

\- Choose NONE when the passage states a conclusion without arguing for it, recites facts or rules without drawing an inference, or makes a move none of the categories fits. NONE is a legitimate answer and you should expect to use it. Do not stretch a category to avoid it.

\- These categories are not equally common. Do not try to balance your answers across a set of passages, and do not assume any category must appear.

Work in this order, then respond with JSON only, no other text:

1. claim: the claim the passage is establishing (S), in your own words, one sentence.

2. support: what the passage rests that claim on (S\*), one sentence.

3. reasoning: what kind of move connects support to claim, one sentence.

4. dominant, subordinate, confidence, as defined above.

{"claim": "...", "support": "...", "reasoning": "...", "dominant": "...", "subordinate": [...], " confidence": N}

confidence: 1 if you are guessing, 2 if plausible, 3 if clear.