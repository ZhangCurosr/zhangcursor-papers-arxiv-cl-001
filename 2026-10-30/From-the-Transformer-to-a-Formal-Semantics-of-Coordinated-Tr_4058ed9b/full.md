# From the Transformer to a Formal Semantics of Coordinated Transformations

Marco Giunti<sup>1</sup> and Fabrizia Giulia Garavaglia<sup>2</sup>

<sup>1</sup>ALOPHIS – Applied Logic, Philosophy, and History of Science, University of Cagliari

<sup>2</sup>Dept. of Pedagogy, Psychology, Philosophy, University of Cagliari

## Abstract

We develop a formal semantics for a general class of coordinated systems of parameterized transformations, taking the Transformer as a paradigmatic starting point but abstracting from its particular mathematical instantiation. Representational vectors receive both extensional reference and ordered in tensional interpretations, while parameter structures acquire relative, second-level interpretations that determine how argument properties are preserved, replaced, or combined. This yields the notion of a transforming concept: a parameterized transformation whose numerical action is coordinated with a corresponding property-level map. We then distinguish the common algebraic-semantic core of these systems from their architecture-specific functional organization, allowing diferent mathematical instan tiations and mechanical realizations of the same abstract system type. The framework is applied to attention heads and residual connections, providing semantic explanations that depend only on the shared algebraic-semantic core and their local functional organization and therefore generalize beyond the standard Transformer. These analyses also reveal the modularity of the framework: complex archi tectures can be decomposed into simpler semantically coordinated subsystems whose organization may recur across diferent abstract system types.

Keywords: formal semantics; Transformer; transforming concepts; coordinated transformations; intensional semantics; attention; residual connections; functional architecture; mathematical instantiation; neura networks

## 1 Introduction

In previous work (Giunti and Garavaglia 2026; hereafter TR1), we proposed interpreting the Transformer as a system that transforms concepts by means of concepts. On that interpretation, the vector representations processed by the system are transformee concepts, whereas the parameterized transformations acting on them are transforming concepts. TR1 developed this interpretation for the Transformer architecture. The present paper asks a more general question: what formal semantics can be given to coordinated systems of transformations of this kind?

The problem is not specific to the standard Transformer (Vaswani et al. 2017), nor even to Transformer architectures in general. We consider a broader class of systems in which matrix- or vector-parameterized transformations may operate singly or in parallel and may be organized through direct or operational output–input interconnections and output–parameter interconnections. The mathematical framework is generalized over an extended value domain and distinguishes these parameterized transformations from the non-parameterized vector-to-vector functions used in operational blocks. This yields a class broad enough to include generalized Transformer systems without being defined by Transformer-specific mathematics.

We then develop an intensional–extensional formal semantics for these systems. Positions of representational vectors are interpreted by properties, while the vectors themselves refer extensionally to objects; the value at a position is therefore the occurrence-relative value of the property associated with that position. Parameter structures may likewise receive intensional interpretations, but they do not thereby refer to objects. Their interpretation is instead relative and second-level: it specifies how properties interpreting an argument are preserved, replaced, or combined in determining the interpretation of a result. This formalsemantic account is kept distinct from the further questions of whether and how such interpretations are representationally grounded.

The coordination between the value level and the property level yields the formal notion of a transforming concept. A parameterized transformation is a transforming concept when its numerical action is coordinated with the property-level map determined by its relative second-level interpretation. Such transformations may be intensionally invariant or intensionally changing independently of whether numerical values are preserved, and dimensional preservation does not entail intensional invariance. The framework also separates the constitutive semantic organization of a transforming concept from the genesis of its parameter values and interpretations.

At a further level, we distinguish the algebraic-semantic core shared by semantically coordinated systems of transformations from their architecture-specific functional organization. This allows diferent mathematical instantiations and mechanical realizations of the same abstract functional architecture and clarifies the distinct routes by which abstract systems may reach the physical level. The framework is also modular: attention heads and residual connections can themselves be treated as abstract types of semantically coordinated systems. Their semantic organization depends only on the common core and their local functional architecture, not on the overall Transformer architecture or on the particular mathematics of its standard instantiation. This makes it possible both to identify semantic invariance across mathematical variation and to distinguish semantic relations that mathematical similarity alone does not determine.

The paper proceeds as follows. Section 2 motivates the move from TR1 to the present problem and situates it within debates on representation, content, and metasemantics in AI. Section 3 develops the generalized value algebra and coordinated system architecture; Section 4 the intensional–extensional semantics; Section 5 the property algebra and transforming concepts; and Section 6 semantic coordination. Section 7 introduces abstract functional architecture, mathematical instantiation, mechanical realization, and functional-architectural invariance. Section 8 applies the framework to attention and residual connections and establishes its modularity. Section 9 discusses the philosophical significance and scope of the proposal and concludes.

## 2 From the Transformer to Coordinated Systems of Transformations

The present framework originates from the interpretation of Transformer processing developed in TR1, but its scope is broader. This section recalls the distinction between transformee and transforming concepts, situates the semantic problem within current debates on representation and content in AI, and abstracts from the Transformer to the class of coordinated systems of transformations analyzed below.

## 2.1 Transformee and transforming concepts

In TR1, we interpreted the Transformer as a system that transforms concepts by means of concepts. The transformee concepts are the vector representations of tokens, including both embeddings and representations generated at later stages of processing. The transforming concepts are parameterized transformations acting on these vectors and defined by parameter matrices or vectors. Some of those parameter structures are fixed through training, whereas others are generated from the current input during inference; TR1 accordingly distinguished static from dynamic transforming concepts.

This terminology was motivated semantically as well as mathematically. A token vector was interpreted as an ordered representation whose positions correspond to properties and whose values specify the values of those properties for the represented token. Parameter matrices and vectors were instead assigned a second-level interpretation relative to the properties associated with the vector on which the corresponding transformation operates.

In the present section, however, transformee concept and transforming concept retain a provisional status. They identify the interpretation inherited from TR1 and the phenomenon to be generalized. A parameterized transformation does not qualify as a transforming concept merely because a matrix or vector parameterizes it. Sections 3 and 4 first specify the relevant mathematical structures and their semantics; Section 5 then distinguishes mathematical parameterization from conceptual parameterization and gives transforming concept its formal definition.

## 2.2 The semantic problem

Recent philosophical work on AI increasingly distinguishes questions concerning concepts, representationa vehicles, semantic content, misrepresentation, and representational grounding (Buckner 2018; Cappelen and Dever 2021; Räz 2023; Freiesleben 2026; Poth and Schuster 2026; Klein 2026; Kieval 2026; Millière and Buckner 2026). These approaches difer over what makes an internal state genuinely represent a property, object, or concept. The present paper addresses a diferent, analytically prior question: once relevant interpretations are specified, what formal semantic organization do the mathematical structures and transformations of such a system exhibit?

This distinction separates three questions. First, formal semantic interpretation concerns which semantic values are assigned to mathematical structures and how those assignments are coordinated. Second, representation concerns the conditions under which interpreted structures actually stand for objects, properties, or other contents. Third, metasemantic grounding concerns what determines or grounds those representational relations. The present paper addresses the first question directly. In doing so, it makes structural commitments about representation: representational vectors are taken to refer to objects, their positions are interpreted by properties, and the framework specifies how such reference and property structure are preserved or transformed through processing. These commitments characterize the formal organization of representations within the framework, but they do not explain what makes a state representational, why it has one content rather than another, how such content arises, or what grounds its representational relations. In this sense, the framework is not a general theory of representation or a theory of metasemantic grounding, but a formal-semantic framework within which diferent such theories may be developed.

## 2.3 Beyond the Transformer

Nothing in this problem restricts the analysis to the Transformer. At the functional level, Transformer processing contains parameterized transformations of vectors, parallel organizations of such transformations, and several ways of coordinating their outputs. Outputs may become later inputs directly, may be regrouped, may pass through mathematically specified operational blocks, or may become parameters of later transformations.

TR1 described the last relation mechanically as an output–weight interconnection, because the relevant parameters function as weights of simple neural networks. At the more general functional level adopted here, we call it an output–parameter interconnection. The same abstraction also requires a distinction between parameterized transformations, which provide the basis for transforming concepts, and non-parameterized vector-to-vector functions used in operational blocks. Operational blocks may modify a signal and mediate output–input relations, but they are not transformations in the technical sense adopted in this paper.

The resulting coordinated systems may contain transformations and/or parallel operations organized through direct output–input, operational output–input, and output–parameter interconnections. The Transformer is an important instance of this broader organization, but the class is not defined by Transformerspecific mathematical operations or by any particular neural realization.

The next section develops the mathematical framework required to characterize this class independently of its particular instances.

## 3 Generalized Transformations, the Value Algebra, and Their Coordination

The semantic framework developed in the following sections requires a mathematical description that does not depend on the particular numerical operations used by the standard Transformer. We therefore begin by generalizing the notions of value, vector, matrix, and parameterized transformation over a blank-extended numerical domain. This yields an abstract value algebra in which ordinary matrix-defined transformations, componentwise multiplicative transformations, and vector additions can be treated within a single construction. We then introduce the forms of parallel organization and interconnection through which transformations can be coordinated into a system, distinguishing parameterized transformations from the non-parameterized vector-to-vector functions involved in operational output–input interconnections.

## 3.1 Generalized values, vectors, and matrices

$$
{ \overline { { \mathbb { R } } } } : = \mathbb { R } \cup \{ \quad \underline { { \mathbf { \delta \pi } } } \} ,
$$

where — is a distinguished blank element. The blank is not a further numerical value and, in particular, is not identified with 0. Its role is to represent the absence of a contribution within the same domain in which ordinary numerical values occur.

Let

$$
\star , \diamond : \mathbb { R } \times \mathbb { R } \longrightarrow \mathbb { R }
$$

be arbitrary binary operations. No assumption is made that they are associative, commutative, distributive, or distinct from one another. Each operation $\omega \in \{ \star , \diamond \}$ can be extended to $\overline { { \mathbb { R } } }$ in two diferent ways.

For each $\omega \in \{ \star , \diamondsuit \} .$ we define its additive and multiplicative extensions to $\overline { { \mathbb { R } } }$ by, for all $x , y \in \mathbb { R }$

$$
\begin{array} { r l } { x \omega ^ { + } y : = x \omega y , } & { \quad x \omega ^ { \times } y : = x \omega y ; } \\ { \quad } & { \quad - \omega ^ { + } - : = - , } \\ { \quad } & { \quad - \omega ^ { + } x : = x , \quad x \omega ^ { + } - : = x , } \end{array} \quad \begin{array} { r l } { x \omega ^ { \times } y : = x \omega y ; } \\ { \quad } & { \quad - \omega ^ { \times } - : = - ; } \\ { \quad } & { \quad - \omega ^ { \times } x : = - , \quad x \omega ^ { \times } - : = - . } \end{array}
$$

Thus, the additive extension $\omega ^ { + }$ makes the blank value — a two-sided identity element, whereas the multi plicative extension $\omega ^ { \times }$ makes it a two-sided absorbing element. On pairs of real numbers, both extensions coincide with the original operation �.

The labels additive and multiplicative therefore characterize the role played by the blank, not the algebraic nature of the underlying operation. The same ordinary operation can consequently occupy either role. Ordinary addition, for example, gives rise both to $+ ^ { + }$ , in which the blank is neutral, and to $+ ^ { \times }$ , in which it is absorbing.

A generalized vector of dimension � is an ordered �-tuple

$$
\pmb { x } = ( x _ { 1 } , \dots , x _ { n } ) \in \overline { { \mathbb { R } } } ^ { n } ,
$$

and a generalized matrix is an array

$$
\begin{array} { r } { \pmb { M } = ( m _ { i j } ) \in \overline { { \mathbb { R } } } ^ { m \times n } . } \end{array}
$$

Once the domain is clear, we will normally omit the adjective generalized.

Since the operation occupying the additive role need not be associative, its finite iteration requires an explicit association convention. Using the generalized finite-sum symbol with the star placed in the concavity of Σ, we define

$$
\sum _ { i = 1 } ^ { 1 } a _ { i } : = a _ { 1 } , \qquad \sum _ { i = 1 } ^ { k + 1 } a _ { i } : = \left( \sum _ { i = 1 } ^ { k } a _ { i } \right) \star ^ { + } a _ { k + 1 } .
$$

Thus, the iteration is left-associated.

This abstract structure defines what we will call the value algebra. Its crucial feature is not the use of any particular pair of numerical operations, but the distinction between two abstract roles: the multiplicative operation $\diamond ^ { \times }$ determines the contribution made by an input value to an output position, while the additive operation $\star ^ { + }$ combines the contributions directed to that position.

## 3.2 Parameterized vector transformations and the value algebra

Parameterized transformations require one additional value used specifically in parameter structures. We introduce the special preserving value ⭒ and define the extended parameter-value domain

$$
\overline { { \mathbb { R } } } _ { * } : = \overline { { \mathbb { R } } } \cup \{ \sharp \} .
$$

The operation occupying the multiplicative role is extended on its right-hand argument by stipulating that ⭒ is a right identity:

$$
x \diamond ^ { \times } \ast : = x \qquad \mathrm { f o r ~ e v e r y ~ } x \in \overline { { \mathbb { R } } } .
$$

The special value ⭒ is therefore a parameter value, not an additional possible value of ordinary argument or resulting vectors. These continue to have values in ℝ.

Let

$$
\pmb { M } = ( m _ { i j } ) \in \overline { { \mathbb { R } } } _ { \ast } ^ { m \times n }
$$

and

$$
\pmb { x } = ( x _ { 1 } , \dots , x _ { m } ) \in \overline { { \mathbb { R } } } ^ { m } .
$$

For each output position $j ,$ the matrix entry $m _ { i j } .$ , together with the input value $x _ { i } ,$ , determines the contribution $x _ { i } \diamond ^ { \times } m _ { i j }$ . The contributions directed to the same output position are then combined by the operation occupying the additive role. We therefore define the matrix-parameterized transformation

$$
L _ { M } ^ { ( \star ^ { + } , \diamond ^ { \times } ) } : \overline { { \mathbb { R } } } ^ { m } \longrightarrow \overline { \mathbb { R } } ^ { n }
$$

by

$$
\left( L _ { M } ^ { ( \star ^ { + } , \diamond ^ { \times } ) } ( \pmb { x } ) \right) _ { j } : = \sum _ { i = 1 } ^ { m } \left( x _ { i } \diamond ^ { \times } m _ { i j } \right) , \qquad j = 1 , \ldots , n .
$$

The construction preserves the division of labor found in ordinary vector-matrix multiplication while dispensing with its specific algebraic operations:

$\diamond ^ { \times }$ determines the contribution of each input position to an output position,

$\star ^ { + }$ combines the contributions directed to that output position.

The construction therefore generalizes the matrix-defined linear transformations obtained by ordinary vectormatrix multiplication. Linearity can still be defined in the usual way relative to the two generalized operations that determine a matrix-defined transformation; however, in the absence of additional algebraic axioms, $L _ { { \pmb { M } } } ^ { ( { \star } ^ { + } , { \diamond } ^ { \times } ) }$ need not be linear relative to those operations.

The matrix � parameterizes the transformation in a purely mathematical sense: fixing � selects one transformation from the corresponding family. Nothing in this definition requires its entries to have been learned, dynamically generated, or semantically interpreted. Mathematical parameterization will therefore remain distinct from the semantic interpretation of parameter structures developed in Sections 4 and 5.

A second important family consists of transformations parameterized by a vector and operating componentwise. Given

$$
\pmb { v } = ( v _ { 1 } , \dots , v _ { n } ) \in \overline { { \mathbb { R } } } _ { \ast } ^ { n } ,
$$

we define the vector-parameterized componentwise transformation

$$
\boldsymbol { C _ { v } ^ { \circ } } ^ { \times } : \overline { { \mathbb { R } } } ^ { n } \longrightarrow \overline { { \mathbb { R } } } ^ { n }
$$

by

$$
{ C _ { \pmb { v } } ^ { \diamond } } ^ { \times } ( \pmb { x } ) : = \pmb { x } \diamond ^ { \times } \pmb { v } ,
$$

that is,

$$
\left( \boldsymbol { C } _ { \boldsymbol { v } } ^ { \diamond ^ { \times } } ( \boldsymbol { x } ) \right) _ { i } : = x _ { i } \diamond ^ { \times } \boldsymbol { v } _ { i } .
$$

We can now define the generalized diagonal matrix associated with �:

$$
\pmb { \mathscr { D } } ( \pmb { v } ) : = \left( \frac { v _ { 1 } } { \mathscr { D } } \begin{array} { c c c c } { - } & { \cdots } & { - } \\ { \mathscr { v } _ { 2 } } & { \cdots } & { - } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { - } & { - } & { \cdots } & { \mathscr { v } _ { n } } \end{array} \right) .
$$

The of-diagonal entries contain blanks rather than zeros because they express the absence of a contribution rather than a numerical contribution equal to zero.

Diagonal Representation Theorem. For every

$$
\pmb { v } \in \overline { { \mathbb { R } } } _ { \ast } ^ { n } ,
$$

$$
\boxed { L _ { D ( v ) } ^ { ( \star ^ { + } , \diamond ^ { \times } ) } = { C _ { v } ^ { \diamond ^ { \times } } } = L _ { D ( v ) } ^ { ( \diamond ^ { + } , \diamond ^ { \times } ) } . }
$$

In the �-th output position, every of-diagonal entry of �(�) is blank. Since the blank is absorbing under $\diamond ^ { \times }$ , all corresponding contributions are blank. The only non-blank contribution is $x _ { j } \diamond ^ { \times } v _ { j }$ . Since the blank is neutral under either $\star ^ { + }$ or $\diamond ^ { + }$ , aggregating that contribution with the remaining blanks leaves it unchanged. Thus, in the diagonal case, the resulting componentwise transformation depends on the operation occupying the multiplicative role, not on which operation occupies the additive role.

When the relevant parameter values are ordinary real values and the two operational roles are instantiated by ordinary addition and multiplication, the usual matrix-defined linear transformation is recovered as

$$
\mathcal { L } [ \pmb { M } ] = L _ { \pmb { M } } ^ { ( + ^ { + } , \times ^ { \times } ) } .
$$

For componentwise multiplication,

$$
\begin{array} { r }  \boxed { \circ [ \pmb { v } ] = L _ { \pmb { D } ( \pmb { v } ) } ^ { ( + ^ { + } , \pmb { \times } ^ { \times } ) } = C _ { \pmb { v } } ^ { \pmb { \times } ^ { \times } } = L _ { \pmb { D } ( \pmb { v } ) } ^ { ( \times ^ { + } , \pmb { \times } ^ { \times } ) } . } \end{array}
$$

For vector addition,

$$
+ [ \pmb { v } ] = L _ { { D } ( \pmb { v } ) } ^ { ( \mathsf { X } ^ { + } , + ^ { \times } ) } = { C _ { \pmb { v } } ^ { + } } ^ { \times } = L _ { { D } ( \pmb { v } ) } ^ { ( + ^ { + } , + ^ { \times } ) } .
$$

These equalities show that, in the diagonal case, the componentwise transformation is determined by the operation occupying the multiplicative role, while the additive role is irrelevant. Thus ordinary matrixdefined transformation ℒ[�], componentwise multiplication ◦[�], and vector addition +[�] are unified within the generalized value algebra.

In what follows, unless explicitly stated otherwise, the term transformation will be used in a technical sense to denote a parameterized transformation of one of the two kinds just defined: a matrix-parameterized transformation $L _ { \pmb { M } } ^ { ( \star ^ { + } , \diamond ^ { \times } ) }$ or a vector-parameterized transformation $C _ { v } ^ { \diamondsuit ^ { \times } }$ . Other vector-to-vector mathematical functions may occur within the systems considered below, most importantly as operational blocks, but they are not transformations in this technical sense.

## 3.3 Parallel transformations

The transformations just defined need not operate sequentially one at a time. Let

$$
T _ { k } : \overline { { \mathbb { R } } } ^ { m } \longrightarrow \overline { { \mathbb { R } } } ^ { n } , \qquad k = 1 , \ldots , s ,
$$

be a finite family of transformations with common input and output arities, and let

$$
\pmb { x } _ { 1 } , \dots , \pmb { x } _ { s } \in \overline { { \mathbb { R } } } ^ { m } .
$$

The transformations may operate in parallel, producing

$$
T _ { 1 } ( { \pmb x } _ { 1 } ) , \dots , T _ { s } ( { \pmb x } _ { s } ) .
$$

The transformations in the family need not be identical. What matters for the present construction is that their input vectors have the same arity � and their output vectors the same arity �. The input vectors can therefore be arranged as the rows of a matrix

$$
\begin{array} { r } { X : = \left( \begin{array} { l } { x _ { 1 } } \\ { \vdots } \\ { x _ { s } } \end{array} \right) \in \overline { { \mathbb { R } } } ^ { s \times m } , } \end{array}
$$

and the corresponding outputs as the rows of a matrix

$$
\begin{array} { r } { \pmb { Y } : = \left( \begin{array} { l } { T _ { 1 } ( \pmb { x } _ { 1 } ) } \\ { \quad \vdots } \\ { T _ { s } ( \pmb { x } _ { s } ) } \end{array} \right) \in \overline { { \mathbb { R } } } ^ { s \times n } . } \end{array}
$$

A particularly important special case is obtained when $T _ { 1 } = \cdots = T _ { s } = T$ , so that the same transformation is applied row-wise to every vector.

The more general case in which the $T _ { k }$ difer is essential. In standard Transformer processing, positionalencoding addition, masking, and residual addition provide important cases in which diferent rows are subjected to transformations parameterized by diferent vectors. If the parameter vectors are $\pmb { p } _ { 1 } , \dots , \pmb { p } _ { s }$ , the corresponding family has the form

$$
+ [ \pmb { p } _ { k } ] : \pmb { x } _ { k } \longmapsto \pmb { x } _ { k } + \pmb { p } _ { k } , \qquad k = 1 , \ldots , s .
$$

At this stage, however, parallelism is considered only in mathematical terms, as a structural characteristic of the architecture. No semantic condition is imposed on the vectors involved. The additional requirement of interpretive uniformity will be introduced only in Section 6.

## 3.4 Direct output–input interconnections

Transformations become parts of a coordinated system when their outputs and inputs are connected according to an architecture. An output–input interconnection is a structural relation through which output values at one stage become available as inputs at a subsequent stage. In a direct output–input interconnection, those values are not modified by any further mathematical function.

A standard output–input interconnection passes an output vector to a subsequent transformation while preserving the relevant processing-stream organization; in the standard Transformer, this includes token-wise correspondence. A routing output–input interconnection instead reorganizes or regroups unmodified outputs to form a required input structure, as in Concat. The two labels need not be exclusive: Transformer Concat also preserves token-wise correspondence.

Thus direct interconnections may transmit, regroup, or redirect values, but do not mathematically modify them.

## 3.5 Operational blocks and operational output–input interconnections

A diferent form of output–input coordination occurs when values produced at one stage become inputs at a subsequent stage only after being mathematically modified. An operational output–input interconnection is the complete architectural relation in which this modification is mediated by an operational block.

Typical operational blocks in the Transformer include the vector functions reLU, softmax, and norm, corresponding respectively to the matrix operations ReLU, Softmax, and Norm considered in TR1.

In the simplest case, $T _ { 1 } \longrightarrow F \longrightarrow T _ { 2 }$ . Here, $T _ { 1 } \longrightarrow F$ identifies a source from which the operational block receives its input, while $F \longrightarrow T _ { 2 }$ identifies a destination to which its output is sent. These relations are not themselves separate operational output–input interconnections. The operational output–input interconnection is the complete mediated relation between the upstream and downstream parts of the system through �.

More generally, an operational block may have multiple sources and/or destinations. At the abstract architectural level it is characterized by these relations, its input–output arity �, and its functional role; a particular mathematical instantiation supplies a dimension-preserving function

$$
F : \overline { { \mathbb { R } } } ^ { n } \longrightarrow \overline { { \mathbb { R } } } ^ { n } , \qquad F ( \pmb { x } ) : = \big ( f _ { 1 } ( \pmb { x } ) , \dots , f _ { n } ( \pmb { x } ) \big ) ,
$$

that performs that role. The semantic reason for dimensional preservation will be given in §5.3. Operational interconnections may also be interwoven in the sense of TR1, drawing values contributing to an output across diferent parallel streams.

The fundamental distinction is therefore

direct output–input interconnection: transmitted values are not mathematically modified, operational output–input interconnection: transmitted values are modified by an operational block.

Operational blocks must also be distinguished from transformations in the technical sense fixed in §3.2. An operational block is mathematically a vector-to-vector function, but it is not a transformation in the terminology adopted here, since it is not parameterized by a matrix or vector of the specified kinds.

## 3.6 Output–parameter interconnections

An output may also determine all or part of the parameter structure of a subsequent transformation. We call this an output–parameter interconnection. Schematically, if $T _ { 1 } ( \pmb { x } ) = \pmb { p } ,$ the output may parameterize a later transformation $T _ { p } ( \mathbf { y } ) ;$ analogously, output matrices or coordinated families of outputs may supply all or part of later parameter matrices or vectors. What defines the relation is that the relevant output values are used as parameters rather than as the input argument of the subsequent transformation.

The present notion generalizes the output–weight interconnections introduced in TR1. There, parameterized vector transformations were identified with simple neural networks, so dynamically generated parameters appeared specifically as neural-network weights. At the present functional and mathematical level, no particular mechanical realization is assumed.

We retain throughout the technical distinction fixed above: a transformation is one of the parameterized vector-to-vector functions of §3.2, whereas an operation is a parallel organization of transformations with common input and output arities (§3.3). This architectural use of operation is distinct from the algebraic operations ⋆ and ⋄.

With this terminology, a coordinated system of transformations consists of transformations and/or operations organized through direct and/or operational output–input interconnections and/or output–parameter interconnections. Its transformations are parameterized transformations of the two kinds defined in §3.2 and are defined within the value algebra, whereas the operational blocks required by its operational output–input interconnections are non-parameterized vector-to-vector functions that perform architecture-specific functional roles through mathematical functions

$$
F : \overline { { \mathbb { R } } } ^ { n } \longrightarrow \overline { { \mathbb { R } } } ^ { n }
$$

supplied by particular mathematical instantiations.

This characterization remains deliberately semantic-free. It specifies how values can be transformed and how transformations and operations can be coordinated, but it does not yet determine what vectors, parameter structures, or transformations mean.

## 4 An Intensional–Extensional Formal Semantics for Representational Vectors

The mathematical framework introduced in Section 3 specifies how values are organized and transformed, but it is deliberately neutral about their interpretation. We now add a semantic level by associating vector and matrix positions with properties and, in the case of representational vectors, by assigning the vector an extensional reference. The resulting semantics is intensional and extensional at the same time, but in diferent respects: properties are intensionally individuated, whereas representational vectors refer extensionally to objects. This distinction will also allow us to clarify why the semantic interpretation of parameter vectors and matrices difers from that of vectors representing objects.

## 4.1 Objects, values, and properties

Let � be a fixed domain of objects, and let $\mathcal { P }$ be a domain of intensional properties that may be used to interpret vector and matrix positions.

Properties are not identified with their extensions, nor is an extension assigned to an intensional property once and for all. It is determined relative to a specific occurrence of that property in a position of a representational vector; the same property may therefore have diferent extensions across occurrences, and distinct properties may coincide extensionally in particular occurrences.

The property domain contains a distinguished null property – –. The null property interprets a position of a representational vector or a parameter structure if and only if the blank value — occupies that position.

For the interpretation of parameter structures, we also introduce the distinguished preservation property ⭒⭒ and the extended parameter-interpretation domain

$$
\mathcal { P } _ { \ast \ast } : = \mathcal { P } \cup \{ \ast \ast \} .
$$

The preservation property is not an ordinary property of objects in � and is therefore not assigned an occurrence-relative extension of the kind defined for properties occurring in representational vectors. It is a special interpretive element used only for parameter structures. Its transformational role will be defined in Section 5.

## 4.2 Intensional interpretation of vector positions

Consider a vector

$$
\pmb { v } = ( v _ { 1 } , \dots , v _ { m } ) \in \overline { { \mathbb { R } } } ^ { m } .
$$

An intensional interpretation of its positions is an ordered �-tuple of properties

$$
\pmb { P } : = ( P _ { 1 } , \hdots , P _ { m } ) , \qquad P _ { i } \in \mathcal { P } .
$$

Thus, each position � is associated both with a value $v _ { i } \in \overline { { \mathbb { R } } }$ and with an intensional property $P _ { i } \in { \mathcal { P } } .$ . These two assignments must be kept distinct. The first belongs to the value structure developed in Section $3 ;$ the second belongs to the intensional interpretation of the vector.

In particular, whenever $P _ { i } = - -$ , the corresponding value is $v _ { i } = - . \mathrm { A t }$ this stage, however, the association of properties with vector positions does not yet make the vector a representation of an object. For that, an extensional component of the semantics is required.

## 4.3 Extensional reference and features

Every vector used as a representation has a fixed extensional interpretation. If � refers to the object $a \in D$ we write $\operatorname { E x t } ( \pmb { v } ) : = a .$ . Suppose that position � of � is interpreted by $P _ { i } .$ The position- and vector-relative extension of $P _ { i }$ is then the function

$$
\begin{array} { r l } { \mathrm { E x t } _ { \pmb { v } , i } ( P _ { i } ) : \{ a \} \longrightarrow \overline { { \mathbb { R } } } } & { { } } \end{array}
$$

defined by

$$
\begin{array} { r } { \operatorname { E x t } _ { v , i } ( P _ { i } ) ( a ) : = v _ { i } . } \end{array}
$$

The relativization to both vector and position is essential. If the same property occurs at two positions of one vector, $P _ { i } = P _ { k } = P$ with $i \neq k$ , while $v _ { i } \neq v _ { k }$ , then $\mathrm { E x t } _ { { \pmb v } , i } ( P )$ and $\mathrm { E x t } _ { v , k } ( P )$ are distinct functions because their values at � difer. The same holds across diferent representational vectors referring to �. Thus identity of an intensional property does not require identity of its extensions across occurrences.

This also fixes the terminology used throughout the paper. Property refers to the intensional entity associated with a vector or matrix position. Feature is used when such a property is considered as a characteristic of a represented object, or when we refer to the value it takes in a particular representational occurrence. Thus, $P _ { i }$ may be considered a feature of �, while $v _ { i }$ is the corresponding feature value in position � of �.

A vector endowed both with an intensional interpretation $\pmb { P } = ( P _ { 1 } , \dots , P _ { m } )$ and with an extensional interpretation $\operatorname { E x t } ( \pmb { v } ) = a$ will be called a representational vector. In the terminology inherited from TR1, it is a transformee concept.

## 4.4 The interpretation of parameter vectors and matrices

Parameter vectors and matrices also admit an intensional interpretation of their positions, but their semantic structure difers in one essential respect from that of representational vectors.

Let

$$
\pmb { p } = ( p _ { 1 } , \dots , p _ { n } ) \in \overline { { \mathbb { R } } } _ { \ast } ^ { n }
$$

be a parameter vector. An intensional interpretation of its positions is an ordered tuple

$$
Q : = ( Q _ { 1 } , \ldots , Q _ { n } ) \in { \mathcal { P } } _ { * * } ^ { n } .
$$

Thus, each parameter position associates a parameter value $p _ { i } \in \overline { { \mathbb { R } } } _ { * }$ with an interpretive element $Q _ { i } \in \mathcal { P } _ { \ast \ast }$ Similarly, let

$$
\begin{array} { r } { \pmb { M } = ( m _ { i j } ) \in \overline { { \mathbb { R } } } _ { \ast } ^ { m \times n } . } \end{array}
$$

An intensional interpretation of its positions assigns an element $P _ { i j } \in \mathcal { P } _ { \ast \ast }$ to each position $( i , j )$ , producing an interpretive matrix $\textbf { I } : = ( P _ { i j } )$ . These are position-wise associations, not evaluation relations between

parameter values and properties. In particular, the special parameter value $\bigstar$ is not a value taken by an ordinary property on an object in �. Its presence therefore does not extend the codomain of ordinary property extensions beyond ℝ.

The following compatibility conditions apply to every position of a parameter structure. The null property – – interprets that position if and only if the blank value — occupies it. Thus, for a matrix position $( i , j )$

$$
P _ { i j } = --- \iff m _ { i j } = \ --- .
$$

If $P _ { i j } \in \mathcal P \setminus \{ - - \} .$ , then $m _ { i j } \in \mathbb { R }$ . If instead $P _ { i j } = \frac { 1 } { \mathbb { A } \mathbb { A } }$ , then $m _ { i j } \in \mathbb { R } \cup \{ \neq \}$ . The same conditions apply, mutatis mutandis, to the corresponding positions of parameter vectors.

What is absent in parameter structures is the extensional interpretation characteristic of representational vectors. A parameter vector or matrix does not, as such, refer to an object $a \in D$ . Consequently, there is no vector extension analogous to $\operatorname { E x t } ( \pmb { v } ) = a$ , and no occurrence-relative property extension analogous to $\mathrm { E x t } _ { v , i } ( P _ { i } )$ . The values occupying parameter positions therefore do not represent features of an object.

A parameter vector or matrix is a parameter structure only relative to the transformation it parameterizes: � is structurally associated with $L _ { { \pmb { M } } } ^ { ( \star ^ { + } , \diamond ^ { \times } ) } ( { \pmb { x } } )$ , and � with ${ C _ { p } ^ { \circ } } ^ { \times } ( { \pmb x } )$ . Section 5 gives this association semantic content by interpreting parameter positions relative to the properties of an arbitrary argument of the corresponding transformation.

## 4.5 Formal semantics, representation, and grounding

The preceding construction distinguishes three components in the semantics of a representational vector: $i \longmapsto P _ { i }$

$$
\operatorname { E x t } ( \pmb { v } ) : = a ,
$$

and

$$
\begin{array} { r } { \operatorname { E x t } _ { v , i } ( P _ { i } ) ( a ) : = v _ { i } . } \end{array}
$$

The first supplies the intensional interpretation of a position, the second the extensional interpretation of the vector, and the third defines the extension of the property relative to its occurrence in that position of that vector.

These formal assignments embody a structural conception of representation: a representational vector is treated as an object-referential structure whose positions are interpreted by properties with occurrencerelative values. Nevertheless, they are not themselves a philosophical theory of representation: $i \mapsto P _ { i }$ does not explain why $P _ { i }$ is the appropriate interpretation, and $\operatorname { E x t } ( \pmb { v } ) : = a$ does not explain what grounds the reference to �. The value $v _ { i }$ is related to � through $\begin{array} { r } { \mathrm { E x t } _ { v , i } ( P _ { i } ) ( a ) : = v _ { i } ; } \end{array}$ parameter values lack this relation to objects. Hence

## formal semantic interpretation ≠ representational relation $\neq$ metasemantic grounding.

The present section specifies the first while distinguishing intensional property identity, extensional vector reference, and occurrence-relative property extension; questions concerning the grounding of these assignments remain metasemantic. Representational vectors possess all three components, whereas parameter structures retain only an intensional interpretation of their positions.

The main result of the present section can therefore be summarized as follows.

C1 — Intensional–extensional formal semantics. A representational vector is semantically characterized by an extensional interpretation $\operatorname { E x t } ( \pmb { v } ) : = a$ together with an intensional interpretation $\pmb { P } = ( P _ { 1 } , \dots , P _ { n } )$ of its positions. For each occurrence of $P _ { i }$ in position � of �, its extension is defined relative to that vector and position by

$$
\begin{array} { r } { \mathrm { E x t } _ { { \pmb v } , i } ( P _ { i } ) : \{ a \}  \overline { { \mathbb { R } } } , \qquad \mathrm { E x t } _ { { \pmb v } , i } ( P _ { i } ) ( a ) : = v _ { i } . } \end{array}
$$

Parameter vectors and matrices may likewise receive intensional interpretations of their positions, but lack an extensional interpretation as representations of objects.

## 5 Transformational Semantics and Transforming Concepts

Section 4 introduced an intensional–extensional semantics for representational vectors and a non-objectreferential interpretation of parameter structures. We now extend the semantics from vectors to parameterized transformations. The central step is to coordinate the value algebra of Section 3 with an algebra of properties. Parameter structures receive interpretations relative to the properties interpreting their arguments, and the resulting property-level maps are paired with the corresponding transformations of values. This provides the formal basis for the notion of a transforming concept.

## 5.1 The property algebra

The special elements introduced in the preceding sections now acquire their transformational role.

At the value level, the preserving parameter value ⭒ is a right identity for the operation occupying the multiplicative role: � $\displaystyle : \diamond ^ { \times } \ast = x .$ . At the property level, the preservation property ⭒⭒ plays the corresponding property-preserving role.

We define

$$
\odot : \mathcal { P } \times \mathcal { P } _ { \times \times } \longrightarrow \mathcal { P } ,
$$

called projection, by

$$
P \odot Q : = { \left\{ \begin{array} { l l } { Q , } & { Q \in { \mathcal { P } } , } \\ { P , } & { Q = { \dot { \pi } } { \dot { \times } } . } \end{array} \right. }
$$

Hence, $P \odot { \overset { \underset { \mathrm { i } } { } } { \approx } } \alpha = P$ . Thus, ⭒⭒ is a right identity for projection.

We also assume a binary operation

$$
\oplus : \mathcal { P } \times \mathcal { P } \longrightarrow \mathcal { P } ,
$$

called property sum, satisfying, for every $P \in { \mathcal { P } } _ { : }$

$$
{ \cal P } \oplus { \cal P } = { \cal P } , \qquad - \oplus { \cal P } = { \cal P } , \qquad { \cal P } \oplus \mathrm { -- } = { \cal P } ,
$$

and, for distinct non-null properties $P _ { 1 }$ and $P _ { 2 }$

$$
P _ { 1 } \oplus P _ { 2 } \neq P _ { 1 } \qquad \mathrm { a n d } \qquad P _ { 1 } \oplus P _ { 2 } \neq P _ { 2 } .
$$

Property sum is therefore idempotent, and the null property is a two-sided identity. The sum of two distinct non-null properties is a further property distinct from either of them.

No associativity assumption is imposed. Accordingly, finite property sums are defined by left association:

$$
\bigoplus _ { i = 1 } ^ { 1 } Q _ { i } : = Q _ { 1 } , \qquad \bigoplus _ { i = 1 } ^ { k + 1 } Q _ { i } : = \left( \bigoplus _ { i = 1 } ^ { k } Q _ { i } \right) \oplus Q _ { k + 1 } .
$$

The value and property algebras exhibit a parallel division of labor:

<table><tr><td>value level</td><td>property level</td></tr><tr><td>xi×mij</td><td>Pi O Pij</td></tr><tr><td>*+</td><td>⊕.</td></tr></table>

The idempotence of ⊕ is especially important: repeated contributions interpreted by the same property � satisfy $Q \oplus \cdots \oplus Q = Q$ , although their numerical contributions remain distinct. This will be central to the analysis of attention in Section 8.

## 5.2 Relative interpretations and second-level interpretations

We can now make explicit the semantic content of the structural association between parameter structures and argument vectors introduced in §4.4.

Let

$$
\pmb { P } : = ( P _ { 1 } , \dots , P _ { m } ) \in \mathcal { P } ^ { m }
$$

be the interpretation of an arbitrary argument of the matrix-parameterized transformation

$$
L _ { { \pmb { M } } } ^ { ( \star ^ { + } , \diamond ^ { \times } ) } ,
$$

where

$$
\begin{array} { r } { \pmb { M } = ( m _ { i j } ) \in \overline { { \mathbb { R } } } _ { \ast } ^ { m \times n } . } \end{array}
$$

An interpretation of � relative to � is a property matrix

$$
\begin{array} { r } { \Pi _ { P } : = ( P _ { i j } ) \in \mathcal { P } _ { * * } ^ { m \times n } . } \end{array}
$$

The subscript � indicates that the interpretation of the parameter matrix is not semantically independent of the interpretation of an argument of the corresponding transformation. Each parameter position $( i , j )$ is interpreted relative to the property $P _ { i }$ associated with the corresponding position of that argument.

For every position $( i , j )$ , the property-level contribution is $P _ { i } \odot P _ { i j }$ , in parallel with the value-level contribution $x _ { i } \diamond ^ { \times } m _ { i j }$ . Each column $j$ of $\Pi _ { P }$ defines an interpretive point $o f$ view. Intuitively, point of view � specifies, for each argument property $P _ { i }$ , whether a property is implicated, which property it is, and to what extent the implication holds.

More specifically, when $P _ { i j } \in \mathcal P \setminus \{ - - \} .$ , the implicated property is $P _ { i j }$ itself, since $P _ { i } \odot P _ { i j } = P _ { i j }$ . The parameter value $m _ { i j }$ specifies to what extent the implication holds. If, in particular, $P _ { i j } = P _ { i }$ , the implication is identical.

When $P _ { i j } = \frac { 1 } { 2 } \sqrt { 2 8 }$ , the implicated property is instead $P _ { i }$ , so that the implication is necessarily identical, again to the extent specified by $m _ { i j }$ , since $P _ { i } \odot ⨏ _ { \ Q } ^ { \therefore } ⨏ _ { \Sigma } ^ { \ Q } = P _ { i }$ . Finally, when $P _ { i j } = - { \bf \Gamma } _ { \mathrm { { ; } } }$ , no property is implicated by $P _ { i }$ according to point of view �, as $P _ { i } \odot \mathrm { - - } = \mathrm { - }$ –. That point of view makes no inferential contribution from $P _ { i }$ By the compatibility condition of §4.4, this occurs exactly when $m _ { i j } = -$ , and therefore $x _ { i } \diamondsuit \mathrm { \ -- } = -$ . Thus, the absence of an implication at the property level corresponds to the absence of a numerical contribution at the value level.

When an implication is present, the parameter value $m _ { i j }$ associated with the implicated property does not by itself contribute to the value of the resulting property. Intuitively, $m _ { i j }$ specifies the strength of the implication from $P _ { i }$ to the implicated property, according to point of view �. For a representational argument, the efective contribution of this implication must therefore also depend on the value $x _ { i }$ with which the implicating property $P _ { i }$ holds of the represented object. For this reason, the contribution to the resulting value is not $m _ { i j }$ alone, but $x _ { i } \diamond ^ { \times } m _ { i j }$ . The operation occupying the multiplicative role thus combines the strength with which the implicating property holds with the strength of the implication itself.

At the property level, the corresponding contribution is the implicated property $P _ { i } \odot P _ { i j }$ , whereas at the value level its contribution is $x _ { i } \diamond ^ { \times } m _ { i j }$ . The resulting property associated with output position � is obtained by combining the properties implicated according to point of view $j \colon$

$$
R _ { j } : = \bigoplus _ { i = 1 } ^ { m } \left( P _ { i } \odot P _ { i j } \right) .
$$

Writing $\pmb { R } : = ( R _ { 1 } , \ldots , R _ { n } )$ , we define the corresponding property-level map by

$$
L _ { \Pi _ { P } } ^ { ( \oplus , \odot ) } ( P ) : = R .
$$

At the value level,

$$
\begin{array} { r } { \pmb { y } : = L _ { \pmb { M } } ^ { ( \star ^ { + } , \diamond ^ { \times } ) } ( \pmb { x } ) , } \end{array}
$$

where

$$
y _ { j } : = \sum _ { i = 1 } ^ { m } \left( x _ { i } \diamond ^ { \times } m _ { i j } \right) .
$$

The value-level and property-level constructions therefore proceed in parallel:

$$
\boxed { y _ { j } : = \sum _ { i = 1 } ^ { m } \left( x _ { i } \diamond ^ { \times } m _ { i j } \right) , }
$$

$$
\left| R _ { j } : = \bigoplus _ { i = 1 } ^ { m } \left( P _ { i } \odot P _ { i j } \right) . \right|
$$

The resulting property $R _ { j }$ is thus obtained by combining the properties implicated according to point of view $j ,$ , while the corresponding resulting value $y _ { j }$ is obtained by combining their numerical contributions through $\star ^ { + }$

When � is a representational argument referring to $a \in D$ , we further assume that the resulting vector � is itself a representational vector referring to the same object �:

$$
\operatorname { E x t } ( \mathbf { x } ) = \operatorname { E x t } ( \mathbf { y } ) = a .
$$

If � interprets �, then by the definition introduced in $\ S 4 . 3$

$$
\begin{array} { r } { \mathrm { E x t } _ { { \bf y } , j } ( R _ { j } ) ( a ) = y _ { j } , \qquad j = 1 , \dotsc , n . } \end{array}
$$

The value-level transformation and the property-level map remain formally distinct. Their results are coordinated semantically insofar as � provides the intensional interpretation of the resulting representational vector ${ \boldsymbol { y } } ,$ whose occurrence-relative property extensions are thereby fixed by its values.

We call $\Pi _ { P }$ a second-level interpretation because projection admits a natural semantic interpretation in terms of implication. From point of view $j , P _ { i } \odot P _ { i j }$ can be read as the property implicated by $P _ { i } \colon$ when $P _ { i j } \in \mathcal P \backslash \{ - - \}$ , right-projection yields $P _ { i j } ;$ when $P _ { i j } = \dot { P } _ { i }$ or $P _ { i j } = \frac { 1 } { 2 } \times \frac { 1 } { 2 8 }$ , the implication is identical; and the null case expresses the absence of an implication. On this interpretation, when $P _ { i j } \neq P _ { i }$ , the implicated property may naturally be regarded as more abstract than $P _ { i } ,$ since it expresses what $P _ { i }$ implies according to point of view � rather than merely reproducing $P _ { i }$ . The subsequent property sum may then compose several such implicated properties into $R _ { j }$

By contrast, property sum admits a natural interpretation as abstraction-level preserving: the property resulting from the combination can be regarded as neither less abstract than any of its component properties nor more abstract than each of them, since their combination introduces no further implication and eliminates none of those already expressed by its components.

On this reading, the framework provides a formal-semantic explanation of the familiar capacity of neural networks to produce progressively more abstract representations: abstraction arises through the replacement, by projection, of argument properties with distinct properties implicated from diferent points of view, while property sum composes those implicated properties without itself producing or destroying abstraction.

The same semantic construction applies to vector-parameterized componentwise transformations, but in a structurally restricted form. Let

$$
\pmb { v } = ( v _ { 1 } , \dots , v _ { n } ) \in \overline { { \mathbb { R } } } _ { \ast } ^ { n }
$$

parameterize

$$
C _ { \pmb { v } } ^ { \diamond ^ { \times } } : \overline { { \mathbb { R } } } ^ { n } \longrightarrow \overline { { \mathbb { R } } } ^ { n } ,
$$

and let

$$
P = ( P _ { 1 } , \ldots , P _ { n } ) \in { \mathcal { P } } ^ { n }
$$

interpret an arbitrary argument. An interpretation of � relative to � is a tuple

$$
\begin{array} { r } { Q _ { P } : = \left( Q _ { 1 } , \ldots , Q _ { n } \right) \in \mathcal { P } _ { * * } ^ { n } . } \end{array}
$$

Each position � of the parameter vector defines a point of view restricted to the corresponding argument property $P _ { j } { \mathrm { : } }$ , rather than ranging over all $P _ { i }$ as a matrix column can. Thus $Q _ { j } \in \mathcal P \setminus \{ - - \}$ implicates $Q _ { j } ,$ with identical implication when $Q _ { j } = P _ { j } ; \ Q _ { j } = \ O _ { \ O } ^ {  } \ O _ { \ O } ^ {  }$ preserves $P _ { j } ;$ and $Q _ { j } = -$ implicates no property. The parameter value $v _ { j }$ specifies the extent of the implication through the value contribution $x _ { j } \diamond ^ { \times } v _ { j }$ . In particular, $Q _ { j } = \frac { 1 } { \sqrt { 2 } }$ together with $v _ { j } = \frac { 1 } { \kappa }$ preserves both property and value. The property-level map is consequently

$$
C _ { \boldsymbol { Q } _ { P } } ^ { \odot } ( P ) : = ( P _ { 1 } \odot Q _ { 1 } , \ldots , P _ { n } \odot Q _ { n } ) ,
$$

in parallel with the value-level transformation

$$
C _ { \pmb { v } } ^ { \diamond ^ { \times } } ( \pmb { x } ) : = ( x _ { 1 } \diamond ^ { \times } v _ { 1 } , \dots , x _ { n } \diamond ^ { \times } v _ { n } ) .
$$

Thus, for each point of view $j ,$

$$
\begin{array} { r } { \boxed { P _ { j } \ \longrightarrow \ P _ { j } \odot Q _ { j } , } } \\ { x _ { j } \ \longrightarrow \ x _ { j } \diamond ^ { \times } v _ { j } . } \end{array}
$$

A vector-parameterized transformation therefore difers from a general matrix-parameterized transformation in one important semantic respect. A matrix-defined point of view may draw implications from several or all argument properties and combine the resulting implicated properties into a single output property. A vector-defined point of view is restricted to the corresponding argument property $P _ { j }$ . It may preserve that property, replace it by a distinct property implicated by it, or imply no property at all, but it cannot combine implications originating from diferent argument positions.

The diagonal representation makes this restriction explicit. Let

$$
D ( Q _ { P } ) : = \left( \begin{array} { c c c c } { { Q _ { 1 } } } & { { \mathrm { --- } } } & { { \cdots } } & { { \mathrm { -- } } } \\ { { \mathrm { --- } } } & { { Q _ { 2 } } } & { { \cdots } } & { { \mathrm { -- } } } \\ { { \vdots } } & { { \vdots } } & { { \ddots } } & { { \vdots } } \\ { { \mathrm { --- } } } & { { \mathrm { --- } } } & { { \cdots } } & { { Q _ { n } } } \end{array} \right) .
$$

In column $j ,$ all of-diagonal positions are null, so point of view $j$ is semantically restricted to the corresponding argument property $P _ { j }$ . Consequently,

$$
L _ { D ( Q _ { P } ) } ^ { ( \oplus , \odot ) } ( P ) = C _ { Q _ { P } } ^ { \odot } ( P ) .
$$

The diagonal correspondence established at the value level in $\ S 3 . 2$ therefore has a precise semantic counterpart: vector parameterization corresponds to a family of interpretive points of view, each restricted to the corresponding argument property.

## 5.3 Transforming concepts

We can now state formally the notion introduced provisionally in $\ S 2$

For a fixed argument interpretation $\pmb { P } \in \mathcal { P } ^ { m }$ and a corresponding relative interpretation $\Pi _ { P }$ of the parameter matrix $M ,$ a matrix-parameterized transforming concept is identified with the ordered pair

$$
\begin{array} { r } { \boxed { \left( L _ { M } ^ { ( \star ^ { + } , \diamond ^ { \times } ) } , L _ { \Pi _ { P } } ^ { ( \oplus , \odot ) } \right) . } } \end{array}
$$

The first member is the value-level transformation, $x \longmapsto y ,$ while the second is the corresponding propertylevel map, $P \longmapsto R .$ . Together they determine

$$
( x , P ) \longmapsto ( y , R ) .
$$

For vector-parameterized componentwise transformations, the corresponding transforming concept is

$$
\left| \left( \boldsymbol { C } _ { v } ^ { \diamond ^ { \times } } , \boldsymbol { C } _ { Q _ { P } } ^ { \odot } \right) . \right|
$$

A transforming concept therefore requires more than mathematical parameterization. It requires a relative interpretation of the parameter structure and a corresponding property-level map. This makes precise the distinction between mathematical parameterization and conceptual parameterization.

Operational blocks are not transforming concepts: they have no parameter structure with a relative, second-level interpretation and hence no parameter-induced property-level map. Since they nevertheless map representational vectors to vectors that may become later arguments, we require them to preserve the input’s extensional reference and ordered intensional interpretation. Thus, if ${ \pmb y } = F ( { \pmb x } )$ , both vectors refer to the same object � and are interpreted by the same tuple $P ,$ although $\mathrm { E x t } _ { x , i } ( P _ { i } ) ( a ) = x _ { i }$ and $\mathrm { E x t } _ { y , i } ( P _ { i } ) ( a ) = y _ { i }$ may difer.

Preserving the complete ordered tuple also preserves its arity, so operational blocks are dimensionpreserving functions

$$
\boxed { F : \overline { { \mathbb { R } } } ^ { n } \longrightarrow \overline { { \mathbb { R } } } ^ { n } . }
$$

They may modify values and occurrence-relative extensions, but not the intensional interpretation. An operational block is therefore intensionally invariant.

The central distinction can therefore be summarized as follows:

<table><tr><td>parameterized transformation</td><td>non-parameterized operational block</td></tr><tr><td>+ relative second-level interpretation</td><td>+ intensional invariance</td></tr><tr><td>↓</td><td>业</td></tr><tr><td>transforming concept,</td><td>not a transforming concept.</td></tr></table>

## 5.4 Intensional invariance and dimensionality

The same notion of intensional invariance can be extended to transforming concepts. In this case, however, intensional invariance is not required by their semantic structure: a transforming concept may either preserve or change the ordered interpretation of its argument.

Let $\pmb { P } = ( P _ { 1 } , \dots , P _ { m } )$ be the interpretation of the argument and $\pmb { R } = ( R _ { 1 } , \dots , R _ { n } )$ the interpretation of the resulting vector. A transforming concept is intensionally invariant relative to � if and only if

$$
\boxed { R = P . }
$$

Otherwise, it is intensionally changing.

Since equality of ordered tuples requires equality of their arities,

$$
R = P \quad \Longrightarrow \quad m = n .
$$

Consequently,

� ≠ � ⟹ intensional invariance is impossible.

When $m = n ,$ , dimensional preservation alone does not guarantee intensional invariance: the transformation may still produce an interpretation $\mathbf { \boldsymbol { R } } \neq \mathbf { \boldsymbol { P } } .$ Let now

$$
Q _ { P } = ( \sharp _ { \mathbb { X } } ^ { \downarrow } , \dots , \sharp _ { \mathbb { X } } ^ { \downarrow } ) .
$$

The componentwise transforming concept

$$
\left( C _ { v } ^ { \circ ^ { \times } } , C _ { Q _ { P } } ^ { \odot } \right)
$$

is then intensionally invariant because $P _ { i } \odot \ O _ { \\approx \times \times } ^ { \downarrow \downarrow } = P _ { i }$ for every �, and therefore

$$
C _ { { Q _ { P } } } ^ { \odot } ( P ) = P .
$$

The corresponding parameter values need not preserve the numerical values. In general,

$$
C _ { \pmb { v } } ^ { \diamond } ( \pmb { x } ) \neq \pmb { x }
$$

may hold while

$$
C _ { \boldsymbol { Q } _ { P } } ^ { \odot } ( P ) = P .
$$

Hence, intensional invariance does not imply value preservation.

The preserving value ⭒ permits both forms of preservation to coincide: if $\pmb { v } = ( \not \kappa , \dots , \not \kappa )$ and $Q _ { P } = $ $( \frac { 1 } { 2 5 } \times \frac { 1 } { 5 } , \frac { 1 } { 2 5 } , \frac { 1 } { 2 5 } \times \frac { 1 } { 5 } )$ , then $C _ { \pmb { v } } ^ { \diamond } ( \pmb { x } ) = \pmb { x }$ and $C _ { Q _ { P } } ^ { \odot } ( P ) = P , { \mathrm { s o } } \left( x , P \right) \mapsto \left( x , P \right)$

The structural possibilities can thus be summarized as follows:

� = �, � = � ⇒ dimension-preserving and intensionally invariant,   
� = �, � ≠ � ⇒ dimension-preserving and intensionally changing,   
� ≠ � ⇒ dimension-changing and necessarily intensionally changing.

The second case is especially important because it shows that a change of intensional interpretation need not be accompanied by a change of dimensionality. Residual additions will provide an important Transformer case in Section 8.

## 5.5 The genesis of values and interpretations

The preceding analysis answers the constitutive question of what makes a parameterized transformation a transforming concept. We distinguish from it a genetic question concerning how parameter values and interpretations arise, addressed here only schematically, and a metasemantic question concerning what grounds the relevant interpretive and representational relations. The latter connects the framework with causal/informational, functional/inferential, environmental, and other accounts of content determination, among which we do not adjudicate.

For the limited genetic classification developed below, we distinguish

$$
G _ { \mathrm { v a l } } : = \mathrm { g e n e s i s ~ o f ~ t h e ~ p a r a m e t e r ~ v a l u e s } ,
$$

from

$$
G _ { \mathrm { i n t } } : = \mathrm { g e n e s i s ~ o f ~ t h e i r ~ i n t e n s i o n a l ~ i n t e r p r e t a t i o n . }
$$

Let � ∶= environmental interaction, and

$$
T : = \mathrm { t h e o r e t i c a l / a r c h i t e c t u r a l ~ d e t e r m i n a t i o n . }
$$

The genetic dimensions represented by $G _ { \mathrm { v a l } }$ and $G _ { \mathrm { i n t } }$ are distinct from the distinction between intensional invariance and intensional change established in §5.4. Whether a transforming concept is intensionally invariant depends on the relation between its argument and output interpretations, not on how either its parameter values or its parameter interpretation originated.

The following table is constructed from distinctions internal to the theory developed above. Its fourth column is added only to indicate Transformer instances of the corresponding configurations. The table is not intended as an exhaustive combinatorial taxonomy of every logically possible assignment of � and �.

<table><tr><td> $G _ { \mathrm { v a l } }$ </td><td> $G _ { \mathrm { i n t } }$ </td><td>Intensional effect</td><td>Illustrative Transformer cases</td></tr><tr><td>E</td><td>E</td><td>Intensionally changing</td><td>Learned matrices such as  $W ^ { Q } , W ^ { K } , W ^ { V } ;$  dynamically generated parameter structures such as 5  $K ^ { \top }$  and V</td></tr><tr><td>E</td><td>T</td><td>Intensionally invariant</td><td>Learned bias and scaling vectors; trainable positional vectors</td></tr><tr><td>T</td><td>T</td><td>Intensionally invariant</td><td>Sinusoidal positional vectors; masking, rescaling, and temperature</td></tr><tr><td>T</td><td>T</td><td>Intensionally changing</td><td></td></tr></table>

The first three configurations have Transformer instances; the fourth is formally permitted but has no clear one. The second and third rows are intensionally invariant for theoretical reasons: bias and scaling vectors, positional vectors, masking, rescaling, and temperature modify values without introducing a new interpretation, so their second-level interpretation contains ⭒⭒ throughout. Their diference lies in value genesis—training in the second row, theoretical or architectural determination in the third.

The first row includes two modes of environmental dependence: learned matrices such as $W ^ { Q } , W ^ { K } , W ^ { V }$ depend on training, whereas dynamically generated parameter structures acquire both values and second-level interpretations during inference from the current input. Thus � covers both modes without identifying them. The fourth row also makes explicit that theoretically or architecturally determined values and interpretations may nevertheless yield an intensionally changing transforming concept. Overall, the table adds a genetic classification without altering the constitutive semantics: genesis does not determine transforming-concept status or intensional invariance. Likewise, the static/dynamic distinction concerns genesis and temporal availability rather than constitutive semantic status.

The most significant results of this section can therefore be summarized as follows.

C2 — Transformational formal semantics. A parameterized transformation constitutes a transforming concept when its value-level transformation is coordinated with a property-level map determined by a relative, second-level interpretation of its parameter structure. The resulting transforming concept may be intensionally invariant or intensionally changing. Its constitutive semantic status is independent both of the genesis of its parameter values and interpretations and of which account, if any, is adopted to ground the relevant interpretive and representational relations. At the same time, by specifying precisely what is interpreted, at which level, how the diferent interpretive relations are connected, and how they contribute to the transformation of representational vectors, the present formal semantics provides the conceptual articulation required for posing metasemantic questions in a form adequate to the complexity of the systems under discussion.

## 6 Semantic Composition in Coordinated Systems

We now combine the architectural organization of Section 3 with the semantics of Sections 4–5. Rather than introducing a recursive semantics for arbitrary compositions, we identify the semantic conditions associated with the principal forms of coordination already distinguished architecturally.

## 6.1 Interpretive uniformity in operations

In §3.3, an operation was defined mathematically as the parallel organization of a finite family of transformations, where transformation is used in the technical sense fixed in §3.2 for the two kinds of parameterized transformations introduced there. At the semantic level developed in $\ S 5$ , we can now consider the corresponding family of transforming concepts.

Let

$$
T _ { k } : \overline { { \mathbb { R } } } ^ { m } \longrightarrow \overline { { \mathbb { R } } } ^ { n } , \qquad k = 1 , \ldots , s ,
$$

be the value-level transformations of such a family, and let

$$
\pmb { x } _ { 1 } , \dots , \pmb { x } _ { s } \in \overline { { \mathbb { R } } } ^ { m }
$$

be their argument vectors. The $T _ { k }$ need not be mathematically identical, and their parameter structures and relative second-level interpretations may also difer.

We call a parallel family of transforming concepts interpretively uniform when all its argument vectors share the same ordered intensional interpretation $\pmb { P } = ( P _ { 1 } , \dots , P _ { m } )$ , and all its resulting vectors share the same ordered intensional interpretation $\pmb { R } = ( R _ { 1 } , \ldots , R _ { n } ) . \mathrm { I f } \ \pmb { y } _ { k } = T _ { k } ( \pmb { x } _ { k } )$ , interpretive uniformity requires that each $\boldsymbol { x } _ { k }$ be interpreted by � and each $\scriptstyle { y _ { k } }$ by �.

Corresponding positions can then be described by the uniform semantic pattern � ↦ �, even when the individual transforming concepts difer in value-level transformations, parameter values, or relative secondlevel interpretations. Thus common arities sufice for a mathematical operation, whereas a semantically coordinated operation additionally requires a common ordered interpretation of corresponding input and output positions. Positional-vector addition and masking provide elementary Transformer examples.

## 6.2 Direct interconnections and semantic transmission

Direct interconnections introduce no further mathematical function. Their semantic efect therefore depends only on their architectural form. In a standard direct interconnection, an output vector becomes a later argument with its values, extensional reference, and ordered interpretation unchanged. A routing interconnection may regroup or reorder output vectors, in which case the interpretive organization follows the same pattern. Transformer Concat illustrates this: output vectors from diferent heads for the same token are concatenated together with their property tuples and retain that token referent. Routing alone, however, cannot determine the reference of a vector formed from representations with diferent referents; that would require additional semantic or architectural information. A direct interconnection therefore transmits already interpreted vectors or reorganizes them according to its architectural form.

## 6.3 Operational interconnections and intensional invariance

Operational interconnections difer because an operational block modifies transmitted values. By §5.3 it is nevertheless intensionally invariant: in $T _ { 1 }  F  T _ { 2 }$ , if the output � of $T _ { 1 }$ refers to � and is interpreted by �, then $F ( { \pmb x } )$ retains both while its values and occurrence-relative extensions may change. Thus no new interpretive level is introduced between source and destination; unlike a parameterized transformation, the value modification is not accompanied by a change of ordered intensional interpretation.

## 6.4 Output–parameter interconnections and second-level interpretation

Output–parameter interconnections introduce a diferent form of semantic composition. Here an output of a preceding transformation does not become the argument of a subsequent transformation, but all or part of its parameter structure.

Schematically, $T _ { 1 } ( { \pmb x } ) = { \pmb p } ,$ followed by $T _ { \pmb { p } } ( \pmb { y } )$ . The same numerical structure therefore participates successively in two distinct semantic roles. As an output of $T _ { 1 } .$ , � may be a representational vector with an extensional reference and an intensional interpretation of its positions. When it is subsequently used as a parameter structure, however, it is no longer interpreted in that role. Qua parameter structure, it receives instead a relative, second-level interpretation with respect to the intensional interpretation of the argument of the destination transformation.

Thus, the transition from output to parameter is not merely a numerical reassignment. It is also a change of semantic role. The output–parameter interconnection connects a first-level representational stage with the second-level interpretive structure required for conceptual parameterization.

An output–parameter interconnection changes the semantic role of the transmitted numerical structure without necessarily erasing its referential provenance. Qua parameter structure, the resulting vector or matrix has no extensional reference of its own. However, proper parts of the parameter structure may retain their relation of provenance to the representational vectors from which they were generated, or even the entire structure as a whole may retain such a relation. This provenance may remain semantically or architecturally relevant in the new role of the parameter structure. It may contribute to the second-level interpretation acquired by the parameter structure, or to the coordination between dynamically generated parameters and the argument vectors to which they are subsequently applied. The precise role of such referential provenance depends on the architecture of the relevant output–parameter interconnection.

With this caveat, the semantic coordination required by an output–parameter interconnection should not be understood as the preservation of the output’s previous representational interpretation. It consists instead in providing the parameter structure, in its new role, with the relative, second-level interpretation required by the transformation in which it now functions as parameter.

This distinction will become especially important in the analysis of attention and residual connections in §8.

## 6.5 Semantic coordination at system level

At the system level, semantic coordination results from the compatibility of the local semantic structures established in the preceding sections with the architectural relations defined in §3. Representational vectors satisfy the intensional–extensional semantics summarized in C1, while transformations receive the transformational semantics summarized in C2. Operations additionally require interpretive uniformity.

Direct interconnections preserve semantic coordination in ways determined by their architectural form, as illustrated paradigmatically by standard token-wise interconnections and routing structures such as Concat in the Transformer. Operational interconnections preserve the intensional interpretation through intensionally invariant operational blocks, while output–parameter interconnections satisfy the semantic conditions established in §6.4.

A coordinated system is therefore semantically coordinated when these local semantic constraints are mutually compatible with its architecture of transformations, operations, operational blocks, and interconnections. No further global semantic mechanism is required: system-level coordination results from the way in which locally interpreted vectors, transformations, and parameter structures are connected according to the functional organization of the system.

At this point, the framework specifies both the mathematical and the semantic coordination of the systems under consideration. The next question concerns a diferent level of description: which aspects of this organization belong to the abstract functional architecture of a system, and which belong instead to the particular mathematical instantiations of that architecture.

## 7 Abstract Functional Architecture, Mathematical Instantiation, and Realization

The preceding sections have characterized semantically coordinated systems of transformations at a level more abstract than any of their particular mathematical instantiations. Their common algebraic-semantic structure has been specified independently of the particular numerical operations and mathematical functions through which it may be instantiated, while their coordination has been described in terms of transformations, operations, operational blocks, and interconnections. We can now make explicit this level of abstraction and distinguish it from the mathematical, mechanical, and physical levels associated with it.

## 7.1 Abstract functional architecture

An abstract functional architecture, equivalently an abstract system type, combines an architectureindependent algebraic-semantic core with an architecture-specific functional organization. The core com prises the value algebra of $\ S 3$ and the semantics of §§4–6, fixing the roles of contribution, combination, projection, property sum, and the semantic organization of vectors, parameters, transforming concepts, and coordination. The functional organization specifies the arrangement and arities of transformations and operations, the pattern of direct, operational, and output–parameter interconnections, and the sources, destinations, and functional roles of operational blocks.

Diferent abstract architectures therefore share the same core and difer in functional organization. The present paper specifies the core in detail while characterizing only some constraints on admissible organizations, thereby providing a common basis for a plurality of abstract functional architectures rather than a unique one.

## 7.2 Mathematical instantiations

An abstract functional architecture does not determine a unique mathematical instantiation. The algebraicsemantic core itself, however, exhibits an important asymmetry between its value and property components.

The domain of the value algebra is fixed. Argument and resulting vectors range over ${ \overline { { \mathbb { R } } } } ,$ , while parameter structures range over $\overline { { \mathbb { R } } } _ { \ast }$ . What varies across mathematical instantiations are the operations occupying the abstract roles represented by

$$
{ \star } ^ { + } \qquad \mathrm { a n d } \qquad { \diamond } ^ { \times } .
$$

The constraints imposed on their underlying binary operations ⋆ and ⋄ are deliberately weak: apart from the specified behavior of their extensions with respect to the blank and preserving value, they are compatible with arbitrary binary operations on the real numbers. Diferent models of the value algebra may therefore assign mathematically very diferent operations to the same abstract roles. Since the value domain is fixed, variation between such models lies in the operations rather than in the domain on which they act.

The property algebra has a diferent structure. Its domain � is not fixed once and for all: diferent interpretations may employ diferent domains of intensional properties. The operations on those properties are constrained by the conditions introduced in $\ S 5 .$ Once � is fixed, projection ⊙ is thereby fixed, whereas property sum $\oplus$ remains only partially determined: for distinct non-null properties, its value must be a property distinct from either of them, but diferent operations may satisfy this condition. Diferent admissible property algebras may therefore difer both in their property domains and, for a fixed domain, in the way property sum is defined on distinct non-null properties.

The algebraic-semantic core consequently combines two complementary forms of model variation:

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>domain</td><td rowspan=1 colspan=1>operations</td></tr><tr><td rowspan=1 colspan=1>value algebraproperty algebra</td><td rowspan=1 colspan=1>fixedvariable</td><td rowspan=1 colspan=1>variable fixed, ⊕ constrained.</td></tr></table>

The object domain � may of course vary between applications of the semantics. Such variation does not alter either of the two algebras or the semantic structure defined over them; it changes only the domain to which that semantics is applied.

At the architecture-specific level, further mathematical variation arises because a mathematical instantiation must supply particular functions

$$
F : \overline { { \mathbb { R } } } ^ { n } \longrightarrow \overline { { \mathbb { R } } } ^ { n }
$$

for operational blocks whose abstract specification determines their functional roles, sources, destinations, and arities but not the mathematical functions performing those roles. Diferent functions may therefore instantiate the same architecture-specific functional role.

A mathematical instantiation of an abstract functional architecture is consequently specified by a model of its value algebra together with mathematical functions instantiating its architecture-specific functional roles. It does not determine a particular model of the property algebra or of the extensional–intensional semantics. Rather, it determines a class of admissible algebraic-semantic models sharing the same valuealgebra model and the same formal semantic constraints, while potentially difering in their object domain, property domain, admissible property-sum operation, and extensional and intensional assignments.

## 7.3 Functional-architectural invariance

Two mathematical instantiations of the same abstract architecture may difer in numerical operations or mathematical functions while preserving the same algebraic-semantic structure and architecture-specific organization—the kinds and arities of transformations and operations, interconnection pattern, and operational roles, sources, and destinations. The system is functionally-architecturally invariant across such instantiations.

This is not mathematical identity: mathematically diferent systems may instantiate the same abstract architecture, while the same function may belong to diferent abstract types when it occupies diferent roles or organizations. Abstract system type is therefore individuated by functional organization within the common algebraic-semantic framework, not by particular numerical operations or functions.

## 7.4 Abstract system types and the Transformer

The Transformer provides a paradigmatic illustration.

At the level considered here, the Transformer can be characterized as an abstract type of semantically coordinated system of transformations. It shares the common algebraic-semantic core developed above, while its architecture-specific functional organization determines its characteristic forms of parallel organization, direct and operational output–input interconnections, output–parameter interconnections, and the functional roles of its operational blocks.

The abstract Transformer type may admit diferent mathematical instantiations involving diferent mathematical choices. Conversely, other abstract system types share the same algebraic-semantic core while difering from the Transformer in their organization of transformations and operations, their patterns of interconnection, or the functional roles assigned to operational blocks.

The Transformer in its standard mathematical instantiation is used throughout this paper as a particularly important exemplar of one such type. Its role is illustrative rather than restrictive.

## 7.5 Mathematical instantiation, mechanical realization, and the physical level

Mathematical instantiation and mechanical realization are distinct relations to an abstract type. A mathematical instantiation of the abstract functional system assigns particular mathematical operations and functions to its abstract roles. A mechanical realization instead specifies an idealized mechanical system whose components and interactions realize the same algebraic-semantic core and the same architecture-specific functional organization.

Neither relation derives from the other. An abstract system type may admit diferent mathematical instantiations and diferent mechanical realizations independently. A particular mathematical instantiation of the abstract system does not determine which kind of mechanism realizes it, and a mechanical realization does not determine which mathematical instantiation is assigned to it.

Because every mechanical realization preserves the functional organization of the abstract system type it realizes, any mathematical instantiation of an abstract type � can also be assigned to any mechanical realization � of �. Thus, if � is a mathematical instantiation of � and � mechanically realizes �, the same � can be assigned to both � and �:

$$
A { \xrightarrow { M } } A _ { M }
$$

and, independently,

$$
A \longrightarrow R \stackrel { M } { \longrightarrow } R _ { M } .
$$

This branching permits two routes to the physical level.

A mathematically instantiated functional system $A _ { M }$ can be computationally implemented through computational simulation without a mechanical realization. Here, computational simulation is understood in the procedural sense of a method for implementing $A _ { M }$ on physical computers (cf. Humphreys 1990). We

do not use simulation here in the representational sense in which a simulation model stands in a distinct representational relation to a target. Thus,

$$
{ \big | } { \mathrm { c o m p u t a t i o n a l ~ s i m u l a t i o n } } = { \mathrm { p r o g r a m m i n g } } + { \mathrm { n u m e r i c a l ~ c o m p u t a t i o n } }
$$

In this characterization, programming reproduces the functional architecture and numerical computation evaluates the mathematical operations and functions specified by �. Thus

$$
A \xrightarrow { M } A _ { M } \xrightarrow { \mathrm { ( n u m e r i c a l ~ c o m p u t a t i o n ) } } \mathrm { c o m p u t a t i o n a l ~ i m p l e m e n t a t i o n } .
$$

The physical infrastructure need only execute this program and computation; it need not itself possess the mechanical organization of �.

A mechanical realization provides the other route. Once the mathematical instantiation � is assigned to it, $R _ { M }$ specifies idealized components and interactions that can guide physical construction:

$$
A \longrightarrow R \stackrel { M } { \longrightarrow } R _ { M } \longrightarrow \mathrm { p h y s i c a l ~ c o n s t r u c t i o n } .
$$

Thus a mathematically instantiated functional system can be physically implemented by simulation, whereas a mathematically instantiated mechanical realization can be materially constructed so that its components realize the mathematically instantiated mechanical organization.

For the Transformer, assigning the standard mathematical instantiation $M _ { \mathrm { s t d } }$ to the abstract type $A _ { T }$ yields a mathematically instantiated system that can be computationally implemented by programming its architecture and numerically evaluating the mathematical operations and functions specified by $M _ { \mathrm { s t d } } .$ Independently, $A _ { T }$ admits a mechanical realization in interconnected groups of simple neural networks, which can be given the same $M _ { \mathrm { s t d } } ;$ other realizations, such as neuromorphic ones, may likewise be given the same mathematical instantiation while motivating diferent physical constructions. The structure is therefore not a linear hierarchy: mathematical instantiation and mechanical realization branch from the abstract type, leading respectively to computational implementation or, through a mathematically instantiated mechanica realization, to physical construction.

The present paper has concentrated on the abstract system level. Its main contribution has been to identify the algebraic-semantic core shared by the class of systems considered here and to show how diferent architecture-specific functional organizations can be specified upon that core. Particular types of architecture, especially the Transformer architecture in its standard mathematical instantiation, have served to illustrate this level. A complete theory of mathematical instantiation, mechanical realization, computational implementation, or physical construction lies beyond the present scope.

The most significant results of this section can therefore be summarized as follows.

## C3 — Functional-architectural invariance and routes to the physical level.

Diferent abstract functional architectures share the same algebraic-semantic core and difer in their architecture-specific functional organization. Each abstract functional architecture may admit diferent mathematical instantiations and diferent mechanical realizations. Variations in particular numerical operations or mathematical functions do not constitute a change of abstract system type insofar as the common algebraic-semantic structure and the relevant organization of transformations, operations, interconnections, and functional roles are preserved.

This distinction also reveals two routes from the abstract functional level to physical systems. A mathematically instantiated abstract functional system can be computationally implemented through computational simulation: programming reproduces its functional organization, while numerical computation evaluates the mathematical operations and functions specified by its mathematical instantiation, without requiring an intervening mechanical realization. Alternatively, a mechanical realization of the same abstract functional architecture can be given the same mathematical instantiation and then physically constructed so that material components and interac tions realize its mathematically instantiated mechanical organization. Computational implementation and physical construction thus constitute distinct routes from abstract functional systems to the physical level.

## 8 Attention and Residual Connections

The preceding framework can now be used to analyze two characteristic processing patterns exemplified by the Transformer: attention heads and residual connections. The purpose is not to reproduce their standard mathematical description, but to identify the semantic organization determined by the local functional architectures of these modular patterns. The resulting explanations depend neither on the particular numerical operations through which they are mathematically instantiated in the standard Transformer nor on the overall functional architecture of the Transformer itself. They depend only on the abstract functional organization of the attention-head and residual-connection structures under consideration, together with the algebraic-semantic core shared by all semantically coordinated systems of transformations. They therefore apply to any abstract type of semantically coordinated system of transformations whose architecture includes structures of the same functional kinds.

## 8.1 Attention scores and token-indexed points of view

Consider an attention head operating on � representational vectors. Let

$$
Q , K \in \overline { { \mathbb { R } } } ^ { n \times d _ { k } } , \qquad V \in \overline { { \mathbb { R } } } ^ { n \times d _ { v } } ,
$$

with rows

$$
\begin{array} { r } { \pmb q _ { i } , \pmb { k } _ { i } \in \overline { { \mathbb { R } } } ^ { d _ { k } } , \qquad \pmb { v } _ { i } \in \overline { { \mathbb { R } } } ^ { d _ { v } } , \qquad i = 1 , \dots , n . } \end{array}
$$

The rows indexed by � originate from the corresponding token $a _ { i }$ . By interpretive uniformity, the rows of each matrix share a common ordered intensional interpretation. Let $\dot { \pmb { P } } ^ { Q } = ( \acute { P _ { 1 } ^ { Q } } , \dots , \bar { P _ { d _ { k } } ^ { Q } } )$

$$
\begin{array} { r } { \pmb { P } ^ { K } = ( P _ { 1 } ^ { K } , \dots , P _ { d _ { k } } ^ { K } ) , } \end{array}
$$

and $\pmb { P } ^ { V } = ( P _ { 1 } ^ { V } , \dots , P _ { d _ { v } } ^ { V } )$ be the interpretations of the query, key, and value vectors respectively.

The first dynamically generated parameter structure of the attention head is

$$
K ^ { \top } \in \overline { { \mathbb { R } } } _ { * } ^ { d _ { k } \times n } .
$$

When it parameterizes the transformation applied to a query vector

$$
\begin{array} { r } { \pmb q _ { i } \in \overline { { \mathbb { R } } } ^ { d _ { k } } , } \end{array}
$$

the resulting raw-attention vector belongs to ${ \overline { { \mathbb { R } } } } ^ { n }$ . Each column $j$ of $K ^ { \top }$ originates from the key vector $k _ { j }$ and therefore constitutes a token-indexed point of view associated by provenance with token $a _ { j }$ . Since the rows of $K$ are interpretively uniform, all these columns share the same ordered second-level interpretation $( P _ { 1 } ^ { K } , \dots , P _ { d _ { k } } ^ { K } )$ relative to the query interpretation $\pmb { P } ^ { Q }$

At the level of the general framework, nothing prevents a second-level key interpretation from containing preserving properties ⭒⭒. For the abstract attention-head type considered here, however, we impose the additional functional constraint that key positions are interpreted by ordinary properties:

$$
\begin{array} { r } { P _ { \ell } ^ { K } \in \mathcal { P } , \qquad \ell = 1 , \dots , d _ { k } . } \end{array}
$$

This constraint is satisfied by the standard Transformer and excludes the limiting degenerate possibility in which every key position is interpreted by ⭒⭒. In that case, the key structure would reproduce the query interpretation componentwise and would therefore fail to perform the semantic role characteristic of an attention head, as explained below.

Under the stated constraint,

$$
P _ { \ell } ^ { Q } \odot P _ { \ell } ^ { K } = P _ { \ell } ^ { K } \qquad ( \ell = 1 , \ldots , d _ { k } ) .
$$

Hence every position of a raw attention vector is interpreted by the same property $A : = \textstyle \bigoplus _ { \ell = 1 } ^ { d _ { k } } P _ { \ell } ^ { K }$ . The property � is therefore characteristic of the attention head under consideration: it results from the combination of its key properties.

The semantically non-trivial case is that in which at least one key property difers from the corresponding query property, $P _ { \ell } ^ { K } \neq P _ { \ell } ^ { Q }$ for some �. In that case, at least one query property is replaced by projection with a more abstract key property, before the resulting key properties are combined into �.

Property sum is abstraction-level preserving: the property resulting from the combination is neither less abstract than any of its component properties nor more abstract than each of them (see §5.2). The abstraction involved in the construction of � is therefore not generated by property sum itself. It is already present in the key interpretation and is preserved when the key properties are combined. In the semantically non-trivial case, � is consequently more abstract than each query property.

However, the identity of � across the � positions does not imply identity of their values. Although all columns of $K ^ { \top }$ share the same second-level intensional interpretation, their numerical values difer because each originates from the key vector associated with a diferent token $a _ { j } .$ The implications associated with the key properties therefore depend not only on the properties involved but also, at the value level, on the particular point of view through which those properties are applied.

For a query originating from token $a _ { i } ,$ the �-th raw attention score is thus an occurrence-relative value of the same property �, obtained from point of view � associated with token $a _ { j } \colon$ the raw-attention vector has the uniform intensional interpretation $( A , \ldots , A )$ , while its diferent positions generally have diferent values.

The subsequent rescaling and masking stages are intensionally invariant transforming concepts, while the normalization stage is an intensionally invariant operational block, instantiated by Softmax in the standard Transformer. They may modify the numerical values of the vector but do not introduce a diferent ordered intensional interpretation. The resulting attention vector

$$
\alpha _ { i } = ( \alpha _ { i 1 } , \ldots , \alpha _ { i n } ) \in \overline { { \mathbb { R } } } ^ { n }
$$

therefore retains the interpretation $( A , \ldots , A )$ . The first part of the attention pattern can thus be understood semantically as producing, for each token $a _ { i } ,$ diferent occurrence-relative values of one and the same more abstract attention property �, each obtained through a diferent token-indexed point of view.

## 8.2 Value aggregation and the semantic output of attention

The second dynamically generated parameter structure involved in the attention pattern is

$$
V \in \overline { { \mathbb { R } } } _ { * } ^ { n \times d _ { v } } ,
$$

when the value matrix assumes its parameter role. The attention vector

$$
\alpha _ { i } \in \overline { { \mathbb { R } } } ^ { n }
$$

$$
\overline { { \mathbb { R } } } ^ { d _ { v } }
$$

is now the argument of a transformation whose result belongs to . The rows $\pmb { v } _ { 1 } , \dots , \pmb { v } _ { n }$ of $V$ share the common intensional interpretation $\pmb { P } ^ { V } = ( P _ { 1 } ^ { V } , \dots , P _ { d _ { v } } ^ { V } )$ , while retaining their distinct referential provenance from the corresponding tokens $a _ { 1 } , \ldots , a _ { n }$

For the abstract attention-head type considered here, we likewise require the second-level interpretation supplied by � to contain only ordinary properties: $P _ { h } ^ { V } \in \mathcal { P } , \qquad h = 1 , \hdots , \bar { d } _ { v }$ . This constraint, also satisfied by the standard Transformer, excludes preserving properties ⭒⭒, which would preserve the attention property � instead of supplying the value property associated with the corresponding output position.

For output position $h ,$ every argument position � is associated with the same second-level property $\boldsymbol { P } _ { h } ^ { V }$ Since the attention vector is interpreted throughout by �, the property-level result is

$$
R _ { h } : = \bigoplus _ { j = 1 } ^ { n } \left( A \odot P _ { h } ^ { V } \right) .
$$

Under the stated constraint, $A \odot P _ { h } ^ { V } = P _ { h } ^ { V }$ . By the idempotence of property sum, $\begin{array} { r } { R _ { h } = \bigoplus _ { j = 1 } ^ { n } P _ { h } ^ { V } = P _ { h } ^ { V } } \end{array}$ . Hence the output vector

$$
z _ { i } \in \overline { { \mathbb { R } } } ^ { d _ { v } }
$$

has the ordered intensional interpretation $\pmb { P } ^ { V }$ . It continues to refer to token $a _ { i } ,$ while its numerical values result from contributions whose provenance lies in the diferent token-indexed value vectors $\pmb { v } _ { j }$

This yields a general semantic explanation of the attention pattern. Its first part constructs a single attention property � from the key interpretation and assigns diferent values of that property according to token-indexed points of view. Its second part uses those values to regulate the contributions associated with the corresponding value vectors, while the resulting vector preserves the ordered intensional interpretation of the value representation and the reference of the query-originating token.

Together with the ordinary-property constraint just stated, two structural conditions are crucial to this result: interpretive uniformity and the idempotence of property sum. Because the value vectors share the same interpretation, each contribution to output position ℎ implicates the same property $\boldsymbol { P } _ { h } ^ { V }$ . Because property sum is idempotent, combining any number of such contributions leaves that property unchanged.

The semantic explanation therefore does not depend on ordinary multiplication and addition as such, nor on the particular mathematical function used by the normalization operational block, such as Softmax in the standard Transformer. Diferent mathematical instantiations can alter the numerical values generated by the attention pattern while preserving the same semantic organization, provided that they instantiate the same local functional roles and satisfy the relevant interpretive constraints.

The resulting invariance is therefore not numerical but semantic and functional: mathematically diferent instantiations may realize the same organization of token-indexed points of view, attention values, regulated contributions, and intensional output.

## 8.3 Residual connections and intensional change

Residual connections exhibit a diferent semantic organization.

Let

$$
\pmb { x } _ { i } = ( x _ { i 1 } , \dots , x _ { i d } ) \in \overline { { \mathbb { R } } } ^ { d }
$$

be a representational vector entering an attention or feed-forward module. Suppose that it refers to token $a _ { i }$ and is interpreted by $\pmb { P } = ( P _ { 1 } , \dots , P _ { d } )$ . Let

$$
\bullet _ { i } = ( o _ { i 1 } , \ldots , o _ { i d } ) \in \overline { { \mathbb { R } } } ^ { d }
$$

be the corresponding module output. By semantic coordination it refers, in its representational role, to the same token $a _ { i } ,$ but it may have a diferent ordered intensional interpretation $\boldsymbol { Q } = ( Q _ { 1 } , \dots , Q _ { d } )$ . In the abstract residual-connection type, $\mathbf { o } _ { i }$ becomes the dynamically generated parameter vector of the componentwise transformation

$$
C _ { \pmb { o } _ { i } } ^ { \diamond ^ { \times } } ( \pmb { x } _ { i } ) .
$$

In the standard Transformer, $\diamond ^ { \times }$ is instantiated by $+ ^ { \times }$ , yielding

$$
{ C _ { { \pmb 0 } _ { i } } ^ { + } } ( { \pmb x } _ { i } ) = { \pmb x } _ { i } + { \pmb 0 } _ { i } .
$$

The referential provenance of $\mathbf { o } _ { i }$ remains architecturally relevant: it identifies the module output associated with the same token as the input vector $\mathbf { \boldsymbol { x } } _ { i }$ , thereby fixing the token-by-token correspondence required by the residual organization. Qua parameter vector, however, $\mathbf { o } _ { i }$ has no extensional reference of its own. Its properties acquire instead a second-level interpretation relative to the interpretation � of the argument.

For the residual-connection type considered here, the second-level properties $Q _ { j }$ are ordinary properties. Hence $P _ { j } \odot Q _ { j } = Q _ { j } \qquad ( j = 1 , \ldots , d )$ . Consequently,

$$
C _ { \boldsymbol { Q } _ { P } } ^ { \odot } ( P ) = Q .
$$

The residual-connection output therefore has the intensional interpretation of the module output used as parameter. The residual transforming concept is intensionally invariant exactly when $Q = P .$ When instead $Q \neq P$ , the residual connection is dimension-preserving but intensionally changing.

This result clarifies a feature that the value-level expression

$$
C _ { \pmb { o } _ { i } } ^ { \diamond } ( \pmb { x } _ { i } )
$$

does not by itself reveal. The two vectors have the same dimensionality, belong to the same numerical domain, and, in their representational roles, refer to the same token. Yet they need not have the same intensional interpretation. Their mathematical similarity therefore does not entail semantic identity.

Moreover, the two vectors occupy diferent semantic roles. The input vector is the argument, whereas the module output becomes the dynamically generated parameter structure. This asymmetry is fixed by the output–parameter interconnection independently of the particular numerical operation instantiating ⋄<sup>×</sup>. In the standard Transformer this point is especially striking because $\diamond ^ { \times }$ is instantiated by $+ ^ { \times }$ , where +, i.e. ordinary addition, is symmetric, although the semantic organization of its two operands is not.

Residual connections therefore provide a concrete instance of the general result established in §5.4: dimensional preservation does not imply intensional invariance.

## 8.4 Complementarity and modularity

The two analyses reveal a complementary pattern, although this is not their principal result. In attention, diferent mathematical operations and functions can preserve semantic organization through interpretive uniformity, referential provenance, projection, and idempotent property sum. Residual connections show the converse possibility: vectors of the same mathematical kind may play diferent semantic roles and carry diferent intensional interpretations, so mathematical similarity does not determine their semantic relations.

More importantly, an attention head and a residual connection can themselves be regarded as diferent abstract types of semantically coordinated systems, each with a local functional architecture governed by the common algebraic-semantic core. This modular organization is already implicit in Figure 1 of TR1; the present framework makes its significance explicit. A complex architecture can therefore be decomposed into simpler modular types that may recur, with the same semantic analysis, across otherwise diferent abstract architectures. The framework thus reveals semantic invariance across mathematical variation, semantic diferences despite mathematical similarity, and modular semantic organization across system types.

## 9 Philosophical Significance, Scope, and Conclusion

The preceding results identify a level at which the semantic organization of artificial systems can be characterized without reducing it to numerical description or presupposing a theory of representational grounding. Their philosophical significance concerns two connected issues: the relation between formal semantics, representation, and metasemantic grounding, and sameness and variation across descriptions and instantiations. The modular analyses of attention and residual connections extend these questions from complete architec tures to functional subsystems.

## 9.1 Formal semantics before representation and grounding

Contemporary debates about representation and content in artificial systems commonly ask under what conditions internal states genuinely represent objects, properties, or concepts. The approaches considered in §2 emphasize diferent candidates for answering this question: coactivation and functional role, informational relations, downstream use, conditions for misrepresentation, structural correspondence, or more general theories of aboutness and metasemantics (Freiesleben 2026; Poth and Schuster 2026; Klein 2026; Millière and Buckner 2026; Cappelen and Dever 2021). Kieval’s (2026) warning against inferring philosophical representationalism directly from the success of representation learning is especially relevant here.

The present proposal does not attempt to decide among these accounts. It changes the order of analysis. Before asking what makes a vector, a position, or a property genuinely representational, one can ask what formal semantic organization is obtained once the relevant interpretations are specified.

C1 provides such an organization by distinguishing extensional vector reference, ordered intensional interpretation, and occurrence-relative property extension, while parameter structures receive intensional interpretations without object-reference. It thereby separates formal interpretation, genuine representational relations, and their grounding.

The priority claimed is analytical rather than metasemantic. Formal interpretation neither creates content nor explains why a given property or referent is correct; it specifies what a theory of representation or grounding must explain, including the assignment of properties to positions and of referents to representational vectors.

This gives formal semantics an autonomous philosophical role without turning it into a substitute for metasemantics. By articulating the structure of interpretation before committing to a particular account of its grounding, the framework also follows a methodological orientation naturally associated with the formalsemantic tradition represented by Dalla Chiara’s work: formal distinctions concerning interpretation can themselves be philosophically informative while leaving stronger questions concerning their foundation open (Dalla Chiara et al. 2018).

## 9.2 From representing contents to transforming them

The second contribution goes beyond the formal semantics of representational states. C2 extends semantic analysis to the transformations acting on them.

This extension matters because the semantic organization of the systems considered here is not exhausted by assigning contents to vectors. Parameter structures acquire relative, second-level interpretations with respect to the properties interpreting their arguments. A parameterized transformation becomes a transforming concept only when its numerical action is coordinated with the corresponding property-level map. Mathematical parameterization and conceptual parameterization are therefore distinct.

If internal representations are transformed during processing, it is not enough to ask what successive vectors represent: one must also characterize how one interpreted representation becomes another and the semantic role of the structures determining that change. Projection, property sum, relative interpretations, and interpretive points of view make this explicit. Parameter structures can therefore play a relational semantic role without representing objects themselves, specifying how argument properties are preserved, replaced, or combined; in output–parameter interconnections this role may arise dynamically without erasing referential provenance.

The distinction between constitutive semantics and genesis becomes especially important here. Whether a transforming concept is static or dynamic, whether its parameter values arise through training or during inference, and whether its interpretation depends on environmental interaction or architectural determination do not define its status as a transforming concept. Those questions concern how the relevant semantic organization comes about; the formal semantics specifies what that organization is once present.

The proposal therefore extends the philosophical problem of content in artificial systems from contentbearing states to content-transforming organization. This is precisely what makes the analyses in §8 possible: attention and residual connections can be explained semantically not merely by interpreting their inputs and outputs, but by reconstructing the semantic roles performed by the transformations and interconnections that relate them.

## 9.3 Sameness, variation, and modular explanation

C3 introduces a second form of abstraction. Diferent mathematical systems can instantiate the same abstract functional architecture when they preserve both the common algebraic-semantic core and the architecturespecific organization of transformations, operations, interconnections, and functional roles. Their identity as systems of the relevant abstract type therefore does not require identity of their numerical operations or of the mathematical functions assigned to particular operational blocks.

This notion of functional-architectural invariance is related to, but should not be conflated with, familiar discussions of multiple realization. The latter typically concern sameness at one level together with physically or mechanically relevant diferences at another (Polger and Shapiro 2016). Recent work has renewed this question in connection with deep learning and has emphasized that multiple realizability does not make differences of implementation explanatorily irrelevant (McGrath and Russin 2024). The invariance established here occurs at a diferent boundary: it concerns diferent mathematical instantiations of the same abstrac functional type, prior to asking how that type is mechanically or physically realized.

This distinction makes the relations among levels more articulated than a single hierarchy from abstract description to physical realization. As shown in §7.5, a mathematically instantiated abstract functional system can be computationally implemented through computational simulation without first being mechanically realized. Alternatively, the abstract system may receive a mechanical realization, which can receive the same mathematical instantiation and subsequently guide a physical construction. Computational implementation and physical construction are therefore diferent routes to the physical level rather than successive stages of one necessary route.

Section 8 adds modularity of semantic explanation: attention heads and residual connections are themselves abstract types whose explanation depends on local functional organization and the common core, so the same module may recur across diferent larger architectures. Attention also shows semantic organization preserved across mathematical instantiations, while residual connections show that vectors of the same mathematical kind may have diferent semantic roles and intensional interpretations. The relevant relations are fixed by coordination between the algebraic-semantic core and functional organization, not by mathematica similarity alone.

## 9.4 Scope and open questions

These results also delimit what the present framework establishes.

First, the framework is not a general theory of representation: it presupposes that some vectors represent objects and does not derive that status or their properties from mathematical structure. Second, it is not a metasemantic theory: causal/informational, functional/inferential, environmental, historical, or other determinants of content may explain why the formal interpretations are appropriate, and first- and secondlevel interpretation, extensional reference, occurrence-relative extension, and referential provenance need not all receive the same metasemantic explanation. Third, the genetic classification of §5.5 is preliminary and does not explain how systems acquire content.

Finally, the paper does not theorize the mechanical, neural, or physical realization of these abstract systems. Section 7 only distinguishes mathematical instantiation, mechanical realization, computational implementation, and physical construction. The further questions of generalized neural realizability and of how the present generalization bears on Conjecture T—the in-principle hypothesis introduced in TR1 that human neural systems may realize a functional architecture relevantly similar to that of the Transformer—are left for future work. Taken together, these limitations preserve distinct explanatory levels so that generation, grounding, realization, implementation, and construction can later be investigated separately.

## 9.5 Conclusion

The starting point of this paper was the interpretation proposed in TR1 of the Transformer as a system that transforms concepts by means of concepts. The aim has been to determine whether the semantic organization suggested by that interpretation can be formulated independently of the particular mathematics and neural organization of the standard Transformer and generalized to a wider class of systems.

The framework yields three main results: C1 distinguishes the intensional–extensional semantics of representational vectors from the non-referential interpretation of parameter structures; C2 defines transforming concepts by coordinating value-level transformations with property-level maps determined by relative, second-level interpretations; and C3 separates abstract functional architecture from mathematical instantiation and mechanical realization, establishing functional-architectural invariance and two routes to the physical level.

Section 8 shows the explanatory payof: attention and residual semantics can remain invariant across diferent mathematical instantiations of the same local organization, and both patterns are modular abstract system types. The resulting generalization is therefore both mathematical and architectural.

The framework does not show that every vector transformation is semantic, that every mathematically adequate model represents, or that formal interpretation is suficient for genuine content. It establishes a more limited but prior result: if coordinated vector transformations are semantically interpreted in the manner specified here, their semantic organization can itself be formally characterized, their transformations can be analyzed as transformations of interpreted properties, and the invariants of that organization can be separated from the particular mathematical, mechanical, and physical forms through which it is instantiated or realized.

This provides a basis for moving to the next question rather than an answer to it: how abstract systems with this semantic and functional organization can be mechanically—and, in particular, neurally—realized, computationally implemented, or physically constructed.

## References

Giunti, Marco, and Fabrizia Giulia Garavaglia. 2026. “The Transformer Revolution, Part 1: Dynamic Processing through Output-Weight Interconnections.” arXiv. https://doi.org/10.48550/arXiv.2608.03921.

Buckner, Cameron. 2018. “Empiricism without Magic: Transformational Abstraction in Deep Convolutional Neura Networks.” Synthese 195 (12): 5339–5372. https://doi.org/10.1007/s11229-018-01949-1.

Cappelen, Herman, and Josh Dever. 2021. Making AI Intelligible: Philosophical Foundations. Oxford: Oxford University Press. https://doi.org/10.1093/oso/9780192894724.001.0001.

Dalla Chiara, Maria Luisa, Roberto Giuntini, Roberto Leporini, and Giuseppe Sergioli. 2018. Quantum Computation and Logic: How Quantum Computers Have Inspired Logical Investigations. Cham: Springer. https://doi.org/10. 1007/978-3-030-04471-8.

Freiesleben, Timo. 2026. “Artificial Neural Nets and the Representation of Human Concepts.” In Philosophy of Science for Machine Learning: Core Issues and New Perspectives, edited by Juan M. Durán and Giorgia Pozzi, 273–294. Cham: Springer. https://doi.org/10.1007/978-3-032-03083-2\_13.

Humphreys, Paul. 1990. “Computer Simulations.” PSA: Proceedings of the Biennial Meeting of the Philosophy of Science Association 1990 (2): 497–506.

Kieval, Phillip Hintikka. 2026. “Representation Learning Without Representationalism: A Non-representational Account of Deep Learning Models in Scientific Practice.” In Philosophy of Science for Machine Learning: Core Issues and New Perspectives, edited by Juan M. Durán and Giorgia Pozzi, 245–272. Cham: Springer. https: //doi.org/10.1007/978-3-032-03083-2\_12.

Klein, Colin. 2026. “Represented Structure versus Representational Structure: A Challenge for Interpreting LLMs.” Philosophy and the Mind Sciences 7 (1): 1–12. https://doi.org/10.33735/phimisci.2026.12137.

McGrath, Sam Whitman, and Jacob Russin. 2024. “Multiple Realizability and the Rise of Deep Learning.” arXiv:2405.13231. https://doi.org/10.48550/arXiv.2405.13231.

Millière, Raphaël, and Cameron Buckner. 2026. “The Philosophy of Language Models.” Philosophy Compass 21 (3): e70095. https://doi.org/10.1111/phc3.70095.

Polger, Thomas W., and Lawrence A. Shapiro. 2016. The Multiple Realization Book. Oxford: Oxford University Press. https://doi.org/10.1093/acprof:oso/9780198732891.001.0001.

Poth, Nina, and Annika Schuster. 2026. “Mental, Scientific, and Artificial Representations.” Philosophy and the Mind Sciences 7 (1). https://doi.org/10.33735/phimisci.2026.12094.

Räz, Tim. 2023. “Methods for Identifying Emergent Concepts in Deep Neural Networks.” Patterns 4 (6): 100761. https://doi.org/10.1016/j.patter.2023.100761.

Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. “Attention Is All You Need.” In Advances in Neural Information Processing Systems 30, 5998–6008.