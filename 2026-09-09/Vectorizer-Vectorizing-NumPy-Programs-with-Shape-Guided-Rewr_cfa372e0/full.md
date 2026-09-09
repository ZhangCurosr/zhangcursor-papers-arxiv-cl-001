# Vectorizer: Vectorizing NumPy Programs with Shape-Guided Rewrite

Jingqian Liu Simon Fraser University Burnaby, British Columbia, Canada jingqian\_liu@sfu.ca

Xiaoyu Liu Simon Fraser University Burnaby, British Columbia, Canada xla411@sfu.ca

Yuepeng Wang Simon Fraser University Burnaby, British Columbia, Canada yuepeng@sfu.ca

## Abstract

NumPy is a widely used Python library for numerical scientific computing, known for its declarative APIs and its optimized implementations. However, writing eficient NumPy programs, which often entails using vectorized array operations instead of explicit Python loops, may not be straightforward. This can be dificult for programmers who are accustomed to imperative array traversal, especially when vector ized API invocations require careful reasoning about shapes, broadcasting, and advanced indexing. This paper presents a rewrite-based approach for vectorizing NumPy programs with explicit loops over array data. Our approach vectorizes loops from the inside out, using array shapes and dataflow analysis to guide a source-to-source transformation that replaces loop bodies with vectorized statements. Following a set of rewrite rules that are correct by construction, our approach is consistently fast. We have implemented the approach as a tool called Vectorizer and evaluated it on 150 benchmarks collected from prior work and Stack Overflow. The evaluation shows that Vectorizer vectorizes 142 of the 150 benchmarks directly and 2 more after minor changes to the original benchmarks, with only 0.53 seconds on average to rewrite each one. The resulting programs are, on average, 74.83× faster than the original loop-based implementations.

## 1 Introduction

NumPy [23] ofers high-level APIs for eficient computation with multidimensional arrays while hiding optimizations and memory management in its low-level implementation. Such APIs are designed around concepts like axes and shapes. Built upon such concepts, techniques such as broadcasting enable concise, declarative array programming. However, correctly aligning axes for broadcasting and advanced indexing can be confusing, even for experienced programmers. As a result, programmers often write nested loops to traverse arrays, which may incur substantial runtime ineficiency.

Vectorizing loop-based NumPy code is challenging. First, Python has a rich syntax. NumPy exposes many operators with flexible interfaces, which makes it dificult to lift arbitrary NumPy code. Second, an efective lifter should incorporate techniques like broadcasting and advanced indexing to improve memory and runtime eficiency. These techniques require properly aligning axes, which can be tricky for code with nested loops and many variables. Third, practical NumPy programs often contain branches, reductions, and other structures whose vectorized forms do not have a straightforward correspondence to the original code.

Existing work attempts to address similar problems. For example, Tenspiler [44] lifts C++ programs to high-level array DSLs by summarizing the input as logical formulas and searching for an equivalent target program. However, it requires program-specific configurations, and extending it with new operators requires additional axioms. Both tasks demand non-trivial user efort. Tensorize [15] summarizes programs via symbolic execution and searches for an equivalent program with pruning based on symbolic algebra solvers. It expects input in the Afine dialect of MLIR [29], which imposes exacting requirements on the source program.

Upon reviewing NumPy questions on Stack Overflow and NumPy code on GitHub, we observe that many loops can be replaced by vectorized statements through deductive program rewrites. The key insight is that vectorizing the innermost loop expands its loop variable to an array with an additional axis. Variables defined in outer scopes can be treated as free variables with fixed shapes. By carefully rewriting expressions in the loop based on shapes, we can propagate the newly added dimension to vectorize the entire loop.

For example, suppose we want to vectorize the code below.

1 for i in range (A. shape [0]) :   
2 for j in range ( A . shape [1]) :   
3 A [i , j ] = i + j

We can first vectorize the innermost loop, treating i as a free variable. Replacing j with np.arange(A.shape[1]), the 1-D array of all values taken by j, yields the following code.

1 for i in range ( A . shape [0]) :   
2 js\_ = np . arange ( A . shape [1])   
3 A[i, js\_] = i + js\_

With broadcasting, we expand both sides of Line 3 to 1-D arrays, updating all the elements of each row of A at once.

In this paper, we present a novel method based on this insight for rewriting Python loops over array data into vectorized NumPy code. The high-level workflow of our method is shown in Figure 1. Rather than attempting to reason about the full NumPy, we introduce a domain-specific language (DSL) that captures the common operations needed for practical loop vectorization while keeping the analysis tractable.

![](images/3a98d3b4ff7005da5814ee334bee1549a6f9834fc3c80b8b80d880a3013912e0.jpg)

![](images/e646e5b0686112fb65e0d8e6f010ce33b18afdd272cf2b1208c4f37eb14c2013.jpg)  
Figure 1. An overview of the workflow.

Within this DSL, we design a set of rewrite rules for vectorizing code with explicit loops over data. The rules are correct by construction, which allows the transformation procedure to avoid enumerative search or a separate equivalencechecking phase. The rewrite process is guided by dataflow analysis and inferred type information about array shape and maskedness. Our rewrite procedure works inside out, repeatedly vectorizing the innermost loop until there is no more loop in the program. After rewriting, we apply postprocessing optimizations, such as indexing simplifications, sum-after-multiplication-to-tensordot replacements, common subexpression elimination and unused variable elimi nation, to produce the final runnable NumPy program.

We have implemented our method in a tool called Vectorizer and evaluated it on 150 benchmarks. The results show that our technique ofers significantly better applicability compared to tools from previous work. Specifically, Vectorizer can successfully vectorize 142 of the 150 benchmarks directly, and 2 more benchmarks after minor adaptations. On average, Vectorizer can rewrite a NumPy function in only 0.53 seconds, and the resulting programs are, on average, 74.83× faster than the original implementations.

Contributions. We make the following main contributions:

• We designed a DSL capturing the core structures and APIs ofreal-world NumPy programs. We formalize its semantics and type inference rules for shapes and maskedness.

• We propose an inside-out loop vectorization algorithm guided by shapes, maskedness, and dataflow analysis.

• We formulate the core vectorization procedure as a set of correct-by-construction rewrite rules over the DSL.

• We implement the approach as a tool called Vectorizer and evaluate it on 150 benchmarks. The results show that it is efective and eficient, and the obtained programs are on average 74.83× faster than original implementations.

## 2 Overview

Let us consider the task to compute the increasing powers of each element in a given 1-D array. Specifically, the task requires implementing a function that takes an input vector � and returns a 2-D array � such that each entry $X _ { i , j } = x _ { i } ^ { j } .$ A simple loop-based implementation is shown below.

```python
1 import numpy as np
2 def looped_power (x):
3 # x : shape (n ,) .
4 X = np . zeros (( x . shape [0] , x . shape [0]) )
5 for i in range (x. shape [0]) :
```

Figure 2. Graphical illustration of the power computations.  
6 for j in range ( x . shape [0]) :   
7 X [i , j ] = np . pow ( x [ i ] , j )   
8 return X

Given that np.pow supports element-wise power on arrays, to vectorize the loops, we can replicate x as columns and repeat numbers from 0 to � − 1 as rows. Then we only need to call np.pow once on the two arrays to get the result. The explicit loops are delegated to the optimized NumPy implementation. This idea is demonstrated graphically in Figure 2a.

In fact, we can further simplify by taking advantage of broadcasting, which implicitly replicates data on axes with dimensionalities of 1 to avoid boilerplate code and unnecessary memory usage. The function looped\_power\_vectorized follows this idea, which is shown below.

```python
1 def looped_power_vectorized (x):
2 exponents = np. arange (x. shape [0])
3 return np. pow (np. expand_dims (x ,1) , exponents )
```

Instead of replicating arrays explicitly, this function expands the x array with one more axis, changing its shape from (�, ) to (�, 1). Here, the shape of an array is a tuple of natural numbers specifying the size of each dimension of the array. In Line 2, np.arange is called to get all the exponents in an array of shape (�, ). Because of broadcasting, we can get the same result. The idea is illustrated in Figure 2b.

From this example, we see that one of the challenges in vectorizing such functions lies in correctly aligning computations on target axes. To automate such transformations, we can first obtain a partially vectorized version of the function.

```python
1 def looped_power_one_loop (x):
2 X = np. zeros ((x. shape [0] , x. shape [0]) )
3 for i in range (x. shape [0]) :
4 js_ = np. arange (x. shape [0])
5 X[i, js_] = np. pow (x[i], js_ )
6 return X
```

Here, the innermost loop is vectorized by replacing j with js\_, the array of values it ranges over. All other variables keep their original shapes, and broadcasting performs row-wise computations and updates. Note that X[i, js\_] broadcasts i to get all the indices of row i, equivalent to X[i, 0:x.shape [0]]. Next, we can eliminate the remaining loop and get the following code.

```python
1 def looped_power_vectorized_v2 (x):
2 X = np. zeros ((x. shape [0] , x. shape [0]) )
3 is_ = np. arange (x. shape [0])
4 is_e = np. expand_dims (is_ , 1)
5 js_ = np . arange ( x . shape [0])
```

```python
6 x_e = np. expand_dims (x[is_], 1)
7 X [ is_e , js_ ] = np . pow( x_e , js_ )
8 return X
```

As before, we want to replace all occurrences of i with an array, which adds a dimension to expressions that use i. Then x[i] becomes x[is\_], changing its shape from () to (�, ). However, a simple replacement will put the newly expanded dimension on the same axis as the one created when we vectorize the previous loop. To properly broadcast the operands, we shift the newly created dimension to the left by expanding a new dimension on the right-most axis, changing its shape from (�, ) to (�, 1). Now, broadcasting can perform elementwise computation for operands with shapes (�, 1) and (�, ), and the left-hand side is handled similarly for selections from X. The resulting program is fully vectorized and can be simplified to looped\_power\_vectorized by postprocessing.

This example suggests a general insight: when vectorizing the innermost loop, variables from the outer scope can be treated as free variables with fixed shapes. Replacing the loop variable with an array adds an extra axis to the expressions that originally contained it. By properly arranging array axes, this new axis can propagate through the program and vectorize computations previously performed by loops.

Branches. Consider this function having a branch, which makes the technique discussed so far not directly applicable.

```python
1 def half_burn ( base ) :
2 # base : shape (M ,) .
3 out = np. zeros (( base . shape [0] ,))
4 for i in range ( base . shape [0]) :
5 if i < 3:
6 out[i] = base [i] / 2
7 return out
```

Specifically, we cannot directly replace the variable i in the branch with np.arange, as the replacement would include values that are not taken in some iterations. We must exclude such values while maintaining the dimensional structure, so the technique discussed above can still be applicable.

Towards this goal, we leverage the masked array in NumPy. Each element in a masked array has an associated boolean mask indicating its validity, and invalid ones are omitted from computations. However, masked arrays have limited out-of-the-box usability and some of their behaviors are not clearly specified or documented. To address these issues, our DSL has more specific semantics for masked arrays and extends NumPy with masked update statements. Unlike normal updates with advanced indexing, whose left-hand side is a variable indexed by arrays, masked updates also accept boolean mask indices. An element is updated only when it is unmasked, the corresponding mask index is true, and the corresponding right-hand-side value is unmasked.

When rewriting code in a branch, we replace variables restricted by the branch with masked arrays. Such variables are identified by a dataflow analysis. As NumPy’s documentation does not explicitly specify the behavior of using masked arrays as indices, our DSL also disallows it. During the rewrite, if an indexing array becomes masked, we fill it with 0 and apply the mask of the indexing array to the indexing result.

![](images/7025067bc86b20df151e4f87f622bc33fbc69f364498c89eec735b8b22145b62.jpg)  
Figure 3. Graphical illustration of masked updates. ⊗ represents masked entries.

This idea is explained graphically in Figure 3, and the corresponding code is shown in the half\_burn\_vectorized function. Here, make\_masked is an operator in the DSL that takes a data array and a mask array to build a masked array. When used on the left-hand side, it turns a statement into a masked update, with its second argument being a mask index. In the original branch, i is restricted by the branch condition, so the vectorized index is\_ must be masked by m. Because our DSL does not allow masked index, we fill it with 0 to unmask it and mask the indexing result instead.

```python
1 # Proof of concept ; not runnable Python code .
2 def half_burn_vectorized ( base ) :
3 out = np . zeros (( base . shape [0] ,) )
4 is_ = np . arange ( base . shape [0])
5 m = is_ < 3
6 idx = np . ma . filled ( make_masked ( is_ , m ) , 0)
7 half = make_masked ( base [ idx ] , m ) / 2
8 make_masked ( out[ idx ], m) = half
9 return out
```

To summarize, we formalize our approach as a set of rewrite rules conditioned on the shape and maskedness of the targets. The rules operate on our DSL, which captures many structures found in practical NumPy programs. After rewriting a program, we apply postprocessing to simplify the resulting code and emit a runnable Python program.

## 3 Preliminaries

We first introduce some concepts used in the paper. NumPy organizes data in multidimensional arrays. We consider two aspects from which arrays are typed: shape and maskedness.

## 3.1 Shapes and Broadcasting

The shape of an array is a tuple of numbers recording the size of the array along each dimension. For example, the shape of np.array([[1,2,3],[4,5,6]]) is (2, 3). We refer to positions in a shape as axes and to the corresponding sizes as dimensionalities. Scalar values have the shape ().

Some NumPy operators accept operands whose shapes are broadcasting-compatible. Broadcasting an array to a target shape can be understood as duplicating the array along axes of size 1 and any missing leading axes. Thus, it allows us to reuse elements in an array to match the required shape in computations. As shown below, binary operators implicitly broadcast operands to the same shape for computation.

```python
1 t = np . array ([[1] , [0]]) # shape (2 , 1)
2 s = np . array ([2 , 0]) # shape (2 ,)
3 r = s + t # [[3 , 1] , [2 , 0]] , shape (2 , 2)
```

Here, t and s are broadcast to [[1, 1], [0, 0]] and [[2, 0], [2, 0]]. The addition is element-wise on the two arrays.

## 3.2 Advanced Indexing

NumPy allows arrays to be indexed by integer arrays, which is a feature known as advanced indexing. We refer to the indexing arrays as indexers, and to the indexed array as the base, or indexee. When several arrays are used together as indexers, their shapes must be broadcasting-compatible. NumPy first broadcasts these indexers to a common shape and then uses them to make element-wise selections from the indexee. Therefore, the result has the common broadcast shape ofthe indexers. In the example below, i2 has shape (2, ) and when used as an indexer, it is broadcast to the shape ofi1, which is (2, 2). Replicated on the implicitly added left-most axis, the efective indexing array is [[0, 2], [0, 2]].

```python
1 i1 = np . array ([[1 , 1] , [1 , 0]]) # shape : (2 , 2)
2 i2 = np . array ([0 , 2]) # shape : (2 ,)
3 w = z[i1 , i2] #[[z[1 ,0] ,z[1 ,2]] ,[z[1 ,0] ,z [0 ,2]]]
```

## 3.3 Masked Arrays

The ma submodule of NumPy provides the MaskedArray class. In addition to the array data, MaskedArray also has an associated boolean array of the same shape called mask. An element whose corresponding mask entry is true is considered masked and is omitted from computations. The reduce operators also skip masked elements. If every element being reduced is masked, the reduction result is masked as well. np.ma.filled can convert a masked array to a normal array by replacing masked entries with a specified value.

1 o = np.ma. MaskedArray ([1 ,2] ,[ False , True ]) #[1,⊗]   
2 n = o + np. array ([100 , 101]) # [101 , ⊗]   
3 m = np.ma. filled (n, 99) # [101 , 99]   
4 l = np. mean (n, axis =0) # 101   
5 k = np . mean ( np . ma . MaskedArray (n ,[ True , False ]) ) #⊗

## 4 Core Language

This section presents the DSL used by our vectorization procedure, which captures core features of NumPy.

## 4.1 Syntax and Semantics

Figure 4 shows the syntax of our DSL, which is high-level and imperative. Its formal semantics is shown in Appendix A.

Here, a program is a function consisting of statements, which are built from expressions. The variable binding statement � := � binds the expression � to the name �. The normal update statement $x [ \overline { { e _ { 1 } } } ]  e _ { 2 }$ modifies elements of � selected by the array indices $\overline { { e _ { 1 } } }$ to the value of $e _ { 2 } { \mathrm { . } }$ , following NumPy’s semantics of updates with advanced indexing. We require the right-hand side of such updates to be unmasked. The masked update statement make\_masked $( x [ \overline { { e _ { 1 } } } ] , e _ { 2 } )  e _ { 3 }$ uses the mask index $e _ { 2 } .$ . The left-hand side may use nested calls to make\_masked to incorporate multiple mask indices. As discussed in Section 2, a selected element is updated only when it is unmasked, the corresponding element in the conjunction of the mask indices is true, and the corresponding element on the right-hand side is unmasked.

Like many imperative languages, our DSL also has the standard skip statement and sequential composition �<sub>1</sub>; �<sub>2</sub>. Branches like if � then $s _ { 1 }$ else $s _ { 2 }$ expect an unmasked scalar as the branch condition �. Since we do not consider specific data types in this DSL, zeros are considered false and non-zero values are considered true. Loops in the form of for � in 0 . . .i do � must have an integer literal or a shape access expression as their bound i, and the loop variable � ranges from zero to the loop bound minus one. To make control flows easy to analyze, each variable name can be bound by at most one statement, and names bound inside a branch or loop may not be used outside that branch or loop.

The semantics of most expressions have a straightforward correspondence to their counterparts in NumPy. The shape access expression S(�) [�] returns the size of axis � of �. The array indexing operator, $e [ e _ { 1 } , ~ . . . , e _ { n } ]$ , is equivalent to integer array indexing in NumPy, but $e _ { 1 } , \ldots , e _ { n }$ cannot be masked arrays. We also require � to equal the number of axes of �. The expression make\_maske $\mathbb { I } ( e _ { 1 } , e _ { 2 } )$ constructs a masked array from the data array $e _ { 1 }$ and the mask array $e _ { 2 } .$ . Unlike the constructor of MaskedArray in NumPy, make\_masked invalidates entries whose masks are falsy. When used on the right-hand side, it broadcasts operands as any other binary operator does. When used on the left-hand side of a masked update statement, only the mask array may be broadcast to the shape of the data array. The replicate operator extends expand\_dims by additionally accepting a sequence of integral expressions, which specifies the sizes of the inserted axes. Operators are assumed to return arrays that do not overlap with their operands in memory.

## 4.2 Typing Shapes and Maskedness

As shown by the formal semantics in Appendix A, expressions evaluate to both values and runtime maskedness. Given type annotations for the input arrays, we can statically infer the type of every expression in a well-typed program. These annotations specify the maskedness and dimensionalities of each argument, using integers for axes with fixed length and symbolic variables for axes with dynamic length.

Due to space limit, the complete set of type inference rules for expressions is provided in Appendix B. Here, we present a subset in Figure 5. The rules for type analysis for statements are in Figure 6. In our formalization, the type environment Γ is a map from variables to pairs of static shapes and maskedness. Judgments of the forms $\Gamma \vdash e : _ { \varsigma } \varsigma$ and $\Gamma \vdash e : _ { m }$ � denote that � is inferred to have static shape � and maskedness �. We use ⊤ to denote masked arrays and use ⊥ to denote unmasked arrays. For example, the (S-Idx) rule infers the shape of an indexing expression by first inferring the shapes of the indexers and then broadcasting them to a common shape. The (M-Idx) rule infers the maskedness of an indexing expression to be the maskedness of the indexee, provided that none of the indexers is masked. In Figure 6, judgments of the form $\Gamma \vdash s \hookrightarrow \Gamma ^ { \prime }$ mean that the statement � is well-typed under the environment Γ, and analyzing � updates Γ to Γ<sup>′</sup>. For example, the (T-Bind) rule states that analyzing a variable binding statement updates Γ to map the variable to the inferred type of the right-hand side, given that the variable has not been bound before. The following theorem formalizes the soundness property of our type system.

Program � ::= f(�) � return <sub>�</sub>   
Statement � ::= skip | � := � | � | if � then � else �   
| for � in 0 . . .� do � | �; �   
Integral � ::= S(�) [�] | �   
Expr � ::= $x \mid I \mid E [ \overline { { { E } } } ] \mid f ( \overline { { { E } } } ) \mid g ( E , c ) \mid { \mathsf { f i l l e d } } ( E , E )$   
| ones((�)) | arange(�) | matmul(�, �)   
| expand\_dims(�, (�))   
| replicate(�, (�), (�))   
Masked LHS �<sup>ˆ</sup> ::= make\_masked( ( �[�] | �<sup>ˆ</sup> ), �)   
Update � ::= ( � [�] | �<sup>ˆ</sup> ) ← ( � )   
�, � ∈ Variables � ∈ Integer literals $f \in o p g \in$ <sub>��</sub>�<sub>���</sub>   
Unary ��: abs, exp, log, negative, logical\_not . . .   
Binary ��: +, −, ∗, /, pow, maximum, minimum, logical\_or,   
logical\_and, ==, <, >, make\_masked . . .   
������: sum, prod, max, min, all, any, mean . . .

Figure 4. DSL syntax. Meta symbols and code constructs are in pink and blue. Bars denote comma-separated sequences.  
Var � 0 ≤ � ≤ � Γ ⊢ �<sub>�</sub> : �<sub>�</sub>   
Γ (� ) = (<sub>�</sub>, �) Broadcast(�<sub>0</sub>, . . . , �<sub>�</sub> ) = �   
(S-Var) (S-Idx)   
Γ ⊢ � :<sub>�</sub> <sub>�</sub> Γ ⊢ � [�<sub>0</sub>, . . . , �<sub>�</sub> ] :<sub>� �</sub>   
1 ≤ � ≤ � Γ ⊢ �<sub>�</sub> : �<sub>�</sub>   
Broadcast(�<sub>1</sub>, . . . , �<sub>�</sub> ) = � <sup>�</sup> <sup>∈</sup> <sup>Unary</sup> <sup>��</sup> <sup>∪</sup> <sup>Binary</sup> <sup>��</sup>Γ ⊢ �(� , . . . , � ) : <sub>�</sub> <sup>Γ (�</sup> <sup>)</sup> <sup>=</sup> <sup>(�, �)</sup>Γ ⊢ � : � Var � (M-Var)   
(S-Op)   
� ∈ Unary �� ∪ Binary ��   
� = make\_masked ∨ (∃�.1 ≤ � ≤ � ∧ Γ ⊢ �� : ⊤)   
(M-Op1)   
Γ ⊢ �(�<sub>1</sub>, . . . , �<sub>�</sub> ) :<sub>�</sub> ⊤   
� ∈ Unary �� ∪ Binary �� Γ ⊢ � :<sub>�</sub> �   
� ≠ make\_masked 0 ≤ � ≤ �   
Γ ⊢ �� :<sub>�</sub> ⊥ 1 ≤ � ≤ � Γ ⊢ �<sub>�</sub> :<sub>�</sub> ⊥   
(M-Op2) (M-Idx)   
Γ ⊢ � (�<sub>1</sub>, . . . , �<sub>�</sub> ) :<sub>�</sub> ⊥ Γ ⊢ � [�<sub>0</sub>, . . . , �<sub>�</sub> ] :<sub>�</sub> �  
Figure 5. Sample type inference rules for expressions.

Γ ⊢ � :<sub>� �</sub>   
(T-Skp) Γ ⊢ � : � � ∉ dom(Γ)   
Γ ⊢ skip ↩→ Γ (T-Bind)   
Γ ⊢ � := � ↩→ Γ[� ↦→ (<sub>�</sub>,�) ]   
Update � (T-Upd) <sup>Γ</sup> <sup>⊢</sup> <sup>�</sup> <sup>:�</sup> <sup>(</sup> <sup>)</sup> <sup>Γ</sup> <sup>⊢</sup> <sup>�</sup> <sup>:�</sup> <sup>⊥</sup><sub>Γ ⊢ → Γ′</sub> <sub>Γ′ ⊢ → Γ′′</sub>   
(T-If)   
Γ ⊢ if � then � else � ↩→ Γ<sup>′′</sup>   
Γ ⊢ � ↩→ Γ<sup>′</sup>   
Γ [� ↦→ ( ( ), ⊥) ] ⊢ � ↩→ Γ<sup>′</sup>   
(T-For) Γ<sup>′</sup> ⊢ � ↩→ Γ<sup>′′</sup>   
Γ ⊢ for � in 0 . . .� do � ↩→ Γ<sup>′</sup> (T-Seq)   
Γ ⊢ � ; � ↩→ Γ<sup>′′</sup>  
Figure 6. Rules for type analysis for statements.

Algorithm 1 Top-level algorithm for vectorizing programs.   
1: procedure Vectorize(P, A)   
Input: A program $\mathcal { P }$ and the input type annotation A.   
Output: The vectorized program.   
2: while HasVectorizableLoops $( \mathcal { P } )$ do   
3: $\Gamma \gets \mathrm { A N A L Y Z E S H A P E } ( \mathcal { P } , \mathcal { R } )$   
4: L ← InnerMostLoop(P)   
5: $\mathcal { L } ^ { \prime } , \mathcal { \longleftarrow } \mathrm { R E W R I T E } ( \mathcal { L } , \Gamma , \Gamma , \emptyset )$   
6: $\mathcal { P }  \mathcal { P } [ \mathcal { L } ^ { \prime } / \mathcal { L } ]$   
7: return $\mathcal { P }$

Theorem 4.1. Let P be a function with body �, A and Σ be the type annotations and an evaluation environment for the arguments. If every variable that can be evaluated under Σ is well-t<sub>y</sub>ped under A, P can be executed under Σ, and ${ \mathcal { A } } \vdash s \hookrightarrow \Gamma ,$ then the value returned is well-t<sub>y</sub>ped under Γ.

Proof. The proof is available in Appendix C.

## 5 Vectorizing with Type-Directed Rewrite 5.1 Problem Statement

We first formally state the vectorization problem: Given a program $\mathcal { P }$ and its input type annotations, our goal is to find a loop-free program $\mathcal { P } ^ { \prime }$ that is observationally equivalent to P. That is, for any evaluation environment Σ under which P returns a value � with maskedness $\mu , \mathcal { P ^ { \prime } }$ under Σ returns a value $v ^ { \prime }$ with maskedness $\mu ^ { \prime }$ such that $\boldsymbol { v } = \boldsymbol { v } ^ { \prime }$ and $\mu = \mu ^ { \prime } .$

Some programs in the DSL cannot be vectorized, such as the ones with strongly coupled loop-carried dependence. Consequently, our rewrite approach is not complete for the full set of programs in the DSL. We will circle back to define rewritable programs more precisely in Section 5.6.

## 5.2 Top-Level Algorithm

The top-level algorithm for vectorization is shown in Algorithm 1. Given a program $\mathcal { P }$ with input type annotations A, Vectorize works inside out, repeatedly rewriting the innermost loop o $\cdot \mathcal { P }$ (Lines 2–6). In each iteration, AnalyzeShape infers the variables’ types (Line 3), as described in Section 4.2. Vectorize then selects the innermost loop (Line 4), invokes Rewrite to rewrite the loop, guided by the inferred types (Line 5), and substitutes the rewritten statements back (Line 6). The procedure repeats until no vectorizable loops remain, and returns the resulting program.

Δ<sup>′</sup> = {� ↦→ arange(i) } Γ<sub>�</sub>, Γ , Δ<sup>′</sup> ⊢ � ↷ �<sup>′</sup>, Γ<sup>′</sup>   
(R-For)   
$\Gamma _ { b } , \Gamma _ { a } , \Delta \vdash \mathsf { f o r } x \mathrm { i n } \emptyset \ldots \mathrm { i d o } s \ \curvearrow \ s ^ { \prime } , \Gamma _ { a } ^ { \prime }$   
Γ<sub>�</sub>, Γ<sub>�</sub>, Δ ⊢ �<sub>1</sub> ↷ �<sup>′</sup><sub>1</sub>, Γ<sup>′</sup><sub>�</sub> Γ<sub>�</sub>, Γ<sup>′</sup><sub>�</sub>, Δ ⊢ �<sub>2</sub> ↷ �<sup>′</sup><sub>2</sub>, Γ<sup>′′</sup><sub>�</sub>   
Γ<sub>�</sub> , Γ<sub>�</sub>, Δ ⊢ �<sub>1</sub> ; �<sub>2</sub> ↷ �<sup>′</sup><sub>1</sub> ; �<sup>′</sup><sub>2</sub>, Γ<sup>′′</sup><sub>�</sub> (R-Seq)   
$\Gamma _ { b } \vdash e : _ { \zeta } \ \varsigma \quad \Gamma _ { a } \vdash e ^ { \prime } : _ { \zeta } \ \varsigma ^ { \prime } \quad \Gamma _ { a } \vdash e ^ { \prime } : _ { m } m ^ { \prime }$   
Γ<sub>�</sub>, Γ<sub>�</sub>, Δ ⊢ � { �<sup>′</sup> <sub>�</sub> ≠ <sub>�</sub><sup>′</sup> Γ<sup>′</sup> = Γ<sub>�</sub> [� ↦→ (<sub>�</sub><sup>′</sup>, �<sup>′</sup> ) ]   
(R-Bind1)   
Γ<sub>�</sub> , Γ<sub>�</sub>, Δ ⊢ � := � ↷ � := �<sup>′</sup>, Γ<sup>′</sup><sub>�</sub>   
$\begin{array} { r } { \Gamma _ { b } , \Gamma _ { a } , \Delta \vdash e _ { c } \iff e _ { c } ^ { \prime } \quad \Gamma _ { b } \vdash e _ { c } : _ { \mathcal { G } } \varsigma _ { c } \quad \Gamma _ { a } \vdash e _ { c } ^ { \prime } : _ { \mathcal { G } } \varsigma _ { c } ^ { \prime } \quad \varsigma _ { c } ^ { \prime } \ne \varsigma _ { c } } \end{array}$   
LoopVarSub(Δ) = �<sub>�</sub> LoopVar(Δ) = <sub>�</sub> �<sup>′′</sup><sub>�</sub> = logical\_not(�<sup>′</sup><sub>�</sub> )   
$\Gamma _ { b } , \Gamma _ { a } , \Delta [ y \mapsto \mathsf { m a k e \_ m a s k e d } ( e _ { s } , e _ { c } ^ { \prime } ) ] \vdash s _ { t } \sim s _ { t } ^ { \prime } , \Gamma _ { a } ^ { \prime }$   
Γ<sub>�</sub>, Γ<sup>′</sup>, Δ[<sub>�</sub> ↦→ make\_masked( $e _ { s } , e _ { c } ^ { \prime \prime } ) ] \vdash s _ { e } \curvearrowright s _ { e } ^ { \prime } , \Gamma _ { a } ^ { \prime \prime }$   
(R-Brch1)   
$\Gamma _ { b } , \Gamma _ { a } , \Delta \vdash \mathrm { i f } \ e _ { c }$ then �<sub>�</sub> else �<sub>�</sub> ↷ �<sup>′</sup><sub>�</sub>; �<sup>′</sup><sub>�</sub>, Γ<sup>′′</sup><sub>�</sub>  
Figure 7. Sample statement rewrite rules.

## 5.3 Type-Directed Rewrite

Rewriting statements. The Rewrite procedure rewrites statements. The complete set of rewrite rules is in Appendix D and a subset of rules for statements is shown in Figure 7. We use judgments of the form $\Gamma _ { b } , \Gamma _ { a } , \Delta \vdash s \ \land ^ { \prime } , \Gamma _ { a } ^ { \prime }$ to represent that rewriting the statement � under environments $\Gamma _ { b } , \Gamma _ { a }$ and Δ produces a new statement $s ^ { \prime }$ and an updated type environment $\Gamma _ { a } ^ { \prime } . \Gamma _ { b }$ is the type environment before the current invocation of Rewrite, and $\Gamma _ { a }$ is the updated environment. Δ maps the current loop variable to its substitute. LoopVar(Δ) and LoopVarSub(Δ) return the current loop variable and the current loop variable substitute stored in Δ.

The key idea is to eliminate a loop by replacing its loop variable with an array containing all the values taken by that variable, and propagating the newly added axis through the loop body. In the rest of the paper, we say that an expression is lifted when it is rewritten to an expression that evaluates to an array of values produced by the original one across loop iterations. We can take advantage of broadcasting to avoid unnecessary replication. Rule (R-For) in Figure 7 initializes the process: before rewriting the loop body, it records in $\Delta$ that the loop variable should be replaced by arange(i) where i is the loop bound. Rule (R-Seq) rewrites sequences compositionally, using the type environment produced by the first rewrite in the second. Bind and update statements are rewritten mainly by rewriting their expressions. In particular, Rule (R-Bind1) handles the case where the right-hand side is lifted, and it updates the type environment accordingly.

Rewriting expressions. Statement-level rewrites involve rewriting expressions. Figure 8 shows some expression-level rules. Judgments of the form $\Gamma _ { b } , \Gamma _ { a } , \Delta \vdash e \ \sim \ e ^ { \prime }$ mean that the expression � is rewritten to $e ^ { \prime }$ under environments $\Gamma _ { b } , \Gamma _ { a }$ and Δ. They use the same environments as the statement level rewrites, but only return the new expression.

As shown by rule (R-LVar) in Figure 8, every occurrence of the loop variable is replaced by its substitute stored in Δ. The substitute is a 1-D array, so each expression involving the loop variable gains a new dimension. The remaining expression rules propagate this new axis recursively. Unary opera tors do not need to change as they are applied element-wise to their sole argument. The ������ operators, expand\_dims, replicate, and the shape access operator take an array and the specified axes as arguments. As shown in the (R-Shape) and (R-Exp) rules, if the array argument is lifted, we shift the target axes one position to the right to ensure the operators are still applied to the same axes as before.

For element-wise operators, rewrites are guided by shapes. The (R-Biop) rule shows how to rewrite binary operations. The shapes before the rewrite indicate whether broadcasting occurred originally. If a lifted operand was broadcast to have additional axes before the rewrite, we insert axes of size 1 after the newly added axis. The number of inserted axes equals the number of axes added by the broadcast. Doing so keeps the positions of the axes added by the broadcast while keeping the newly added axis as the left-most one.

The (R-Indx) rule rewrites indexing expressions. As indexers are broadcast to a common shape, the rule first rewrites each indexer and inserts new axes when needed, similar to rewriting binary operations. It then rewrites the base when the expression is not on the left-hand side of an update. If the base is lifted, we add an indexer for the new left-most axis. This indexer is the loop variable substitute stored in $\Delta ,$ which is an array containing numbers of iterations where the expression is evaluated. The new indexer needs to be expanded to ensure proper broadcasting and propagate the new axis. Finally, the rule applies UnmaskIndexer (discussed in Section 5.4), to ensure that there is no masked indexer.

Example 5.1. Consider the code below.

```prolog
1 # a: shape (L, M, N), not masked
2 for x in 0... S(a) [0] do
3 ys := arange(S(a) [1]) ; # (M ,)
4 t := a[x, ys , 0]; # (M ,)
5 ns := arange(S(t) [0]) ; #(M ,)
6 a[x, ys , 0] ← t[ns] + x;
```

To rewrite the for-loop, we set arange $\mathrm { ( } \mathsf { S } \mathrm { ( } \mathsf { a ) } [ 0 ] \mathrm { ) }$ as the loop variable substitute in Δ. The first statement in the loop is unchanged by the rewrite. The next statement defines t using an expression involving x. For this statement, we replace x with the loop variable substitute. Since ys has shape (�, ), the indexers were broadcast originally. To preserve the broadcast axis, we expand a new right-most axis for the first indexer. As a is not lifted, no new indexer is needed. The returned type environment shows that t is lifted to the new shape (�, �). Thus, we increment the axis argument in the shape access expression in the next statement. For the last update statement, we rewrite its left-hand side in the same way as before. On the right-hand side, the base t is lifted, so we add an indexer for its new axis. Because ns was 1-D before, the new indexer is expanded for proper broadcasting.

```javascript
4 a [ i ] ← b [ i ];
```

1 # a, b: shape (M ,) , not masked   
2 for i in 0... S(a)[0] do   
3 if i > 2 then  
Figure 8. Sample rewrite rules for expressions. OnLHS(�) checks if � is on the left-hand side of an update.

$$
\begin{array} { r l r l r l } & { \quad \mathbf { V a r s } \quad } & { \quad \bot \mathrm { c o n y u t ~ d i m } ( \boldsymbol { c } _ { i } , \ \mathbf { c } _ { i } , \mathbf { s } _ { i } ) ( \boldsymbol { \Delta } ) = \boldsymbol { c } _ { s } } & { \quad \textrm { F i r b e } ^ { c } : \boldsymbol { s } \cdot \boldsymbol { \varsigma } _ { i } , } & { \quad \textrm { F i r b e } ^ { c } : \boldsymbol { s } \cdot \boldsymbol { \varsigma } _ { i } , } & { c _ { i } ^ { c } = \beta ; } \\ & { \quad \quad \quad \frac { \mathrm { V a r s } ~ \mathrm { V a r s } ~ \boldsymbol { \alpha } ( \Delta ) = \boldsymbol { x } } { \mathrm { { I b } } _ { i } , \mathrm { { I G } } _ { i } , \Delta \boldsymbol { \kappa } \cdot \boldsymbol { \kappa } _ { i } } } & { \quad \frac { \mathrm { I } _ { i } ^ { c } , \mathrm { I } \kappa _ { i } \Delta \cdot \boldsymbol { \epsilon } _ { f } - \boldsymbol { s } \cdot \boldsymbol { \omega } _ { e } ^ { c } } { \mathrm { I { I G } } _ { i } + \boldsymbol { c } _ { i } ^ { c } , \cdots \cdot \boldsymbol { s } _ { i } ^ { c } } } & { \quad \frac { \mathrm { I } _ { i } ^ { c } \times \boldsymbol { \epsilon } _ { f } - \boldsymbol { s } ^ { c } } { \mathrm { I { I b } } _ { i } , \mathrm { ~ I } _ { \alpha } + \boldsymbol { c } _ { i } ^ { c } , \cdots \cdot \boldsymbol { s } _ { i } ^ { c } } } & { \quad \textrm { I } _ { i } ^ { c } = \epsilon ^ { - \epsilon ^ { \prime } } ; \boldsymbol { s } _ { i } ^ { c } , \cdots \quad \epsilon ^ { c } _ { i } ^ { c } } & { \quad } \\ &  \quad \quad \quad \quad \frac { \mathrm { I } _ { i } ^ { c } = \mathrm { I } _ { i } ^ { c } \times \boldsymbol { \kappa } _ { i } \boldsymbol { \kappa } _ { i } \boldsymbol { \kappa } _ { i } \boldsymbol { \kappa } _ { i } \boldsymbol { \kappa } _ { i } \boldsymbol { \kappa } _ { i } \boldsymbol { \kappa } _ { i } } &   \end{array}
$$

Here, x is also replaced by the loop variable substitute and then expanded. The final result is shown below.

```lisp
1 xs := arange( S ( a ) [0]) ;
2 xse := expand_dims( xs , (1 ,) ) ;
3 ys := arange( S ( a ) [1]) ;
4 t := a [ xse , ys , 0]; # (L , M )
5 ns := arange( S ( t ) [1]) ;
6 a[xse , ys , 0] ← t[xse , ns] + xse;
```

## 5.4 Rewriting Branches

Recall from Section 2 that rewriting branches requires excluding values from invalid iterations. For such rewrites, we first rewrite the condition. If it is not lifted, we rewrite the two branch bodies and keep the branch structure. If lifted, we flatten the branches by composing the rewritten branch bodies sequentially, following the (R-Brch1) rule in Figure 7.

Before rewriting each branch body, the rule updates the loop variable substitute stored in Δ by masking the original substitute with the lifted condition using a make\_masked call. The substitute then has all the entries corresponding to invalid iterations masked. In other words, the then-branch receives the iteration numbers for which the condition is true, with all others masked, and the else-branch uses the negated condition. Nested branches work in the same way, because the efective mask of nested make\_masked calls is the conjunction of the masks from each call. The rewrite also masks loop-local variables defined outside the flattened branch, ensuring invalid values not leaked into the branch.

Because the DSL disallows masked indexers, we use UnmaskIndexer to check if any indexer is masked under the updated type environment, and if so, wrap them in filled with 0 as the fill value. We also wrap the entire indexing expression in make\_masked, masking it with the conjunction of the indexers’ masks. Such rewrites ensure that the entries selected by the supposedly masked indices remain masked in the result. Normal updates may be turned into masked updates, if the left-hand side gets masked after such rewrites.

Example 5.2. Consider the code below as an example.

The code block below shows the rewrite result.

1 is\_ := arange(S(a) [0]) ;   
2 mis\_t := make\_masked( is\_ , is\_ > 2) ;   
3 bis\_ := make\_masked(b[filled(mis\_t ,0)], is\_ > 2);   
4 make\_masked( a [filled( mis\_t ,0) ] , is\_ > 2) ← bis\_ ;

Here, replacing i with is\_ lifts the branch condition to the array is\_ > 2. Before rewriting the then-branch, the rewrite records mis\_t as the loop variable substitute, masking out iterations where the condition is false. All occurrences of i in the then-branch are replaced by this masked substitute. Because masked arrays cannot be indexers, the rewrite fills the masked indices with 0 and moves the same mask to the indexing result. The last statement is a masked update, so it updates exactly the elements of a corresponding to iterations that execute the then-branch. Were there an else-branch, it would be handled analogously with the negated condition.

## 5.5 Dataflow Analysis for the Depends-On Relation

There are two code patterns that need special handling. The first is false dependence [43]: a loop-local variable defined by an expression not lifted is updated by a lifted expression. In this case, the defining expression must be lifted explicitly to fit the updates. The second pattern is branch-restrained variables defined outside ofthe loop. If the flattened branch puts conditions on other loop variables, only masking looplocal variables may incorrectly remove those restrictions. To address these issues, we first give the following definition.

Definition 5.3 (Depends-on). A non-variable expression � depends on a loop variable � if evaluating � directly involves a variable depending on �. A variable � defined inside the loop of � depends on � if its defining or updating expressions depends on �, or if � is updated inside a branch whose condition depends on �. Loop variables are self-dependent.

Before rewriting, a dataflow analysis maps each variable to a set of loop variables on which it depends. The analysis result allows us to identify variables incurring false dependence and use replicate to explicitly expand them. It also allows us to find variables escaping the restrictions when flattening branches and thus mask them.

Example 5.4. This example shows both uses of the analysis.

```matlab
1 # a: shape (M, N), not masked ; assume M > N
2 for i in 0... S(a)[0] do
3 for j in 0... S ( a ) [1] do
4 b := ones((1 ,) ) ;
5 if i < j then
6 b [0] ← a[j, i];
```

After rewriting the inner loop, we obtain the code below.

```prolog
1 for i in 0... S ( a ) [0] do
2 js := arange( S ( a ) [1]) ;
3 b := replicate(ones((1 ,) ) ,(0 ,) ,( S ( a ) [1] ,) ) ;
4 m := i < js ; # shape (N ,)
5 ir := make_masked(i, m); # shape (N ,)
6 jsm := make_masked( js , m ) ;
7 jsmf := filled( jsm , 0) ;
8 irf := filled( ir ,0) ;
9 make_masked(b[jsmf , 0],m) ← \
10 make_masked( a [ jsmf , irf ] , logical_and(m , m ) ) ;
```

The dataflow analysis reports that b depends on j. Therefore, the rewrite wraps its definition in replicate, creating a copy for each j. Since b is now lifted, indexing into b needs a new indexer. The analysis also shows that i is restricted by the branch. When the branch is flattened, the i in a[j, i] is masked by i < js. The masked indexers are then filled as before, leaving jsmf and irf as the final indexers. As the potentially out-of-bound values are masked before indexing into a, the rewritten expression remains well-defined.

## 5.6 Vectorizability

In general, our rewrite rules cannot handle unvectorizable loops. Loops’ vectorizability is determined by the presence of loop-carried dependence [8]. As discussed in prior works [10, 25], if two statements access the same memory location, and at least one of them writes, the later one is said to be dependent on the earlier one. In most cases, cycles or backward dependence in dependence graphs prevent direct vectorization [25, 34]. Thus, our technique cannot vectorize programs with loop-carried dependence, except for one special case.

Since loops are commonly used to implement reduction operators such as sum, we add a rule that specifically rewrite such statements. First, we introduce the following definition.

Definition 5.5 (Rewritable reduce statements). An update statement in a loop is a rewritable reduce statement if it meets the following conditions: (1) the top-level operator on the right-hand side is one of +, ∗, maximum, or minimum; (2) the first operand of the operator is the same as the update target; (3) the updated variable is not defined in the same loop; and (4) the left-hand side depends neither on the loop variable of the enclosing loop nor any restricted loop variable.

For loops with one such statement, we can replace it with the corresponding reduction operator. If the reduced expression is not lifted, we explicitly replicate it. If the statement is in a branch, the reduction is done selectively via masked arrays. To avoid reductions on masked arrays returning a masked value, we fill the reduced array with the reduction’s identity value.

Example 5.6. Consider the code snippet below. The first loop adds up a sequence of 10s, which can be replaced by a call to sum. The second finds the smallest j such that j ≥ a[0], which is a min reduction on a subset of the values taken by j. Both loops contain rewritable reduce statements.

1 # a: shape (M ,) , not masked ; assume M < 100.   
2 acc := ones((2 ,) ) \* 100;   
3 for i in 0... S ( a ) [0] do   
4 acc [0] ← acc [0] + 10;   
5 for j in 0... S(a) [0] do   
6 if j >= a [0] then   
7 acc [1] ← minimum( acc [1] , j ) ;

After the write we get this code:

```lisp
1 acc := ones((2 ,)) * 100;
2 rep_10 := replicate(10 , (0 ,) , ( S ( a ) [0] ,) ) ;
3 acc [0] ← acc [0] + sum( rep_10 , 0) ;
4 js := arange(S(a) [0]) ;
5 jsm := make_masked(js , js >= a [0]) ;
6 jsmf := filled(jsm , inf);
7 acc [1] ← minimum( acc [1] ,min( jsmf , 0) ) ;
```

In the first loop, because 10 is not lifted, we replicate it explicitly for reduction. Adding the reduction expression to the accumulator is equivalent to performing the additions iteratively. In the second loop, the reduced array is jsm, which is lifted from j and no replication is needed. If a[0] is greater than every value of j, jsm is fully masked, and reducing it would result in a masked value. To avoid this, we fill jsm with inf, which denotes infinity, the identity value for min.

Next, we define rewritability more precisely in the context of our DSL and state the main correctness theorem.

Definition 5.7 (Rewritability). A loop is rewritable if (1) it has no loop-carried dependence, or (2) it contains only one rewritable reduce statement and the only loop-carried dependence is the rewritable statement on itself. A program is rewritable if all of its loops are rewritable.

Theorem 5.8 (Correctness of the Vectorize Routine). Let P be a rewritable <sub>p</sub>ro<sub>g</sub>ram, and A be the t<sub>yp</sub>e annotation for its ar<sub>g</sub>uments. If Vectorize(P, A) returns P<sup>′</sup>, then, for any evaluation environment Σ under which the execution ofP returns a value, executin<sub>g</sub> P<sup>′</sup> under Σ returns the same value.

Proof. The proof is provided in Appendix E.

## 6 Implementation

We have implemented the proposed technique in a tool called Vectorizer based on standard NumPy and the ast module of Python, making Vectorizer lightweight and easy to run. Furthermore, many constructs in practical NumPy programs map directly to our DSL, which allows Vectorizer to process source programs with only minor adaptations, such as renaming variables to satisfy the requirements.

Postprocessing the rewrite results. Code generated by the rewrite rules may incur unnecessary runtime overhead or contain masked updates that cannot be expressed directly in Python because function calls are not l-values. To ad dress these issues, Vectorizer applies a pipeline of postprocessing passes that transforms rewritten programs to executable Python while eliminating common ineficiencies. For example, the pipeline simplifies boolean mask manipulations, replaces masked updates with np.where and advanced indexing with slices and transposes, collapses sum after-multiplications into np.tensordot calls, and removes unnecessary materializations, copies, calls to filled, and masked array constructions. It also performs standard opti mizations such as common-subexpression elimination (CSE) and unused-variable elimination. Because the main rewrite procedure emits mostly straight-line code, the resulting programs are amenable to dataflow analyses such as reaching definitions, making these postprocessing passes efective and straightforward.

## 7 Evaluation

We evaluated Vectorizer on NumPy programs that use explicit loops over arrays to answer the following questions.

RQ1. Is Vectorizer efective at vectorizing programs?

RQ2. How much time is needed to vectorize programs?

RQ3. How much performance improvement does the program vectorized by Vectorizer achieve?

## 7.1 Evaluation Set-up

Benchmarks. We collected 150 benchmarks from 12 datasets, which are blas [12], blend [6], darknet [46], dsp [4], dspstone [56], llama [1], makespeare [47], mathfu [2], polybench [42], simpl\_array [50], utdsp [48], and Stack Overflow.

Among these datasets, blas, darknet, dsp, dspstone, make speare, mathfu, simpl\_array, and utdsp were collected by Magalhães et al. [32] and are available as C source code. The blend and llama datasets were collected by Ahmad et al. [6] and Qiu et al. [44], available as C++ code. The 15 polybench benchmarks were selected by Brauckmann et al. [15] from the polybench suite [42]. For these benchmarks, we translated the original C or C++ source code into Python programs using NumPy while preserving their logic and structure.

In addition, we also created 51 benchmarks based on questions from Stack Overflow. Specifically, we searched for code snippets and functionality descriptions in questions related to NumPy and vectorization, and created benchmarks based on the problem descriptions or code snippets in the questions. Compared to the benchmarks translated from C or C++, some of these Stack Overflow benchmarks already use high-level NumPy features and are therefore more representative of practical NumPy programs.

![](images/0761ebd32edc96876945b71baab39586d75212c4b40454f0f2b41d593bdf9547.jpg)  
Figure 9. Running time. The x-axis is on a log scale. The table reports the mean and standard deviation of the perbenchmark vectorization time for each tool.

Baselines. We compared Vectorizer with two state-of-theart tools for vectorizing array programs: Tensorize [15] and Tenspiler [44]. Specifically, Tensorize uses symbolic execution and a symbolic algebraic solver to search for a program using high-level APIs that is equivalent to the original program. It expects input in the Afine dialect of MLIR, though we were unable to generate such code that Tensorize accepts. Hence, we compared with it only on the benchmarks whose MLIR code is available in its repository, which covers all datasets except Stack Overflow. Tenspiler uses verified lifting techniques based on program synthesis to search for a vectorized program equivalent to the original. To compare with Tenspiler, we manually translated the benchmarks not used in its evaluation into C++ as input. Since Tenspiler requires users to provide driver scripts to configure synthesis for each benchmark, we reused the driver scripts from its repository for benchmarks included in its evaluation. For other benchmarks, we wrote automatic driver scripts with which Tenspiler infers the synthesis configurations.

Configurations. We set a 30-minute time limit for synthesis per benchmark for each tool. All experiments were conducted on a machine with an Intel Core i5-13600K CPU and 64 GB of memory, running Linux Mint 22.

## 7.2 Efectiveness and Eficiency

Efectiveness. Across the 150 benchmarks, Vectorizer vectorizes 142, corresponding to an approximately 95% success rate. Upon further inspection of the 8 failed benchmarks, we found that 2 can be solved after minor changes to the code. Specifically, one needs to replace a branch guarding an update with the maximum operator. The other requires adding more axes to an argument to eliminate false dependence, which our technique cannot handle automatically. The remaining 6 failures are primarily due to complex loop-carried dependence that our technique cannot handle.

By contrast, Tenspiler solves 70 of the 150 benchmarks. Upon inspection, we found that Tenspiler can eficiently find solutions with driver scripts that have manually specified configurations, but it is less efective with automatic driver scripts and imposes several syntactic restrictions on inputs. For example, with automatic drivers, Tenspiler can only handle benchmarks with a single top-level loop that cannot be more than two levels deep.

Among the 99 benchmarks lowered to MLIR in the Tensorize repository, Tensorize generates output for 97. However, many outputs are not directly executable because they contain undefined symbols or incorrect arguments to NumPy APIs. Therefore, we considered these outputs to be intermediate representations and used coding agents based on large language models (LLMs) to analyze and normalize them while preserving their structure, operator sequence, and logic as closely as possible. LLM-assisted inspection of the normalized code suggested that 15 of the 97 outputs are semantically diferent from their source programs.

Eficiency. Figure 9 presents the running time of diferent tools. On average, Vectorizer takes 0.53 seconds to vectorize a program, which is about 10.91× faster than Tenspiler (5.78 seconds) and 34.85× faster than Tensorize (18.47 seconds). Because Vectorizer does not rely on search or symbolic reasoning, it is fast even when input programs contain semantically complex operators or control-flow structures.

Answer to RQ1 and RQ2. The results show that Vectorizer is both efective and eficient. It correctly vectorizes substantially more benchmarks than other tools and frees users from per-benchmark configuration and normalizing the emitted code.

## 7.3 Performance Improvement

To measure performance improvement, we compared each vectorized program against its original loop-based version and computed the speedup of the vectorized program. To measure the execution time, we generated random input data whose sizes correspond to 25,000,000 iterations of the inner-most loops. The workload is selected to minimize the variance caused by overly simple executions while preventing the experiments from being prohibitively slow or running out of memory on the host machine. We applied this setup to all benchmarks except 3 from the Stack Overflow dataset, whose program structures require fixed, benchmarkspecific dimensionalities. We ran each program 10 times and computed the speedups using the median measurements.

Results. Table 1 summarizes the geometric mean speedups, and Figure 10 shows their distributions. Overall, Vectorizer’s emissions gain modestly higher speedups than those emitted by other tools. Nevertheless, Vectorizer slows down 7 benchmarks from Stack Overflow, and our inspection reveals two primary causes. First, vectorization can eagerly materialize large intermediate arrays, whose memory management overhead may outweigh the computational benefits of vectorization. In such cases, loops that operate on small, cache-friendly array chunks can be more eficient. Second, for some benchmarks involving branches or data filtering, vectorization may replace control flow with predicated execution or reduction, which eliminates the branch and memory pruning available to the original loop-based implementation. In a few extremely complex benchmarks, the current postprocessing pipeline may not be suficient to simplify mask manipulations. One way to address these issues is to design a workload-aware cost model for deciding whether vectorization is desirable. We leave it as future work.

Table 1. Geometric mean of speedups. The two benchmarks solved after minor changes are included. Benchmarks solved by Vectorizer are a superset of others. The speedup in row � and column � denotes the speedup of tool � on the benchmarks that can be solved by tool �.
<table><tr><td>Vectorized by</td><td>VECTORIZER (144)</td><td>TENSPILER (70)</td><td>TENSORIZE (82)</td><td>All (59)</td></tr><tr><td>VECTORIZER</td><td>74.83×</td><td>107.16×</td><td>163.25×</td><td>106.71×</td></tr><tr><td>TENSPILER</td><td></td><td>110.27×</td><td></td><td>93.89×</td></tr><tr><td>TENSORIZE</td><td>一</td><td>一</td><td>157.67×</td><td>101.54×</td></tr></table>

On 2 benchmarks, Tensorize and Tenspiler produce shallow copies of the inputs instead of the intended deep copies, yielding apparent speedups exceeding 22, 000×. Tensorize assumes real number arithmetics and uses real division for a few benchmarks instead of integer division, which is slower.

Compared to Tenspiler, Vectorizer picks up more sumafter-multiplication snippets and replaces them with fast np.tensordot calls. The CSE pass also eliminates some redundancy kept by Tenspiler. Tensorize, aided by the symbolic algebra solver, finds more compact computation routes for some benchmarks. However, it often uses np.full to explicitly materialize large arrays of constants, while Vectorizer can take advantage of implicit broadcasting.

Comparing with Numba. All comparisons so far focus on source-to-source translations. Next, let us consider whether Vectorizer’s vectorization also complements just-in-time (JIT) compilation. To this end, we evaluate Numba [27], a popular JIT compiler for Python and NumPy programs, on both the original benchmarks and Vectorizer’s emissions. Numba only supports a limited subset of NumPy. Adding the @njit decorator allows Numba to compile 139 of the 150 original benchmarks, producing a geometric mean speedup of 77.56×. However, some operations in Vectorizer’s emissions, such as np.tensordot, fall outside the subset of NumPy supported by Numba. We thus apply an additional postprocessing pass that replaces common unsupported operations with Numba-compatible equivalents. After postprocessing, Numba compiles 122 of Vectorizer’s emissions, which achieve a geometric mean speedup of 111.91× over the original, non-compiled benchmarks. For a direct comparison, we consider the 121 benchmarks for which Numba compiles both the original program and Vectorizer’s emission. On this common subset, applying Numba to Vectorizer’s emissions yields a geometric mean speedup of 1.45× over applying Numba directly to the original programs.

![](images/10e824ae2aaf0fbf7c840990701ea83bf0ca3a29c02690314d94d25d93983383.jpg)  
Figure 10. Speedup distribution by tool and dataset. Numbers on the x-axis show the number of benchmarks included in the box. Labels below boxes show geometric mean speedups of the benchmarks counted. The y-axis is on a log scale

Answer to RQ3. Vectorizer substantially improves loopbased programs with a geometric mean speedup of 74.83×. Such vectorization also benefits Numba JIT compilations, bringing a geometric mean speedup of 1.45×.

## 8 Related Work

DSLs for programming multidimensional arrays. Many specialized libraries, DSLs, and IRs support array programming. NumPy [23] is widely used, underpins libraries such as SciPy [53] and OpenCV-Python [14], and has influenced machine-learning frameworks with similar APIs [5, 13, 18, 41]. Other systems adopt specialized models: Halide [45] separates image-processing algorithms from schedules, TACO [26] compiles symbolic index notation into eficient tensor kernels, and StableHLO [3] provides a high-level IR for compiler stacks such as MLIR [29]. We target NumPy because its API and programming model are broadly adopted and have influenced many subsequent array-programming systems.

Program synthesis for multidimensional array DSLs. Diferences in programming models and interfaces make array DSLs dificult to adopt and complicate the migration of legacy code, motivating automated synthesis. TF-Coder [49] uses enumerative search to synthesize TensorFlow programs from input-output examples, while C2TACO [32] lifts arraymanipulating C code to TACO. Dexter [6] combines pattern matching with enumerative search to lift C++ imageprocessing code to Halide, and Tensorize [15] uses symbolic algebra to guide lifting from MLIR Afine to NumPy and MLIR HLO. Cobbler [31] uses syntactic canonicalization to synthesize equivalent rewrites of 1-D NumPy programs.

Built on Metalift [11], Tenspiler [44] searches for logical program summaries to lift C++ code to high-level DSLs. Other approaches predict PyTorch API usage with machine learning [36] or employ LLMs for lifting [19, 20, 30]. In contrast, our type-directed rewrites lift loop-based array programs without costly search and are more deterministic and cost-efective than learning-based approaches.

Program rewrites. Rewriting is a core compiler optimization technique. The Glasgow Haskell Compiler lets programmers express domain-specific optimizations as rewrite rules [24], while modular rewriting systems and DSLs [54, 55] separate transformation rules from strategies governing their application. This approach has been applied to instruction selection, program interpretation, and constant propagation [16, 21, 38]. In high-performance computing, Steuwer et al. [52] reshape high-level programs to generate OpenCL [35] code, Panyala et al. [39] use term rewriting to migrate applications across platforms, and Elevate [22] optimizes functional programs for parallel architectures. We also use source-tosource rewrites, but for vectorizing NumPy programs.

Predicated execution. Predicated expressions have long been used to transform control flow into guarded computation. Allen et al. [7] introduce logical guards for this purpose, and Park and Schlansker [40] use predicates to flatten branched loops. Subsequent work applies predication to architectures supporting instruction-level parallelism [9, 17, 33]. Modern compilers similarly eliminate branches through selection and if-conversion: LLVM provides llvm.vp.select for vector selection [28], while GCC performs tree-level ifconversion for vectorization [51]. Predicated execution is also supported at the hardware level. For example, NVIDIA PTX provides predicated instructions [37], and divergent warp paths execute serially with inactive threads disabled. Vectorizer similarly represents conditional computation using masked arrays. While the idea is inspired by predicated execution, Vectorizer focuses on source-to-source transformation of high-level NumPy programs.

## 9 Conclusion

We presented Vectorizer, a correct-by-construction approach for vectorizing loop-based NumPy programs using inside-out rewrites guided by array types and dataflow analysis. Vectorizer does not rely on search or symbolic reasoning, so it is consistently fast in practice. Vectorizer vectorizes 142 of the 150 benchmarks directly, and 2 more benchmarks after minor adaptations. The average time needed for vectorizing a benchmark is 0.53 seconds, and the resulting programs achieved a geometric mean speedup of 74.83× over the original implementations.

## References

[1] [n. d.]. llama2.cpp. htps://github.com/leloykun/llama2.cpp/

[2] [n. d.]. mathfu. htps://github.com/google/mathfu

[3] [n. d.]. StableHLO. htps://github.com/openxla/stablehlo

[4] [n. d.]. Texas Instrument Digital Signal Processing (DSP) Library for MSP430 Microcontrollers. htps://www.ti.com/tool/MSP-DSPLIB.

[5] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,Jefrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geofrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vin cent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xi aoqiang Zheng. 2015. TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. htps://www.tensorflow.org/ Software available from tensorflow.org.

[6] Maaz Bin Safeer Ahmad, Jonathan Ragan-Kelley, Alvin Cheung, and Shoaib Kamil. 2019. Automatically translating image processing li braries to halide. ACM Trans. Graph. 38, 6 (2019). doi:10.1145/3355089. 3356549

[7] John R Allen, Ken Kennedy, Carrie Porterfield, and Joe Warren. 1983. Conversion of control dependence to data dependence. In Proceedings of the 10th ACM SIGACT-SIGPLAN symposium on Principles of programming languages. 177–189.

[8] Randy Allen and Ken Kennedy. 1987. Automatic translation of FOR TRAN programs to vector form. ACM Trans. Program. Lang. Syst. 9, 4 (1987), 491–542. doi:10.1145/29873.29875

[9] David I August, Wen-mei W Hwu, and Scott A Mahlke. 1997. A framework for balancing control flow and predication. In Proceedings of30th Annual International Symposium on Microarchitecture. IEEE, 92–103.

[10] A. J. Bernstein. 1966. Analysis of Programs for Parallel Processing. IEEE Transactions on Electronic Computers EC-15, 5 (1966), 757–763. doi:10.1109/PGEC.1966.264565

[11] Sahil Bhatia, Sumer Kohli, Sanjit A. Seshia, and Alvin Cheung. 2023. Building Code Transpilers for Domain-Specific Languages Using Program Synthesis (Experience Paper). In European Conference on Object-Oriented Programming (ECOOP). 38:1–38:30. doi:10.4230/LIPICS. ECOOP.2023.38

[12] L. Susan Blackford, James Demmel, Jack Dongarra, Iain Duf, Sven Hammarling, Greg Henry, Michael Heroux, Linda Kaufman, Andrew Lumsdaine, Antoine Petitet, Roldan Pozo, Karin Remington, and

R. Clint Whaley. 2002. An Updated Set of Basic Linear Algebra Subprograms (BLAS). ACM Trans. Math. Software 28, 2 (June 2002), 135–151. doi:10.1145/567806.567807

[13] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. 2018. JAX: composable transformations of Python+NumPy programs. htp://github. com/jax-ml/jax

[14] G. Bradski. 2000. The OpenCV Library. Dr. Dobb’s Journal ofSoftware Tools (2000).

[15] Alexander Brauckmann, Luc Jaulmes, José W de Souza Magalhães, Elizabeth Polgreen, and Michael FP O’Boyle. 2025. Tensorize: Fast Synthesis of Tensor Programs from Legacy Code using Symbolic Tracing, Sketching and Solving. In Proceedings of the 23rd ACM/IEEE International Symposium on Code Generation and Optimization. 15–30.

[16] Martin Bravenboer and Eelco Visser. 2002. Rewriting strategies for instruction selection. In International Conference on Rewriting Techniques and Applications (RTA). 237–251.

[17] L. Carter, B. Simon, B. Calder, L. Carter, andJ. Ferrante. 1999. Predicated static single assignment. In 1999 International Conference on Parallel Architectures and Compilation Techniques (Cat. No.PR00425). 245–255. doi:10.1109/PACT.1999.807561

[18] Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, and Zheng Zhang. 2015. Mxnet: A flexible and eficient machine learning library for heterogeneous distributed systems. arXiv preprint arXiv:1512.01274 (2015).

[19] José Wesley de Souza Magalhães, Shideh Hashemian, Alexander Brauckmann, Jackson Woodruf, Elizabeth Polgreen, and Michael F. P. O’Boyle. 2026. Accelerating Sparse Algebra with Program Synthesis. In Proceedings ofthe ACM SIGPLAN International Conference on Compiler Construction (CC). Association for Computing Machinery, 106–118. doi:10.1145/3771775.3786281

[20] José Wesley de Souza Magalhães, Jackson Woodruf, Jordi Armengol-Estapé, Alexander Brauckmann, Luc Jaulmes, Elizabeth Polgreen, and Michael FP O’Boyle. 2025. Guess, Measure & Edit: Using Lowering to Lift Tensor Code. In International Conference on Parallel Architectures and Compilation Techniques (PACT). 216–228.

[21] Eelco Dolstra and Eelco Visser. 2002. Building Interpreters with Rewriting Strategies. Electronic Notes in Theoretical Computer Science 65, 3 (2002), 57–76. htps://doi.org/10.1016/S1571-0661(04)80427-4

[22] Bastian Hagedorn, Johannes Lenfers, Thomas Koehler, Xueying Qin, Sergei Gorlatch, and Michel Steuwer. 2020. Achieving highperformance the functional way: a functional pearl on expressing high-performance optimizations as rewrite strategies. Proceedings of the ACM on Programming Languages 4, ICFP (2020), 1–29.

[23] Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. 2020. Array programming with NumPy. Nature 585, 7825 (Sept. 2020), 357–362. doi:10.1038/s41586-020-2649-2

[24] Simon Peyton Jones, Andrew Tolmach, and Tony Hoare. 2001. Playing by the rules: rewriting as a practical optimisation technique in GHC. In Haskell workshop, Vol. 1. 203–233.

[25] Ken Kennedy and John R. Allen. 2001. Optimizing compilersfor modern architectures: a dependence-based approach. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, Chapter CHAPTER 2 Dependence: Theory and Practice, 57–98.

[26] Fredrik Kjolstad, Shoaib Kamil, Stephen Chou, David Lugato, and Saman Amarasinghe. 2017. The tensor algebra compiler. Proceedings of the ACM on Programming Languages 1, OOPSLA (2017), 1–29.

[27] Siu Kwan Lam, Antoine Pitrou, and Stanley Seibert. 2015. Numba: a LLVM-based Python JIT compiler. In Proceedings ofthe Second Workshop on the LLVMCompilerInfrastructure in HPC (Austin, Texas) (LLVM ’15). Association for Computing Machinery, New York, NY, USA, Article 7, 6 pages. doi:10.1145/2833157.2833162

[28] Chris Lattner and Vikram Adve. 2004. LLVM: A compilation framework for lifelong program analysis & transformation. In International symposium on code generation and optimization, 2004. CGO 2004. IEEE, 75–86.

[29] Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasi lache, and Oleksandr Zinenko. 2021. MLIR: Scaling Compiler Infrastructure for Domain Specific Computation. In International Symposium on Code Generation and Optimization (CGO). 2–14. doi:10.1109/ CGO51591.2021.9370308

[30] Yixuan Li, José Wesley de Souza Magalhães, Alexander Brauckmann, Michael F. P. O’Boyle, and Elizabeth Polgreen. 2025. Guided Tensor Lifting. Proc. ACM Program. Lang. 9, PLDI, Article 227 (June 2025), 23 pages. doi:10.1145/3729330

[31] Justin Lubin, Jeremy Ferguson, Kevin Ye, Jacob Yim, and Sarah E Chasins. 2024. Equivalence by Canonicalization for Synthesis-Backed Refactoring. Proceedings of the ACM on Programming Languages 8, PLDI (2024), 1879–1904.

[32] José Wesley de Souza Magalhães, Jackson Woodruf, Elizabeth Polgreen, and Michael F. P. O’Boyle. 2023. C2TACO: Lifting Tensor Code to TACO. In ACM SIGPLAN International Conference on Generative Programming: Concepts and Experiences (GPCE). Association for Computing Machinery, 42–56. doi:10.1145/3624007.3624053

[33] Scott A Mahlke, Richard E Hank, James E McCormick, David I August, and Wen-Mei W Hwu. 1995. A comparison offull and partial predicated execution support for ILP processors. In Proceedings of the 22nd annual international symposium on Computer architecture. 138–150.

[34] Charith Mendis. 2024. Compiler Auto-vectorization. htps://charithm. web.illinois.edu/cs526/sp2024/lec11.pdf

[35] Aaftab Munshi. 2009. The opencl specification. In IEEE Hot Chips 21 Symposium (HCS). 1–314.

[36] Daye Nam, Baishakhi Ray, Seohyun Kim, Xianshan Qu, and Satish Chandra. 2022. Predictive synthesis of API-centric code. In Proceedings of the ACM SIGPLAN International Symposium on Machine Programming (MAPS). 40–49.

[37] NVIDIA. [n. d.]. Parallel Thread Execution ISA Version 9.3. ([n. d.]). htps://docs.nvidia.com/cuda/parallel-thread-execution/index.htm

[38] Karina Olmos and Eelco Visser. 2002. Strategies for Source-to-Source Constant Propagation. Electronic Notes in Theoretical Computer Science 70, 6 (2002), 156–175. htps://doi.org/10.1016/S1571-0661(04)80605-4

[39] Ajay Panyala, Daniel Chavarría-Miranda, and Sriram Krishnamoorthy. 2012. On the use of term rewriting for performance optimization of legacy HPC applications. In International Conference on Parallel Processing (ICPP). 399–409.

[40] Joseph C. H. Park and Michael S. Schlansker. 1991. On Predicated Execution. Technical Report HPL-91-58. Hewlett-Packard Laboratories, Palo Alto, CA, USA. htps://web.eecs.umich.edu/\~mahlke/courses/ 583f19/reading/HPL-91-58.pdf

[41] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems 32 (2019).

[42] Louis-Noel Pouchet and Tomofumi Yuki. [n. d.]. polybench. htp: //polybench.sourceforge.net

[43] William W. Pugh and David Wonnacott. 1992. Eliminating False Data Dependences using the Omega Test. In Proceedings ofthe ACM SIG-PLANConference on Programming Language Design and Implementation (PLDI). 140–151. doi:10.1145/143095.143129

[44] Jie Qiu, Colin Cai, Sahil Bhatia, Niranjan Hasabnis, Sanjit A. Seshia, and Alvin Cheung. 2024. Tenspiler: A Verified-Lifting-Based Compiler for Tensor Operations. In 38th European Conference on Object-Oriented Pro<sub>g</sub>rammin<sub>g</sub> (ECOOP 2024) (Leibniz International Proceedin<sub>g</sub>s in Informatics (LIPIcs), Vol. 313), Jonathan Aldrich and Guido Salvaneschi (Eds.). Schloss Dagstuhl – Leibniz-Zentrum für Informatik, Dagstuhl, Germany, 32:1–32:28. doi:10.4230/LIPIcs.ECOOP.2024.32

[45] Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, and Saman P. Amarasinghe. 2013. Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines. In ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI). 519–530. doi:10.1145/2491956.2462176

[46] Joseph Redmon. 2013–2016. Darknet: Open Source Neural Networks in C. htps://pjreddie.com/darknet/.

[47] Christopher D Rosin. 2019. Stepping stones to inductive synthesis of low-level looping programs. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33. 2362–2370.

[48] Mazen AR Saghir. 1998. Application-specific instruction-set architectures for embedded DSP applications. Ph. D. Dissertation. University of Toronto.

[49] Kensen Shi, David Bieber, and Rishabh Singh. 2022. Tf-coder: Program synthesis for tensor manipulations. ACM Transactions on Programming Languages and Systems (TOPLAS) 44, 2 (2022), 1–36.

[50] Sunbeom So and Hakjoo Oh. 2017. Synthesizing imperative programs from examples guided by static analysis. In International Static Analysis Symposium. Springer, 364–381.

[51] Richard M. Stallman and GCC Developer Community. 2003. Using GCC: The GNU Compiler Collection Reference Manual. Free Software Foundation. For GCC version 3.3.1.

[52] Michel Steuwer, Christian Fensch, Sam Lindley, and Christophe Dubach. 2015. Generating performance portable code using rewrite rules: from high-level functional expressions to high-performance OpenCL code. In Proceedings ofthe ACM SIGPLANInternational Conference on Functional Programming (ICFP). 205–217. doi:10.1145/2784731. 2784754

[53] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, JosefPerktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. 2020. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods 17 (2020), 261–272. doi:10.1038/s41592-019-0686-2

[54] Eelco Visser. 2001. Stratego: A language for program transformation based on rewriting strategies system description of stratego 0.5. In International Conference on Rewriting Techniques and Applications (RTA). 357–361.

[55] Eelco Visser, Zine-El-Abidine Benaissa, and Andrew P. Tolmach. 1998. Building Program Optimizers with Rewriting Strategies. In Proceedings of the ACM SIGPLAN International Conference on Functional Programming (ICFP). 13–26. doi:10.1145/289423.289425

[56] Vojin Zivojnovic. 1994. DSPstone: A DSP-oriented benchmarking methodology. Proc. Signal Processing Applications & Technology, Dallas, TX, 1994 (1994), 715–720.

## A Formal Semantics of the Proposed DSL

The formal semantics of the proposed DSL is given in Figure 11, 12, 13, 14, and 15. Figure 11 specifies how to determine the maskedness of an expression at runtime under an evaluation environment. Figure 12 specifies how statements in a program are executed. Figure 13 specifies the rule for evaluating a program in the DSL. Finally, Figure 14 and 15 specify how expressions in the DSL are evaluated to values under an evaluation environment.

Following the format ofthe paper, we use blue texts for literal code in the proposed DSL. In our formalism, the state of each variable at runtime is a pair $( v , \mu )$ . Here, � is the value of the variable and $\mu$ is the maskedness of the variable. We use Σ to represent a program state, which is a stack of maps from variables to such pairs. We refer to each map in the stack as a scope. We use � to represent a scope. Peek(Σ), Pop(Σ), and Push(�, Σ) return the scope at the top of the stack, the stack with the top scope removed, and the stack with a newly pushed-in scope �. We use $\Sigma \vdash s \searrow \Sigma ^ { \prime }$ to denote that executing the statement � under the environment Σ results in a new state environment $\Sigma ^ { \prime }$ . We use $\Sigma \vdash e \downarrow \downarrow _ { v }$ � to denote an expression node � is evaluated to a value � under the state Σ. Similarly, $\Sigma \vdash e \Downarrow _ { \mu } \mu$ denotes that the maskedness of � at runtime is $\mu$ when evaluated under Σ. We use $[ f ] ( \boldsymbol { v } _ { 1 } , \ldots , \boldsymbol { v } _ { n } )$ to denote the returned value of a routine $f$ with arguments $v _ { 1 } , \ldots , v _ { n } .$ . We use ⊗ to denote arrays that are masked at runtime and ⊙ to denote arrays that are unmasked at runtime. We use the notation $\Sigma \vdash \textsf { f } ( x _ { 0 } , \ \ldots \ , \ x _ { n } )$ � return $y \Downarrow ( v , \mu )$ to denote that a program taking arguments $x _ { 0 } , \ldots , x _ { n }$ returns a value � whose runtime maskedness is $\mu$ under the initial environments Σ.

To conveniently define the semantics of our DSL, we also included a few auxiliary operators and defined their semantics. The auxiliary operators are technically not part of our DSL. The rules specifying the semantics of these auxiliary operators have names starting with “Aux.” Auxiliary operators’ names are in the small caps font. The auxiliary indexing operator � [i], which means picking out a top-level element from an array based on the given index, is in purple, so it is diferentiated from the indexing operator in the DSL.

At a high-level, executing statements under an evaluation environment results in an updated evaluation statement. Variable binding statements create a new mapping from the variable name to a runtime value-maskedness pair in the top scope (SEM-Bind). The execution of a branch entails the evaluation of the branch condition first (SEM-If). The corre sponding branch is run under the evaluation environment with an empty scope pushed to the top. After the execution of the branch, the top-level scope is removed. Loops’ execution is defined recursively, which is shown by (SEM-For0) and (SEM-Fori). When the loop bound evaluates to zero, no update to the execution environment is made. When the evaluation of the loop bound is non-zero, we first try to execute the loop with a decremented loop bound and then execute the loop under the current loop bound. Similar to the execution of branches, a new scope with only the state of the loop variable is pushed to the evaluation environment before each iteration’s evaluation. The top scope is removed after each iteration’s evaluation. Update(Σ, �, �) is an auxiliary operator that digs into Σ to find the scope in which � is defined and updates the value bound to � to �. The semantics of this auxiliary operator is shown by the rules (SEM-Auxupd1) and (SEM-Auxupd2). The semantics of normal update statements (SEM-Upd) and masked update statements (SEM-MskUpd) are defined in terms of this auxiliary operator.

The rules for evaluating expressions’ value and maskedness are pretty straightforward. For the value evaluations, we use the Shape(�) operator to specify the runtime shapes of the evaluation results. The semantics of this auxiliary operator is shown by the rules (RTV-AuxSBase) and (RTV-AuxSInd). We use the auxiliary indexing operator to specify how each individual element in the evaluation result is related to the elements in the operands. The semantics of this auxiliary indexing operator is shown by the rule (RTV-AuxIdx). As the denotational semantics for most of the operators in our DSL are well-known, we omit them here and use � to represent the result of calling these operators.

The rules for maskedness evaluations are also straightforward. In general, if one argument of an operator is masked, the evaluation result is also masked. There are some exceptions. Masked arrays cannot be indexers of indexing operators, and cannot be arguments to the matmul operator. Also, the second argument of the filled should not be masked.

Finally, the semantics of executing a program in the DSL is simply executing the program body under an evaluation environment that can evaluate all the arguments to the program. The returned value and its maskedness are then fetched from the updated evaluation environment after the execution of the loop body.

## B Type Inference Rules

Figure 16 shows the complete set of rules for static shape inference. Figure 17 shows the complete set of rules for static maskedness inference.

The type environment Γ is a map from variable names to pairs of shape and maskedness. Here, $\Gamma \vdash e : _ { \varsigma } \varsigma$ and $\Gamma \vdash e : _ { m }$ � denote that � is inferred to have a static shape � and a static maskedness �. We use the ⊤ symbol to denote masked arrays and use the ⊥ symbol to denote unmasked arrays. We only support typing arrays with a fixed number of axes, though the dimensionality of each axis may be dynamic. Therefore, static shapes are presented by tuples of symbols and integers. We use the $[ [ \mathbf { i } ] ] _ { \Gamma }$ notation to represent evaluating an integral expression statically, which may return an integer or a symbol.

Vectorizer: Vectorizing NumPy Programs with Shape-Guided Rewrite

$$
\begin{array} { r l } { \underbrace { \frac { \mathrm { P u r a l i t } } { \sum \star \mathrm { i } \mathrm { i } \cdot \mathrm { l } \mathrm { i } \cdot \mathrm { l } \mathrm { i } \cdot \mathrm { l } \mathrm { i } \cdot \mathrm { \Lambda } } } } _ { \geq \mathrm { ~ k } \mathrm { ~ i } \mathrm { ~ i } \mathrm { ~ j } _ { \mu } \otimes \mathrm { ~ ( ~ K T a i t ) } } \quad  & { \underbrace { \mathrm { ~ V a r a r a ~ } \sigma = \mathrm { P o r ( \vec { \Lambda } ) } } _ { \geq \mathrm { ~ k } \times \mathrm { ~ U } , \mu } } \\ { \underbrace { \Sigma \star \mathrm { d } _ { \mu } \mu _ { \mu } \odot \Delta \mathrm { ~ ( ~ R T a ~ ) } } _ { \Sigma \star \mathrm { ~ x } \mathrm { ~ U } , \mu } \quad } & { \underbrace { x \in \mathrm { d a n } \sigma \mathrm { ~ ( ~ } \sigma \mathrm { ) } } _ { \Sigma \star \mathrm { ~ x } \mathrm { ~ U } , \mu } \quad } & { \underbrace { x \notin \mathrm { d o n } \sigma \mathrm { ~ ( ~ } \sigma \mathrm { ) } } _ { \Sigma \star \mathrm { ~ x } \mathrm { ~ U } , \mu } \quad } \\ { \sum \star \epsilon \mathrm { ~ g } _ { \mu } \mu _ { \mu } \ \ \ \mathrm { \scriptsize { = ~ } \delta } \mathrm { ~ ( ~ B a r a ~ ) } } & { f \in \mathrm { U n a r a ~ } \mathrm { ~ V o r ~ } } \\ { \frac { \Sigma \star \mathrm { ~ c } \mathrm { ~ c } \mathrm { ~ g } _ { \mu } \mu _ { \mu } \diamond } { \sum \star \mathrm { ~ c } \mathrm { ~ g } _ { \mu } \mathrm { ~ l } \mathrm { ~ g } _ { \mu } \mu } } & { \underbrace { f \in \mathrm { ~ U n a r a ~ } \sigma \mathrm { ~ V o r ~ } \mathrm { ~ g } } _ { \Sigma \star \mathrm { ~ o } \mathrm { ~ p } } } &  \underbrace { 1 \mathrm { ~ S t } \cdot \mathrm { a } \mathrm { ~ n } \Sigma \sigma \mathrm { ~ ( ~ } \sigma \mathrm { ~ k } \mathrm { ~ n } \mathrm { ~ , ~ } \sigma \mathrm { ) } } _ { \Sigma \star \mathrm { ~ } \mathrm { ~ C } \mathrm { ~ p } , \mu } \end{array}
$$

Figure 11. The rules for evaluating maskedness of expressions at runtime.

$$
\begin{array} { r l r l r l r l } & { \frac { 1 } { \sqrt { 5 + \eta ( \eta ) } } \frac { \eta } { \sqrt { 5 + \eta ( \eta ) } } \lesssim \beta \epsilon _ { \eta } ^ { \frac { \epsilon } { \epsilon } } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { 2 } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { 2 } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { 2 } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { \sqrt { 5 + \eta ( \epsilon ) } } } & & { \le \frac { \eta } { \sqrt { 5 + \eta ( \epsilon ) } } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { 2 } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { \sqrt { 5 + \eta ( \epsilon ) } } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { 2 } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { \sqrt { 5 + \eta ( \epsilon ) } } } & & { \le \eta \le \eta } \\ & { } & { \frac { \epsilon } { \sqrt { 5 + \eta ( \epsilon ) } } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { \sqrt { 5 + \eta ( \epsilon ) } } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { 2 } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { \sqrt { 5 + \eta ( \epsilon ) } } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { 2 } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { \sqrt { 5 + \eta ( \epsilon ) } } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { \sqrt { 5 + \eta ( \epsilon ) } } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { \sqrt { 5 + \eta ( \epsilon ) } } } & & { \le \frac { \eta } { \sqrt { 5 + \eta ( \epsilon ) } } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { 2 } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { \sqrt { 5 + \eta ( \epsilon ) } } \frac { \sqrt { 5 + \eta ( \epsilon ) } } { \sqrt { 5 + \eta ( \epsilon ) } } } & &  \le \frac { \eta } { \sqrt { 5 + \eta ( \epsilon ) } } \frac  \sqrt   \end{array}
$$

Figure 12. The semantics of the statements.

$$
{ \begin{array} { r l } & { 0 \leq k \leq n \quad \Sigma \vdash x _ { k } \nsubseteq \Sigma \vdash x _ { k } \nsubseteq \Sigma \vdash x _ { k } \nsubseteq \mu _ { \mu } \mu _ { k } } \\ & { { \frac { \Sigma \vdash s \searrow \Sigma ^ { \prime } \quad \Sigma ^ { \prime } \vdash y \nsubseteq \Sigma ^ { \prime } \vdash y \qquad \Sigma ^ { \prime } \vdash y \nwarrow \mu } { \Sigma \vdash f ( x _ { 0 } , \ \dots , x _ { n } ) \ s \ r e t u r n \ y \ \downarrow \ ( v , \mu ) } } \quad ( { \mathrm { S E M \mathrm { - } P r o g } } ) } \end{array} }
$$

Figure 13. The semantics of programs.

## C Proof of Soundness of Type Analysis

We want to prove Theorem C.1 shown below, which is a more formally restated version ofTheorem 4.1. We use $\alpha _ { \varsigma } ( v ) , \alpha _ { m } ( \mu )$ to denote the application of abstraction functions for shapes and maskedness. The abstraction function for shapes can be simply defined as the Shape() function in Figure 14.

The abstraction function for maskedness can be defined as $\alpha _ { m } ( \mu ) = \mathsf { I T E } ( \mu = \otimes , \mathsf { T } , \perp )$

Theorem C.1 (Soundness of type analysis). Suppose that <sub>we</sub> h<sub>ave</sub> $\mathsf { f } ( x _ { 0 } , \ldots , \ x _ { n } )$ � return <sub>�</sub>, which is a pro<sub>g</sub>ram, A be a type annotation and Σ an evaluation for $x _ { 0 } , \ldots , x _ { n }$ Assume that ∀ Variable $x . \Sigma \ \vdash \ x \ \Downarrow _ { v } \ v \land \Sigma \ \vdash \ x \ \Downarrow _ { \mu } \ \mu \ $ ${ \mathcal { A } } \vdash x \ : _ { \varsigma } \ \varsigma \wedge \varsigma \ = \ \alpha _ { \varsigma } ( \upsilon ) \wedge { \mathcal { A } } \vdash x \ : _ { m } \ m \wedge m \ = \ \alpha _ { m } ( \mu ) . \ I f$ $\Sigma \ \vdash \ \mathsf { f } \left( x _ { 0 } , \ \ldots , \ x _ { n } \right)$ � return $y \ \Downarrow \ ( v , \mu )$ <sub>,</sub> th<sub>en</sub> ${ \mathcal { A } } \vdash s \hookrightarrow$ $\Gamma \wedge \Gamma \vdash y : _ { \zeta } \varsigma ^ { \prime } \wedge \Gamma \vdash y : _ { m } m ^ { \prime } \ a n d \alpha _ { \zeta } ( v ) = \varsigma ^ { \prime } \wedge \alpha _ { m } ( \mu ) = m ^ { \prime } .$

First, we state the following lemmas and prove them. These lemmas will eventually lead to the proof of Theorem C.1.

$$
\begin{array}{c} \begin{array}{c} \begin{array} { r l } { \frac { \mathrm { I n t e g e r l i t e r a l } c } { \boldsymbol { \Sigma } \star \boldsymbol { \epsilon } \| \boldsymbol { \epsilon } \cdot \boldsymbol { \epsilon } \| \boldsymbol { \epsilon } } } & { \qquad \mathrm { V a r i a b l e x } \quad \sigma = \mathrm { P e k } ( \boldsymbol { \Sigma } ) } \\ { \boldsymbol { \Sigma } \star \boldsymbol { \epsilon } \| \boldsymbol { \epsilon } \cdot \boldsymbol { \Sigma } \| _ { \boldsymbol { \sigma } } \boldsymbol { \epsilon } } & { \qquad \boldsymbol { \Sigma } \in \mathrm { d o m } ( \boldsymbol { \sigma } ) \quad \sigma ( \boldsymbol { \kappa } ) = ( \boldsymbol { \sigma } , \boldsymbol { \mu } ) } \\ { \boldsymbol { \Sigma } \star \boldsymbol { \epsilon } \| \boldsymbol { \epsilon } \cdot \boldsymbol { \Sigma } \| _ { \boldsymbol { \sigma } } \boldsymbol { v } } & { \qquad \mathrm { ( R T N - \boldsymbol { \epsilon } ) } } \\ { \boldsymbol { \sigma } \in \mathbb { R } \cup \{ \boldsymbol { \epsilon } \} } & { \qquad \boldsymbol { v } = ( v _ { 1 } , \dots , v _ { m } ) \quad 1 \leq j \leq m } \\ { \mathrm { S u b r e } ( \boldsymbol { \sigma } ) = ( \boldsymbol { \sigma } ) } & { \qquad \mathrm { S u b r e } ( v _ { j } ) = ( a _ { n } , \dots , a _ { 0 } ) } \\ { \mathrm { S u b r e } ( \boldsymbol { \sigma } ) = ( \boldsymbol { \sigma } ) } & { \qquad \mathrm { S u b r e } ( v ) = ( m , a _ { n } , \dots , a _ { 0 } ) } \end{array} \quad \frac { \mathrm { V a r i a b l e ~ \boldsymbol { \Sigma } ~ } \boldsymbol { \epsilon } } { \mathrm { A u s ~ S u r e } \boldsymbol { \Sigma } \boldsymbol { } \boldsymbol { \epsilon } }  & { \qquad \mathrm { ( R T N - A u c k ) } \quad \sigma = \mathrm { P e } \boldsymbol { \Sigma } ^ { \boldsymbol { \epsilon } } \boldsymbol { \epsilon } \boldsymbol { \epsilon } \| \boldsymbol { \epsilon } \| _ { \boldsymbol { \sigma } } } \\ & { \qquad \boldsymbol { \Sigma } \star \boldsymbol { \epsilon } ( \boldsymbol { \epsilon } ) [ \boldsymbol { \epsilon } ] \| _ { \boldsymbol { \sigma } } u _ { \boldsymbol { \epsilon } } } \end{array} ( \mathrm { R e T N } \times \boldsymbol { \epsilon } )  \end{array} ( \mathrm
$$

$$
\begin{array} { r l } { v = ( v _ { 1 } , \ldots , v _ { n } ) \quad n \geq 0 \qquad } & { \mathrm { S u a r r } ( v ) = ( a _ { m } , \ldots , a _ { 0 } ) \quad \mathrm { B o a n c s c i e } ( ( a _ { m } , \ldots , a _ { 0 } ) , ( b _ { n } , \ldots , b _ { 0 } ) ) = ( b _ { n } , \ldots , b _ { 0 } ) } \\ { v = ( s _ { 0 } , \ldots , v _ { n } ) \quad c \in \mathbb { Z } \qquad } & { \mathrm { S u a r r } ( v ) = ( b _ { n } , \ldots , b _ { n } ) \qquad \Rightarrow \ j \leq m \ c _ { j } = \| \nabla ( a _ { j } = 1 , 0 , 1 ) | } \\ { v | c | = \pi _ { c } \qquad } & { \mathrm { ( R T N - A u w r d a y ) } \qquad } & { \mathrm { B i a s . } \cdots \forall i _ { 0 } , \ldots 0 \leq i _ { j } < b _ { j } \cdots > v ^ { \prime } [ a _ { 1 } ] \cdot [ i _ { 0 } ] = v | c _ { m } + i _ { m } ] \cdot [ c _ { 0 } + i _ { 0 } ] \qquad } \\ { \mathrm { B e a n c s t r } \mathrm { I n } ( v , \ldots , b _ { 0 } ) \big ) = \sigma ^ { \prime } } & { \mathrm { ( R T N - A u w r d a y ) } } \end{array} \qquad ( \mathrm { R T N - A u s b } \mathrm { R T } )
$$

$$
\begin{array} { r l r } { \mathfrak { o } \le l \le m \quad \Sigma \left. e _ { l } \parallel _ { \infty } \eta _ { l } } & { \operatorname* { B a o n c a s t r a f } ( S \cup \kappa \cup \kappa ( \sigma _ { 0 } ) ) , \ldots , \ldots , \operatorname { S u a n g } ( \upsilon _ { m } ) ) = ( a _ { n } , \ldots , a _ { 0 } ) } & { \Sigma \vdash e \parallel _ { \infty } \mathcal { D } } & { \operatorname* { S u a n g } ( \upsilon ) = ( b _ { m } , \ldots , b _ { 0 } ) } \\ { \operatorname { S u a r c } ( \upsilon ^ { \prime } ) = ( a _ { n } , \ldots , a _ { 0 } ) } & { \operatorname* { B a o n c a s t r a f } ( \upsilon _ { l } , ( a _ { n } , \ldots , a _ { 0 } ) ) = \eta _ { l } ^ { \prime } } & { \forall i _ { n } , \ldots , \forall i _ { n } , \sum _ { n = 0 } ^ { n } \le i _ { k } < a _ { k } - \omega _ { k } - \sum _ { n = 0 } ^ { n } \zeta _ { l } ^ { \prime } \left| i _ { n } \right| \ldots \left| i _ { 0 } \right| \in \mathbb { N } \cup \left\{ 0 \right\} }  \\ & { \forall i _ { n } , \ldots , \forall i _ { n } , \lambda _ { l } ^ { \prime } \ne 0 } & { \le i _ { j } < a _ { j } \to \upsilon ^ { \prime } \left| i _ { n } \right| , \ \ldots \left| i _ { 0 } \right| = \left| \eta _ { l } ^ { \prime \prime } \left| i _ { n } \right| \ldots \left[ i _ { 0 } \right| \right] \ldots \left[ i _ { 0 } \right| \left| i _ { n } \right| \ldots \left| i _ { 0 } \right| \right] } & { \quad \mathrm { ( R e l r - l a ) } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}
$$

$$
\small \begin{array} { r l } & { f \in \mathrm { ~ U n a r y ~ o p ~ U B b m a r y } \ o p \ \textit { \textbf { 1 leq 1 \leq m } } \sum \textit { \textbf { 1 \leq 1 \leq m } } \sum \textit { \textbf { 1 \leq n } } \gamma \delta _ { 1 } \quad \mathrm { B a c a n \leq s x } \{ \mathrm { S u b s t } ( \it \{ n _ { 1 } \leq n _ { 1 } \} , \dots , \mathrm { S u b s t } ( \it \{ n _ { n } \} ) ) = ( a _ { 1 } , \dots , a _ { 0 } ) \} \quad \mathrm { B u o a n \times A s s } \mathrm { T i o } ( \it \{ n _ { 1 } , ( a _ { 1 } , \dots , a _ { 0 } ) \} ) = v _ { i } ^ { \prime } } \\ & { \qquad \mathrm { S u b s t } ( \it { \textbf { 1 \leq 1 , \dots , a 0 } ) } \quad \forall i _ { n _ { 1 } , \dots , \times } \forall i _ { 0 } , \dots \delta _ { i j _ { \tau } } \circ A \textit { \textbf { 1 \leq 1 \leq n } } \gamma \delta [ i _ { n _ { 1 } } ] \dots [ i _ { 0 } ] = [ f ] ( v _ { i } ^ { \prime } [ i _ { n _ { 1 } } ] , \dots [ i _ { 0 } ] , \dots , v _ { i n _ { 1 } } ^ { \prime } [ i _ { 0 } ] \dots [ i _ { 0 } ] ) } \\ & { \qquad \textrm { 1 \dots \delta _ { i } f e , \quad \qquad \quad \ge ~ \gamma 1 \vdash \gamma \delta _ { i } \eta \textit { \textbf { 1 \leq 1 \leq n } } } } \end{array}\tag{RTV-Op}
$$

$$
\begin{array} { r l r } & { \forall i _ { 0 } , . . . , \nabla i _ { \bar { x } - 1 } , \nabla i _ { \bar { x } + 1 } , . . . , \nabla i _ { n } , \gamma _ { \bar { x } } ( 0 , . . . , i _ { n - 1 } , 0 ) \leq i _ { j } < a _ { j } \to } \\ & { \quad \underline { { v ^ { \prime } [ i _ { 0 } ] . . . [ i _ { \bar { x } - 1 } ] [ i _ { \epsilon + 1 } ] . . . [ i _ { n } ] = [ [ f ] ] ( v [ i _ { 0 } ] . . . [ i _ { n - 1 } ] [ 0 ] [ i _ { \epsilon + 1 } ] . . . , [ i _ { n } ] , . . . , [ i _ { 0 } ] . . . , [ i _ { \bar { x } - 1 } ] [ i _ { \epsilon - 1 } ] . . . , [ i _ { n } ] ) } } } & { \mathrm { ( R T V - R e l e c ) } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \Sigma \vdash f ( \epsilon , c ) \downarrow _ { v } \underline { { v ^ { \prime } } } } & \end{array}
$$

$$
\begin{array} { r } { \frac { \sqrt { k _ { 0 } \ldots \bigvee k _ { n + 1 } \ldots \bigvee \bar { k } _ { m + 0 } } \circ \xi _ { m } \prec k _ { m } \right. \vee l . \bar { j } . \bar { j } = \left| \mathbb { T } \{ l < c , k _ { l } , k _ { l + 1 } \} \wedge \sigma ^ { \prime } \left[ k _ { 0 } \right] \ldots \left[ k _ { n + 1 } \right] = v \left[ j _ { 0 } \right] \ldots \left[ j _ { n } \right] } { \Sigma \star \exp \mathrm { a d } \mathrm { d } \mathrm { a s } \langle e , ( c , \varepsilon ) \rangle \mathrm { l } _ { \sigma } v ^ { \sigma } } \qquad \Longrightarrow \qquad ( \mathbb { R } \mathrm { T } \mathrm { V } \mathrm { E } \mathrm { S u s } ) } \end{array}
$$

$$
\begin{array} { r l r } { \underset { \mathbb { X } \times \mathrm { e x p a n d } \underline { { \mathrm { d i m } } } \times \mathrm { g a n d } , \mathrm { d i m s } ( \epsilon _ { \nu } , ( \epsilon _ { 0 } , \dots , \epsilon _ { k - 1 } ) ) , ( \epsilon _ { m } , \nu ) } { \le _ { k } \le _ { \mathcal { S } } \le _ { k } \le _ { \mathcal { S } } \le _ { k } } } & { \underset { \mathbb { X } \times \mathrm { e x p a n d } \textnormal {  { V } } = \nu } { \le _ { k } \le _ { \mathcal { S } } \le _ { k } \le _ { \mathcal { S } } } } & \\ { \underline { { \texttt { X } \times \mathrm { e x p a n d } } } \underline { { \mathrm { d i m s } } } ( \epsilon \mathrm { s g n d } , \mathrm { d i m s } ( \epsilon _ { \nu } , ( \epsilon _ { 0 } , \dots , \epsilon _ { m - 1 } ) ) , ( \epsilon _ { m } , \nu ) ) } & { \mathrm { ( R e r v e r g e n d ) } } & { } & { \frac { \mathrm { V o r } _ { k } \cdot \sqrt { \kappa _ { k } \cdot _ { \mathcal { N } } } = _ { 0 } \le _ { k } \le _ { \mathcal { S } } G _ { l } \le C _ { l } \to v [ k _ { k } ] \dots [ k _ { k } ] = 1 } { \Sigma \cdot \mathrm { c o n s } ( \epsilon ( k _ { 0 } , \dots , \underline { { \cdot } } \mu _ { n } ) ) \mathbb { P } _ { 0 } \nu } \quad \mathrm { ( R T V - O n e s ) } } \end{array}
$$

$$
\frac { \Sigma \vdash \textnormal { i } \big \| _ { v } c \textnormal { \texttt { S H A P E } } ( v ) = ( c , ) \quad \forall j . 0 \le j < c \to v [ j ] = j } { \Sigma \vdash \mathsf { a r a n g e } ( \mathrm { i } ) \iint _ { v } v } \quad ( \mathrm { R T V - A r a n } ) \int _ { v } ^ { \infty } d v .
$$

# Figure 14. The rules for evaluating values of expressions at runtime.

$$
\begin{array} { r l } { \sum \textsf { e } q _ { 0 } \underbrace { \sum \textsf { e } q _ { 0 } } _ { \geq \leq \leq \leq \leq 0 } + \sum \textsf { a n a x } ( \sigma ) = ( a _ { 0 } , \ldots , a _ { n } ) \operatorname* { S a x } q _ { 0 } ( \sigma ^ { \prime } ) = ( b _ { 0 } , \ldots , b _ { n + 1 } ) } \\ { \sum \textsf { h } _ { 0 } \ldots \underbrace { y _ { 0 } \sum \textsf { h } _ { 0 } } _ { \geq + \leq \leq \leq \leq n + 1 } + b _ { \mathrm { ~ a t } } = \sum \textsf { h } _ { i = \sigma } ( \sigma - \varepsilon _ { 0 } , d _ { 1 } \boxed { C ( i , \ldots , d _ { n - 1 } ) } } \\ { \underbrace { \sqrt { k _ { 0 } \ldots \sqrt { k _ { n + 1 } } , \ldots , N _ { n + 1 } , \ldots , N _ { n - 1 } } } _ { \geq + \varepsilon _ { 0 } } \leq \underbrace { k _ { n } \ldots , N _ { n - 1 } } _ { \geq + \sqrt { k _ { n } } } \ldots \underbrace { \sqrt { | \varepsilon _ { n } | } } _ { \geq \varepsilon _ { 0 } } \ldots \underbrace { | \sqrt { k _ { n } k _ { 1 } } + 1 } _ { \sqrt { 0 } } \ldots [ h _ { 0 } ] \ldots [ \sigma \middle | h ] \ldots [ \middle | h ] } _ { \leq \sigma } \quad \mathrm { ( \mathbb { R } \nabla \times \nabla \times \mathbb { R } _ { 0 } ) }  \\ { \underbrace { \mathrm { v } k , 0 \le k \le m } _ { \le \le \le \le \le k _ { \delta } } \sigma _ { \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le \le } } \\  \sum \textsf { e } q _ { 0 } \underbrace  \sum \textsf { e } q _ { 0 } \sum \sigma ( \sigma , \varepsilon _ { 0 } + \varepsilon _ { 0 } , \ldots , \sigma _ { n - 1 } , \sigma _ { n - 1 } , \sigma _ { n - 1 } , \sigma _ { n - 1 } , \sigma _  n \end{array}
$$

Figure 15. The rules for evaluating values of expressions at runtime, cont.

Lemma C.2 (Soundness of shape analysis for expressions). Given a program state Σ, a type environment Γ, and an expression node �, (∀ Variable $x . \Sigma \vdash x \downarrow _ { v } v  \Gamma \vdash x : _ { \varsigma } \varsigma \land \varsigma =$ $\alpha _ { s } ( v ) ) \implies ( \Sigma \vdash e \iint _ { v } v ^ { \prime }  \Gamma \vdash e : _ { S } \varsigma ^ { \prime } \wedge \varsigma ^ { \prime } = \alpha _ { s } ( v ^ { \prime } ) )$

Proof. We prove this lemma by structural induction.

Base case (1): � is an integral expression. This case is typed by the rule (S-Lit). There are two possible integral expressions: integer literals and shape access expressions. By the semantics of integer literals (RTV-Lit), integer literals evaluate to their literal value, which are scalars. By the semantics of shape access expressions (RTV-SAcc), such expressions

$$
\begin{array} { r l } { \frac { 1 } { \Gamma + \gamma } \left( \begin{array} { l l l l l l l l l } { \log \operatorname* { m a d } 1 } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & & & { \xi } & \\ { \Gamma + \xi } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & & { \sqrt { 1 + \xi } } & & { \sqrt { 1 + \xi } } \\ & & { \Gamma + \xi } & & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } \\ & & & { \Gamma + \xi } & & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } \end{array} \right) } & { \begin{array} { l l l l l l l } { \Gamma + \xi } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } \\ { \xi } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } \\ & & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } \end{array} } &  \begin{array} { l l l l l l l } { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } & { \sqrt { 1 + \xi } } &  \sqrt  \end{array} \end{array}
$$

Figure 16. Rules for analyzing shapes of expressions. $[ [ \mathrm { i } ] ] _ { \Gamma }$ means symbolically evaluating the integral expression i under environment Γ.

$$
\begin{array} { r }  \begin{array} { c c c c c c c c } { \underline { { \mathrm { I m a g e a l i } } } } & { \mathrm { V a r a t } } & { \mathrm { T e : } m } & { \mathrm { P r } + \underline { { e _ { 2 } m } } } & { 0 \leq k \leq n } & { f = \mathrm { U a k a t e } \underline { { p } } \varphi ( \mathrm { U b a r y } \varphi ) \mathrm { B i n a r y } \varphi } \\ { \mathrm { T e ~ t i s t . } } & { \mathrm { T e ~ } \mathrm { U s } = ( \zeta , m ) } & { \mathrm { ( W ~ v a r ) } } & { \frac { \Gamma \vdash \epsilon _ { 1 } \gamma _ { 2 } } { \Gamma \vdash \epsilon _ { 1 } \gamma _ { 2 } } \mathrm { ~ . ~ } } & { \mathrm { ( M : d a ) } } & { \frac { f = \mathrm { U d k e } \underline { { \mathrm { ~ R a s t e d } } } \mathrm { ~ . } \epsilon _ { 2 } \gamma _ { 2 } } { \Gamma \vdash \epsilon _ { 1 } \epsilon \epsilon _ { 1 } \epsilon _ { 2 } \gamma _ { 2 } } \mathrm { ~ . ~ } } & { \mathrm { ( M : d o r ) } } \\ { f \in \mathrm { ~ U n a v o p o p o p t b a n e y } } & { \mathrm { T e : } r } & { \mathrm { T e ~ t e } _ { 2 } } & { \mathrm { ~ . ~ } } & { \mathrm { T e ~ t : } m } \\ { \mathrm { P e ~ p a r a b e } \mathrm { ~ a s t a e g o p } } & { \mathrm { T e ~ } } & { \mathrm { P e ~ } \mathrm { ~ } \mathrm { , ~ } } & { \mathrm { P e ~ } \mathrm { ~ } } \\ { \mathrm { T ~ F e ~ f e ~ n a m } \mathrm { ~ . ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ } \mathrm { ~ I s ~ } \epsilon _ { 2 } \mu _ { 1 } } & { \mathrm { ~ } } & { \mathrm { T e ~ t e ~ } _ { 2 } \mathrm { ~ , ~ } } & { \mathrm { ( M : d r a ) } } & { \mathrm { ~ T e ~ t e ~ } \epsilon _ { 2 } \mathrm { ~ , ~ } } \\ { \mathrm { T ~ F e ~ f e ~ n } , \mathrm { ~ } \epsilon _ { n } \rangle \mathrm { ~ . ~ } \omega _ { n } } &  \mathrm  ~ U s p a d e \end{array} \end{array}
$$

Figure 17. Mask analysis for expressions. We use overlined symbols to represent comma-spliced sequences in code.

evaluate to a number in the shape tuples, which are scalars.   
So integral expressions always have the scalar type.

Base case (2): � is a variable. Γ stores the correct type of the variable by assumption.

Inductive case (1): � is an indexing expression of the form $e [ e _ { 0 } , \ldots , e _ { m } ]$ . The semantics of indexing expressions (RTV-Index) say the value to which � evaluates has a runtime shape $\left( a _ { n } , \ldots , a _ { 0 } \right)$ . This runtime shape is calculated by broadcasting the runtime shapes of the indexers together. The (S-Idx) rule also types � by broadcasting the static shapes of the indexers together. If we assume all the static shapes of the indexers correctly reflect the runtime shapes of the indexers, then the static shape of � correctly reflects the runtime shape of �.

Inductive case (2): � is a routine call, which is in the form of $f ( e _ { 1 } , \ \ldots , \ e _ { n } )$ . Similar to the last inductive case, the semantics of routine calls (RTV-Op) say the runtime shape of the returned value is the result of broadcasting all the runtime shapes of the operands. The static shape is calculated by broadcasting the static shapes of all the operands. This type inference rule is (S-Op). So if the static shapes of operands correctly reflect the runtime shapes of operands, the static shape � is also correct.

Inductive case (3): � is a reduction call of the form $f ( e _ { t } , c )$ The semantics of reduction operators (RTV-Rdce) say the returned value has a runtime shape $( a _ { 0 } , \dotsc , a _ { c - 1 } , a _ { c + 1 } , \dotsc , a _ { n } )$ assuming the runtime shape of $\boldsymbol { e _ { t } } ^ { \prime }$ s evaluation has the shape $( a _ { 0 } , \ldots , a _ { n } )$ . The static shape of � is inferred in the same way based on the static shape of $e _ { t }$ . This type inference rule is (S-Rdce). If we assume the static shape of $e _ { t }$ is correctly inferred, the static shape of � is correctly inferred.

Inductive case (4): � is call to expand\_dims or replicate. First, we discuss the base case, in which only one axis is added by the expand\_dims call or the replicate call. The (RTV-ExpBase) rule and the (RTV-RepBase) rule show the semantics of the calls in this case. The runtime shape of the returned value in this case has one more dimension added to the specific axis, and all the axes after the position at which the new dimension is added are shifted to one position to the right. If the call is expand\_dims, the newly added dimension has a dimensionality of one, and if the call is replicate, the newly added dimension has a dimensionality of the specified number. The static shape inference rules are (S-ExpBase) and (S-RepBase). As the static shape inference is done in the same way on the static shape of the input argument, as long as the static shape of the input argument correctly reflects the runtime shape of the input argument, the static shape correctly reflects the runtime shape of the expression. For situations where more than one dimension is to be expanded, the returned value is calculated by recursive calls with one less dimension to expand, as shown by rules (RTV-ExpInd) and (RTV-RepInd). The static shapes are inferred in the same recursive fashion, which is shown by rules (S-ExpInd) and (S-RepInd). As the base case correctly infers static shapes, any recursive calls should continue to correctly infer the static shapes.

Inductive case (5): � is a call to matmul. That is, � is of the form matmul $( e _ { 1 } , e _ { 2 } )$ . The semantics of matmul (RTV-Matmul) say that the returned runtime value has the runtime shape $( c _ { l } , \ldots , c _ { 2 } , a _ { 1 } , b _ { 0 } )$ when the runtime shape of $\boldsymbol { e _ { 1 } { \dot { s } } }$ evaluation is $( a _ { m } , \ldots , a _ { 0 } )$ and that of $e _ { 2 }$ is $( b _ { n } , \ldots , b _ { 2 } , a _ { 0 } , b _ { 0 } )$ Here, $( c _ { l } , \ldots , c _ { 2 } )$ is the result of broadcasting $\left( a _ { m } , \ldots , a _ { 2 } \right)$ and $\left( b _ { n } , \ldots , b _ { 2 } \right)$ . The static shape inference is calculated similarly on the static shapes of $e _ { 1 }$ and $e _ { 2 } .$ . So as long as the static shapes of the arguments are correctly reflecting the runtime shapes of the arguments, the static shape inferred for this expression correctly reflects the runtime shape of the expression.

Inductive case (6): � is a call to arange or ones. The semantics of the two calls, (RTV-Aran) and (RTV-Ones), say that the integral expressions used as the arguments to the call are first evaluated, and the returned value has a runtime shape specified by the evaluations ofthe integral expressions. The static typing rules (S-Aran) and (S-Ones) symbolically evaluate the integral expressions, and then use the symbols obtained from the symbolic evaluation to represent the static shape of the created array. For integer literals, the symbolic evaluation always returns the literal value, and for shape access expressions, symbolic evaluations return symbolic dimensionalities on the specified axes of the shapes of the argument expressions. If we assume the static shape of the argument to the shape access expression is correct, the symbolic evaluation ofthe shape access expression then correctly reflects its runtime value. The shape that is inferred based on the symbolic evaluation is also correct then.

Inductive case (7): � is a call to filled. The semantics of filled (RTV-Fill) say the runtime shape of the returned value is the same as the runtime shape of the first argument. It also requires the second argument to be a scalar. The static typing rule for the shape of filled, (S-Fill), also infers the shape of the call by the shape of its first argument. It also checks the static shape of the second argument to ensure its static shape is a scalar. So ifthe static shapes ofthe arguments are correctly inferred, the inferred static shape of � correct reflects �’s runtime shape. □

Lemma C.3 (Soundness of maskedness analysis for expressions). Given a program state Σ, a type environment Γ, and an ex<sub>p</sub>ression no<sup>d</sup>e $e , \ ( \forall \ V a r i a b l e x . \Sigma \ \vdash \ x \ \downarrow _ { \mu } \ \mu \ \to \ \Gamma \ \vdash$ $x : m$ � $\wedge \ m = \alpha _ { m } ( \mu ) ) \ \implies \ ( \Sigma \ \vdash \ e \ \Downarrow _ { \mu } \ \mu ^ { \prime } { } ^ { \curvearrowright } \stackrel { } { \to } \Gamma \ \vdash \ e \ : _ { m }$ 2 <sub>�</sub>′ $\wedge \alpha _ { m } ( \mu ^ { \prime } ) = m ^ { \prime } )$

Proof. This lemma may be proven by structural induction.

Base case (1): � is an integral expression. Because integral expressions’ runtime maskedness is always ⊙ by (RTM-Lit) and the (M-Lit) rule always types them as ⊥, the static type inference rule correctly infers the maskedness of �.

Base case (2): � is an identifier for a variable. The inferred type is correct by assumption.

Inductive cases: � is one of the following types of expression: indexing expression, unary or binary routine call, call to expand\_dims or replicate, call to matmul, call to a reduction operator, call to arange or ones, call to filled. Note that the static maskedness inference rules’ syntax can be directly mapped to that of the rules for calculating runtime maskedness. For all these expressions, if the arguments’ static maskedness is correctly inferred, the inferred static maskedness of the expression correctly reflects the runtime maskedness of the expression. □

Lemma C.4 (Soundness of type analysis for statements). Let Σ be a program state, Γ be a type environment and � be a statement. (∀ Variable $x . \Sigma \vdash x \Downarrow _ { v } v \land \Sigma \vdash x \Downarrow _ { \mu } \mu  \Gamma \vdash x : _ { \varsigma }$ $\begin{array} { r } { \varsigma \wedge \varsigma = \alpha _ { \varsigma } ( v ) \wedge \Gamma \vdash x : _ { m } m \wedge \alpha _ { m } ( \mu ) ) \implies ( ( \Sigma \vdash s \searrow \Sigma ^ { \prime } \wedge \Gamma \vdash \mu ) ) \lor \alpha ( \mu ) } \end{array}$ $s \hookrightarrow \Gamma ^ { \prime } ) \ \Longrightarrow \ ( \forall$ Variable � ${ . \Sigma ^ { \prime } } \vdash y \Downarrow _ { v } v ^ { \prime } \land \Sigma ^ { \prime } \vdash y \Downarrow _ { \mu } \mu ^ { \prime } $ $\Gamma ^ { \prime } \vdash y : _ { \varsigma } \varsigma ^ { \prime } \wedge \varsigma ^ { \prime } = \alpha _ { \varsigma } ( v ^ { \prime } ) \wedge \Gamma ^ { \prime } \vdash y : _ { m } m ^ { \prime } \wedge m ^ { \prime } = \alpha _ { m } ( \mu ^ { \prime } ) ) )$

Proof. This can be proven by structural induction on the statements in the DSL.

Base case (1): � is a skip statement. As the semantics of skip (SEM-Skp) say the evaluation environment is unchanged after the evaluation of this statement, the type inference rule (T-Skp) also does not change the type environment before typing this statement. So the correctness of the type environment is maintained.

Base case (2): � defines a variable, or in other words, binds a value from the evaluation of an expression to a variable. In this case, the semantics of the statement (SEM-Bind) say that the statement stores the runtime value and maskedness of the expression’s evaluation to the top scope of the evaluation environment. From Lemma C.2 and Lemma C.3, we know that with Γ we can correctly infer the static shape and maskedness of the expression on the right-hand side. The (T-Bind) rule maps the variable on the left-hand side to the static shape and maskedness of the expression on the right-hand side. Since all variables that can be evaluated under Σ before the execution of � have the correct types in $\Gamma ,$ and the newly added variable also has the correct type in Γ, Γ continues to correctly type all the variables that can be evaluated under Σ.

Base case (3): � is an update statement. The rules stipulating the semantics of update statements, which are (SEM-Upd) and (SEM-MskUpd), use two auxiliary rules (SEM-Auxupd1) and (SEM-Auxupd2), which specify how the runtime evaluation environments are changed. Note that the actual changes to the runtime evaluation environment are made by the (SEM-Auxupd1) rule, which does not change the runtime maskedness of the variable being updated. The updated value to be bound to the variable is specified by the (SEM-Upd) rule and the (SEM-MskUpd) rule. Both rules specify that the updated value has the same runtime shape as the original value. So the update statements do not change the runtime shape and maskedness of variables. The (T-Udp) rule also types update statements by directly returning the original type environment. So the correctness of the type environment is maintained.

Inductive case (1): � is an if-else branch. The semantics of such statements are given by (SEM-If). Before the execution of a branch, an empty scope is pushed onto the evaluation environment. All the statements in the scope are evaluated under this evaluation environment. The actual returned evaluation environment after the execution of the branch will have the top scope removed. Because the (SEM-Bind) rule says all the newly defined variables are stored in the top scope of the evaluation environment, the evaluation environment after the branch is executed cannot evaluate any new variable that cannot be evaluated under Σ. All the variables that can be evaluated under Σ can still be evaluated under Σ<sup>′</sup>, as no statement can explicitly remove variables from an evaluation environment, and no statement can remove a scope from the evaluation environment without adding one first. Also, because no statement can change the runtime shape and maskedness of variables in an evaluation environment, all the variables that can be evaluated under Σ still have the same runtime shape and maskedness when evaluated under Σ<sup>′</sup>. The (T-If) rule types an if-else branch by first typing the statement in the then-branch and using the updated type environment to type the statement in the else-branch. As no statement-level type inference rule can remove or update a mapping in the type environment, all the variables that can be typed by Γ will be typed the same under Γ<sup>′</sup>. Thus Γ<sup>′</sup> still correctly types all the variables that can be evaluated under Σ<sup>′</sup>. We also note that since Γ contains the correct types of all the variables that can be evaluated under the evaluation environment before the execution of the branch body, by the inductive hypothesis of the structural induction, Γ<sup>′</sup> can also correctly type variables in the branch body.

Inductive case (2): � is a for-loop. The rules (SEM-For0) and (SEM-Fori) specify the semantics of for-loops. For-loops are executed in iterations. Before the execution of the loop body for each iteration, a new scope containing a mapping from the loop variable to the iteration count is pushed onto the evaluation environment. The loop body is executed under the evaluation environment, and the top scope in evaluation is removed before the execution of the next iteration. For the same reasons in the previous inductive case, Σ<sup>′</sup>, the evaluation environment after the execution of the loop, can evaluate the same set ofvariables as Σ and Σ<sup>′</sup> cannot evaluate any new variable that cannot be evaluated under Σ. Also, for the reason discussed in the previous inductive case, all the variables that can be evaluated under Σ will have the same runtime shape and maskedness when evaluated under Σ<sup>′</sup>. Because the mappings in Γ cannot be altered or removed, Γ<sup>′</sup> still correctly type all the variables that can be evaluated under Σ<sup>′</sup>. We also note that before typing the loop body, Γ is updated to map the loop variable to the type of unmasked scalar, which reflects the runtime shape and maskedness of the loop variable, as specified by (SEM-Fori). So the updated Γ can correctly type all the variables that can be evaluated under the evaluation environment before the execution of the loop body, and by the inductive hypothesis for the structural inductive, Γ<sup>′</sup> also contains the correct types for the variables in the loop body.

Inductive case (3): � is a sequence of statements. The semantics of sequences (SEM-Seq) say that in executions of a sequence, the first statement in the sequence is executed first, and then the updated evaluation environment is used to execute the second statement. The (T-Seq) also first types the first statement in the sequence and then uses the updated type environment to type the second statement. By the inductive hypothesis of the structural induction, the updated type environment after typing the first statement in the sequence can correctly type all the variables that can be evaluated under the evaluation environment after the execution of the first statement. Then the updated type environment after typing the whole sequence can correctly type all the variables that can be evaluated under the evaluation environment after the execution of the whole sequence. □

Now, we back to the proof of Theorem C.1.

Proof. By Lemma C.4, Γ can correctly type any variable that can be evaluated under the evaluation environment before the return of the program. By the semantics of programs (SEM-Prog), a program can only return the value ofa variable that can be evaluated under the evaluation environment before the return. Thus the type of the returned variable inferred under Γ must be correct. □

## D Rewrite Rules

Figure 18 and Figure 19 show the complete set of rules for rewriting statements and expressions. Some of the rules are already included in Figure 7 and Figure 8 in Section 5. The additional rewrite rules included in Figure 18 and Figure 19 follow the informal descriptions of the rewrite procedure described in Section 5.

The (R-Bind2) rule states that nothing needs to be changed, and the type environment does not need to be updated if the right-hand side of a bind statement is not lifted after rewrite and the variable being bound does not depend on the current loop variable. The (R-Bind3) rule explains how to do bind-site replication to eliminate pseudo-dependence. The premise for such rewrites is that the right-hand side of a bind statement depends on the current loop variable but is not lifted after the rewrite. Depending on whether the statement is in a branch, we need to explicitly replicate the defining expression either by calling replicate or calling make\_masked and let the broadcasting mechanism handle the replication. The (R-Updt) rule states how to rewrite update statements that are not rewritable reduce statements. It first checks that the update statement is not a rewritable reduce statement and then rewrites the left-hand side and the right-hand side of the statement. Because the right-hand side expression may be implicitly broadcast to the shape of the left-hand side expression, we need to ensure that if the right-hand side is lifted, the axes that were broadcast before the rewrite are still properly broadcast after the rewrite. Thus, we may need to expand new axes on the right-hand side expression, depending on the shapes of the expressions before and after the rewrite. This part of the rewrite is similar to the handling of binary operators. The (R-Brch2) rule handles the cases where the branch condition is not lifted after the rewrite. As discussed in Section 5.4, in this case, we simply rewrite the branch bodies of the then-branch and the else-branch and put them back into the corresponding branches, maintaining the branch structure. Finally, the (R-Rstmt) rule specifies how to rewrite rewritable reduce statements formally. We first check if an update statement is indeed a rewritable reduce statement. Then we rewrite the expression on which the reduction is made. Depending on whether the statement is in a branch and if the reduced expression is lifted after the rewrite, we handle the statement diferently. If the reduced expression is not lifted after the rewrite, we need to explicitly create a new axis on which the reduction is made. Depend ing on whether the statement is in a branch, we may create this new axis by calling replicate or calling make\_masked. Then we can just replace the reduced expression to call to the corresponding reduction operator on the reduced expression. Also, as discussed in Section 5.4, we need to fill the reduced expression with the identity value of the reduction operator if the statement is in a branch to ensure that the result of the reduction is not a masked value.

The rule (R-Rdce) specifies how to rewrite ������ operators. Because such operators operate on a specified axis, we need to ensure that the reduction is made on the same axis after the rewrite. So if the reduced expression is lifted, we need to increment the argument specifying the reduced axis by one. The (R-Idxr) rule and the (Unmask) rule state how indexing expressions are rewritten. Aligned with the informal descriptions in Section 5, the (R-Idxr) rule states that if the indexers are broadcast before the rewrite and some of the broadcast indexers are lifted after the rewrite, we expand these indexers to ensure the broadcast is still properly applied on the right axes. The (Unmask) rule specifies how to deal with indexers that become masked after the rewrite by extracting the mask from these masked indexers and wrapping the whole indexing expression in a call to make\_masked. The conjunction of the extracted masks is used as the mask to the make\_masked call. The (R-Var) rule specifies how to handle two cases where the appearance of a variable needs to be replicated. The first case is when a variable that is not loop-local escapes the restrictions from the branch condition when the branch is flattened. The second case is when a loop-local variable that depends on the current loop variable is used in a branch deeper than the branch in which it is defined. For both cases, we need to wrap the appearance of the variable in a make\_masked call to mask them by the mask corresponding to the current branch. By doing so, we ensure values corresponding to the iterations where the variable’s appearance is not evaluated are masked. The (R-Uop) rule and the (R-Fill) rule are simple as the operators are unary element-wise operators. If the operand is lifted, the operators are still applied element-wise on the newly added axis, propagating the newly added axis. The (R-Shape) rule and the (R-Rep) rule are similar to the (R-Exp) rule, incrementing the arguments specifying the target axes so the operators are applied to the same axes. The (R-Int) rule and the (R-Init) rule state that the integer literals and the array initialization calls are never changed, as there is no argument that may be lifted by rewrites.

## E Proof of Correctness of Rewrite Rules

We can state the soundness of loop-level rewrites as a lemma.

Lemma E.1 (Soundness of loop rewrites). Suppose that we h<sub>ave</sub> <sub>rewr</sub>it<sub>a</sub>bl<sub>e</sub> l<sub>oop</sub> $\mathcal { L } = \mathsf { f o r } x \mathrm { i n } 0 \ldots \mathrm { i n } 0 \ldots \mathrm { i n }$ do � in a pro<sub>g</sub>ram $\mathcal { P }$ and � does not contain an<sub>y</sub> loop. Let Γ be the t<sub>y</sub>pe environment inferredfrom P. Let Σ be an evaluation environment. If $\Gamma , \Gamma , \emptyset \vdash \mathcal { L } \ \sim \ s ^ { \prime } , \Gamma _ { a } ^ { \prime }$ <sub>an</sub>d $\mathopen { } \mathclose \bgroup  \Sigma \vdash \mathcal { L } \searrow \sum \prime$ <sub>,</sub> th<sub>en</sub> $\Sigma \vdash s ^ { \prime } \searrow \Sigma ^ { * }$ <sub>an</sub>d ∀ Variable $y . \Sigma ^ { \prime } \vdash y \Downarrow _ { v } v  \Sigma ^ { * } \vdash y \Downarrow _ { v } v .$

We want to prove Lemma E.1. The proof has two parts. First, we want to prove that for a loop with no loop-carried dependence at all, the assertion on the rewrite result is true. Then we prove that the rule for rewriting rewriteable reduce statements is valid while maintaining the correctness of other rewrite rules.

To prove the first part, we cite a theorem proven by [25] stating that “it is valid to convert a sequential loop to a parallel loop if the loop carries no dependence.” If the loop can be executed in parallel, then clearly each statement in the loop can be executed in parallel. We only need to prove that the rewrite rules transform each statement into a new statement that is equivalent to executing the original statement at each iteration of the loop in parallel. In the context of our DSL, we define “the parallel evaluation of $\cdot \ e ^ { \mathfrak { w } }$ and “the parallel execution of $s ^ { \mathfrak { n } }$ as follows.

Definition E.2 (Parallel evaluation of an expression). Assume that $\mathcal { L } = \mathsf { f o r } x \mathrm { i n } 0 \ldots \mathrm { i n } 0 \ldots \mathrm { i n }$ do � is the innermost loop in a program $\mathcal { P }$ . Let � be an expression node in �. Let Σ be an evaluation environment such that $\Sigma \vdash \mathcal { L } \setminus \Sigma ^ { \prime }$ and $\textstyle \sum \vdash \vdash \bigcup \psi _ { v } ~ c$

Γ ⊢ � : Γ , Γ , Δ ⊢ � { �<sup>′</sup> Γ<sub>�</sub> ⊢ � : <sub>�</sub> Γ<sub>�</sub>, Γ , Δ ⊢ � { �<sup>′</sup>   
Γ<sub>�</sub>, Γ , {� ↦→ arange(i) } ⊢ � ↷ �<sup>′</sup>, Γ<sup>′</sup> Γ<sub>�</sub> ⊢ �<sup>′</sup> :<sub>� �</sub><sup>′</sup> Γ<sub>�</sub> ⊢ �<sup>′</sup> :<sub>�</sub> �<sup>′</sup> <sub>�</sub> ≠ <sub>�</sub><sup>′</sup> Γ<sub>�</sub> ⊢ �<sup>′</sup> : <sub>�</sub><sup>′</sup> <sub>�</sub> = <sub>�</sub><sup>′</sup>   
Γ<sub>�</sub>, Γ , Δ ⊢ for � in 0 . . .i do � ↷ �<sup>′</sup>, Γ<sup>′</sup> (R-For) Γ<sup>′</sup> = Γ<sub>�</sub> [� ↦→ (<sub>�</sub><sup>′</sup>, �<sup>′</sup> ) ] (R-Bind1) LoopVar(Δ) = <sub>� �</sub> ∉ DependsOn(�) (R-Bind2)   
Γ , Γ , Δ ⊢ � := � ↷ � := �<sup>′</sup>, Γ<sup>′</sup> Γ , Γ , Δ ⊢ � := � ↷ � := �, Γ   
$\Gamma _ { b } \vdash e : _ { \zeta } \ \varsigma \quad \Gamma _ { b } , \Gamma _ { a } , \Delta \vdash e \ \sim \ e ^ { \prime } \quad \Gamma _ { a } \vdash e ^ { \prime } : _ { \zeta } \ \varsigma ^ { \prime } \quad \varsigma = \varsigma ^ { \prime } \quad \Gamma _ { a } \vdash e ^ { \prime } : _ { m } m ^ { \prime }$   
Γ<sub>�</sub>, Γ , Δ ⊢ � ↷ �<sup>′</sup>, Γ<sup>′</sup> $\varsigma ^ { \prime } = \left( a _ { n } , \ldots , a _ { 1 } \right)$ LoopVar $( \Delta ) = y \quad y \in$ DependsOn(� ) LoopVarSub(Δ) = �<sub>�</sub>   
<sup>�</sup>   <sup>1</sup>  <sup>1</sup> <sup>�</sup>Γ<sub>�</sub> , Γ<sup>′</sup><sub>�</sub>, Δ ⊢ �<sub>2</sub> ↷ �<sup>′</sup><sub>2</sub>, Γ<sup>′′</sup><sub>�</sub> Γ ⊢ � : (�, ) Γ ⊢ � : � $\begin{array} { r } { e ^ { \prime \prime ^ { \circ } } = \bar { \mathsf { I T E } } ( m _ { s } = \top , e _ { m } ^ { \prime \prime } , } \end{array}$ replicate(�<sup>′</sup>, (0, ), (�, ) ) )   
(R-Seq) �<sup>′′</sup><sub>�</sub> = make\_masked(�<sup>′</sup>, expand\_dims(logical\_not(getmaskarray(�<sub>�</sub> ) ), (1, . . . , �) ) )<sub>′′</sub> <sub>′</sub>   
Γ<sub>�</sub> , Γ<sub>�</sub>, Δ ⊢ �<sub>1</sub> ; �<sub>2</sub> ↷ �<sup>′</sup><sub>1</sub> ; �<sup>′</sup><sub>2</sub>, Γ<sup>′′</sup><sub>�</sub>   
(R-Bind3)   
Γ<sub>�</sub>, Γ , Δ ⊢ � := � ↷ � := �<sup>′′</sup>, Γ [�<sup>′′</sup> ↦→ (� :: <sub>�</sub>, �<sup>′′</sup> ) ]   
¬IsRewritableRdce(�<sub>�</sub> ← �<sub>�</sub> ) Γ<sub>�</sub>, Γ<sub>�</sub>, Δ ⊢ �<sub>�</sub> { �<sup>′</sup><sub>�</sub> Γ<sub>�</sub>, Γ<sub>�</sub>, Δ ⊢ �<sub>�</sub> { �<sup>′</sup><sub>�</sub> Γ<sub>�</sub> ⊢ �<sub>�</sub> :<sub>� ��</sub> Γ<sub>�</sub> ⊢ �<sub>�</sub> :<sub>� �� ��</sub> = (�<sub>�+�</sub>, . . . , �<sub>0</sub> ) <sub>��</sub> = (�<sub>�</sub>, . . . , �<sub>0</sub> )   
Γ<sub>�</sub> ⊢ �<sup>′</sup> :<sub>�</sub> �<sup>′</sup> Γ<sub>�</sub> ⊢ �<sup>′</sup><sub>�</sub> :<sub>�</sub> �<sup>′</sup><sub>�</sub> �<sup>′</sup> = ITE(�<sub>�</sub> ≠ �<sup>′</sup><sub>�</sub> ∧ � ≥ 1, �<sup>′</sup> ← expand\_dims(�<sup>′</sup><sub>�</sub>, (1, . . . , �) ), �<sup>′</sup> ← �<sup>′</sup><sub>�</sub> )   
(R-Updt)   
Γ<sub>�</sub>, Γ<sub>�</sub>, Δ ⊢ �<sub>�</sub> ← �<sub>�</sub> ↷ �<sup>′</sup>, Γ<sub>�</sub>   
Γ<sub>�</sub>, Γ<sub>�</sub>, Δ ⊢ �<sub>�</sub> { �<sup>′</sup> Γ<sub>�</sub> ⊢ �<sub>�</sub> : <sub>��</sub> Γ<sub>�</sub> ⊢ �<sup>′</sup> : <sub>�</sub><sup>′</sup> <sub>�</sub><sup>′</sup> ≠ <sub>��</sub> LoopVarSub(Δ) = �<sub>�</sub> LoopVar(Δ) = <sub>�</sub>   
Γ<sub>�</sub>, Γ<sub>�</sub>, Δ[<sub>�</sub> ↦→ make\_masked(�<sub>�</sub> , �<sup>′</sup><sub>�</sub> ) ] ⊢ �<sub>�</sub> ↷ �<sup>′</sup><sub>�</sub>, Γ<sup>′</sup><sub>�</sub> Γ<sub>�</sub>, Γ<sup>′</sup><sub>�</sub>, Δ[<sub>�</sub> ↦→ make\_masked(�<sub>�</sub> , logical\_not(�<sup>′</sup><sub>�</sub> ) ) ] ⊢ �<sub>�</sub> ↷ �<sup>′</sup><sub>�</sub>, Γ<sup>′′</sup><sub>�</sub>   
(R-Brch1)   
Γ<sub>�</sub>, Γ<sub>�</sub>, Δ ⊢ if �<sub>�</sub> then �<sub>�</sub> else �<sub>�</sub> ↷ �<sup>′</sup><sub>�</sub>; �<sup>′</sup><sub>�</sub>, Γ<sup>′′</sup><sub>�</sub>   
Γ<sub>�</sub>, Γ<sub>�</sub>, Δ ⊢ �<sub>�</sub> { �<sup>′</sup><sub>�</sub> Γ<sub>�</sub> ⊢ �<sub>�</sub> :<sub>�</sub> <sub>��</sub> Γ<sub>�</sub> ⊢ �<sup>′</sup><sub>�</sub> :<sub>�</sub> <sub>�</sub><sup>′</sup><sub>� �</sub><sup>′</sup><sub>�</sub> = <sub>��</sub> Γ<sub>�</sub>, Γ<sub>�</sub>, Δ ⊢ �<sub>�</sub> ↷ �<sup>′</sup><sub>�</sub>, Γ<sup>′</sup><sub>�</sub> Γ<sub>�</sub>, Γ<sup>′</sup><sub>�</sub>, Δ ⊢ �<sub>�</sub> ↷ �<sup>′</sup><sub>�</sub>, Γ<sup>′′</sup><sub>�</sub>   
(R-Brch2)   
Γ<sub>�</sub>, Γ<sub>�</sub>, Δ ⊢ if �<sub>�</sub> then �<sub>�</sub> else �<sub>�</sub> ↷ if �<sup>′</sup><sub>�</sub> then �<sup>′</sup><sub>�</sub> else �<sup>′</sup><sub>�</sub>, Γ<sup>′′</sup><sub>�</sub>   
IsRewritableRdce(�<sub>�</sub> ← <sub>�</sub>(�<sub>�</sub>, �<sub>�</sub> ) ) Γ<sub>�</sub> ⊢ �<sub>�</sub> :<sub>� ��</sub> Γ<sub>�</sub>, Γ<sub>�</sub>, Δ ⊢ �<sub>�</sub> { �<sup>′</sup> Γ<sub>�</sub> ⊢ �<sup>′</sup> :<sub>� �</sub><sup>′</sup> <sub>�</sub><sup>′</sup> = (�<sub>1</sub>, . . . , �<sub>�</sub> ) LoopVarSub(Δ) = �<sub>�</sub> Γ<sub>�</sub> ⊢ �<sub>�</sub> :<sub>�</sub> �   
�<sup>′′</sup> = replicate(�<sup>′</sup> , (0, ), (S(�<sub>�</sub> ) [0], ) ) �<sub>1</sub> = � (�<sup>′</sup> , 0) �<sub>2</sub> = � (�<sup>′′</sup>, 0) �<sub>3</sub> = � (filled(�<sup>′</sup> , �<sub>��</sub> ), 0) �<sub>4</sub> = � (filled(make\_masked(�<sup>′′</sup>, �<sub>�</sub> ), �<sub>��</sub> ), 0)   
�<sub>�</sub> = expand\_dims(logical\_not(getmaskarray(�<sub>�</sub> ) ), (1, . . . , �) ) �<sub>�</sub> = ITE(� = ⊥, ITE(<sub>��</sub> ≠ <sub>�</sub><sup>′</sup> , �<sub>1</sub>, �<sub>2</sub> ), ITE(<sub>��</sub> ≠ <sub>�</sub><sup>′</sup> , �<sub>3</sub>, �<sub>4</sub> ) ) �<sup>′</sup> = �<sub>�</sub> ← <sub>�</sub>(�<sub>�</sub>, �<sub>�</sub> )   
� = {+ ↦→ sum, − ↦→ sum, ∗ ↦→ prod, / ↦→ prod, maximum ↦→ max, minimum ↦→ min} (<sub>�</sub>)   
�<sub>��</sub> = {+ ↦→ 0, − ↦→ 0, ∗ ↦→ 1, / ↦→ 1, maximum ↦→ −∞, minimum ↦→ ∞} (<sub>�</sub>)   
(R-Rstm   
Γ<sub>�</sub>, Γ<sub>�</sub>, Δ ⊢ �<sub>�</sub> ← <sub>�</sub> (�<sub>�</sub>, �<sub>�</sub> ) ↷ �<sup>′</sup>, Γ<sub>�</sub>  
Figure 18. Statement-level rewrite rules. LoopVar(Δ) returns the loop variable of the current loop. LoopVarSub(Δ) returns the expression that the current loop variable maps to in Δ. IsRewritableRdce(�) checks if the top-level operator on the right-hand side is one of ${ \dot { } } + , - , * , / ,$ maximum, minimum, the first operand of the right-hand side is the same as the update target, the variable being updated is not defined in the loop being rewritten, and the left-hand side does not depend on the current loop variable or any restricted loop variable.

Le ${ \mathfrak { k } } { \bar { \mathbf { j } } } \subseteq \{ 0 , \dots , c { - } 1 \}$ be a set of integers such that during an iteration that � evaluates to $n , n \in { \bar { \mathrm { j } } } ,$ , � is evaluated. Suppose $n \in \bar { \mathrm { j } }$ and let $\Sigma _ { n }$ be the updated Σ used to evaluate � during the iteration when � evaluates to � such that $\textstyle \sum _ { n } \vdash e \bigcup _ { v } v _ { n } .$

The parallel evaluation of� with respect to $\Sigma , \mathcal { L }$ is defined diferently depending on whether � is an identifier for a variable defined outside of L.

If � is not an identifier for a variable defined outside of $\mathcal { L } ,$ , then the parallel evaluation of � with respect to $\Sigma , \mathcal { L }$ is either $v _ { n }$ for any $n \in { \bar { \mathrm { j } } } ,$ , or the array $( v _ { 0 } ^ { \prime } , \ldots , v _ { c - 1 } ^ { \prime } )$ . Here, $v _ { k } ^ { \prime } = v _ { k } { \mathrm { i f } } k \in { \overline { { j } } }$ and $v _ { k } ^ { \prime } = v _ { k } ^ { \otimes }$ , an array of the shape of � filled with ⊗, otherwise. If the parallel evaluation of � is $v _ { n } ,$ then ∀�, $b . a , b \in \overline { { j } } \to v _ { a } = v _ { b }$ . If � is an identifer for a variable depending on the loop variable of $\mathcal { L } ,$ the parallel evaluation of � must be of the form $( v _ { 0 } ^ { \prime } , \ldots , v _ { c - 1 } ^ { \prime } )$

If� is an identifier for a variable defined outside of ${ \mathcal { L } } ,$ then the parallel evaluation of � with respect to $\Sigma , \mathcal { L }$ is the result of merging the updates on the variable in each iteration from the statements before �. That is, suppose in iteration �, the statements before � in the loop body update $\boldsymbol { e } ^ { \prime } \boldsymbol { s }$ array cells $\overline { { k } } _ { n }$ to values ${ \overline { { v } } } _ { n }$ . Suppose $\Sigma \vdash e \Downarrow _ { v } v _ { e } .$ The parallel evaluation of � with respect to $\Sigma , \mathcal { L }$ is then $v _ { e } [ \overline { { k } } _ { 0 } \longmapsto \overline { { v } } _ { 0 } ] \ldots [ \overline { { k } } _ { c - 1 } \mapsto \overline { { v } } _ { c - 1 } ]$

Definition E.3 (Parallel execution of a loop). Suppse that $\mathcal { L } = \mathsf { f o r } x \mathrm { i n } 0 \ldots \mathrm { i n } $ do � is the innermost loop in a program

P. Let Σ be an evaluation environment such that $\Sigma \vdash \mathcal { L } \setminus \Sigma ^ { \prime }$ and $\Sigma \vdash \textsf { i } \Downarrow _ { v } c$

The parallel execution of � with respect to $\Sigma , \mathcal { L }$ results in $\Sigma ^ { * }$ , which has the following properties.

• If� is a bind statement of the form $y : = e$ , then we have $\mathsf { P e e k } ( \Sigma ^ { * } ) ( y ) = v$ , where � is the parallel evaluation of� with respect to Σ, L. If� depends on the loop variable of $\mathcal { L } ,$ � must be an array of values to which � evaluates in each iteration.

• If � is an update statement of the form $e _ { l } \gets e _ { r } ,$ , which updates array cells of a variable � selected by $e _ { l }$ with values specified by $e _ { r } ,$ then $\Sigma ^ { * } \vdash y \downarrow _ { v }$ � such that � has all the array cells selected by $e _ { l }$ in each iteration updated to the values that are from the evaluation of $e _ { r }$ in each iteration.

• If � is a sequence of the form $s _ { 1 } ; s _ { 2 }$ , then $\Sigma ^ { * }$ should be the result of parallel execution of �<sub>2</sub> with respect to $\Sigma ^ { * * }$ and loop for � in $0 \ldots \mathrm { i }$ do $s _ { 2 } .$ Here, $\Sigma ^ { * * }$ is the result of the parallel execution of $s _ { 1 }$ with respect to Σ, for � in $0 \ldots \mathrm { i }$ do �<sub>1</sub>.

• If � is a branch of the form $\mathrm { i } \mathsf { f } \ e$ then $s _ { t }$ else $s _ { e } ,$ , then the parallel execution of� with respect to $\Sigma , \mathcal { L }$ is either – the parallel execution of $s _ { t } ; s _ { e }$ with respect to $\Sigma , \mathcal { L }$ or

$\mathsf { P o p } ( \Sigma ^ { * * * } )$ . Here, the evaluation environment $\Sigma ^ { * * * }$ is the result of parallel execution of $s _ { e }$ with respect to

$$
\begin{array} { r l } &  \begin{array} { r l } & { \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \sin \theta , \quad \sin \theta , \cos \theta } \\ & { - \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \sin \theta , } \\ & { \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } } \\ & { - \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \cos \theta } \frac { \sin \theta } { \sin \theta } } \\ & { \frac { \sin \theta } { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \cos \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \cos \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \cos \theta } \frac { \sin \theta } { \sin \theta } } \\ &  \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta } { \sin \theta } \frac { \sin \theta }  \cos \theta \end{array} \end{array}
$$

Figure 19. Rewrite rules for expressions. R-Biop applies to matmul if operands are not masked after rewrites. Integer literals, ones, and arange are never changed in rewrites. The rewrite rules for replicate and shape access are similar to that of expand\_dims. These rules are omitted here. LoopVar(Δ) and LoopVarSub(Δ) means getting the current loop variable and its substitute from Δ. DefiningLoopVar(�) returns the loop variable of the loop in which � is defined. OnLHS(�) checks if � is on the left-hand side of an update. DefiningBranch(�) ≠ CurBranch(Δ) checks that the current branch level we are trying to flatten is not the same branch level defining �.

$\mathsf { P u s h } ( \emptyset , \mathsf { P o p } ( \Sigma ^ { * * } ) )$ and for $x$ in $0 \ldots \mathrm { i }$ do $s _ { e }$ and $\Sigma ^ { * * }$ is the parallel execution result of $s _ { t }$ with respect to Push(∅, Σ) and for � in $0 \ldots \mathrm { i }$ do $s _ { t } .$

Given the two definitions above, we can more formally restate the theorem by[25] with respect to our DSL.

Theorem E.4. Let $\mathcal { L } = \mathsf { f o r } x \mathrm { i n } 0 \ldots \mathrm { i n } $ do � be the innermost loop in a pro<sub>g</sub>ram P. Let Σ be an evaluation environment such th<sub>a</sub>t $\Sigma \vdash \mathcal { L } \setminus \Sigma ^ { \prime }$ . L does not have an<sub>y</sub> loo<sub>p</sub>-carried de<sub>p</sub>endence when executed under Σ. Then, the parallel execution of� with respect to Σ and L results in $\Sigma ^ { * }$ such thatfor any variable $y , \Sigma ^ { \prime } \vdash y \Downarrow _ { v } v  \Sigma ^ { * } \vdash y \Downarrow _ { v } v .$

Now we just need to prove the following two lemmas.

Lemma E.5. Let $\mathcal { L } = \mathsf { f o r } x \mathrm { i n } 0 \ldots \mathrm { i n } 0 .$ do � be the innermost <sup>l</sup>oop <sup>i</sup>n a program $\mathcal { P }$ whose t<sub>y</sub>pe environment is Γ. Let � be an expression node in �. Let Σ be an evaluation environment <sub>suc</sub>h th<sub>a</sub>t $\Sigma \vdash \mathcal { L } \setminus \Sigma ^ { \prime }$ <sub>an</sub>d $\Sigma \vdash \textsf { i } \Downarrow _ { v } c . \mathcal { L }$ d<sub>oes</sub> <sub>no</sub>t h<sub>ave</sub> <sub>any</sub> loo -carried de endence when executed under Σ. Assume that $\Gamma , \Gamma , \emptyset \vdash \mathcal { L } \ : \sim \ : s ^ { \prime }$ <sub>en</sub>t<sub>a</sub>il<sub>s</sub> $\Gamma _ { b } , \Gamma _ { a } , \Delta \vdash e \ \sim \ e ^ { \prime }$ <sub>an</sub>d th<sub>a</sub>t $\Sigma ^ { * }$ is <sub>an eva</sub>l<sub>ua</sub>ti<sub>on env</sub>i<sub>ronmen</sub>t th<sub>a</sub>t <sub>eva</sub>l<sub>ua</sub>t<sub>es eac</sub>h th<sub>e var</sub>i<sub>a</sub>bl<sub>e</sub> appearin<sub>g</sub> in � to its parallel evaluation with respect to Σ, L. Th<sub>en,</sub> $\Sigma ^ { * } \vdash e ^ { \prime } \Downarrow _ { v } \tau$ such that � is the parallel evaluation of� wit<sup>h</sup> res<sub>p</sub>ect to $\Sigma , \mathcal { L }$

Lemma E.6. Let $\mathcal { L } = \mathsf { f o r } x \mathrm { i n } 0 \ldots \mathrm { i n } 0 \ldots \mathrm { i n }$ do � be the innermost <sup>l</sup>oop <sup>i</sup>n a program $\mathcal { P }$ whose type environment is Γ. Let Σ be an <sub>eva</sub>l<sub>ua</sub>ti<sub>on env</sub>i<sub>ronmen</sub>t <sub>suc</sub>h th<sub>a</sub>t $\Sigma \vdash \mathcal { L } \setminus \Sigma ^ { \prime }$ <sub>an</sub>d $\textstyle \sum \vdash \ i \ \Downarrow _ { v } c .$ $\mathcal { L }$ d<sub>oes no</sub>t h<sub>ave any</sub> l<sub>oop-carr</sub>i<sub>e</sub>d d<sub>epen</sub>d<sub>ence w</sub>h<sub>en execu</sub>t<sub>e</sub>d under Σ. $I f \Gamma , \Gamma , \emptyset \vdash \mathcal { L } \ \sim \ s ^ { \prime } , \Gamma ^ { \prime }$ <sub>,</sub> th<sub>en</sub> $\Sigma \vdash s ^ { \prime } \searrow \Sigma ^ { * }$ <sub>suc</sub>h th<sub>a</sub>t $\Sigma ^ { * }$ is the result of the parallel execution of � with respect $\Sigma , \mathcal { L }$

For any variable � that can be evaluated under Σ<sup>∗</sup> to a value $v , \Gamma ^ { \prime }$ correctly maps � to the type of �.

Lemma E.5 is proven by structural induction.

Proof. Base case (1): � is an integer literal. Then � evaluates to the same value always. The rule (R-Int) returns the same $e ,$ which always evaluates to the same value. This value is the parallel evaluation of �.

Base case (2): � is an identifer of a variable. There are four sub-cases then.

Base case (2.1): The first one is that � is the loop variable of ${ \mathcal { L } } ,$ which is handled by the rule (R-LVar). In this case � is rewritten to the loop variable substitute currently recorded in $\Delta .$ There are only two rules changing the loop variable substitute stored in $\Delta ,$ which are (R-Brch1) and (R-For). The rule (R-For) sets the substitute to a call to arange which is evaluated to an array of all the values the loop variable can take in each iteration. The rule (R-Brch1) sets the substitute to a call to make\_masked, masking the previous loop variable substitute with the condition of the branch being flattened. In this case the loop variable substitute will be evaluated to an array of all the values that the loop variable takes in each iteration, but the positions corresponding to iterations not evaluating � are evaluated to a $\otimes$ value. Under $\Sigma ^ { * }$ , both possibilities of the loop variable substitute will be evaluated to the parallel evaluation of �.

Base case (2.2): The second sub-case is that � is a variable defined outside of $\mathcal { L }$ and the intersection ofthe loop variables on which � depends and the loop variables on which the current loop variable substitute stored in Δ depends is non empty. That is, the branch of � puts restrictions on at least one of the loop variables on which � depends. The (R-Var) rule handles this case. To rewrite �, it masks � by the mask extracted from the current loop variable substitute. Because the loop variable substitute is always a 1-D array, its mask is also a 1-D array. Being a variable defined outside of $\mathcal { L } ,$ � does not depend on the loop variable of $\mathcal { L }$ and always evaluates to the same value in each iteration of $\mathcal { L } .$ . The evaluation of � under $\Sigma ^ { * }$ is just this value. When we expand dimensions of the extracted mask on the axes $( 1 , \ldots , n )$ , where � is the number of dimensions of � originally, and use it to mask $e ,$ the broadcasting semantics duplicates the value to which � evaluates under $\Sigma ^ { * }$ on the left-most axis. So under $\Sigma ^ { * }$ , the rewrite result evaluates to an array of values that � evaluates to in each iteration. The locations corresponding to iterations for which � is not evaluated are fully masked. So under $\Sigma ^ { * }$ $e ^ { \prime }$ evaluates to the parallel evaluation of � with respect to $\Sigma , \mathcal { L }$

Base case (2.2): The third sub-case is that � is a variable defined in $\mathcal { L }$ and depends on the loop variable of $\mathcal { L } .$ However, the occurrence of � is in a branch that is deeper than the scope in which it is defined. This case is also handled by (R-Var). Because � is defined in $\mathcal { L }$ and depends on the loop variable of $\mathcal { L } , \Sigma ^ { * }$ evaluates � to an array of values that � evaluates to in each iteration. To mask elements in this array that correspond to the iterations in which � is not evaluated, we do similarly as in the case (2.3). In this case, the evaluation of � under $\Sigma ^ { * }$ already has one more dimension on the left compared to its evaluation in each iteration so we just need to expand the dimension of the extracted mask on axes $\left( 1 , \ldots , n - 1 \right)$ . Now the rewrite result is an array that � evaluates to in each iteration and the elements corresponding to the iterations not evaluating � are masked. This is the parallel evaluation of � with respect to Σ, L.

Base case (2.4): The last sub-case is all the cases not handled by the previous sub-cases. Rule (R-Var) handles this sub-case by not making any changes in the rewrite result. There are three possibilities of what happens in this sub-case. The first one is that � is a variable defined outside of $\mathcal { L }$ and is not restricted by the condition of the current branch. The second possibility is that � is defined in $\mathcal { L }$ but it does not depend on the loop variable or any restricted loop variable of an outer loop. The third possibility is that � is defined in $\mathcal { L } ,$ , depends on the loop variable and the definition of � is in the same branch level as �. Because it is in the same branch level as its defining statement, all the entries corresponding to unevaluated iterations must have been correctly masked. So in all three possibilities, � does not need to be changed in the rewrite process and evaluates to its parallel evaluation under $\Sigma ^ { * }$ by assumption.

Inductive case (1): � is a shape access expression of the form $\mathsf { S } ( e _ { t } ) [ c ]$ . The (R-Shape) rule handles this case. Under $\Sigma ^ { * }$ , the result of rewriting $e _ { t }$ will evaluate to the parallel evaluation of $\boldsymbol { e } _ { t } ,$ which may be of the same shape or have one more dimension on the left-most axis, by the definition of parallel evaluation. If it has the same shape, then the rule does not change anything to rewrite the expression as the parallel evaluation of the expression should read the length of the same axis as before. If it does not have the same shape, which means one more dimension is added on the left-most axis, the index of the axis whose length we are trying to read should be one greater than the index before the rewrite. In either case, under $\Sigma ^ { * }$ , the evaluation of the rewritten expression always equals the same value as we are reading in each iteration of the loop. This is the parallel evaluation of a shape access expression with respect to $\Sigma , { \mathcal { L } } .$

Inductive case (2): � is an array creating expression in the form of ones $( \mathfrak { i } _ { 1 } , \ldots , \mathfrak { i } _ { n } )$ or arange(i). Note that an integral expression always evaluates to the same value in each iteration of $\mathcal { L }$ and the rewrite rules always rewrite them to an expression that evaluates to this value under $\Sigma ^ { * }$ . The (R-Init) rule rewrites array creating routines without changing anything. So the rewrite result evaluates to the same value under $\Sigma ^ { * }$ as � does in each iteration. This is a parallel evaluation result of � with respect to $\Sigma , \mathcal { L }$

Inductive case (3): � is a unary routine. The (R-Uop) rule handles this case. Because unary operations apply the function to each scalar element in the argument, when the rewritten argument evaluates to the parallel evaluation of the original argument with respect to $\Sigma , \mathcal { L }$ under $\Sigma ^ { * }$ , applying the function to each scalar element naturally results in the parallel evaluation of � with respect to $\Sigma , { \mathcal { L } } .$

Inductive case (4): � is a binary routine ofthe form $f ( e _ { 1 } , e _ { 2 } )$ or matmul $( e _ { 1 } , e _ { 2 } )$ . The (R-Biop) rule handles this case. Without loss of generality, we assume the number of axes of $e _ { 1 }$ is no less than that of $e _ { 2 } .$ The rewritten $e _ { 1 } , e _ { 2 }$ , or $e _ { 1 } ^ { \prime } , e _ { 2 } ^ { \prime } ,$ evaluate to their parallel evaluations with respect to $\Sigma , \mathcal { L }$ under $\Sigma ^ { * }$ , which may be of the same shapes as their evaluations in each iteration or have one more dimension on the leftmost axes, by the definition of parallel evaluation. To make $e ^ { \prime }$ evaluate to the parallel evaluation of � with respect to $\Sigma , \mathcal { L }$ under $\Sigma ^ { * }$ , each element-wise function application in $e ^ { \prime }$ must align with the corresponding element-wise function application in each iteration. The evaluation of � in each iteration involves broadcasting the evaluation of $e _ { 2 }$ in each iteration to the shape of the evaluation of $\dot { \boldsymbol { e } } _ { 1 }$ in each iteration for element-wise function application. Thus, if $e _ { 2 } \ ' _ { s }$ parallel evaluation has one more dimension on the left-most axis, we need to make some changes to ensure the axes at the positions that are broadcast in the evaluation in each iteration are still broadcast to the same numbers corresponding to the axes of $e _ { 1 }$ . The rewrite rule ensures this by expanding new dimensions of one after the left-most axis on $e _ { 2 } ^ { \prime }$ so the $e _ { 2 } ^ { \prime \prime }$ s new dimension does not take over an axis that should be broadcast. If the parallel evaluation of $e _ { 2 }$ does not have one more dimension, the semantics of broadcast continue to ensure the same element-wise function is applied and $e _ { 2 } ^ { \prime }$ is properly duplicated on the right axes. Thus, the rewrite result evaluates to the parallel evaluation of � under $\Sigma ^ { * }$

Inductive case (5): � is either a call to expand\_dims or a call to replicate. These two types of expressions are handled by rules (R-Rep) and (R-Exp). In each iteration, the expression inserts new axes to the first argument of the call. If the parallel evaluation of the argument is the same value it evaluates to in each iteration, and there is not an extra dimension added to the left-most axis, then the rewrite rules do not need to change the expression. Then, � still evaluates to the same value under $\Sigma ^ { * }$ as it is evaluated to in each iteration. This value is the parallel evaluation of �. If the parallel evaluation of the argument is an array of the values that it takes in each iteration, a new dimension is added to the left-most axis of the argument. In this case, the numbers noting the positions to add axes are all incremented by one. Then each value in the array that the rewritten expression evaluates to under $\Sigma ^ { * }$ still corresponds to each value that � takes in each iteration. So in either case the rewritten expression evaluates to the parallel evaluation of � under Σ.

Inductive case (6): � is a call to filled. Because the second argument to filled is always a scalar, which is stipulated by the semantics of the call, we just need to require that the rewrite result of the second argument is also a scalar. Then the reason why the (R-Fill) rule rewrites � to an expression that evaluates to the parallel evaluation of� follows the same reason why (R-Uop) is correct.

Inductive case (7): � is a an indexing expression of the form $e _ { t } [ e _ { m } , ~ . . . , ~ e _ { 1 } ]$ . This case is handled by the rules (R-Indx), (R-Idxr) and (Unmask). At a high-level, because we assume that $\mathcal { L }$ does not have any loop-carried dependence when evaluated under Σ and thus the evaluation of � in each iteration does not read values that are written or updated in other iterations, the values read when the base and the indexers are parallelly evaluated are the parallel evaluation of the indexing expression. First, if � is not on the left-hand side of an update statement, the base of the indexing expres sion is rewritten. Then all the indexers are rewritten. If the base of the indexing expression’s parallel evaluation is an array of values that it takes in each iteration, a new indexer is added to select all the values corresponding to iterations in which � is evaluated in this array. This new indexer is simply loop variable substitute stored in Δ. Similar to binary routines, the indexing operator broadcasts all the indexers to the same shape for element-wise value selection. So the same trick used to rewrite binary routines is used here to ensure indexers that are broadcast for evaluation in each iteration are still properly broadcast. Now the evaluation of the $e ^ { \prime }$ under $\Sigma ^ { * }$ selects the elements that � selects in each iteration. Because the semantics of the indexing expression require indexers to be unmasked and the parallel evaluation of each indexer may be masked arrays, we need to handle indexers that are masked arrays. The (Unmask) rule does this. We fill the masked indexers with 0, which is always a valid index for any meaningful array. Then, if there is any masked indexer, we wrap the entire indexing expression by a call to make\_masked, masking the indexing result by the mask extracted from the indexers. If there is any iteration in which the indexing expression is not evaluated, the corre sponding element in the evaluation of the rewrite result is still masked. Thus, the rewrite result of (R-Indx) evaluates to the parallel evaluation of � with respect to $\Sigma , \mathcal { L }$ under $\Sigma ^ { * }$ . We do not rewrite the base if � is on the left-hand side to avoid the rewrite strategies described in base cases (2.2) and (2.3) duplicating the base variable, which results in an invalid update statement. We do not need to worry about whether the base variable is escaping the restriction imposed by branches. Note that because there is no loop-carried dependence, if� is on the left-hand side of an update statement, at least one indexer must depend on the loop variable of L and its parallel evaluation is an array of values that this indexer takes in each iteration. This parallel evaluation of the indexer can pass the mask to the entire indexing expression, so the rewrite result’s evaluation under $\Sigma ^ { * }$ is still �’s parallel evaluation. □

Lemma E.6 is also proven by structural induction on the expressions.

Proof. Base case (1): � is a variable binding statement in the form of $y : = e$ . There are three sub-cases.

Base case (2.1): � is rewritten to $e ^ { \prime } ,$ , which under $\Sigma ^ { * }$ evaluates to an array of values that � evaluates to in each iteration. This case is handled by the rule (R-Bind1). In this case, the evaluation of $e ^ { \prime }$ under $\Sigma ^ { * }$ is $\mathrm { ^ a }$ parallel evaluation of $e$ by Lemma E.5. So the execution of $\mathit { \Pi } _ { \boldsymbol { s } ^ { \prime } }$ stores this evaluation of $e ^ { \prime }$ in the current evaluation environment, which is a parallel execution of $s .$ Because $e ^ { \prime }$ has the type $( \varsigma ^ { \prime } , m ^ { \prime } )$ under the current type environment, we just need to update the current type environment to map � to $( \varsigma ^ { \prime } , m ^ { \prime } )$ to ensure the variable � has the correct type under the updated type environment.

Base case (2.2): � is not changed after the rewrite, and the static analysis says � does not depend on the loop variable of $\mathcal { L }$ . This case is handled by the rule (R-Bind2). Because $y$ does not depend on the current loop variable, by the definition of parallel execution, the value bound to $y$ after executing $s ^ { \prime }$ does not need to have one more dimension on the left-most axis. It just needs to be the parallel evaluation of $e ,$ which is true by Lemma E.5. So the execution of $\cdot _ { s ^ { \prime } }$ is the parallel execution of �. Because the variable’s defining expression is not changed, we do not need to update the typing environment. The old typing environment still correctly type all the variables.

Base case (2.3): � is not changed after the rewrite, and the static analysis says � depends on the loop variable of $\mathcal { L } .$ This case is handled by the rule (R-Bind3). By the definition of parallel execution, �’s parallel evaluation must have one more dimension on the left-most axis. However, � is not changed by the rewrite process. Hence, we need to rewrite to manually duplicate � to have one more dimension. This is achieved through either calling replicate on axis 0 to replicate � � times or replication by masking. The premise of the first solution is that the current loop variable substitute stored in Δ is unmasked, meaning � is not in a branch being flattened. In this case we just need to replicate it on the left-most axis. Similarly, the premise of the second solution means � is in a branch being flattened. In this case, calling make\_masked with the expanded mask extracted from the current loop variable substitute will duplicate the value that $e ^ { \prime }$ evaluates to by broadcasting. All the elements in the leftmost axis, which is created by broadcasting, will either be the same value that $e$ evaluates to in iterations that it is evaluated or be a fully masked array. So the evaluation of this rewritten expression under $\Sigma ^ { * }$ is the parallel evaluation of�. This statement, $s ^ { \prime } { } _ { \mathrm { : } }$ , stores this evaluation into the updated evaluation environment, which means the execution of $s ^ { \prime }$ is a parallel execution of �. The shape that the rewritten result evaluates to under $\Sigma ^ { * }$ is just the length of the loop variable substitute prepended to the old shape. The maskedness is either the old maskedness if we do not use the extracted mask to replicate or ⊤ otherwise. Storing the updated shape and maskedness to the environment ensures that the typing environment is updated to type � correctly.

Base case (2): � is an update statement of the form $e _ { l } \gets e _ { r }$ This case is handled by the rule (R-Updt). First, the expression on the left-hand side of the statement and the expression on the right-hand side of the statement are rewritten. So by Lemma $\mathrm { E } . 5 ,$ , the rewritten left-hand side should evaluate to the parallel evaluation of the original left-hand side under $\Sigma ^ { * }$ Because the semantics of updates (SEM-Upd) and masked updates (SEM-MskUpd) state that elements corresponding to the evaluation of the left-hand side are selected for update, the elements in the array bound to the variable being updated in each iteration are all selected at once. The right-hand side is also rewritten, which will evaluate to the parallel evaluation of $e _ { r }$ under $\Sigma ^ { * }$ . This parallel evaluation may or may not have one more dimension on the left-most axis. To ensure that the broadcasting semantics can continue to correctly duplicate dimensions so each scalar array cell can be updated to the value it is updated to in each iteration, we need to expand new dimensions on the rewritten right-hand side. The expanded dimensions are on axes $( 1 , \ldots , n )$ , which corresponds to the axes being broadcast in each iteration. � is the diference in numbers of axes of $e _ { l }$ and $e _ { r }$ before the rewrite. After the dimension expansion, the broadcast semantics ensure that if there is any broadcasting used to duplicate the values for update, the same duplications are applied to the values so they can be used to update the same locations as in each iteration. Since the update statement cannot change the type of a variable, the typing environment does not need to be updated, and it still correctly stores all the variables’ types.

Inductive case (1): � is a sequence of the form $s _ { 1 } ; s _ { 2 } .$ . In this case, � is rewritten to the sequence of the rewrite results of $s _ { 1 }$ and $s _ { 2 } .$ The semantics of sequence statements (SEM-Seq) say the semantics of the rewrite result matches the definition of parallel execution of sequences. Also, if after rewriting $s _ { 1 }$ we get an updated type environment that can correctly type all the variables defined no later than $s _ { 1 } ,$ , the rewrite process for $s _ { 2 }$ can be correctly guided by this type environment and we get an updated type environment that can correctly type all the variables defined no later than $s _ { 2 } .$ . This can be concluded under the assumption that each statement level rewrite correctly updates the type environment.

Inductive case (2): � is a branch, and it is in the form of if $e _ { c }$ then $s _ { t }$ else $s _ { e } .$ First, depending on whether $e _ { c }$ has one more dimension on the left-most axis after the rewrite, branch statements are handled by either (R-Brch1) or (R-Brch2).

Inductive case (2.1): If $e _ { c }$ does not have one more dimension on the left-most axis, which is handled by (R-Brch2), the branch is not flattened, and we just need to rewrite the statements in both branches. The rewritten statements are still kept in the branches where they used to reside. When the rewrite result is executed, new scopes are pushed into the evaluation environment before executing a branch and popped after executing the branch. The rewritten statements in the branches are executed under the evaluation environment with the newly added scope, which corresponds to the parallel execution of the statements in the branches under this evaluation environment. This execution meets the second case of the definition for parallel execution of branches. Because the type environment is not scoped and the updated type environment after rewriting the then-branch is used to rewrite the else-branch, the rewrite process continues to correctly update the types of variables defined in both branches.

Inductive case (2.2): If $e _ { c }$ has one more axis after the rewrite, the branches are flattened. The rewritten statements in the then-branch and the else-branch are concatenated into a sequence. Note that before rewriting the statements in the branches, the loop variable substitute stored in $\Delta$ is updated to be the previous loop variable substitute masked by the expanded condition expression. The previous loop variable substitute should have all the elements corresponding to the iterations during which the outer branch is not executed masked. By the semantics of the branch statements, the condition expression should be a scalar, and after the rewrite, it should be a one-dimensional array of the values it evaluates to in each iteration. The non-zero values in the array correspond to the iterations that the then-branch is executed, the zeroes in the array correspond to the iterations that the else-branch is executed, and the ⊗ values in the array correspond to the iterations that neither branch is executed. Masking the previous loop variable substitute with this flat tened condition will mask out all elements corresponding to the iterations that the then-branch is not executed, as by the semantics of make\_masked and binary ${ \boldsymbol { o } } { \boldsymbol { p } } ;$ , if an element whose corresponding value in the mask argument is zero or $\otimes ,$ , the corresponding element in the returned array is ⊗. The statements in the else-branch are handled similarly. When this sequence of rewritten statements is executed, the stack of the evaluation environment is not changed, and all the variables defined in this sequence stay in the evaluation en vironment after the sequence is evaluated. This is the same as first using the current evaluation environment to execute the statement in the then-branch in parallel, then using the updated evaluation environment to execute the else-branch in parallel. This behavior of the execution meets the firstcase of the definition of the parallel execution of a branch, and all the variables are correctly typed in the updated typ ing environment after the rewrite for the same reason as in Inductive case (2.1).

Inductive case (3): The current target for the rewrite process is $\mathcal { L }$ itself. This case is trivially true. Note that when we rewrite a loop, we set the loop variable substitute stored in $\Delta$ to an arange call that generates all the values that the loop variable takes in each iteration. Because the scope immediately under the loop is always executed in each iteration of the loop, the loop variable substitute stored in $\Delta$ when rewriting this scope does not have a value masked. So the loop variable substitute is correctly evaluated to the parallel evaluation of the loop variable under the current scope. Because we are assuming the statement rewrite process correctly returns an updated type environment, the type environment returned from the rewrite process is correct by assumption. □

Finally we need to prove that if we relax the assumption from $\mathcal { L }$ being a loop without any loop-carried dependence to $\mathcal { L }$ being a rewritable loop, the rewrite procedure can still correctly rewrite the loop. The proof goal for this part can be summarized into an enhanced Lemma $\operatorname { E } . 6 .$

Lemma E.7. Let $\mathcal { L } = \mathsf { f o r } x \mathrm { i n } 0 \ldots \mathrm { i n } 0 \ldots \mathrm { i n }$ do � be the innermost <sup>l</sup>oop <sup>i</sup>n a program $\mathcal { P }$ whose type environment is Γ. Let Σ be an <sub>eva</sub>l<sub>ua</sub>ti<sub>on env</sub>i<sub>ronmen</sub>t <sub>suc</sub>h th<sub>a</sub>t $\Sigma \vdash \mathcal { L } \setminus \Sigma ^ { \prime }$ <sub>an</sub>d $\textstyle \sum \vdash \ i \ \Downarrow _ { v } c .$ $\mathcal { L }$ is a rewritable loop when executed under Σ. IfΓ, Γ, ∅ ⊢ $\mathcal { L } \ \sim \ s ^ { \prime } , \Gamma ^ { \prime }$ <sub>,</sub> th<sub>en</sub> $\Sigma \vdash s ^ { \prime } \searrow \Sigma ^ { * }$ such that ∀ Variable $y . \Sigma ^ { \prime }$ ⊢ 1 $\ d s \cup \mathcal { V } _ { v } v  \Sigma ^ { * } \vdash y \ d \bigcup _ { v } v .$

Proof. By the definition of rewritable loops, $\mathcal { L }$ is either free of loop-carried dependence or has one rewriteable reduce statement.

The first case is covered by Lemma E.6. In this case $\Sigma ^ { * }$ is the result of the parallel execution of � with respect to $\Sigma , \mathcal { L }$ In this case, ∀ Variable $y . \Sigma ^ { \prime } \vdash y \ \Downarrow _ { v } \ v  \Sigma ^ { * } \vdash y \ \Downarrow _ { v } \ v$ , by Theorem E.4.

Now we discuss the case where the loop has one rewritable reduce statement. By the definition of rewriteable reduce statements, if this rewriteable reduce statement is removed from $\mathcal { L } ,$ the resulting loop, $\mathcal { L } ^ { \prime }$ , is completely free of loopcarried dependence. Because no other statement reads from the array cells updated by the rewritable reduce statement, the evaluation environment that is the result of the parallel execution of the body of $\mathcal { L } ^ { \prime }$ should evaluate all the variables that $\Sigma ^ { \prime }$ can evaluate to the same value as their evaluation under $\Sigma ^ { \prime }$ , except the variable updated by the rewritable reduce statement. This follows the proof of the first case. Now we add back this rewritable reduce statement. The rule for rewriting such statements is (R-Rstmt). This rule first rewrites the second argument of the right-hand side, which is the argument diferent from the left-hand side. Because by definition the rewritable reduce statement does not read array cells that are updated by other statements in other iterations, the rewritten second argument evaluates to its parallel evaluation when all the variables making up the second argument evaluate to their parallel evaluation, by Lemma E.5. The (R-Rstmt) rule rewrites the iterative reduce to one single declarative call that does the same reduce operation on the left-most axis. If the second argument does not have one more dimension on the left-most axis after the rewrite, we need to manually duplicate it by either calling replicate or using replication by masking. The masked elements in the array to be reduced are filled with the “identity value” of the reduce operation, so in case that the entire array being reduced is masked, the reduction result is not a masked value. If the entire array being reduced is masked, the reduction is never performed in the iterative execution, and the array cells being updated should retain their initial value. Applying the operation to the array cells with the “identity value” being the second argument keeps the array cells unchanged. Executing this rewritten statement updates the array cells being updated to the values they should hold at the end of iterative execution. As no other statement in the loop updates these array cells, the updated values remain unchanged after the parallel execution of all other statements. Thus, $\Sigma ^ { \prime }$ and $\Sigma ^ { * }$ evaluate the variable updated by this statement to the same value. Γ remains correct after rewriting the rewritable reduce statement as the rewrite of the rewritable reduce statement does not change the Γ. □

Lemma E.1 can then be proved.

Proof. Lemma E.7 implies Lemma E.1.

From Lemma E.1, we can prove the following theorem.

Theorem E.8 (Soundness of rewriting the innermost loop). Let P be a rewritable pro<sub>g</sub>ram, L be its innermost loop, Γ be its type environment. IfRewrite( $\mathcal { L } , \Gamma , \Gamma , \emptyset )$ returns a statement�<sub>,</sub> the program after rewrite is $\mathcal { P } ^ { \prime } = \mathcal { P } [ s / \mathcal { L } ]$ <sub>.</sub> F<sub>or an eva</sub>l<sub>ua</sub>ti<sub>on</sub> environment Σ such that $\Sigma \vdash \mathcal { P } \downarrow \downarrow ( v , \mu )$ <sub>,</sub> it h<sub>o</sub>ld<sub>s</sub> th<sub>a</sub>t $\Sigma \vdash \mathcal { P } ^ { \prime } \Downarrow$ $( v ^ { \prime } , \mu ^ { \prime } ) \wedge v = v ^ { \prime } \wedge \mu = \mu ^ { \prime }$

Proof. Because each unique variable name can only be de fined in one statement in the program and no statement can move variables across scopes in the evaluation environment, neither the original loop nor � can change the scope of variables that are already in the evaluation environment. From

Lemma E.1 we know all the variables that can be evaluated after the original loop was executed can be evaluated to the same values after � is executed. We also know that these variables remain in the scopes that they are defined after the execution of the original loop or the execution of �. Lastly, by the semantics of statements in the DSL, the evaluation environment after evaluating the original loop or � must have the same number of scopes as before. So as far as all the variables that are defined before the loop are concerned, the results of the execution of the original loop and the execution of � are the same. The remaining executions in the program will consequently return the same value. □

Finally, we can prove Theorem 5.8, which is restated below.

Theorem E.9 (Correctness of the Vectorize Routine). Let $\mathcal { P }$ be a rewritable program. The types of the input arguments to $\mathcal { P }$ are correctl<sub>y</sub> annotated b<sub>y</sub> A. Assume that Vectorize(P, A) returns $\mathcal { P } ^ { \prime }$ . Then for all evaluation environment Σ such that $\Sigma \vdash \mathcal { P } \downarrow \downarrow _ { v } ( v , \mu ) , \Sigma \vdash \mathcal { P } ^ { \prime } \downarrow \downarrow _ { v } ( v ^ { \prime } , \mu ^ { \prime } ) \land v = v ^ { \prime } \land \mu = \mu ^ { \prime } .$

Proof. The AnalyzeShape routine correctly returns a type environment for all the variables in the program, which follows Theorem 4.1. Suppose we call the program at the �th iteration of the while-loop $\mathcal { P } _ { n }$ . By Theorem $\operatorname { E . 8 , }$ for any evaluation environment $\Sigma$ such that $\Sigma \vdash { \mathcal { P } } \downarrow \downarrow _ { v } v \implies \Sigma$ ⊢ $\mathcal { P } _ { n } \Downarrow _ { v } v$ is clearly a loop invariant, which can be shown by a simple induction. As the routine directly returns the $\mathcal { P }$ after the while-loop, the property obviously holds on the returned $\mathcal { P }$ □