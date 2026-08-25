# FormuEvo: LLM-Guided Evolution for Discovering Solver-Efficient Mixed-Integer Programming Formulationsssover <sup>Mutation</sup> MIP

Haofeng Yuan<sup>1</sup>, Jianing Peng<sup>1</sup>, Jieyi Bi<sup>2</sup>, Ni Zhang<sup>3</sup>, Shiji Song<sup>1,</sup> <sup>†</sup>, Zhiguang Cao<sup>3Diagonstic</sup><sub>LLM</sub><sup>Memory</sup><sub>Retrieval</sub> <sup>Diagonstic</sup><sub>LLM</sub><sup>Memory</sup><sub>Retrieval</sub> <sup>Solver-Informed</sup><sub>Feedback</sub>

<sup>1</sup>Department of Automation, BNRist, Tsinghua University<sup>LLM LLM</sup>   <sup>-</sup> <sup>...</sup> <sup>2</sup>College of Computing and Data Science, Nanyang Technological University <sup>N</sup> </> <sup>Offspring</sup>Formulation </> <sup>Offspring</sup>Formulation </> <sup>Repair</sup><sub>LLM</sub>... <sup>3</sup>School of Computing and Information Systems, Singapore Management University

yhf22@mails.tsinghua.edu.cn, pjn24@mails.tsinghua.edu.cn, jieyi001@e.ntu.edu.sgLibrary LLM<sup>...-</sup> <sup>Strategy:...Mem</sup> <sup>1 -</sup> <sup>Strategy:...Mem</sup> <sup>2</sup> ni.zhang.2025@phdcs.smu.edu.sg, shijis@mail.tsinghua.edu.cn, zgcao@smu.edu.sg

## Abstract

Mixed-integer programming (MIP) lies at theariables? Compactness? Modeling Code core of operations research and industrial optimization. While large language models (LLMs)<sup>Data</sup> have recently shown promise in automated MIP<sup>ledge</sup> modeling from natural language, they prioritize semantic correctness but overlook formulation(a) Human Expert (b) Fine-tuned LLM strength, severely bottlenecking the efficiency of downstream solvers. We propose FormuEvo,<sup>MIP</sup> <sup>Solver</sup>Solver Info an LLM-guided evolutionary framework for automated discovery of solver-efficient MIP<sup>agnosis</sup> formulations. FormuEvo frames MIP formulation design as evolutionary optimization overEvolution the symbolic space of MIP formulations, repNext Generation resented as executable modeling programs, by iteratively generating, evaluating, and selecting stronger candidates via LLM-driven crossover, mutation, and repair operations. To move beMemory i yond blind exploration, FormuEvo introduces a solver-informed diagnosis mechanism that<sup>g-M</sup> <sup>constraints</sup> <sup>are</sup> <sup>present</sup> <sup>and</sup> <sup>the</sup> <sup>LP</sup> <sup>relaxation</sup> <sup>is</sup> exploits fine-grained solver statistics as verbal gradients for targeted refinement. Additionally,eplace big-M formulations with tighter convex hull a structured memory abstracts prior experience into reusable modeling strategies, avoiding re-<sup>/Effect></sup> dundant exploration while enabling zero-shotumber of branch-and-bound is reduced transfer to unseen problems and bootstrapping smaller LLMs. Experiments across diverse linear and non-linear problems demonstrate that FormuEvo discovers formulations that signif icantly outperform both expert-designed formulations and existing LLM-based approaches, accelerating solvers by up to 5.5×, with distilled knowledge transferring effectively across problems and model scales.

## 1 Introduction

Mixed-integer programming (MIP) plays a fundamental role in decision-making across domains such as manufacturing, transportation, and management (Bixby, 2012; Wolsey, 2020; Clautiaux and Ljubic´, 2025). The practical success of MIP has been largely driven by decades of advances in powerful solvers such as Gurobi (Gurobi, 2026) and COPT (Ge et al., 2022), which integrate sophisticated algorithmic components including branchand-bound, cutting planes, and various heuristics to solve problems of growing scale and complexity.

![](images/a722f8f3cc0e42bebd1c0dcd407555d349b1b267a73655d28b19aeaf426d70f1.jpg)  
Figure 1: MIP formulation design via (a) human experts, (b) LLM fine-tuning with end-to-end generation, and (c) the LLM-guided evolution framework of FormuEvo.

A well-established principle in mathematical programming is that a single optimization problem may admit multiple mathematically equivalent MIP formulations, which guarantee the same optimal solution but can differ by orders of magnitude in computational efficiency (Wolsey, 2020). Consequently, the practical tractability of MIP depends not only on constructing a mathematically correct formulation, but also on designing one that is computationally strong, namely, whose structural properties enable MIP solvers to operate efficiently.

Despite its fundamental importance, discovering solver-efficient MIP formulations has long been regarded as one of the most challenging and expertiseintensive problems in operations research (Vielma, 2015). Even experienced experts require substantial domain knowledge and structural insights to craft a strong formulation (Fig. 1(a)). This challenge is further complicated by the fact that modeling intuitions can become outdated: with the development of modern MIP solvers, classical modeling strategies once regarded as “best practices”, such as certain strengthening constraints or symmetrybreaking rules, can inadvertently interfere with the advanced presolving procedures, cut generation, and heuristic routines, leading to counter-intuitive performance degradation (Achterberg and Wunderling, 2013; Salvagnin, 2018). This motivates a new automated approach capable of discovering solverefficient MIP formulations and adaptable to the advancing algorithmic internals of modern solvers.

Large language models (LLMs) have recently demonstrated remarkable promise in automating MIP modeling from natural language (Wang et al., 2025; Qian et al., 2025; Zhou et al., 2026). By fine-tuning based on curated formulation datasets, they have effectively bridged the gap between natural language problem descriptions and executable MIP modeling programs via end-to-end generation (Fig. 1(b)). However, their objective is correctness and executability, namely, producing a formulation that accurately encodes the target problem and can be successfully solved by a downstream solver. As a result, the formulations they generate, though correct, are often computationally naive, falling far short of strong formulations required for largescale real-world optimization.

To address this gap, we propose FormuEvo, a novel LLM-guided evolutionary framework for automated discovery of solver-efficient MIP formulations. Rather than treating formulation design as single-pass generation, FormuEvo reframes it as evolutionary optimization over the symbolic space of MIP formulations, which are represented as executable modeling programs. Given a natural language problem description and a formulation template, FormuEvo maintains a population of candidate formulations, iteratively applying LLMdriven crossover, mutation, and repair operations guided directly by the solver performance, progressively evolving toward solver-efficient formulations (Fig. 1(c)). To move beyond blind exploration, FormuEvo introduces two key mechanisms: 1) a solver-informed diagnosis mechanism, where a diagnostic LLM converts fine-grained solver statistics into interpretable verbal gradients, guiding evolution toward targeted refinement, and 2) a structured memory, which abstracts prior experience into reusable modeling strategies, avoiding redundant exploration while further enabling zeroshot transfer to unseen problems and bootstrapping smaller LLMs by distilling a generalizable knowledge base. Our main contributions are as follows:

• We propose FormuEvo, an LLM-guided evolutionary framework for automated discovery of solver-efficient MIP formulations.

• We introduce a solver-informed diagnosis mechanism that leverages fine-grained solver statistics for targeted, solver-aware evolution.

• We develop a structured memory that distills reusable knowledge, enabling efficient search and transfer across problem and model scales.

• FormuEvo discovers non-obvious MIP formulations that outperform the state-of-the-art baselines, accelerating MIP solvers by up to 5.5×.

## 2 Related Work

## 2.1 MIP Modeling and Formulation

Designing strong MIP formulations is a longstanding challenge in operations research (Wolsey, 2020). Classical strengthening techniques include valid inequalities and cutting planes (Nemhauser and Wolsey, 1988; Cornuéjols, 2008), symmetrybreaking constraints (Margot, 2009), and extended variables and reformulations (Conforti et al., 2013). However, with rapid development of modern MIP solvers, classical techniques can sometimes conflict with sophisticated solver internals and lead to counter-intuitive performance degradation (Achterberg and Wunderling, 2013; Salvagnin, 2018), motivating the need for automated, adaptive approaches to formulation design.

## 2.2 LLM for Automated MIP Modeling

Recent advances in LLMs have automated MIP modeling from natural language through structured formulations (Ramamonjison et al., 2022), agent systems (AhmadiTeshnizi et al., 2024), or finetuning (Huang et al., 2025; Wang et al., 2026; Zhou et al., 2026). Despite improving domain fidelity, they suffer from two major flaws. First, they operate at the instance level, generating formulations with parameters tied to particular instances rather than discovering generalized, instance-agnostic formulations at the problem level. Second, as these open-loop methods prioritize mathematical correctness rather than formulation strength, the generated formulations are structurally weak and inefficient for large-scale optimization.

## 2.3 LLM-Guided Evolutionary Search

LLM-guided evolutionary search has emerged as a powerful paradigm for automated algorithm design and heuristic discovery (Liu et al., 2024; Novikov et al., 2025). FunSearch (Romera-Paredes et al., 2024) pioneered this by using LLMs as evolutionary operators over executable code, while subsequent works (Ye et al., 2024, 2025; Zhang et al., 2026; Hou et al., 2026; Gungordu et al., 2026) extended this framework for combinatorial heuristic design. However, little work has adapted this paradigm to MIP formulation. The most closely related work, EvoCut (Yazdani et al., 2025), applies LLM-based evolution to cutting plane generation. However, EvoCut remains limited to local relaxation tightening on an existing model and lacks the capability of holistic structural redesign. In contrast, FormuEvo evolves over the global symbolic space of MIP formulations, and incorporates solver-informed diagnosis and structured memory to enable systematic and directed improvement.

## 3 FormuEvo

## 3.1 Problem Formalization

Given a problem $p$ described in natural language, an MIP formulation can be modeled as:

$$
\begin{array} { r l } { \underset { x , y } { \operatorname* { m i n } } } & { c ( x , y ) } \\ { \mathrm { s . t . } } & { g _ { i } ( x , y ) \leq 0 , \quad i = 1 , \ldots , m , } \\ & { h _ { j } ( x , y ) = 0 , \quad j = 1 , \ldots , n , } \\ & { x \in X \subseteq \mathbb { R } ^ { q } , } \\ & { y \in Y \subseteq \mathbb { Z } ^ { r } , } \end{array}\tag{1}
$$

where x and $y$ denote continuous and integer decision variables, $c ( x , y )$ is the objective function, and $g _ { i } ( x , y ) , h _ { j } ( x , y )$ denote inequality and equality constraints, respectively. In practice, the MIP formulation is instantiated as executable programs (e.g., Python code with Gurobi) that specify these components and can be directly executed by an MIP solver. We denote by $\mathcal { F }$ the symbolic space of MIP formulations of problem $p ,$ namely, the set of all semantically correct and syntactically valid programs encoding problem $p .$ . By definition, any two formulations $f _ { 1 } , f _ { 2 } \in \mathcal { F }$ , possibly involving different structures of variables and constraints, are equivalent in the sense of optimization, which guarantees the same optimal solution for problem $p .$

Existing LLM-based approaches treat MIP modeling as a satisfaction problem, aiming to generate any formulation that can be successfully solved by a downstream MIP solver and produce the correct solution. However, they largely overlook a fundamental principle in mathematical optimization: although formulations in $\mathcal { F }$ are mathematically equivalent, they can induce dramatically different solver behaviors and computational efficiency. In practice, two equivalent formulations may differ by orders of magnitude in runtime due to differences in relaxation tightness, symmetry structure, and even solver-specific implementation details. Motivated by this, we reframe MIP formulation design as an optimization problem over the symbolic space of MIP formulations toward minimizing the downstream computational cost:

$$
f ^ { \star } = \underset { f \in \mathcal { F } } { \arg \operatorname* { m i n } } \ \phi ( f ) ,\tag{2}
$$

where $\phi ( f )$ evaluates the computational cost $( \mathrm { e . g . }$ runtime) of formulation $f$ with a downstream MIP solver. This defines an optimization problem over a vast, discrete, and highly structured symbolic space, where the objective is non-differentiable and expensive to evaluate. Therefore, we frame the discovery of solver-efficient MIP formulations as an evolutionary search, with LLMs as operators to navigate the symbolic space of MIP formulations.

## 3.2 LLM-Guided Evolutionary Search

FormuEvo takes a natural language description and a formulation template $f _ { 0 }$ as input, maintaining a population $\mathcal { P } _ { t } = \{ f _ { 1 } , \ldots , f _ { N } \}$ of $N$ candidate formulations at generation t. To effectively explore the vast and structured symbolic space ${ \mathcal { F } } _ { : }$ , FormuEvo employs an evolutionary loop coordinated by five specialized LLM modules: 1) a generator LLM that produces new candidate formulations, 2) a diagnostic LLM that interprets solver feedback to guide generation, 3) a repair LLM that debugs failed formulations, 4) a reflector LLM that abstracts evolution experience into structural memories, and 5) a distiller LLM that distills accumulated memory into transferable knowledge for unseen problems. The overall framework is illustrated in Fig. 2.

Initialization. Given the problem description and a formulation template, the evolution starts by prompting the generator LLM to produce $N$ candidate formulations as the initial population $\mathcal { P } _ { 0 }$ . The generator is encouraged to explore diverse formulation variants, such as alternative variables, logically Modelin<sub>g</sub>equivalent but structurally different constraints, as <sup>Constraints? LP</sup> <sup>Strength?</sup>well as variations in implementation details, to prevent premature convergence to sub-optimal local <sub>Knowledge</sub>regions in the symbolic space.

![](images/a61f38d91c6ce4ed382a439751db2673c9b8b7be7862bbb7da6f14502e50fa88.jpg)  
Figure 2: The overview of FormuEvo. FormuEvo iteratively evolves formulations through LLM-driven crossover, MIP Fomulation MIP Fomulation (Program)mutation, and repair toward better solver efficiency, guided by solver-informed diagnosis and structured memory.

Evaluation. Each candidate formulation $f \in \mathcal { P } _ { t }$ <sup>(a)</sup> <sup>Human</sup> <sup>Expert (b)</sup> <sup>Fine-tu</sup>is evaluated by executing it on a downstream MIP solver (e.g., Gurobi) over a set of instances of prob-<sup>MIP</sup> <sup>Solv</sup>Solver Infolem p. If execution fails due to compilation errors or correctness violations (by verifying whether the <sup>Diagnosis</sup>solved objective values match the ground-truth optimal value), the error tracebacks and solver logs are fed into a repair LLM, which attempts to fix the candidate formulation while preserving its intended modeling logic. Candidates exceeding the <sup>c</sup> <sup>FormuEvo:</sup> <sup>LLM-Guided</sup> <sup>Evolution</sup> <sup>of</sup> <sup>MIP F</sup>maximum repair budget are discarded, ensuring that all maintained formulations strictly belong to F. Notably, since the generator LLM may produce bold yet error-prone formulation variants, the repair <sup></Condition></sup>operations can still be valuable for exploration.

weak with a large integrality gap.For each successfully evaluated formulation, its fitness is measured by the solver cost ϕ(f), com-<sup>Replace</sup> <sup>big-M</sup> <sup>constraints</sup> <sup>with</sup> <sup>tighter</sup> <sup>convex</sup>puted as the shifted geometric mean (SGM) of perinstance runtimes, a standard evaluation metric for solver performance. Meanwhile, the reflector LLM <sup>number</sup> <sup>of</sup> <sup>branch-and-bound</sup> <sup>nodes</sup> <sup>is</sup> <sup>reduced</sup>abstracts the modeling decisions and solver feedback from each evaluated formulation into reusable strategies, which are stored in a structured memory library for future retrieval (Section 3.4).

Evolution. At each generation, FormuEvo preserves the top-N formulations according to their fitness. The next generation is produced through two LLM-based operators: crossover and mutation.

For crossover, two distinct parent formulations are sampled from the current population $\mathcal { P } _ { t } .$ , with Codeprobability according to their fitness rank. They are first processed by the diagnostic LLM to pro-<sub>Data</sub>duce a structural diagnosis (Section 3.3). Guided <sup>Collection</sup>by this diagnosis, the generator LLM produces an <sup>ine-tune</sup>offspring formulation that integrates complementary strengths or explores an orthogonal modeling direction. For mutation, a high-performing elite is selected. Similar to the crossover workflow, the diagnostic LLM analyzes the elite formulation and <sub>aluation</sub>pinpoints specific refinement opportunities to ex-<sup>Solver-Informe</sup>plore its structural neighborhood. Guided by this Memorytargeted diagnosis, the generator LLM then implements directed refinements instead of blind perturbation to produce a mutated offspring.

All offspring candidates are subject to the evalu-<sup>(c)</sup> <sup>FormuEvo:</sup> <sup>LLM-Guided</sup>ation and repair procedure described above. This process iterates for T generations, to evolve the population toward increasingly solver-efficient formulations. The best formulation across all genera-<sup>on</sup> <sup>is</sup>tions is returned as the final formulation $\hat { f } ^ { \star }$

## 3.3 Solver-Informed Diagnosis

<sup>ities.</sup>A key bottleneck of evolutionary search is that it overly relies on a scalar fitness signal to guide the <sup>the</sup>evolution (Romera-Paredes et al., 2024; Shi et al., 2026). In our case, the runtime fitness can indicate whether a formulation is slow, but provides little insight into why it is slow, making the search space exploration highly sample-inefficient.

To overcome this bottleneck, FormuEvo introduces a solver-informed diagnosis mechanism. We note that modern MIP solvers are not mere blackbox evaluators; instead, they expose rich, finegrained internal statistics during optimization, such as presolving outcomes, relaxation quality, and branching dynamics (right of Fig. 2). These statistics reflect important solver behaviors and reveal structural bottlenecks that a single runtime cannot capture. For example, a weak relaxation may indicate excessive fractional solutions that require strengthening, while a large branch-and-bound tree may suggest a highly symmetric structure.

To exploit this fine-grained feedback for targeted, solver-informed evolution, the diagnostic LLM takes the parent or elite formulations alongside their solver statistics to produce a structured verbal diagnosis before crossover or mutation. The diagnosis identifies major computational bottlenecks, attributes them to specific structural properties, and proposes targeted directions for improvement. For instance, if parent $f _ { i }$ is compact but suffers from weak relaxations that induce excessive branching, while parent $f _ { j }$ yields stronger bounds but incurs costly presolving, the diagnostic LLM may guide the generator LLM to selectively incorporate the bound-tightening constraints of $f _ { j }$ where they matter most, while retaining the compact structure of $f _ { i }$ elsewhere. This diagnosis functions as an interpretable verbal gradient, guiding the generator LLM toward principled, solver-informed structural improvement rather than undirected exploration.

## 3.4 Structured Memory

We note that in long-horizon evolutionary search, LLMs are prone to redundant exploration, such as repeatedly revisiting failed modifications or forgetting effective sub-structures that can be directly retrieved for future refinements. To mitigate this issue, FormuEvo incorporates a structured memory library to abstract, organize, and store both successful and failed experiences into reusable modeling strategies for guiding future evolution steps.

Specifically, after each evaluation, the reflector LLM abstracts the modeling modifications and solver feedback between parent and offspring formulations into a structured memory entry with three fields: 1) a condition that describes the problem context and formulation characteristics under which the strategy was applied; 2) a strategy that captures the specific modeling decision or structural modification; and 3) an effect that summarizes its observed impact on solver behavior and optimization performance. This structured format, as illustrated in Fig. 3, enables precise retrieval and reuse of prior experience. During subsequent evolution, the diagnostic LLM retrieves relevant memory entries based on the current bottleneck conditions and incorporates them as additional context to enrich the diagnosis, leveraging prior experience to prune the search space and avoid known pitfalls.

![](images/ff6e7000ff1fac5b10f695bade695c7bd9e9e79095589aa9635be0a15f0731d5.jpg)  
Figure 3: Example of a structured memory entry.

While the memory efficiently accelerates the search within a single problem, we note that optimization problems often share common modeling logic and structural properties (e.g., routing logic, assignment constraints), even when their mathematical formulations may appear different. Therefore, upon completion of the evolutionary search, we introduce a distiller LLM to summarize the accumulated memory library into high-level problemagnostic knowledge by removing instance-specific details. This distilled knowledge base not only enables FormuEvo to perform efficient zero-shot transfer when tasked with entirely unseen optimization problems, but can also serve as a lightweight yet high-quality reasoning resource for bootstrapping smaller LLMs with domain expertise distilled from a more powerful model (see Section 4.3).

## 4 Experiments

## 4.1 Experiment Setup

Benchmarks. We evaluate FormuEvo on classical mixed-integer linear programming (MILP) and non-linear programming (MINLP) benchmarks, including: traveling salesman problem (TSP), jobshop scheduling problem (JSSP), bin packing problem (BPP), capacitated facility location problem (CFLP), and non-linear quadratic assignment problem (QAP). These problems span diverse optimization structures, covering routing, scheduling, packing, location, and assignment. To further assess the capability of FormuEvo beyond well-studied textbook settings, we additionally consider two challenging and less explored optimization tasks: neural network verification (NNV) (Tjeng et al., 2019) and a mathematical challenge derived from

IMO 2025 Problem 6 (Art of Problem Solving Wiki, 2025), where human-designed formulation priors are limited or absent. Following prior practices (Gleixner et al., 2021), instances for classical problems are categorized into three levels: Easy, Medium, and Hard. The Easy set consists of small instances used for training and validating in evolution, while the Medium and Hard sets contain larger instances held out to test the final performance. For the two novel optimization problems, we use Easy instances for evolution, and evaluate final performance on a separate set of Hard instances. More detailed problem descriptions and the corresponding instance splits are provided in Appendix A.

Baselines. We compare FormuEvo with two categories of baselines. First, we include both standard and state-of-the-art expert-designed formulations from the operations research literature, tracing the advancement of human expertise. For instance, TSP baselines range from the textbook MTZ formulation (Miller et al., 1960), to the stronger single-commodity flow (SCF) formulation (Gavish and Graves, 1978), up to the MCF-RLT (Balma et al., 2018), which achieves the theoretically tightest known relaxation bound, representing the current limits of human design. We also compare with recent LLM-based modeling approaches, including ORLM (Huang et al., 2025) and StepORLM (Zhou et al., 2026), to demonstrate the gap between correctness-oriented generation and efficiency-oriented evolution. Finally, we include EvoCut (Yazdani et al., 2025), an LLM-guided cutting-plane generation method, to highlight the difference between local formulation strengthening and our exploration of the broad symbolic space of MIP formulations. Additional introductions for the baseline formulations of each problem are also provided in Appendix A.

Implementation. All experiments are conducted in Python using Gurobi 10.0 as the downstream MIP solver, with a single thread and default parameters, on a server equipped with AMD EPYC 9654 processors. We use a population size of N = 8 and evolve for T = 5 generations, producing 8 offspring per generation via crossover and mutation with a mutation rate of 0.3. Memory-augmented generation is applied with probability 0.7 to avoid search overfitting toward previously explored regions. Each candidate is evaluated on 100 Easy instances, with fitness measured by SGM runtime with a 1-second shift (Gleixner et al., 2021). Failed candidates receive one repair attempt. Operator prompts are provided in Appendix B.

FormuEvo uses GPT-5.4-mini as the backbone LLM (a comparison across different backbones is reported in Section 4.5). ORLM and StepORLM use their fine-tuned LLMs, and are allowed the same number of generation attempts as FormuEvo for a fair comparison. For EvoCut, we adopt the reported formulation from their original paper on problems covered by their benchmark, noting that EvoCut uses a stronger backbone model than ours. For problems outside their benchmark suite, we rerun EvoCut using the same backbone and experimental settings as FormuEvo.

## 4.2 Main Results

Tables 1, 2 present results on classical MILP and MINLP benchmarks and the two novel problems, respectively. We report three metrics: 1) “Time”, SGM runtime over 100 test instances and 5 independent runs, with a per-instance time limit of 600 seconds; 2) “Wins”, the number of instances on which a formulation achieves the best runtime among all formulations; and 3) “Solved”, the number of instances solved to optimality within the time limit.

Overall, FormuEvo discovers formulations that outperform both expert-designed and LLM-based baselines, with acceleration of up to 5.5× over the best baselines. The best discovered formulations are presented in Appendix A. Moreover, FormuEvo attains the best runtime on the majority of instances across nearly all benchmarks, demonstrating robustness across diverse optimization structures. Beyond the numerical improvements, the results reveal several important insights.

First, FormuEvo successfully discovers formulations that outperform not only textbook models but also highly optimized expert-designed formulations, often by substantial margins. For instance, in TSP, while the human-designed MCF-RLT formulation has the theoretically tightest relaxation bound, it completely fails (0/100 solved) on Hard instances. This highlights a known but often overlooked fact: theoretical tightness does not necessarily translate to solver efficiency, as a heavily extended formulation can overwhelm the computation in MIP solvers. In contrast, FormuEvo is efficiency-oriented and guided by solver-informed statistics, enabling targeted optimization toward the practical computational objective of solvers.

Second, the results of ORLM and StepORLM further illustrate the gap between correctness and efficiency. Although they successfully generate correct and executable formulations on standard MILPs, their outputs largely remain naive textbook formulations, which are weak and scale poorly in practice. Moreover, both approaches fail to produce valid formulations for the novel NNV and IMO problems. This also underscores the limited generalization of fine-tuning approaches to more complex problems beyond their training distribution. See Appendix C.5 for further discussion.

Table 1: Performance on classical MILP and MINLP problems.
<table><tr><td colspan="4">Easy</td><td colspan="2">Medium</td><td colspan="3">Hard</td></tr><tr><td></td><td>Formulation</td><td>Time↓</td><td>Wins↑</td><td>Time↓</td><td>Wins↑</td><td>Time↓</td><td>Wins↑</td><td>Solved↑</td></tr><tr><td rowspan="8">TSP</td><td>MTZ (1960)</td><td>0.9578</td><td>0/100</td><td>6.1826</td><td>5/100</td><td>95.1315</td><td>0/100</td><td>62/100</td></tr><tr><td>SCF (1978)</td><td>0.3079</td><td>3/100</td><td>1.0065</td><td>10/100</td><td>8.3315</td><td>3/100</td><td>100/100</td></tr><tr><td>MCF-RLT (2018)</td><td>3.2729</td><td>0/100</td><td>84.1649</td><td>0/100</td><td>600.0570</td><td>0/100</td><td>0/100</td></tr><tr><td>ORLM (2025)</td><td>0.9575</td><td>2/100</td><td>6.3270</td><td>0/100</td><td>92.5953</td><td>0/100</td><td>62/100</td></tr><tr><td>StepORLM (2026)</td><td>1.7667</td><td>0/100</td><td>17.0081</td><td>0/100</td><td>173.8792</td><td>0/100</td><td>64/100</td></tr><tr><td>EvoCut (2025)</td><td>0.5268</td><td>8/100</td><td>2.5013</td><td>6/100</td><td>13.8359</td><td>17/100</td><td>92/100</td></tr><tr><td>FormuEvo (Ours)</td><td>0.1410 (+54.2%)</td><td>87/100</td><td>0.5640 (+44.0%)</td><td>79/100</td><td>3.9469 (+52.6%)</td><td>80/100</td><td>100/100</td></tr><tr><td>Disj. (1960)</td><td>0.3264</td><td>19/100</td><td>1.8389</td><td>9/100</td><td>17.6653</td><td>25/100</td><td>98/100</td></tr><tr><td rowspan="6">JSP</td><td>Enh. Disj. (2016)</td><td>0.4886</td><td>1/100</td><td>7.9405</td><td>0/100</td><td>158.1492</td><td>0/100</td><td>77/100</td></tr><tr><td>ORLM (2025)</td><td>600.0012</td><td>0/100</td><td>600.0032</td><td>0/100</td><td>600.0019</td><td>0/100</td><td>0/100</td></tr><tr><td>StepORLM (2026)</td><td>0.3040</td><td>28/100</td><td>1.7270</td><td>22/100</td><td>18.7024</td><td>20/100</td><td>99/100</td></tr><tr><td>EvoCut (2025)</td><td>0.3005</td><td>19/100</td><td>1.7627</td><td>20/100</td><td>18.1329</td><td>14/100</td><td>98/100</td></tr><tr><td>FormuEvo (Ours)</td><td>0.2781 (+7.5%)</td><td>33/100</td><td> $1 . 5 3 1 6 _ { ( + 1 1 . 3 \% ) }$ </td><td>49/100</td><td>15.4237 (+12.7%)</td><td>41/100</td><td>98/100</td></tr><tr><td>Kant. (1960)</td><td>18.6428</td><td>0/100</td><td>191.9222</td><td>0/100</td><td>351.7605</td><td>0/100</td><td>28/100</td></tr><tr><td rowspan="7">BP</td><td>AF (1999)</td><td>0.2710</td><td>0/100</td><td>0.9930</td><td>1/100</td><td>4.9778</td><td>1/100</td><td>100/100</td></tr><tr><td>VPSolver (2016)</td><td>0.0610</td><td>53/100</td><td>0.4064</td><td>40/100</td><td>1.4425</td><td>38/100</td><td>100/100</td></tr><tr><td>ORLM (2025)</td><td>18.3614</td><td>0/100</td><td>287.7057</td><td>0/100</td><td>341.1659</td><td>0/100</td><td>24/100</td></tr><tr><td>StepORLM (2026)</td><td>12.3430</td><td>0/100</td><td>117.0032</td><td>0/100</td><td>95.0336</td><td>1/100</td><td>42/100</td></tr><tr><td>EvoCut (2025)</td><td>3.0279</td><td>0/100</td><td>26.7969</td><td>4/100</td><td>29.0496</td><td>5/100</td><td>60/100</td></tr><tr><td>FormuEvo (Ours)</td><td> $\mathbf { 0 . 0 5 4 3 } _ { ( + 1 1 . 0 \% ) }$ </td><td>47/100</td><td> $\mathbf { 0 . 3 4 1 8 } _ { ( + 1 5 . 9 \% ) }$ </td><td>55/100</td><td> $\mathbf { 0 . 8 3 8 4 } _ { ( + 4 1 . 9 \% ) }$ </td><td>55/100</td><td>100/100</td></tr><tr><td>Agg. (1965)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="8">CLP</td><td>Disagg. (1989)</td><td>2.7178</td><td>19/100</td><td>22.5520</td><td>8/100</td><td>87.3078</td><td>6/100</td><td>96/100 95/100</td></tr><tr><td>Enh. Disagg. (2021)</td><td>4.8283</td><td>0/100</td><td>31.4982</td><td>0/100</td><td>113.4126 113.5033</td><td>1/100</td><td>95/100</td></tr><tr><td></td><td>5.0126</td><td>0/100</td><td>33.1480</td><td>0/100</td><td></td><td>1/100</td><td></td></tr><tr><td>ORLM (2025)</td><td>2.8439</td><td>0/100</td><td>24.0395</td><td>0/100</td><td>88.4319</td><td>2/100</td><td>95/100</td></tr><tr><td>StepORLM (2026) EvoCut (2025)</td><td>3.2663</td><td>1/100</td><td>26.8000</td><td>3/100</td><td>104.8201 59.4309</td><td>3/100</td><td>94/100 100/100</td></tr><tr><td></td><td>2.7698</td><td>5/100</td><td>17.1088</td><td>14/100</td><td></td><td>16/100</td><td></td></tr><tr><td>FormuEvo (Ours) KB Quad. (1957)</td><td> $\mathbf { 1 . 6 2 5 8 } _ { ( + 4 0 . 2 \% ) }$ </td><td>75/100</td><td> $1 1 . 6 9 5 2 _ { ( + 3 1 . 6 \% ) }$ </td><td>75/100</td><td> $4 4 . 3 4 4 7 _ { ( + 2 5 . 4 \% ) }$ </td><td>71/100</td><td>100/100</td></tr><tr><td rowspan="8">OAPP</td><td>McC. Lin. (1976)</td><td>9.9323 9.5458</td><td>0/100</td><td>23.7309</td><td>0/100 0/100</td><td>538.7275</td><td>0/100 0/100</td><td>28/100 0/100</td></tr><tr><td>AJ Lin. (1994)</td><td>1.2154</td><td>0/100 1/100</td><td>67.9480 8.0087</td><td>0/100</td><td>600.0062 237.7423</td><td>0/100</td><td>97/100</td></tr><tr><td>XY-KB Lin. (2006)</td><td>0.4044</td><td>28/100</td><td>1.6163</td><td>25/100</td><td>37.8213</td><td>28/100</td><td>100/100</td></tr><tr><td>IPQAPR (2010)</td><td>0.3929</td><td>35/100</td><td>1.6883</td><td>26/100</td><td>40.2989</td><td>19/100</td><td>100/100</td></tr><tr><td>ORLM (2025)*</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>StepORLM (2026)</td><td></td><td></td><td>68.3794</td><td>0/100</td><td>600.0061</td><td>0/100</td><td>0/100</td></tr><tr><td>EvoCut (2025)</td><td>9.7489 10.1303</td><td>0/100 0/100</td><td>23.9718</td><td>0/100</td><td>538.5478</td><td>0/100</td><td>28/100</td></tr><tr><td>FormuEvo (Ours)</td><td> $0 . 3 9 3 6 _ { ( - 0 . 2 \% ) }$ </td><td>36/100</td><td> $1 . 5 6 4 8 ( + 3 . 2 \% )$ </td><td>49/100</td><td> $3 4 . 5 0 3 4 _ { ( + 8 . 8 \% ) }$ </td><td>53/100</td><td>100/100</td></tr></table>

<sup>\*</sup>ORLM failed to produce a correct formulation for QAP.

Third, EvoCut, which strengthens fixed formulations through LLM-generated cutting planes, yields moderate improvements over baseline formulations but remains fundamentally limited by the underlying model structure. For instance, on BPP, EvoCut improves upon standard formulations but remains significantly weaker than both VPSolver and FormuEvo, because the primary bottleneck of BPP lies in its assignment structure rather than LP relaxation strength. Similarly, on QAP, the standard quadratic formulation defines a Birkhoff polytope that already represents the convex hull (von Neumann, 1953), and EvoCut fails to achieve any improvement through strengthening cuts. In contrast, FormuEvo breaks through this limitation by searching over the entire symbolic space of MIP formulations, enabling the discovery of structural reformulations such as alternative linearizations and variable extension that fundamentally alter the search landscape rather than merely modifying a fixed one.

Table 2: Performance on two novel MIP problems.
<table><tr><td>Formulation</td><td></td><td>Time↓</td><td>Wins↑</td><td>Solved↑</td></tr><tr><td>ΛNN</td><td>Std. Formu. (2019) ORLM / StepORLM EvoCut (2025)</td><td>69.5200 67.6093</td><td>9/100 13/100</td><td>71/100 73/100</td></tr><tr><td></td><td>FormuEvo (Ours)</td><td>21.4139 (+68.3%)</td><td>78/100</td><td>86/100</td></tr><tr><td>IMI</td><td>Std. Formu. (2025) ORLM / StepORLM EvoCut (2025)</td><td>114.0912 99.4400</td><td>0/4 - 0/4</td><td>3/4 - 3/4</td></tr><tr><td></td><td>FormuEvo (Ours)</td><td>17.9341 (+82.0%)</td><td>4/4</td><td>4/4</td></tr></table>

Table 3: Ablation study of FormuEvo components.
<table><tr><td rowspan=1 colspan=2>Method</td><td rowspan=1 colspan=1>Easy</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>Hard</td></tr><tr><td rowspan=3 colspan=2>Best BaselineTSP FormuEvow/o Memoryw/o Diagnosis</td><td rowspan=1 colspan=1>0.3079</td><td rowspan=1 colspan=1>1.0065</td><td rowspan=1 colspan=1>8.3315</td></tr><tr><td rowspan=2 colspan=1>0.14100.17760.2308</td><td rowspan=1 colspan=1>0.5640</td><td rowspan=2 colspan=1>3.94694.66496.5207</td></tr><tr><td rowspan=1 colspan=1>0.61020.7350</td></tr><tr><td rowspan=5 colspan=2>Best BaselineJSP FormuEvow/o Memoryw/o Diagnosis</td><td rowspan=1 colspan=1>0.3005</td><td rowspan=1 colspan=1>1.7270</td><td rowspan=5 colspan=1>17.665315.423716.322318.0159</td></tr><tr><td rowspan=2 colspan=1>ory</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.2781</td><td rowspan=3 colspan=1>1.53161.56981.7648</td></tr><tr><td rowspan=1 colspan=1>0.2846</td></tr><tr><td rowspan=1 colspan=1>0.3048</td></tr></table>

## 4.3 Transfer Performance

We evaluate the transferability of distilled knowledge to unseen problems and smaller LLMs. Fig. 4 shows evolution trajectories using a smaller LLM (GPT-5.4-nano), with distilled knowledge aggregated from all problems except the target. Without transfer, GPT-5.4-nano exhibits substantially slower convergence and higher final runtime, struggling to move beyond the baseline formulations. In contrast, when augmented with the distilled knowledge, GPT-5.4-nano significantly narrows the gap, achieving convergence trajectories and final performance close to the larger GPT-5.4-mini model. These results suggest that the distilled knowledge can serve as a lightweight but effective reasoning source, compensating for the capacity of smaller models by providing structured domain expertise that would otherwise be difficult to infer independently, which offers a practical pathway toward strong MIP formulation discovery at lower costs.

![](images/500213eb09c98a4bd2382c6dbd4221c7ef36e169a53d43678b6e864e4872304c.jpg)  
Figure 4: Transfer performance on smaller LLMs.

Table 4: Impact of different backbone LLMs.
<table><tr><td rowspan=1 colspan=2>Method</td><td rowspan=1 colspan=1>Cost</td><td rowspan=1 colspan=1>Easy</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=2>Hard</td></tr><tr><td rowspan=2 colspan=2>Best BaselineTSPGPT-5.4-miniClaude-Sonnet-4.6DeepSeek-V4-Flash</td><td rowspan=2 colspan=1>-～$2～$10~$0.5</td><td rowspan=1 colspan=1>0.3079</td><td rowspan=2 colspan=1>1.00650.56400.47770.6283</td><td rowspan=2 colspan=2>8.33153.94694.01485.3068</td></tr><tr><td rowspan=1 colspan=1>0.14100.10170.1498</td></tr><tr><td rowspan=4 colspan=2>Best BaselineJSPGPT-5.4-miniClaude-Sonnet-4.6DeepSeek-V4-Flash</td><td rowspan=1 colspan=1>.</td><td rowspan=1 colspan=1>0.3005</td><td rowspan=1 colspan=1>1.7270</td><td rowspan=1 colspan=2>17.6653</td></tr><tr><td rowspan=1 colspan=1>～$3</td><td rowspan=1 colspan=1>0.2781</td><td rowspan=1 colspan=1>1.5316</td><td rowspan=1 colspan=2>15.4237</td></tr><tr><td rowspan=1 colspan=1>4.6</td><td rowspan=1 colspan=1>～$16</td><td rowspan=1 colspan=1>0.2808</td><td rowspan=1 colspan=1>1.5319</td><td rowspan=1 colspan=2>15.8340</td></tr><tr><td rowspan=1 colspan=1>～$0.8</td><td rowspan=1 colspan=1>0.3050</td><td rowspan=1 colspan=1>1.6399</td><td rowspan=1 colspan=2>16.1327</td></tr></table>

## 4.4 Ablation Study on Key Components

We analyze the contributions of two key components: solver-informed diagnosis and memory, by comparing FormuEvo against two ablated variants. As shown in Table 3, removing either component leads to clear performance degradation, indicating that the diagnosis mechanism plays a crucial role in providing the directional signal for FormuEvo, while the memory reuse also facilitates the evolution by improving the search efficiency.

## 4.5 Impact of Different LLM Backbones

We further evaluate FormuEvo with different backbone LLMs (Table 4). All three consistently outperform the best baselines on both problems, with only modest differences in formulation quality and generalization, where no backbone model consistently dominates across all settings. These results indicate that the gains stem primarily from the evolutionary framework and are robust to the choice of backbone. We adopt GPT-5.4-mini as the default because it offers the best overall trade-off between performance and API cost.

Additional analyses on different downstream solvers, backbone LLMs, and evolutionary hyperparameters are provided in Appendix C.2 - C.4.

## 5 Conclusions

We presented FormuEvo, an LLM-guided evolutionary framework for automatically discovering solver-efficient MIP formulations, which incorporates solver-informed diagnosis and structured memory for directed and systematic improvement. Experiments show that FormuEvo successfully discovers formulations that significantly outperform state-of-the-art baselines, accelerating modern MIP solvers by up to 5.5×. Notably, FormuEvo is transferable to unseen problems and smaller LLMs by distilling a generalizable knowledge base, demonstrating its generalizability and applicability.

## Limitations

FormuEvo focuses on discovering static MIP formulations that can be directly solved by off-theshelf general-purpose MIP solvers, which represents the most commonly adopted approach in realworld practice. However, for many large-scale or highly complex optimization problems, advanced problem-specific solution approaches in operations research often rely on dynamic relaxation or decomposition algorithms, such as column generation, which iteratively adds variables by solving a pricing sub-problem, and Benders decomposition, which relaxes the original problem and then alternates between a master problem and subproblems to progressively tighten the formulation. These methods typically require iterative interactions between master problems and sub-problems, along with dynamic generation of variables and constraints during the solving process. In such approaches,formulation design and algorithm development are inherently coupled: the effectiveness of the formulation depends on the decomposition strategy, while the decomposition procedure itself is tightly determined by the underlying formulation structure. Extending FormuEvo beyond static formulations to jointly evolve reformulations and decomposition-based solution algorithms is a promising but substantially more challenging direction. We leave this extension for future work.

## Acknowledgments

This work was done during the first author’s internship at Singapore Management University. This work is supported in part by the BNRist Project (No: BNR2024TD03003), the National Natural Science Foundation of China (No: 42327901), and the Program of China Scholarship Council (No: 202506210230). This research is supported by the National Research Foundation Singapore under the AI Singapore Programme (AISG Award No: AISG3-RPGV-2025-017). We would also like to thank the AI4Opt group for helpful discussions, and the reviewers for their valuable feedback.

## References

Tobias Achterberg and Roland Wunderling. 2013. Mixed integer programming: Analyzing 12 years of progress. In Facets of Combinatorial Optimization, pages 449–481. Springer.

Warren P Adams and Terri A Johnson. 1994. Improved linear programming-based lower bounds for the quadratic assignment problem. DIMACS Series in Discrete Mathematics and Theoretical Computer Science, 16:43–77.

Ali AhmadiTeshnizi, Wenzhi Gao, and Madeleine Udell. 2024. OptiMUS: Scalable optimization modeling with (MI)LP solvers and large language models. In International Conference on Machine Learning.

Art of Problem Solving Wiki. 2025. 2025 IMO Problem 6. https://artofproblemsolving.com/wiki/ index.php/2025\_IMO\_Problems/Problem\_6. Accessed: May 2026.

Pasquale Avella, Maurizio Boccia, Sara Mattia, and Fabrizio Rossi. 2021. Weak flow cover inequalities for the capacitated facility location problem. European Journal ofOperational Research, 289(2):485–494.

Michel Louis Balinski. 1965. Integer programming: Methods, uses, computations. Management Science, 12(3):253–313.

Ali Balma, Safa Ben Salem, Mehdi Mrad, and Talel Ladhari. 2018. Strong multi-commodity flow formulations for the asymmetric traveling salesman problem. European Journal of Operational Research, 271(1):72–79.

Robert E Bixby. 2012. A brief history of linear and mixed-integer programming computation. In Optimization Stories, pages 107–121.

Filipe Brandao and Joao Pedro Pedroso. 2016. Bin packing and related problems: General arc-flow formulation with graph compression. Computers & Operations Research, 69:56–67.

François Clautiaux and Ivana Ljubic. 2025. Last fifty´ years of integer linear programming: A focus on recent practical advances. European Journal ofOperational Research, 324(3):707–731.

Michele Conforti, Gérard Cornuéjols, and Giacomo Zambelli. 2013. Extended formulations in combinatorial optimization. Annals ofOperations Research, 204(1):97–143.

Gérard Cornuéjols. 2008. Valid inequalities for mixed integer linear programs. Mathematical Programming, 112(1):3–44.

Bezalel Gavish and Stephen C Graves. 1978. The travelling salesman problem and related problems. MIT Operations Research Center.

Dongdong Ge, Qi Huangfu, Zizhuo Wang, Jian Wu, and Yinyu Ye. 2022. Cardinal optimizer (COPT) user guide. arXiv preprint arXiv:2208.14314.

Ambros Gleixner, Gregor Hendel, Gerald Gamrath, Tobias Achterberg, Michael Bastubbe, Timo Berthold, Philipp Christophel, Kati Jarck, Thorsten Koch, Jeff Linderoth, Marco Lübbecke, Hans D. Mittelmann, Derya Ozyurt, Ted K. Ralphs, Domenico Salvagnin, and Yuji Shinano. 2021. MIPLIB 2017: Data-driven compilation of the 6th mixed-integer programming library. Mathematical Programming Computation, 13(3):443–490.

Oguzhan Gungordu, Siheng Xiong, and Faramarz Fekri. 2026. Pathwise: Planning through world model for automated heuristic design via self-evolving LLMs. In International Conference on Machine Learning.

Gurobi. 2026. Gurobi Optimizer Reference Manual. https://www.gurobi.com. Accessed: May 2026.

Christopher Hojny, Mathieu Besançon, Ksenia Bestuzheva, Sander Borst, João Dionísio, Johannes Ehls, Leon Eifler, Mohammed Ghannam, Ambros Gleixner, Adrian Göß, Alexander Hoen, Jacob von Holly-Ponientzietz, Rolf van der Hulst, Dominik Kamp, Thorsten Koch, Kevin Kofler, Jurgen Lentz, Marco Lübbecke, Stephen J. Maher, and 15 others. 2025. The SCIP optimization suite 10.0. Technical report, Optimization Online.

Zhinan Hou, Xingchen Li, Yankai Zhang, Tianxun Li, and Keyou You. 2026. LLM4Branch: Large language model for discovering efficient branching policies of integer programs. In International Conference on Machine Learning.

Chenyu Huang, Zhengyang Tang, Shixi Hu, Ruoqing Jiang, Xin Zheng, Dongdong Ge, Benyou Wang, and Zizhuo Wang. 2025. ORLM: A customizable framework in training large models for automated optimization modeling. Operations Research, 73(6):2986– 3009.

Leonid V Kantorovich. 1960. Mathematical methods of organizing and planning production. Management Science, 6(4):366–422.

Tjalling C Koopmans and Martin Beckmann. 1957. Assignment problems and the location of economic activities. Econometrica: Journal ofthe Econometric Society, pages 53–76.

Wen-Yang Ku and J Christopher Beck. 2016. Mixed integer programming models for job shop scheduling: A computational analysis. Computers & Operations Research, 73:165–173.

Janny MY Leung and Thomas L Magnanti. 1989. Valid inequalities and facets of the capacitated plant location problem. Mathematical Programming, 44(1):271–291.

Fei Liu, Tong Xialiang, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. 2024. Evolution of heuristics: Towards efficient automatic algorithm design using large language model. In International Conference on Machine Learning.

Alan S Manne. 1960. On the job-shop scheduling problem. Operations Research, 8(2):219–223.

François Margot. 2009. Symmetry in integer linear programming. 50 Years of Integer Programming 1958–2008: From the Early Years to the State-of-the-Art, pages 647–686.

Garth P McCormick. 1976. Computability of global solutions to factorable nonconvex programs: Part I—convex underestimating problems. Mathematical Programming, 10(1):147–175.

Clair E Miller, Albert W Tucker, and Richard A Zemlin. 1960. Integer programming formulation of traveling salesman problems. Journal ofthe ACM, 7(4):326– 329.

George L Nemhauser and Laurence A Wolsey. 1988. Integer and Combinatorial Optimization, volume 18. Wiley New York.

Alexander Novikov, Ngân Vu, Marvin Eisenberger, Em-˜ ilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. 2025. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131.

Cheng Qian, Hongyi Du, Hongru Wang, Xiusi Chen, Yuji Zhang, Avirup Sil, Chengxiang Zhai, Kathleen McKeown, and Heng Ji. 2025. ModelingAgent: Bridging LLMs and mathematical modeling for realworld challenges. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 1599–1633.

Rindra Ramamonjison, Haley Li, Timothy Yu, Shiqi He, Vishnu Rengan, Amin Banitalebi-Dehkordi, Zirui Zhou, and Yong Zhang. 2022. Augmenting operations research with auto-formulation of optimization models from problem descriptions. In Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 29–62.

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. 2024. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475.

Domenico Salvagnin. 2018. Symmetry breaking inequalities from the schreier-sims table. In International Conference on the Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 521–529. Springer.

Yiding Shi, Jianan Zhou, Wen Song, Jieyi Bi, Yaoxin Wu, Zhiguang Cao, and Jie Zhang. 2026. Generalizable heuristic generation through LLMs with metaoptimization. In International Conference on Learning Representations.

Marius M Solomon. 1987. Algorithms for the vehicle routing and scheduling problems with time window constraints. Operations Research, 35(2):254–265.

Eric Taillard. 1993. Benchmarks for basic scheduling problems. European Journal ofOperational Research, 64(2):278–285.

Vincent Tjeng, Kai Y. Xiao, and Russ Tedrake. 2019. Evaluating robustness of neural networks with mixed integer programming. In International Conference on Learning Representations.

JM Valério de Carvalho. 1999. Exact solution of bin-packing problems using column generation and branch-and-bound. Annals ofOperations Research, 86(0):629–659.

Juan Pablo Vielma. 2015. Mixed integer linear programming formulation techniques. SIAM Review, 57(1):3–57.

John von Neumann. 1953. A certain zero-sum twoperson game equivalent to the optimal assignment problem. Contributions to the Theory of Games II, 28:5–12.

Yilin Wang, Heng Zhou, Dongxing Mao, Linjie Li, Jingru Tan, Haochen Han, Zhengyuan Yang, Alex Jinpeng Wang, and Min Li. 2026. OR-PRM: A process reward model for algorithmic problem in operations research. In International Conference on Learning Representations.

Zhiyuan Wang, Bokui Chen, Yinya Huang, Qingxing Cao, Ming He, Jianping Fan, and Xiaodan Liang. 2025. ORMind: A cognitive-inspired end-to-end

reasoning framework for operations research. In Annual Meeting ofthe Associationfor Computational Linguistics: Industry Track, pages 104–131.

Laurence A Wolsey. 2020. Integer Programming. John Wiley & Sons.

Yong Xia and Ya-Xiang Yuan. 2006. A new linearization method for quadratic assignment problems. Optimisation Methods and Software, 21(5):805–818.

Milad Yazdani, Mahdi Mostajabdaveh, Samin Aref, and Zirui Zhou. 2025. EvoCut: Strengthening integer programs via evolution-guided language models. arXiv preprint arXiv:2508.11850.

Haoran Ye, Jiarui Wang, Zhiguang Cao, Federico Berto, Chuanbo Hua, Haeyeon Kim, Jinkyoo Park, and Guojie Song. 2024. ReEvo: Large language models as hyper-heuristics with reflective evolution. Advances in Neural Information Processing Systems, 37:43571– 43608.

Huigen Ye, Hua Xu, An Yan, and Yaoyang Cheng. 2025. Large language model-driven large neighborhood search for large-scale MILP problems. In International Conference on Machine Learning.

Huizhen Zhang, Cesar Beltran-Royo, and Miguel Constantino. 2010. Effective formulation reductions for the quadratic assignment problem. Computers & Operations Research, 37(11):2007–2016.

Ni Zhang, Zhiguang Cao, Jianan Zhou, Cong Zhang, and Yew-Soon Ong. 2026. An agentic framework with LLMs for solving complex vehicle routing problems. In International Conference on Learning Representations.

Chenyu Zhou, Tianyi Xu, Jianghao Lin, and Dongdong Ge. 2026. StepORLM: A self-evolving framework with generative process supervision for operations research language models. In International Conference on Learning Representations.

## A Problems and Formulations

This appendix details the benchmark problems used in our experiments, including the descriptions, standard formulations, baselines, and the best formulations discovered by FormuEvo. Detailed instance splits in the experiments are shown in Table 5.

Table 5: Instance splits for experiments.
<table><tr><td rowspan=1 colspan=1>Problem</td><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Configuration</td></tr><tr><td rowspan=1 colspan=1>TSP</td><td rowspan=1 colspan=1>EasyMediumHard</td><td rowspan=1 colspan=1>30 cities50 cities100 cities</td></tr><tr><td rowspan=1 colspan=1>JSSP</td><td rowspan=1 colspan=1>EasyMediumHard</td><td rowspan=1 colspan=1>8 jobs, 8 machines10 jobs, 10 machines12 jobs, 12 machines</td></tr><tr><td rowspan=1 colspan=1>BPP</td><td rowspan=1 colspan=1>EasyMediumHard</td><td rowspan=1 colspan=1>100 items, 150 capacity150 items, 300 capacity200 items, 400 capacity</td></tr><tr><td rowspan=1 colspan=1>CFLP</td><td rowspan=1 colspan=1>EasyMediumHard</td><td rowspan=1 colspan=1>100 facilities, 200 customers200 facilities, 400 customers300 facilities, 600 customers</td></tr><tr><td rowspan=1 colspan=1>QAP</td><td rowspan=1 colspan=1>EasyMediumHard</td><td rowspan=1 colspan=1>8 facilities / locations9 facilities / locations10 facilities / locations</td></tr><tr><td rowspan=1 colspan=1>NNV</td><td rowspan=1 colspan=1>EasyHard</td><td rowspan=1 colspan=1>MLP dim [20, 80, 10]MLP dim [40, 160, 20]</td></tr><tr><td rowspan=1 colspan=1>IMO</td><td rowspan=1 colspan=1>EasyHard</td><td rowspan=1 colspan=1>Grid size {3, 4, 5, 6}Grid size {7, 8, 9, 10}</td></tr></table>

## A.1 Traveling Salesman Problem (TSP) Problem description

The traveling salesman problem aims to determine an optimal Hamiltonian cycle (route) to visit a set of n cities exactly once and return to the starting city. The problem is defined on a complete graph where each pair of cities $( i , j )$ is associated with a travel distance (or cost) $c _ { i j }$ . The decision involves sequencing the visits to ensure that every city is entered exactly once and exited exactly once. The selected route must form a single continuous tour covering all n cities. Any disconnected, smaller loops (subtours) that do not include all cities are strictly prohibited. The objective is to minimize the total travel distance of the complete tour.

## Standard MIP formulation

$V = \{ 1 , 2 , \dots , n \}$ : the set of n cities.

$C = V \setminus \{ 1 \}$ : the set of cities except city 1.

$\begin{array} { r } { c _ { i j } \colon } \end{array}$ travel distance from city i to city j.

$x _ { i j } \colon$ decision variable that equals 1 if and only if the route travels directly from node i to node j.

$u _ { i } \mathrm { : }$ auxiliary ordering variable used for subtour elimination, where $u _ { 1 } = 1$

Below is the standard MTZ formulation for TSP:

$$
\sum _ { i = 1 } ^ { n } \sum _ { j = 1 , \ j \neq i } ^ { n } c _ { i j } x _ { i j }
$$

$$
{ \mathrm { s . t . } } \sum _ { j = 1 , \ j \neq i } ^ { n } x _ { i j } = 1 , \quad \forall i \in V ,
$$

$$
\sum _ { i = 1 , \ i \neq j } ^ { n } x _ { i j } = 1 , \quad \forall j \in V ,
$$

$$
u _ { i } - u _ { j } \le ( n - 1 ) ( 1 - x _ { i j } ) , \forall i , j \in C , i \neq j ,
$$

$$
x _ { i j } \in \{ 0 , 1 \} , \quad \forall i , j \in V , \ i \neq j ,
$$

$$
2 \leq u _ { i } \leq n , \quad \forall i \in C .
$$

## Baseline formulations

• MTZ: The standard formulation using continuous auxiliary sequencing variables to enforce a strictly increasing visit order for subtour elimination, as described above.

• SCF: A flow-based formulation that eliminates subtours by tracking a single continuous flow where the depot dispatches n − 1 units and each subsequent city consumes exactly one unit.

• MCF-RLT: An advanced flow-based formulation using reformulation-linearization technique, which linearizes higher-order products of flow and precedence variables, achieving the theoretically tightest known relaxation bound.

• ORLM: Generated by ORLM, same as MTZ.

• StepORLM: Generated by StepORLM, based on MTZ with explicit self-loop elimination.

• EvoCut: A strengthened MTZ with lifted ordering constraints, envelope bounds, and elimination of 2-cycles and depot-associated short cycles.

## FormuEvo formulation

$\bar { c } _ { i j } = c _ { i j } + c _ { j i } \mathrm { : }$ bidirectional distance of $( i , j )$

$f _ { i j } { \mathrm { : } }$ non-negative flow variable on arc $( i , j )$ , defined for $i , j \in C , i \neq j$

The FormuEvo formulation preserves the global connectivity guarantee of SCF formulation while selectively incorporating small lifted DFJ subtour elimination cuts as local strengthening constraints:

$$
\operatorname* { m i n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 , \ j \neq i } ^ { n } c _ { i j } x _ { i j }
$$

$$
{ \mathrm { s . t . } } \sum _ { j = 1 , \ j \neq i } ^ { n } x _ { i j } = 1 , \quad \forall i \in V ,
$$

$$
\sum _ { i = 1 , \ i \neq j } ^ { n } x _ { i j } = 1 , \quad \forall j \in V ,
$$

$$
x _ { i j } + x _ { j i } \leq 1 , \quad \forall i , j \in V , i < j ,
$$

$$
\sum _ { i \in S } \sum _ { \substack { j \in S , j \neq i } } x _ { i j } \leq | S | - 1 , \quad \forall S \in \mathcal { F } ,
$$

$$
( n - 1 ) \cdot x _ { 1 j } + \sum _ { i \in C \backslash \{ j \} } f _ { i j } - \sum _ { k \in C \backslash \{ j \} } f _ { j k } = 1 ,
$$

$$
\forall j \in C ,
$$

$$
x _ { i j } \le f _ { i j } \le ( n - 2 ) \cdot x _ { i j } , \quad \forall i , j \in C , i \neq j ,
$$

$$
x _ { i j } \in \{ 0 , 1 \} , \quad \forall i , j \in V , \ i \neq j ,
$$

$$
f _ { i j } \geq 0 , \quad \forall i , j \in C , i \neq j .
$$

The subtour elimination family F is defined as the set of all unique collection consisting of:

• Depot 3-cuts: $\{ 1 , a , b \} , \forall a , b \in C , a < b .$

• Local 3-cuts: For each $i \in C ,$ , let $N _ { K } ( i )$ be the $K = \operatorname* { m i n } ( 4 , n - 2 )$ nearest neighbors of i w.r.t. $\bar { c } _ { i j }$ . For every pair $\{ a , b \} \subseteq N _ { K } ( i )$ , add the set $\{ i , a , b \} \mathrm { i f } | \{ i , a , b \} | < n .$

• Depot 4-cuts $( { \mathrm { i f ~ } } n > 4 ) \colon$ Let $N _ { K } ( 1 )$ be the $K =$ min $( 6 , n - 1 )$ nearest neighbors of the depot w.r.t. $\bar { c } _ { 1 j }$ . For every triple $\{ a , b , c \} \subseteq N _ { K } ( 1 )$ , add the set $\left\{ 1 , a , b , c \right\} \mathrm { i f } \left| \left\{ 1 , a , b , c \right\} \right| < n .$

• Local 4-cuts (if $n > 4 )$ : For each $i \in C$ and every triple $\{ a , b , c \} \subseteq N _ { K } ( i ) , K = \operatorname* { m i n } ( 4 , n -$ 2), add the set $\{ i , a , b , c \} \mathrm { i f } | \{ i , a , b , c \} | < n .$

## A.2 Job-Shop Scheduling Problem (JSSP)

## Problem description

The job-shop scheduling problem aims to determine the optimal scheduling of a set of n jobs on a set of m machines. Each job i consists of m operations, where each operation must be processed on a distinct machine. The processing order of these operations is predefined for each job, representing its technological order. Each machine can process at most one operation at a time, and preemption is strictly prohibited. The decision involves determining the exact start time of each operation such that both machine capacity constraints and job precedence constraints are satisfied. The objective is to minimize the makespan, defined as the completion time of the last operation across all jobs.

## Standard MIP formulation

$J = \{ 1 , \ldots , n \}$ : the set of n jobs.

$M = \{ 1 , \dots , m \}$ : the set of m machines (also the number of operations per job).

$\mu _ { i j } \in M :$ : the machine that processes the $j \mathrm { - t h }$ operation of job i.

$\pi _ { i } ( k ) \in M \colon$ : the index of the operation in job i that executes on machine $k , \mathrm { i } . \mathrm { e } . , \mu _ { i , \pi _ { i } ( k ) } = k$

$p _ { i j } \colon$ processing time of $j \cdot$ -th operation of job i.

$z _ { i _ { 1 } i _ { 2 } k } \mathrm { : }$ : binary variable that equals 1 if and only if job $i _ { 1 }$ is processed before job $i _ { 2 }$ on machine k.

• $s _ { i j } \colon$ decision variable representing the start time of the j-th operation of job i.

$C _ { \mathrm { m a x } } \mathrm { : }$ decision variable representing makespan.

• $L \colon$ a large enough constant, set to $\textstyle \sum _ { i \in J , j \in M } p _ { i j }$ Below is the disjunctive formulation for JSSP:

min $C _ { \mathrm { m a x } }$

$$
s _ { i , j + 1 } \geq s _ { i j } + p _ { i j } , \forall i \in J , \forall j \in M \setminus \{ m \} ,
$$

$$
s _ { i _ { 1 } , \pi _ { i _ { 1 } } ( k ) } + p _ { i _ { 1 } , \pi _ { i _ { 1 } } ( k ) } \leq s _ { i _ { 2 } , \pi _ { i _ { 2 } } ( k ) } +
$$

$$
L \cdot ( 1 - z _ { i _ { 1 } i _ { 2 } k } ) , \forall i _ { 1 } , i _ { 2 } \in J , i _ { 1 } < i _ { 2 } , k \in M ,
$$

$$
\begin{array} { r l r } & { } & { s _ { i _ { 2 } , \pi _ { i _ { 2 } } ( k ) } + p _ { i _ { 2 } , \pi _ { i _ { 2 } } ( k ) } \le s _ { i _ { 1 } , \pi _ { i _ { 1 } } ( k ) } + L \cdot z _ { i _ { 1 } i _ { 2 } k } , } \\ & { } & { \forall i _ { 1 } , i _ { 2 } \in J , i _ { 1 } < i _ { 2 } , k \in M , } \end{array}
$$

$$
C _ { \mathrm { m a x } } \geq s _ { i j } + p _ { i j } , \quad \forall i \in J , j \in M ,
$$

$$
z _ { i _ { 1 } i _ { 2 } k } \in \{ 0 , 1 \} , \forall i _ { 1 } , i _ { 2 } \in J , i _ { 1 } < i _ { 2 } , k \in M .
$$

$$
s _ { i j } \geq 0 , \quad \forall i \in J , j \in M ,
$$

$$
C _ { \mathrm { m a x } } \geq 0 .
$$

## Baseline formulations

• Disj.: The standard disjunctive formulation using $\mathrm { b i g - } M$ constraints for non-overlapping requirements, as described above.

• Enh. Disj.: An enhanced disjunctive formulation that replaces standard big-M inequalities with continuous surplus variables to explicitly track and bound the idle time between operations.

• ORLM: Generated by ORLM, a time-indexed formulation that enforces the precedence and nonoverlapping requirements using time constraints.

• StepORLM: Generated by StepORLM, same as the disjunctive formulation.

• EvoCut: A strengthened disjunctive formulation that tightens the linear relaxation with static valid inequalities derived from maximum job lengths and machine-level critical-path bounds.

## FormuEvo formulation

$\begin{array} { r } { \underline { { s } } _ { i j } = \sum _ { l = 1 } ^ { j - 1 } p _ { i l } . } \end{array}$ : lower bound on $s _ { i j }$

$\begin{array} { r } { \sigma _ { i j } = \sum _ { l = j } ^ { m } p _ { i l } \colon } \end{array}$ tail processing time from the j-th operation of job i onward.

$\bar { s } _ { i j } = \operatorname* { m a x } ( \underline { { s } } _ { i j } , \hat { C } - \sigma _ { i j } )$ : upper bound on $s _ { i j }$ where $\hat { C }$ is a heuristic makespan.

$\mu _ { i } = ( \mu _ { i 1 } , \ldots , \mu _ { i m } ) \colon$ the machine assignment vector of job i.

$p _ { i } = ( p _ { i 1 } , \dots , p _ { i m } )$ : the processing time vector of job i.

$\omega _ { j } = m - j + 1 \colon$ symmetry-breaking weight for operation j.

$y _ { i _ { 1 } i _ { 2 } k } \mathrm { : }$ binary disjunctive variable on machine $k ,$ defined only for free pairs (pairs whose ordering cannot be inferred from bounds).

$L _ { a b k } = \bar { s } _ { a , \pi _ { a } ( k ) } + p _ { a , \pi _ { a } ( k ) } - \underline { { s } } _ { b , \pi _ { b } ( k ) } ;$ pair-specific tight big-M for precedence $a \to b$

The FormuEvo formulation strengthens the disjunctive model through tight variable bounds, tailbased makespan constraints, symmetry-breaking constraints, and bound-driven fixing of resolvable disjunctions to reduce binary variables.

min $C _ { \mathrm { m a x } }$

$$
\begin{array} { r l } & { \mathrm { s . t . s } _ { i , j + 1 } \geq \mathrm { s } _ { i , j + 1 } > \mathrm { s } _ { i , j } \leq N _ { i , j + 1 } \leq i , j \leq i , j \leq i , j \leq i , j \leq i , j \leq i , j \leq i , j \leq i , j \leq i , j \leq i , j \leq i , j } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad  \quad \quad \quad \quad \quad \quad \quad \quad \quad  \quad \quad \quad \quad \quad \quad  \\ & & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad  \quad \quad \quad \quad \quad \quad \quad \quad \quad  \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

where $\mathcal { P } _ { k }$ and ${ \mathcal { V } } _ { k }$ are defined as follows. For each machine k, consider the operations processed on k and sort them lexicographically by $( \underline { { s } } _ { i j } + p _ { i j } , \ : \bar { s } _ { i j } )$ For every ordered pair $( i _ { 1 } , j _ { 1 } )$ and $( i _ { 2 } , j _ { 2 } )$ in this order, if $\underline { { s } } _ { i _ { 1 } , j _ { 1 } } + p _ { i _ { 1 } , j _ { 1 } } > \overline { { s } } _ { i _ { 2 } , j _ { 2 } }$ , then $( i _ { 2 } , i _ { 1 } )$ is added to a fixed precedence relation. Its transitive closure defines a partial order on machine k. Pairs resolved by this order are placed in $\mathcal { P } _ { k }$ , while the remaining unresolved pairs form $\mathcal { V } _ { k }$ and retain the binary big-M disjunction. The heuristic makespan $\hat { C }$ is initialized from the best schedule produced by constructive heuristics, including serial list scheduling and priority dispatch rules.

## A.3 Bin Packing Problem (BPP)

## Problem description

The bin packing problem aims to determine the optimal assignment of a set of n discrete items into a set of identical bins. Each bin is homogeneous with a fixed, deterministic maximum weight capacity W. Each item i has a predefined weight w , and all item weights are assumed to be positive integers. The packing decision involves determining exactly which bin each item is placed into. Every item must be assigned to exactly one bin. For any bin that is used, the cumulative weight of all items assigned to it cannot exceed W. Sufficient bins are assumed to be available, bounded by a maximum possible number m (typically $m = n )$ . The objective is to minimize the total number of bins actively used to accommodate all n items.

## Standard MIP formulation

## Notation

$I = \{ 1 , 2 , \dots , n \}$ : the set of n items.

$B = \{ 1 , 2 , \ldots , m \}$ : the set of m available bins.

$w _ { i } \colon$ weight of item i.

• $W { : }$ maximum weight capacity of each bin.

• $x _ { i j } { \mathrm { : } }$ binary variable that equals 1 if and only if item i is placed into bin j.

• $y _ { j } { \mathrm { : } }$ binary variable that equals 1 if and only if bin j is actively used.

Below is the Kantorovich formulation for BPP:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { j = 1 } ^ { m } y _ { j } } \\ { \displaystyle \mathrm { s . t . } \sum _ { j = 1 } ^ { m } x _ { i j } = 1 , \quad \forall i \in I , } \\ { \displaystyle \sum _ { i = 1 } ^ { n } w _ { i } x _ { i j } \le W y _ { j } , \quad \forall j \in B , } \\ { \displaystyle x _ { i j } \in \{ 0 , 1 \} , \quad \forall i \in I , \forall j \in B , } \\ { \displaystyle y _ { j } \in \{ 0 , 1 \} , \quad \forall j \in B , } \end{array}
$$

## Baseline formulations

• Kant.: The Kantorovich assignment formulation, as described above.

• AF: A pseudo-polynomial formulation that reformulates BPP as a flow network problem.

• VPSolver: An AF-style formulation with graph compression and side constraints.

• ORLM: Generated by ORLM, same as Kant.

• StepORLM: Generated by StepORLM, based on Kantorovich formulation with symmetrybreaking constraints and objective bounds.

• EvoCut: A strengthened Kantorovich variant that breaks symmetry via lexicographical ordering inequalities, continuous weight bounds, and indexbased assignment restrictions for sorted items.

## FormuEvo formulation

## Notation

$g = \operatorname* { g c d } ( W , w _ { 1 } , \dots , w _ { n } )$ : greatest common divisor (GCD) of W and all weights $w _ { i }$

${ \mathcal { W } } = \{ \bar { w } _ { 1 } , . . . , \bar { w } _ { K } \}$ : set of distinct item weights after GCD scaling (W and $w _ { i }$ are divided by $g )$ .

$b _ { w } \mathrm { : }$ the demand quantity of items with scaled weight $w \in \mathcal W$

$B _ { 0 } { : }$ the number of items whose weight equals the capacity W, which exactly fill a bin, and are pre-packed and excluded from the problem.

$\mathcal S = \{ 0 = s _ { 0 } < s _ { 1 } < \cdots < s _ { T } \leq W \}$ : the sorted set of valid cumulative bin loads (nodes in the flow network).

$x _ { w , s } \colon$ item flow variable representing the number of items of weight w placed into bins that currently have an accumulated load of state s.

• $\ell _ { t } \colon$ loss flow variable representing the unused bin capacity between states $s _ { t }$ and $s _ { t + 1 }$

• z: variable representing the number of bins required to pack all items lighter than W.

The FormuEvo formulation constructs a directed acyclic graph where valid residual capacities are compressed via subset-sum filtering. Loss arcs $( \ell _ { t } )$ ensure flow conservation without requiring explicit bin indexing, while z and $\ell _ { t }$ are naturally integers.

$$
\begin{array} { r l } & { \mathrm { ~ \operatorname* { m i n } ~ } \ R _ { 0 } + z } \\ & { \mathrm { s . t . } } \\ & { \quad \times : \displaystyle \sum _ { u \in \mathcal { S } } x _ { u , v } = z _ { v } , } \\ & { \quad \quad \times : \displaystyle \sum _ { u = s } z _ { u , v } = x _ { u , v } + z _ { v - 1 } = \sum _ { u \in \mathcal { S } } x _ { u , v } + z _ { v - 1 } } \\ & { \quad \quad \times : \displaystyle \sum _ { u \in \mathcal { S } } x _ { u , v } = ( x _ { u , v } + x _ { v - 1 } - x _ { v - 1 } ) } \\ & { \quad \quad \quad \times : \displaystyle \prod _ { u \in \mathcal { S } } x _ { u , v } = ( y _ { v - 1 } - x _ { v - 1 } - x _ { v - 1 } ) } \\ & { \quad \quad \quad \times : \displaystyle \sum _ { u \in \mathcal { S } } x _ { u , v } = y _ { v - 1 } = z _ { v } , } \\ & { \quad \quad \quad \times \displaystyle \sum _ { u \in \mathcal { S } } x _ { v , v } = b _ { u , v } \ \forall w \in \mathcal { W } , } \\ & { \quad \quad \times : \displaystyle z \leq z \leq z _ { v } , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}
$$

where $\underline { { z } } , \overline { { z } }$ are valid lower and upper bounds obtained from capacity-based bounds and a first-fit decreasing heuristic. The state set S is generated by subset-sum reachability over the scaled item sizes. Starting from 0, every reachable load $s + w$ with $s \in { \mathcal { S } }$ and $w \in \mathcal W$ is iteratively added whenever $s + w \le W$ . Only transitions between reachable states are retained in the compressed network.

## A.4 Capacitated Facility Location Problem (CFLP)

## Problem description

The capacitated facility location problem aims to determine the optimal subset of facilities to open from a set of m candidate locations and the corresponding shipment quantities to satisfy the demands of n customers. Each facility i incurs a fixed opening cost $f _ { i }$ and has a maximum supply capacity $C _ { i }$ . Each customer $j$ has a demand $d _ { j }$ that must be fully met. It is assumed that the total available capacity is sufficient to satisfy total demand. Shipping one unit from location i to customer j incurs a transportation cost $c _ { i j }$ , and a facility can supply customers only if it is opened. The decision involves selecting which facilities to open and determining the exact shipment quantity from each opened facility to each customer such that all demands are met without exceeding any facility’s capacity. The objective is to minimize the total cost, comprising the sum of fixed opening costs and variable transportation costs.

## Standard MIP formulation

## Notation

$I = \{ 1 , 2 , \dots , m \}$ : the set of m locations.

$J = \{ 1 , 2 , \dots , n \}$ : the set of n customers.

• $f _ { i } { \mathrm { : } }$ fixed cost of opening facility i.

$C _ { i } \colon$ maximum supply capacity of facility i.

• $d _ { j } \colon$ demand of customer j.

$c _ { i j } \colon$ unit transportation cost from facility i to customer $j .$

• y<sub>i</sub>: binary variable that equals 1 if and only if facility i is opened.

• x<sub>ij</sub>: continuous variable representing the shipment quantity from facility i to customer j.

The standard MIP formulation is as follows.

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { i = 1 } \displaystyle \sum _ { i = 1 } ^ { m } f _ { i } y _ { i } + \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } c _ { i j } x _ { i j } } \\ { \mathrm { s . t . } \displaystyle \sum _ { i = 1 } ^ { m } x _ { i j } = d _ { j } , \quad \forall j \in J , } \\ { \displaystyle \sum _ { j = 1 } ^ { n } x _ { i j } \leq C _ { i } y _ { i } , \quad \forall i \in I , } \\ { y _ { i } \in \{ 0 , 1 \} , \quad \forall i \in I , } \\ { x _ { i j } \geq 0 , \quad \forall i \in I , \forall j \in J . } \end{array}
$$

Baseline formulations

• Agg.: The aggregated formulation using facilitylevel capacity constraints, as described above.

• Disagg.: A disaggregated formulation that replaces the capacity constraint with per-customer linking inequalities for a tighter LP relaxation.

• ORLM: Generated by ORLM, same as the aggregated formulation.

• StepORLM: Generated by StepORLM, based on the aggregated formulation with explicitly decomposed demand constraints.

• EvoCut: A strengthened disaggregated formulation with tightened variable upper bounds, singlecustomer coverage inequalities, global capacity bounds, minimum-open-facility bounds, and subset coverage cuts for the largest customers.

## FormuEvo formulation

## Notation

$u _ { i j } = \operatorname* { m i n } ( C _ { i } , d _ { j } ) ;$ : an upper bound on $x _ { i j }$

$\mathcal { L } _ { j } = \{ i \in I \mid C _ { i } \geq d _ { j } \}$ : set of large facilities

whose individual capacity can satisfy demand $d _ { j }$ FormuEvo tightens the formulation through perarc upper bounds, global and per-customer cover inequalities, minimum-open-count constraints, and symmetry breaking based on dominance.

$$
\operatorname* { m i n } \sum _ { i = 1 } ^ { m } f _ { i } y _ { i } + \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } c _ { i j } x _ { i j }
$$

$$
{ \mathrm { s . t . ~ } } \sum _ { i = 1 } ^ { m } x _ { i j } = d _ { j } , \quad \forall j \in J ,
$$

$$
\sum _ { j = 1 } ^ { n } x _ { i j } \leq C _ { i } y _ { i } , \quad \forall i \in I ,
$$

$$
\sum _ { i = 1 } ^ { m } C _ { i } y _ { i } \geq \sum _ { j = 1 } ^ { n } d _ { j } ,
$$

$$
\sum _ { i \in \mathcal { L } _ { j } } C _ { i } y _ { i } \ge d _ { j } - \sum _ { i \in I \backslash \mathcal { L } _ { j } } \operatorname* { m i n } ( C _ { i } , d _ { j } ) , \forall j \in J ,
$$

$$
\sum _ { i = 1 } ^ { m } y _ { i } \geq \bigg \lceil \frac { \sum _ { j = 1 } ^ { n } d _ { j } } { \operatorname* { m a x } _ { i \in I } C _ { i } } \bigg \rceil ,
$$

$y _ { k } \le y _ { i } , \quad \forall i , k \in I$ with $C _ { i } \geq C _ { k } , f _ { i } \leq f _ { k }$ and $c _ { i j } \leq c _ { k j } , \forall j \in J ,$

$$
0 \leq x _ { i j } \leq u _ { i j } , \quad \forall i \in I , \forall j \in J ,
$$

$$
y _ { i } \in \{ 0 , 1 \} , \quad \forall i \in I .
$$

## A.5 Quadratic Assignment Problem (QAP)

## Problem description

The quadratic assignment problem aims to determine the optimal one-to-one assignment of a set of n facilities to a set of n distinct locations. Each facility must be assigned to exactly one location, and each location must be occupied by exactly one facility, establishing a strict bijective mapping. There is a deterministic flow $f _ { i j }$ between every pair of facilities i and $j ,$ and a deterministic distance $d _ { k l }$ between every pair of locations k and l. The decision involves determining a bijective assignment between facilities and locations. The objective is to minimize the total interaction cost, computed as the sum of flow–distance products induced by all assigned facility pairs.

## Standard MIP formulation

## Notation

$N = \{ 1 , \ldots , n \}$ : the set of facilities / locations.

$f _ { i j } { \mathrm { : } }$ flow (interaction weight) between facility i and facility $j .$

$d _ { k l } \colon$ distance between location k and location l.

• x<sub>ik</sub>: binary variable that equals 1 if and only if facility i is assigned to location k.

Below is the standard Koopmans–Beckmann binary quadratic formulation:

$$
\operatorname* { m i n } \ \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \sum _ { k = 1 } ^ { n } \sum _ { l = 1 } ^ { n } f _ { i j } d _ { k l } x _ { i k } x _ { j l }
$$

$$
\mathrm { s . t . } \sum _ { k = 1 } ^ { n } x _ { i k } = 1 , \quad \forall i \in N ,
$$

$$
\sum _ { i = 1 } ^ { n } x _ { i k } = 1 , \quad \forall k \in N ,
$$

$$
x _ { i k } \in \{ 0 , 1 \} , \quad \forall i , k \in N .
$$

## Baseline formulations

• K-B Quad.: The standard Koopmans–Beckmann binary quadratic formulation, as described above.

• McC. Lin.: A classical linearization that introduces auxiliary variables $y _ { i j k l }$ to represent products $x _ { i k } x _ { j l }$ and enforces equivalence via standard McCormick envelope inequalities.

• AJ Lin.: The Adams–Johnson linearization that lifts continuous $y _ { i j k l }$ variables with tight linking constraints, yielding a stronger relaxation.

• XY-KB Lin.: The Xia–Yuan linearization that precomputes minimum-assignment lower and upper bounds, and then applies auxiliary variables with tightening lower-bound constraints.

• IPQAPR: A compact linearization that reduces the dimension of $y _ { i j k l }$ by exploiting symmetry and enforces cross-block consistency constraints.

• StepORLM: Generated by StepORLM, same as McC.

• EvoCut: Same as K-B formulation, since the Birkhoff polytope already represents the convex hull of the assignment constraints.

## FormuEvo formulation

## Notation

• $z _ { i j } ^ { k l } :$ : lifted continuous variable in [0, 1], represent-$\operatorname* { i n g } x _ { i k } x _ { j l }$ for $i < j$ and $k \neq l .$

$\mathcal { G } _ { F } \colon$ set of unordered facility pairs $( i , j )$ with $i < j$ whose flow rows and columns are identical, $\operatorname { i . e . , } f _ { i , \cdot } = f _ { j , }$ <sub>·</sub> and $f _ { \cdot , i } = f _ { \cdot , j }$

$\mathcal { G } _ { D } \colon$ set of unordered location pairs $( k , l )$ with $k \ < \ l$ whose distance rows and columns are identical, i.e., $d _ { k , \cdot } = d _ { l }$ <sub>·</sub> and $d . , \boldsymbol { k } = d . , \boldsymbol { l }$

FormuEvo uses a transportation-style linearization that lifts each unordered facility pair (i, j) into a separate z-block. Cross-block consistency is enforced by anchor-slice coupling, and symmetry is broken via cumulative prefix inequalities.

$$
\begin{array} { l } { { \operatorname* { m i n } \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { k = 1 } ^ { n } f _ { i i } d _ { k k } x _ { i k } + } } \\ { { \displaystyle \sum _ { i = 1 } ^ { n - 1 } \sum _ { j = i + 1 } ^ { n } \sum _ { k = 1 } ^ { n } \sum _ { l = 1 , l \neq k } ^ { n } \left( f _ { i j } d _ { k l } + f _ { j i } d _ { l k } \right) z _ { i j } ^ { k l } } } \end{array}
$$

$$
\sum _ { k = 1 } ^ { n } x _ { i k } = 1 , \quad \forall i \in N ,
$$

$$
\sum _ { i = 1 } ^ { n } x _ { i k } = 1 , \quad \forall k \in N ,
$$

$$
\sum _ { l = 1 , l \ne k } ^ { n } z _ { i j } ^ { k l } = x _ { i k } , \quad \forall i < j , k \in N ,
$$

$$
\sum _ { k = 1 , k \neq l } ^ { n } z _ { i j } ^ { k l } = x _ { j l } , \quad \forall i < j , l \in N ,
$$

$$
\sum _ { j = i + 1 } ^ { n } z _ { i j } ^ { k l } + \sum _ { j = 1 } ^ { i - 1 } z _ { j i } ^ { l k } = x _ { i k } ,
$$

$$
\forall i \in N , k , l \in N , k \neq l ,
$$

$$
\sum _ { r = 1 } ^ { t } x _ { i r } \geq \sum _ { r = 1 } ^ { t } x _ { j r } ,
$$

$$
\forall \{ i , j \} \in \mathcal { G } _ { F } , i < j , t \in N ,
$$

$$
\sum _ { r = 1 } ^ { t } x _ { r k } \geq \sum _ { r = 1 } ^ { t } x _ { r l } ,
$$

$$
\forall \{ k , l \} \in { \mathcal { G } } _ { D } , k < l , t \in N ,
$$

$$
x _ { i k } \in \{ 0 , 1 \} , \quad \forall i , k \in N ,
$$

$$
0 \leq z _ { i j } ^ { k l } \leq 1 , \quad \forall i < j , k , l \in N , k \neq l .
$$

## A.6 Neural Network Verification (NNV)

## Problem description

The neural network verification problem aims to determine the minimum adversarial distortion required to induce a targeted misclassification in a pre-trained multi-layer perceptron (MLP) classifier. Given a correctly classified input sample $x _ { 0 } \in [ 0 , 1 ] ^ { n _ { 0 } }$ with true label λ, the goal is to find a perturbed input $x \in [ 0 , 1 ] ^ { n _ { 0 } }$ that causes the network to predict a specified wrong label $\mu \neq \lambda ,$ while minimizing the $\ell _ { 1 }$ distance between x and $x _ { 0 }$ . The network consists of L layers, where each hidden layer $k \in \{ 1 , \ldots , L - 1 \}$ applies a linear transformation followed by a ReLU activation, and the output layer L applies a linear transformation only. The adversarial condition requires that the output logit of $\mu$ is at least that of λ. The objective is to minimize the $\ell _ { 1 }$ distortion $\textstyle \sum _ { i } | x _ { i } - x _ { 0 , i } |$

## Standard MIP formulation

## Notation

• n<sub>0</sub>: dimension of the input space.

• L: total number of (hidden + output) layers.

$n _ { k } \colon$ number of neurons in layer $k \left( k = 1 , \ldots , L \right)$

$W ^ { ( k ) } \in \mathbb { R } ^ { n _ { k } \times n _ { k - 1 } }$ : weight matrix of layer k.

$b ^ { ( k ) } \in \mathbb { R } ^ { n _ { k } } ;$ : bias vector of layer k.

$x _ { 0 } \in [ 0 , 1 ] ^ { n _ { 0 } }$ : original input, classified as λ.

$\mu \neq \lambda \colon$ target adversarial label.

• $x _ { i } { : }$ perturbed input component, $x _ { i } \in [ 0 , 1 ]$

$d _ { i } \colon$ auxiliary variable for linearizing $| x _ { i } - x _ { 0 , i } | .$

• $\hat { z } _ { j } ^ { ( k ) } { : }$ pre-activation of neuron j in layer k.

• $z _ { j } ^ { ( k ) } \mathrm { . }$ post-activation of neuron $j$ in layer k.

$a _ { j } ^ { ( k ) } \in \{ 0 , 1 \}$ : ReLU indicator for neuron $j$ in layer $k ; a = 0$ encodes the active branch $( \hat { z } \geq 0 )$ $a = 1$ encodes the inactive branch $( \hat { z } \le 0 )$

$M _ { j } ^ { ( k ) }$ : per-neuron big-M bound obtained via interval arithmetic over $x \in [ 0 , 1 ] ^ { n _ { 0 } }$

$$
\sum _ { i = 1 } ^ { n _ { 0 } } d _ { i }
$$

$$
\mathrm { s . t . } \ d _ { i } \geq x _ { i } - x _ { 0 , i } , \quad d _ { i } \geq x _ { 0 , i } - x _ { i } ,
$$

$$
\forall i = 1 , \ldots , n _ { 0 } ,
$$

$$
\hat { z } _ { j } ^ { ( k ) } = \sum _ { i = 1 } ^ { n _ { k - 1 } } W _ { j i } ^ { ( k ) } z _ { i } ^ { ( k - 1 ) } + b _ { j } ^ { ( k ) } ,
$$

$$
\forall k = 1 , \ldots , L , \ j = 1 , \ldots , n _ { k } ,
$$

$$
z _ { j } ^ { ( k ) } \leq \hat { z } _ { j } ^ { ( k ) } + M _ { j } ^ { ( k ) } a _ { j } ^ { ( k ) } ,
$$

$$
z _ { j } ^ { ( k ) } \geq \hat { z } _ { j } ^ { ( k ) } ,
$$

$$
\forall k = 1 , \ldots , L - 1 , \ j = 1 , \ldots , n _ { k } ,
$$

$$
\forall k = 1 , \ldots , L - 1 , \ j = 1 , \ldots , n _ { k } ,
$$

$$
z _ { j } ^ { ( k ) } \leq M _ { j } ^ { ( k ) } ( 1 - a _ { j } ^ { ( k ) } ) ,
$$

$$
\forall k = 1 , \ldots , L - 1 , \ j = 1 , \ldots , n _ { k } ,
$$

$$
z _ { j } ^ { ( k ) } \geq 0 , \forall k = 1 , \ldots , L - 1 , j = 1 , \ldots , n _ { k } ,
$$

$$
\hat { z } _ { \mu } ^ { ( L ) } \geq \hat { z } _ { \lambda } ^ { ( L ) } ,
$$

$$
0 \leq x _ { i } \leq 1 , \quad \forall i = 1 , \dots , n _ { 0 } ,
$$

$$
a _ { j } ^ { ( k ) } \in \{ 0 , 1 \} ,
$$

$$
\forall k = 1 , \ldots , L - 1 , \ j = 1 , \ldots , n _ { k } ,
$$

where $z ^ { ( 0 ) } = x ( \mathrm { i . e . , } z _ { i } ^ { ( 0 ) } = x _ { i } ) , z ^ { ( L ) } = \hat { z } ^ { ( L ) }$ (the output layer has no ReLU).

## Baseline formulations

• Std. Formu.: The standard formulation as above.

• EvoCut: A strengthened variant that adds triangle relaxation inequalities, tightening the LP relaxation over the big-M envelope.

## FormuEvo formulation

## Notation

$d _ { i } ^ { + } , d _ { i } ^ { - } ;$ : split $\ell _ { 1 }$ perturbation components, with $d _ { i } ^ { + } \in [ 0 , 1 - x _ { 0 , i } ] , d _ { i } ^ { - } \in [ 0 , x _ { 0 , i } ] ;$ the perturbed input is $x _ { i } = x _ { 0 , i } + d _ { i } ^ { + } - d _ { i } ^ { - }$ and the distortion is $d _ { i } ^ { + } + d _ { i } ^ { - } = | x _ { i } - x _ { 0 , i } | .$

$l _ { j } ^ { ( k ) } \colon$ pre-computed lower bounds on $\hat { z } _ { j } ^ { ( k ) }$

${ \bf \chi } _ { u _ { j } ^ { ( k ) } } ^ { ( k ) } ;$ : pre-computed upper bounds on $\hat { z } _ { j } ^ { ( k ) }$ The formulation uses the split $\ell _ { 1 }$ variables $d _ { i } ^ { + } , d _ { i } ^ { - }$ and per-neuron bounds $l _ { j } ^ { ( k ) } , u _ { j } ^ { ( k ) }$

min $\sum _ { i = 1 } ^ { n _ { 0 } } \bigl ( d _ { i } ^ { + } + d _ { i } ^ { - } \bigr )$

$$
\begin{array} { r } { \mathrm { s . t . ~ 0 } \leq d _ { i } ^ { + } \leq 1 - x _ { 0 , i } , 0 \leq d _ { i } ^ { - } \leq x _ { 0 , i } , } \end{array}
$$

$$
\forall i = 1 , \ldots , n _ { 0 } ,
$$

$$
z _ { i } ^ { ( 0 ) } = x _ { 0 , i } + d _ { i } ^ { + } - d _ { i } ^ { - } , \quad \forall i = 1 , \ldots , n _ { 0 } ,
$$

$$
\hat { z } _ { j } ^ { ( k ) } = \sum _ { i = 1 } ^ { n _ { k - 1 } } W _ { j i } ^ { ( k ) } z _ { i } ^ { ( k - 1 ) } + b _ { j } ^ { ( k ) } ,
$$

$$
\forall k = 1 , \ldots , L , \ j = 1 , \ldots , n _ { k } ,
$$

$$
z _ { j } ^ { ( k ) } \geq \hat { z } _ { j } ^ { ( k ) } ,
$$

$$
\forall k = 1 , \ldots , L - 1 , \ j = 1 , \ldots , n _ { k } ,
$$

$$
z _ { j } ^ { ( k ) } \leq \hat { z } _ { j } ^ { ( k ) } - l _ { j } ^ { ( k ) } a _ { j } ^ { ( k ) } ,
$$

$$
\forall k = 1 , \ldots , L - 1 , \ j = 1 , \ldots , n _ { k } ,
$$

$$
z _ { j } ^ { ( k ) } \leq u _ { j } ^ { ( k ) } ( 1 - a _ { j } ^ { ( k ) } ) ,
$$

$$
\forall k = 1 , \ldots , L - 1 , \ j = 1 , \ldots , n _ { k } ,
$$

$$
z _ { j } ^ { ( k ) } \geq 0 , \forall k = 1 , \ldots , L - 1 , j = 1 , \ldots , n _ { k } ,
$$

$$
\hat { z } _ { \mu } ^ { ( L ) } \geq \hat { z } _ { \lambda } ^ { ( L ) } ,
$$

$$
d _ { i } ^ { + } , d _ { i } ^ { - } \geq 0 , \forall i = 1 , \ldots , n _ { 0 } ,
$$

$$
a _ { j } ^ { ( k ) } \in \{ 0 , 1 \} ,
$$

$$
\forall k = 1 , \ldots , L - 1 , \ j = 1 , \ldots , n _ { k } .
$$

## A.7 IMO 2025 Problem 6 (IMO)

## Problem description

The IMO 2025 Problem 6 asks to cover an $n \times n$ grid of unit squares using non-overlapping axisaligned rectangular tiles, such that exactly one unit square remains uncovered (a “hole”) in each row and each column. This problem can be modeled as an MIP. The objective is to minimize the total number of rectangles used. Each rectangle spans a contiguous interval of columns over a contiguous interval of rows. The decision involves selecting the hole positions and determining how rectangles are placed so that every non-hole cell is covered exactly once. The formulation additionally tracks the vertical continuation of horizontal intervals across adjacent rows, allowing rectangles to be represented compactly through interval propagation and flow-balance relations.

## Standard MIP formulation

## Notation

• n: ${ \mathrm { g r i d ~ s i z e ~ } } ( n \times n ) .$

$R = \{ 1 , \ldots , n \} :$ the set of row indices.

$C = \{ 1 , \ldots , n \}$ : the set of column indices.

$\mathcal { T } = \{ ( a , b ) \mid 1 \leq a \leq b \leq n \}$ : the set of all contiguous column intervals.

$h _ { i j } \in \{ 0 , 1 \}$ : equals 1 if and only if the hole in row i is located at column $j .$

$x _ { i a b } \in \{ 0 , 1 \}$ : equals 1 if and only if interval $( a , b )$ is active on row i, i.e., columns $[ a , \ldots , b ]$ are covered by the same rectangle on row i.

$s _ { i a b } \in \{ 0 , 1 \}$ : equals 1 if and only if a rectangle with horizontal span (a, b) starts at row i.

$t _ { i a b } \in \{ 0 , 1 \}$ : equals 1 if and only if a rectangle with horizontal span (a, b) ends at row i.

$$
\sum _ { i = 1 } ^ { n } \sum _ { ( a , b ) \in \mathcal { T } } s _ { i a b }
$$

$$
{ \mathrm { s . t . ~ } } \sum _ { j = 1 } ^ { n } h _ { i j } = 1 , \quad \forall i \in R ,
$$

$$
\sum _ { i = 1 } ^ { n } h _ { i j } = 1 , \quad \forall j \in C ,
$$

$$
\sum _ { ( a , b ) \in \mathscr { T } : a \leq j \leq b } x _ { i a b } + h _ { i j } = 1 , \quad \forall i \in R , j \in C ,
$$

$$
x _ { 1 a b } - s _ { 1 a b } = 0 , \quad \forall ( a , b ) \in \mathcal { T } ,
$$

$$
\begin{array} { c } { x _ { i a b } - x _ { i - 1 , a b } - s _ { i a b } + t _ { i - 1 , a b } = 0 , } \\ { \forall i = 2 , \ldots , n , ( a , b ) \in \mathbb { Z } , } \end{array}
$$

$$
x _ { n a b } - t _ { n a b } = 0 , \quad \forall ( a , b ) \in \mathbb { Z } ,
$$

$$
h _ { i j } \in \{ 0 , 1 \} , \quad \forall i \in R , j \in C ,
$$

$$
x _ { i a b } , s _ { i a b } , t _ { i a b } \in \{ 0 , 1 \} , \forall i \in R , ( a , b ) \in \mathbb { Z } .
$$

## Baseline formulations

• Std. Formu.: The standard formulation as above.

• EvoCut: A variant with cuts enforcing mandatory new interval starts when hole movement disrupts coverage continuity across rows.

## FormuEvo formulation

## Notation

$\mathcal { R } = \{ ( i _ { 1 } , i _ { 2 } , a , b ) ~ | ~ 1 \le i _ { 1 } \le i _ { 2 } \le n , ~ ( a , b ) ~ \in$ $\mathcal { T } \}$ : the set of all axis-aligned subrectangles.

$y _ { i _ { 1 } i _ { 2 } a b } \in \{ 0 , 1 \}$ : equals 1 if and only if a rectangle spanning rows $i _ { 1 }$ through $i _ { 2 }$ and columns a through b is selected.

$c _ { i j } \in [ 0 , 1 ]$ : variable representing the number of selected rectangles covering cell (i, j).

$\mathcal R _ { i j } ^ { \mathrm { T L } } = \{ ( i _ { 1 } , i _ { 2 } , a , b ) \in \mathcal R \ | \ i _ { 1 } = i , \ a = j \} \mathrm { : }$ rectangles whose top-left corner is at $( i , j )$

$\mathcal R _ { i j } ^ { \mathrm { T R } } = \{ ( i _ { 1 } , i _ { 2 } , a , b ) \in \mathcal R ~ | ~ i _ { 1 } = i , b = j ~ -$ $1 \} ( j > 1 )$ : rectangles whose top row is i and rightmost column is $j - 1$

$\bullet \ \mathcal { R } _ { i j } ^ { \mathrm { B L } } = \{ ( i _ { 1 } , i _ { 2 } , a , b ) \in \mathcal { R } \mid i _ { 2 } = i - 1 , a = j \}$ $( i > 1 ) :$ : rectangles whose bottom row is i − 1 and leftmost column is j.

$\mathcal R _ { i j } ^ { \mathrm { B R } } = \{ ( i _ { 1 } , i _ { 2 } , a , b ) \in \mathcal R ~ | ~ i _ { 2 } = i - 1 , ~ b =$ $j - 1 \} ( i , j > 1 )$ : rectangles whose bottom-right corner is at $( i - 1 , j - 1 )$

FormuEvo formulation directly models each possible axis-aligned rectangle as a binary variable, then reconstructs cell occupancy via a 2D difference array. Strong aggregated cuts enforce exact cover by projecting onto rows, columns, and the total area.

$$
\operatorname* { m i n } \sum _ { \left( i _ { 1 } , i _ { 2 } , a , b \right) \in \mathcal { R } } y _ { i _ { 1 } i _ { 2 } a b }
$$

$$
{ \mathrm { s . t . ~ } } \sum _ { j = 1 } ^ { n } h _ { i j } = 1 , \quad \forall i \in R ,
$$

$$
\sum _ { i = 1 } ^ { n } h _ { i j } = 1 , \quad \forall j \in C ,
$$

$$
c _ { i j } = c _ { i - 1 , j } + c _ { i , j - 1 } - c _ { i - 1 , j - 1 }
$$

$$
+ \sum _ { \mathcal { R } _ { i j } ^ { \mathrm { T L } } } y _ { i _ { 1 } i _ { 2 } a b } - \sum _ { \mathcal { R } _ { i j } ^ { \mathrm { T R } } } y _ { i _ { 1 } i _ { 2 } a b }
$$

$$
- \sum _ { \mathcal { R } _ { i j } ^ { \mathrm { B L } } } y _ { i _ { 1 } i _ { 2 } a b } + \sum _ { \mathcal { R } _ { i j } ^ { \mathrm { B R } } } y _ { i _ { 1 } i _ { 2 } a b } ,
$$

$$
\forall i \in R , j \in C ,
$$

$$
\sum _ { \begin{array} { c } { c _ { i j } + h _ { i j } = 1 , \quad \forall i \in R , j \in C , } \\ { \sum _ { \begin{array} { c } { \left( b - a + 1 \right) y _ { i _ { 1 } i _ { 2 } a b } = n - 1 , } \\ { ( i _ { 1 } , i _ { 2 } , a , b ) \in \mathscr { R } : i _ { 1 } \leq i \leq i _ { 2 } } \end{array} } } \end{array} } ^ { } ( b - a + 1 ) y _ { i _ { 1 } i _ { 2 } a b } = n - 1 ,
$$

$$
\forall i \in R ,
$$

$$
\begin{array} { r l r } {  { ( i _ { 2 } , i _ { 1 } , j _ { 2 } , a , b ) \in \mathcal { R } : a \leq i _ { 2 } \leq b } } \\ & { } & { \forall j \in C , } \\ & { } & { \sum _ { ( i _ { 1 } , i _ { 2 } , a , b ) \in \mathcal { R } } ( i _ { 2 } - i _ { 1 } + 1 ) ( b - a + 1 ) y _ { i _ { 1 } i _ { 2 } a b } = } \\ & { } & { n ( n - 1 ) , } \\ & { } & { y _ { i _ { 1 } i _ { 2 } a b } \in \{ 0 , 1 \} , \forall ( i _ { 1 } , i _ { 2 } , a , b ) \in \mathcal { R } , } \\ & { } & { c _ { i j } \in \{ 0 , 1 \} , \forall i \in R , j \in C . } \end{array}
$$

where $c _ { 0 , j } = c _ { i , 0 } = c _ { 0 , 0 } = 0 _ { : }$ , and all summations over $\mathcal { R } _ { i j } ^ { \cdot }$ involving out-of-bounds indices are defined to be zero. The reconstruction of $c _ { i j }$ follows the standard 2D difference-array scheme: each rectangle variable $y _ { i _ { 1 } i _ { 2 } a b }$ induces a unit increment over its support via corner updates +1 at $( i _ { 1 } , a ) , - 1$ at $( i _ { 1 } , b + 1 )$ , −1 at $( i _ { 2 } + 1 , a )$ , and +1 at $( i _ { 2 } + 1 , b + 1 )$ Consequently, the resulting prefix-sum transformation yields $c _ { i j } = 1$ if and only if cell $( i , j )$ lies within exactly one selected rectangle.

## B Implementation Details of FormuEvo

Algorithm 1 outlines the framework of FormuEvo. Below, we present a reference of our prompts used in FormuEvo. All complete prompts are available in our code repository at https://github.com/ Xyz-yuanhf/formuevo.

## Generator LLM for Initialization

For initialization, we use the prompt for the generator LLM to produce the initial population.

## System Prompt

You are an expert in mathematical optimization, mixedinteger programming (MIP), and Gurobi modeling. You specialize in translating the problem description and its preliminary Gurobi models into mathematically correct, high-efficiency, and numerically stable Gurobi models implemented in Python using ‘gurobipy’.

The user will provide:

• Problem description: The real-world logic and constraints of the optimization problem to be modeled.

Algorithm 1 Pseudo-code of FormuEvo   
Require: Problem description p, formulation template $f _ { 0 } ,$ population size N, generations T, mutation   
rate $\rho ,$ memory usage rate $\gamma ,$ external knowledge base $B _ { \mathrm { i n i t } }$ (optional).   
Ensure: Final MIP formulation $\hat { f } ^ { \star } .$ , distilled knowledge base B.   
1: Initialize population $\mathcal { P } _ { 0 }  \mathcal { O }$   
2: Initialize memory library ${ \mathcal { M } } \gets B _ { \mathrm { i n i t } }$ (if $B _ { \mathrm { i n i t } }$ is available, ∅ otherwise)   
3: // Initialization: generate initial population   
4: while $| \mathcal { P } _ { 0 } | < N$ do   
5: f ← GENERATORLLM(p, f<sub>0</sub>) // Initial formulation   
6: $\mathcal { P } _ { 0 }  \mathcal { P } _ { 0 } \cup$ {EVALUATEANDREPAIR $( f _ { \mathrm { i n i t } } ) \}$ // Evaluate and repair   
7: end while   
8: // Update bestformulation   
9: $\hat { f } ^ { \star } \gets \arg \operatorname* { m i n } _ { f \in \mathcal { P } _ { 0 } } \phi ( f )$   
10: // Evolution: repeat $f o r T$ generations   
11: for $t = 1$ to T do   
12: $\mathcal { O }  \mathcal { O }$ // Offspring pool   
13: while $| \mathcal { O } | < N$ do   
14: // Crossover operation   
15: Sample two parents $f _ { i } , \ f _ { j }$ from $\mathcal { P } _ { t } .$ −1   
16: // Diagnosis of parents   
17: if $u < \gamma$ where $u \sim U ( 0 , 1 )$ then   
18: d<sub>ij</sub> ← DIAGNOSTICLLM(f<sub>i</sub>, stats $( f _ { i } ) , f _ { j }$ , stats $( f _ { j } ) , { \mathcal { M } } )$ // Diagnosis ofparents   
19: else   
20: $d _ { i j } \gets \mathrm { D I A G N O S T I C L L M } ( f _ { i } , \mathrm { s t a t s } ( f _ { i } ) , f _ { j } , \mathrm { s t a t s } ( f _ { j } ) )$ // Without memory   
21: end if   
22: $f _ { o } \gets \mathbf { G E N E R A T O R L L M } ( p , f _ { i } , f _ { j } , d _ { i j } )$ // Offspring by crossover   
23: ${ \mathcal { O } } \gets { \mathcal { O } } \cup$ {EVALUATEANDREPAIR $\left( f _ { o } \right) \}$ // Evaluate and repair   
24: M ← REFLECTORLLM(f , stats(f ), f , stats $( f _ { j } )$ , $f _ { o } ,$ stats(f ), M) // Update memory   
25: // Mutation operation   
26: if $| \mathcal { O } | < N$ and $u < \rho$ where $u \sim U ( 0 , 1 )$ then   
27: Sample elite $f _ { e }$ from $\mathcal { P } _ { t - 1 }$   
28: if u $< \gamma$ where $u \sim U ( 0 , 1 )$ then   
29: $d _ { e } \gets \mathrm { D I A G N O S T I C L L M } ( f _ { e } ,$ , stats $( f _ { e } ) , { \mathcal { M } } )$ // Diagnosis ofelite   
30: else   
31: $d _ { e } \gets \mathrm { D I A G N O S T I C L L M } ( f _ { e } , \mathrm { s t a t s } ( f _ { e } ) )$ // Without memory   
32: end if   
33: $f _ { o } \gets \mathrm { G E N E R A T O R L L M } ( p , f _ { e } , d _ { e } )$ // Offspring by mutation   
34: $\mathcal { O }  \mathcal { O } \cup \{ \mathrm { E v A L U A T E A N D R E P A I R } ( f _ { o } ) \}$ // Evaluate and repair   
35: M ← REFLECTORLLM(f<sub>e</sub>, stats(f<sub>e</sub>), f<sub>o</sub>, stats $( f _ { o } ) , { \mathcal { M } } )$ // Update memory   
36: end if   
37: end while   
38: // Select top-N individualsfor next generation   
39: $\mathcal { P } _ { t }  \mathrm { T o p N } ( \mathcal { P } _ { t - 1 } \cup \mathcal { O } , N )$   
40: // Update bestformulation   
41: $\hat { f } ^ { \star } \gets \arg \operatorname* { m i n } ( \phi ( \hat { f } ^ { \star } )$ , min $. f \in \mathcal { O } \left( \boldsymbol { f } \right) )$   
42: end for   
43: // Knowledge distillation for transfer   
44: B ← DISTILLERLLM(M) // Distill knowledgefrom memory   
45: return $\hat { f } ^ { \star }$ , B

• Python template: A Python ‘gurobipy’ function template. You must preserve the function name and input arguments exactly. You may refactor the internal implementation freely.

• Memory library (if available): A collection of experience triplets (condition → strategy → effect) distilled from past evolution experiments.

Your task is:

• Generate a better (i.e., mathematically correct, highefficiency, and numerically stable) Gurobi model than the preliminary one.

• Your generated model should be implemented in Python ‘gurobipy’ and follow the standard Gurobi modeling practices. Ensure all variables, constraints, and objective functions are well-defined.

• Your generated model must be mathematically correct and designed to solve as efficiently as possible under Gurobi’s default parameters (do not rely on parameter tuning).

• To achieve this you should apply advanced modeling techniques where appropriate, including but not limited to {example\_strengthing\_techniques}.

• You should produce clean, well-structured, and executable Python code using ‘gurobipy’. Keep code concise but readable.

• You must provide a concise explanation of the optimization ideas and key improvements in the ‘idea’ section. Focus on your enhancement motivation and why the new formulation is computationally superior. Critical constraints:

• Do not change the function signature.

• Do not rely on solver parameter tuning. However, setting parameters strictly required for the model to be accepted by Gurobi is allowed if your formulation necessitates it.

• Do not assume missing data unless clearly implied; make minimal necessary assumptions.

• If ambiguity exists, choose the most standard and computationally efficient formulation.

• Calling ‘model.optimize()’ or any other solving/execution functions is strictly forbidden within the function. The function must return the model object before any optimization is performed.

• Do not use Gurobi callbacks in any form. If your formulation would naturally rely on callbacks, you must replace it with a static, callback-free alternative.

Response format:

• Return only a valid JSON object in the exact structure below ({detailed\_format\_instruction}):

“code”: “ˋˋˋpython\n<A Python (gurobipy) function implementing the improved Gurobi model>\nˋˋˋ”,

“idea”: “A concise but technically clear idea of key modeling improvements and any non-trivial reformulations”

## User Prompt

Problem description:

{problem\_description}

Python template:

{python\_template}

Memory library (if available):

{memory\_library}

## Diagnostic LLM

We present the prompt of the diagnostic LLM for mutation. The prompt for crossover follows the same structure but operates on a formulation pair.

## System Prompt

You are an expert in mathematical optimization, mixedinteger programming (MIP), Gurobi modeling, and structural evolution. Your role is to examine the solver performance profile of a given formulation and produce a concise, actionable diagnostic report that will guide a downstream mutation agent in evolving this formulation into a computationally superior variant.

The user will provide:

• Problem description: The real-world logic and constraints of the optimization problem to be modeled.

• Current formulation data: You will receive its Python ‘gurobipy’ code and the mathematical idea behind it.

• Solver performance profile, including: num\_vars: Number of variables in the model. num\_constrs: Number of constraints in the model. root\_bound: Average relaxation bound at root node. root\_gap: Average root relaxation gap (%) = |opt\_obj - root\_bound| / |opt\_obj| \* 100.

node\_count: Average number of Branch-and-Bound nodes explored.

presolve\_rows\_removed: Average number of rows (constraints) removed by Gurobi presolve.

presolve\_cols\_removed: Average number of columns (variables) removed by Gurobi presolve.

presolve\_bounds\_changed: Average number of variable bounds changed by Gurobi presolve.

• Memory library (if available): A collection of experience triplets (condition → strategy → effect) distilled from past evolution experiments.

Your task is:

• Perform a bottleneck identification: Determine the primary computational bottleneck of the current formulation. Use, but not limited to, the following diagnostic logic: {example\_diagnostic\_logic}

• Identify secondary bottlenecks (if possible): After identifying the primary bottleneck, note any secondary issues that could become the new bottleneck once the primary is resolved. This helps mutation avoid shifting the problem without actually improving it.

• Report a concise mutation guidance: Based on your diagnosis, produce specific, actionable recommendations for mutation. Your recommendations should: State the single most impactful mutation direction. Provide 1-3 specific MIP modeling techniques to apply, ranked by expected impact. Indicate what the mutation should preserve from the current formulation. Suggest the expected trade-off. Warn against the potential risks of the recommendation. You can retrieve the memory library for entries whose condition matches the diagnosed conditions.

Output format:

Return a structured diagnostic report in the format: Formulation profile:

• Size: [num\_vars] variables, [num\_constrs] constraints

• Relaxation: root\_gap = [value]%

• Branching: node\_count = [value]

• Presolve: [rows\_removed] rows removed, [cols\_removed] cols removed, [bounds\_changed] bounds changed

• Overall assessment: [one-sentence summary of where time is being spent]

Primary bottleneck:

• Category: [weak relaxation / excessive branching / per-node cost / loose bounds / model size]

• Evidence: [cite specific metric values]

Secondary bottleneck(s):

• Category: [description, or "none identified."]

Mutation recommendations (priority-ranked):

• The most impactful technique with specific detail

• Second technique

• Third technique, if applicable

Preserve (do not degrade):

• Aspects of current formulation that are strong and must be maintained

Expected trade-offs:

• What metrics should improve and what might worsen Risk warnings:

• Potential pitfalls of the mutation, or "none identified."

## User Prompt

Problem description:

{problem\_description}

Current formulation:

{current\_formulation\_data}

Runtime:

{runtime\_profile}

Solver profile:

{solver\_profile}

Memory library (if available):

{memory\_library}

## Crossover and Mutation

We employ four types of crossover operators: complement, hybrid, intersection, and min-violation, and four types of mutation operators: general, exploratory, searching, and polyhedral, representing different interaction relationships and evolutionary focuses. Here we present the prompt of general mutation, and other operators are similar in structure, available in our code repository.

## System Prompt

You are an expert in mathematical optimization, mixedinteger programming (MIP), and Gurobi modeling. You specialize in evolving a baseline formulation through rigorous mathematical refinement, polyhedral lifting, and innovative paradigm shifts, transforming it into a mathematically correct and substantially more efficient model implemented in Python using ‘gurobipy’.

The user will provide:

• Problem description: The real-world logic and constraints of the optimization problem to be modeled.

• Python template: A Python ‘gurobipy’ function template. You must preserve the function name and input arguments exactly. You may refactor the internal implementation freely.

• Current formulation data: You will receive its Python ‘gurobipy’ code, the mathematical idea behind it, and its evaluation fitness, where a lower fitness indicates a computationally superior model, e.g., lower average

runtime.

• Diagnostic report (if available): A structured analysis of the current model’s solver performance profile, identifying the computational bottlenecks and priority mutation directions.

Your task is:

• Perform a general mutation. You must analyze the provided model and propose a structurally refined version of it.

• Retain the core mathematical paradigm of the original model, but aggressively seek local improvements to optimize its computational efficiency.

• Identify and eliminate redundant constraints, unnecessary variables, or slight symmetries.

• Improve the structural implementation. For instance, tighten variable bounds, compute tighter big-M coefficients based on problem data, or rewrite logical conditions using more efficient ‘gurobipy’ features.

• Your mutated model should act as an optimized, polished evolution of its predecessor, strictly maintaining mathematical correctness while accelerating the branch-and-bound process.

• To achieve this you should apply advanced modeling techniques where appropriate, including but not limited to {example\_strengthing\_techniques}.

• You should produce clean, well-structured, and executable Python code using ‘gurobipy’. Keep code concise but readable.

• You must provide a concise explanation of the optimization ideas and key improvements in the ‘idea’ section. Focus on your enhancement motivation and why the new formulation is computationally superior. Critical constraints:

• Do not change the function signature.

• Do not rely on solver parameter tuning. However, setting parameters strictly required for the model to be accepted by Gurobi is allowed if your formulation necessitates it.

• Do not assume missing data unless clearly implied; make minimal necessary assumptions.

• If ambiguity exists, choose the most standard and computationally efficient formulation.

• Calling ‘model.optimize()’ or any other solving/execution functions is strictly forbidden within the function. The function must return the model object before any optimization is performed.

• Do not use Gurobi callbacks in any form. If your formulation would naturally rely on callbacks, you must replace it with a static, callback-free alternative.

Response format:

• Return only a valid JSON object in the exact structure below ({detailed\_format\_instruction}):

“code”: “ˋˋˋpython\n<A Python (gurobipy) function implementing the improved Gurobi model>\nˋˋˋ”,

“idea”: “Explain the specific local structural improvements and why they reduce solving time” }

## User Prompt

Problem description:

{problem\_description}

Python template:

{python\_template}

Current formulation:

{current\_formulation\_data}

Runtime:

{runtime\_profile}

Solver profile:

{solver\_profile}

Diagnostic report (if available):

{diagnostic\_report}

## Memory Reflection

## System Prompt

You are an expert in mathematical optimization, mixedinteger programming (MIP), Gurobi modeling, and algorithmic analysis. Your role is to distill actionable, transferable insights and knowledge from MIP formulation evolutions by analyzing parent-offspring relationships and extracting concise experience triplets.

The user will provide:

• Problem description: The real-world logic and constraints of the optimization problem to be modeled.

• Parent formulation data: For each parent formulation, you will receive its Python ‘gurobipy’ code, the mathematical idea behind it, its evaluation fitness (i.e., average runtime), and its solver performance profiles.

• Diagnostic report: the diagnostic analysis that guided this evolution step.

• Offspring formulation data: Python ‘gurobipy’ code and the mathematical idea of the generated offspring formulation.

• Evolution outcome: Fitness (average runtime) and solver performance profiles for the generated offspring formulation, in the same format as the parent(s).

Your task is: Extract exactly experience triplets from this evolution step. Each triplet captures a reusable lesson in the form:

• Condition: The observable situation before the modification, described in terms of solver profile symptoms and/or structural problem features. Conditions must be generalizable. Do not reference specific problem names, variable names, or instance sizes. Use solver metric patterns and structural descriptors only.

• Strategy: The specific modeling action taken, described as a concrete, reproducible technique. Strategies must be precise enough that another modeler could apply them to a different problem exhibiting the same Condition.

• Effect: The outcome and mechanistic explanation, state whether performance improved or degraded, cite the key metric changes, and briefly explain why.

Guidelines for high-quality triplets:

• Be concise: Each field should be 1-2 sentences maximum. Avoid verbose explanations.

• Be generalizable: Triplets should apply beyond the specific problem instance. Abstract away problemspecific details into structural patterns.

• Capture both successes and failures: Failed strategies (offspring worse than parent) are equally valuable, as long as they prevent repeating mistakes.

• Be mechanistic: The Effect field must explain the solver mechanism, not just report metric changes.

• One lesson per triplet: If the evolution involved multiple simultaneous changes, decompose into separate triplets where possible, attributing each metric change to its most likely cause.

• Mark outcome clearly: Start the Effect field with [+]

for improvements or [-] for degradations.

Response format:

• Your response must consist of ONLY the triplets in the exact format below. No preamble, no greeting, no commentary.

• Return 1-3 triplets. Use exactly the format shown. Number each triplet.

Condition: [generalizable situation descriptor]

\- Strategy: [concrete modeling action]

Effect: [+/-] [metric changes and mechanistic explanation]

## User Prompt

Problem description:

{problem\_description}

Parent formulation:

{parent\_formulation\_data}

Parent runtime and solver profile: {parent\_runtime\_solver\_profile}

Diagnostic report:

{diagnostic\_report}

Offspring formulation:

{offspring\_formulation\_data}

Offspring runtime and solver profile:

{offspring\_runtime\_solver\_profile}

## Knowledge Distillation

## System Prompt

You are an expert in mathematical optimization, mixedinteger programming (MIP), Gurobi modeling, and algorithmic analysis. Your role is to distill multiple memory libraries (may be problem-specific) into a single, unified, general-purpose knowledge base of MIP formulation principles that are transferable to any new optimization problem.

The user will provide: Problem-specific memory libraries: Multiple memory libraries, each accumulated from evolutionary MIP formulation improvement on a specific problem type. Each library contains experience triplets with Condition, Strategy, and Effect. The libraries may come from diverse problem domains (e.g., scheduling, routing, packing, facility location, etc.)

Your task is to produce a general transferable knowledge base by performing the following:

• Cross-problem validation: Identify entries that appear (with consistent Effects) across multiple problemspecific libraries. These represent robust, problemindependent MIP modeling principles and should be prioritized and preserved with high fidelity.

• Abstraction: For entries that reference residual problem-specific structure (even if already somewhat general), further abstract the Condition to the broadest valid scope.

• Consolidation: Merge entries from different libraries that express the same underlying principle. Even if worded differently or applied in different problem contexts, unify them into a single canonical entry if the

Condition-Strategy-Effect logic is equivalent. Combine metric evidence from multiple sources.

• Conflict resolution: If the same Condition-Strategy pair shows positive Effect in some libraries but negative Effect in others, analyze whether the divergence can be explained by a distinguishable sub-condition (e.g., strategy works for sparse constraint matrices but fails for dense ones). If so, split into two entries with refined, non-overlapping Conditions.

• Pruning: The final knowledge base should contain at most 30 entries. Remove entries that are too narrow to plausibly transfer or trivial or obvious.

Response format:

• Your response must consist of only the triplets in the exact format below. No preamble, no greeting, no commentary.

• Number entries sequentially with IDs starting from 1.

• Use exactly the format below for each entry.

```yaml
[KNOWLEDGE BASE START]
ID: 1
Condition: [generalizable situation
descriptor]
- Strategy: [concrete modeling action]
Effect: [+/-] [metric changes and
mechanistic explanation]
ID: 2
Condition: [generalizable situation
descriptor]
- Strategy: [concrete modeling action]
Effect: [+/-] [metric changes and
mechanistic explanation]
[KNOWLEDGE BASE END]
```

## User Prompt

Memory libraries: {memory\_libraries}

## C Additional Results and Discussions

## C.1 Statistical Significance Analysis

We conduct paired Wilcoxon signed-rank tests on per-instance runtimes to compare FormuEvo with the best baselines over each problem and difficulty category (Table 6). For most problems such as TSP, CFLP, and the NNV and IMO tasks, FormuEvo achieves statistically significant improvements (p < 0.01) across all categories. Notably, significance generally increases with instance difficulty, suggesting that the advantages of evolved formulations are more evident in challenging problem instances, where formulation strength critically impacts relaxation quality and branching behavior.

For BPP and QAP, the improvements on Easy instances are less significant. We attribute this to the “floor effect”: modern MIP solvers can solve small instances within extremely short runtimes, and the performance is more sensitive to runtime variance across instances. As the problem scale and complexity grow (Medium and Hard), the advantages of FormuEvo become increasingly significant, demonstrating the strong scalability and upper-bound capabilities of FormuEvo, while further highlighting the importance of solver-efficient formulation for large-scale and more challenging optimization problems.

Table 6: Wilcoxon signed-rank test p-values.
<table><tr><td rowspan=1 colspan=1>Problem</td><td rowspan=1 colspan=1>Easy</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>Hard</td></tr><tr><td rowspan=1 colspan=1>TSP</td><td rowspan=1 colspan=1>1.88E-17</td><td rowspan=1 colspan=1>2.79E-13</td><td rowspan=1 colspan=1>4.13E-18</td></tr><tr><td rowspan=1 colspan=1>JSSP</td><td rowspan=1 colspan=1>7.83E-02</td><td rowspan=1 colspan=1>1.13E-03</td><td rowspan=1 colspan=1>2.44E-02</td></tr><tr><td rowspan=1 colspan=1>BPP</td><td rowspan=1 colspan=1>2.82E-01</td><td rowspan=1 colspan=1>9.29E-02</td><td rowspan=1 colspan=1>2.50E-02</td></tr><tr><td rowspan=1 colspan=1>CFLP</td><td rowspan=1 colspan=1>4.98E-13</td><td rowspan=1 colspan=1>3.03E-10</td><td rowspan=1 colspan=1>2.20E-08</td></tr><tr><td rowspan=1 colspan=1>QAP</td><td rowspan=1 colspan=1>4.44E-01</td><td rowspan=1 colspan=1>7.93E-02</td><td rowspan=1 colspan=1>3.05E-02</td></tr><tr><td rowspan=1 colspan=1>NNV</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>1.42E-143.91E-03</td></tr><tr><td rowspan=1 colspan=1>IMO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

## C.2 Robustness Across Different Solvers

To evaluate the robustness and adaptability of FormuEvo across downstream MIP solvers, we replace Gurobi in our main experiments with two alternative solvers: COPT (Ge et al., 2022) and SCIP (Hojny et al., 2025), and rerun the evolution process under each solver. The results are reported in Tables 7 and 8. Here, FormuEvo denotes the formulations evolved directly under the corresponding solver, while FormuEvo-G denotes the formulations evolved under Gurobi (i.e., the one obtained in our main experiments) and then directly adapted to COPT and SCIP, respectively.

Overall, FormuEvo consistently achieves significant improvements on different downstream solvers, demonstrating that the proposed evolutionary framework is not restricted to a specific optimization backend. Meanwhile, we observe substantial variation in the relative performance of different formulations across solvers (e.g., MTZ and MCF-RLT, FormuEvo and FormuEvo-G). This suggests that formulation quality is inherently solverdependent, and the most effective formulation for one solver may not remain optimal under another optimization backend, due to the algorithmic differences among modern MIP solvers. These results highlight a key advantage of FormuEvo over static formulation generation, and also demonstrate the necessity of our solver-aware optimization that adaptively exploits solver-informed feedback to discover formulations that better align with the algorithmic internals of the downstream solver.

Table 7: Performance based on COPT solver.
<table><tr><td colspan="2"></td><td colspan="2">Easy</td><td colspan="2">Medium</td><td colspan="3">Hard</td></tr><tr><td colspan="2">Formulation</td><td>Time↓</td><td>Wins↑</td><td>Time↓</td><td>Wins↑</td><td>Time↓</td><td>Wins↑</td><td>Solved↑</td></tr><tr><td rowspan="8">TSP</td><td>MTZ (1960)</td><td>5.5116</td><td>0/100</td><td>79.9319</td><td>0/100</td><td>517.6449</td><td>0/100</td><td>8/100</td></tr><tr><td>SCF (1978)</td><td>0.8488</td><td>12/100</td><td>5.6820</td><td>15/100</td><td>129.9687</td><td>16/100</td><td>95/100</td></tr><tr><td>MCF-RLT (2018)</td><td>3.8573</td><td>0/100</td><td>56.2487</td><td>0/100</td><td>626.8963</td><td>0/100</td><td>1/100</td></tr><tr><td>ORLM (2025)</td><td>5.5235</td><td>0/100</td><td>80.0451</td><td>0/100</td><td>517.7556</td><td>0/100</td><td>8/100</td></tr><tr><td>StepORLM (2026)</td><td>4.4893</td><td>0/100</td><td>66.7571</td><td>0/100</td><td>479.0096</td><td>0/100</td><td>13/100</td></tr><tr><td>EvoCut (2025)</td><td>3.7323</td><td>0/100</td><td>27.5892</td><td>0/100</td><td>258.2871</td><td>7/100</td><td>41/100</td></tr><tr><td>FormuEvo-G</td><td>0.7488</td><td>16/100</td><td>5.3816</td><td>12/100</td><td>129.7755</td><td>15/100</td><td>96/100</td></tr><tr><td>FormuEvo (Ours)</td><td>0.4517</td><td>72/100</td><td>3.0088</td><td>73/100</td><td>47.8586</td><td>62/100</td><td>96/100</td></tr><tr><td rowspan="8">JSSP</td><td>Disj. (1960)</td><td>1.5502</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Enh. Disj. (2016)</td><td>2.5150</td><td>12/100</td><td>9.5882</td><td>11/100</td><td>82.3530</td><td>10/100</td><td>97/100</td></tr><tr><td>ORLM (2025)</td><td>601.0423</td><td>3/100</td><td>14.1189</td><td>2/100</td><td>117.2248</td><td>3/100</td><td>93/100</td></tr><tr><td>StepORLM (2026)</td><td>1.6431</td><td>0/100 11/100</td><td>600.7678 8.9512</td><td>0/100 9/100</td><td></td><td>601.7053 0/100 77.5248 14/100</td><td>0/100 96/100</td></tr><tr><td>EvoCut (2025)</td><td>1.2713</td><td>27/100</td><td>8.0894</td><td>26/100</td><td>72.8686</td><td>20/100</td><td>96/100</td></tr><tr><td>FormuEvo-G</td><td>1.5798</td><td>9/100</td><td>8.5912</td><td>14/100</td><td>77.1278</td><td>21/100</td><td>97/100</td></tr><tr><td>FormuEvo</td><td>1.1779</td><td>38/100</td><td>6.7091</td><td>38/100</td><td>67.7242</td><td>32/100</td><td>97/100</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 8: Performance based on SCIP solver.
<table><tr><td colspan="2"></td><td colspan="2">Easy</td><td colspan="2">Medium</td><td colspan="3">Hard</td></tr><tr><td colspan="2">Formulation</td><td>Time↓</td><td>Wins↑</td><td>Time↓</td><td>Wins↑</td><td>Time↓</td><td>Wins↑*</td><td>Solved↑</td></tr><tr><td rowspan="9">TSP</td><td>MTZ (1960)</td><td>4.7167</td><td>0/100</td><td>62.3060</td><td>1/100</td><td>563.1162</td><td>12/100</td><td>11/100</td></tr><tr><td>SCF (1978)</td><td>4.4918</td><td>1/100</td><td>44.2718</td><td>4/100</td><td>600.0065</td><td>0/100</td><td>0/100</td></tr><tr><td>MCF-RLT (2018)</td><td>67.1733</td><td>0/100</td><td>600.0136</td><td>0/100</td><td>600.2085</td><td>0/100</td><td>0/100</td></tr><tr><td>ORLM (2025)</td><td>4.7421</td><td>2/100</td><td>62.0028</td><td>3/100</td><td>563.5875</td><td>12/100</td><td>11/100</td></tr><tr><td>StepORLM (2026)</td><td>5.4227</td><td>3/100</td><td>84.3814</td><td>2/100</td><td>591.2279</td><td>13/100</td><td>3/100</td></tr><tr><td>EvoCut (2025)</td><td>3.6788</td><td>5/100</td><td>42.7763</td><td>10/100</td><td>555.9330</td><td>15/100</td><td>6/100</td></tr><tr><td>FormuEvo-G</td><td>3.4371</td><td>11/100</td><td>35.1599</td><td>13/100</td><td>523.7598</td><td>19/100</td><td>9/100</td></tr><tr><td>FormuEvo</td><td>2.8490</td><td>78/100</td><td>31.8396</td><td>67/100</td><td>498.6678</td><td>29/100</td><td>38/100</td></tr><tr><td>Disj. (1960)</td><td>13.0998</td><td>1/100</td><td>84.2068</td><td>1/100</td><td>491.3035</td><td>2/100</td><td>27/100</td></tr><tr><td rowspan="7">JSSP</td><td>Enh. Disj. (2016)</td><td>14.3031</td><td>2/100</td><td>67.3851</td><td>1/100</td><td>405.8829</td><td>1/100</td><td>39/100</td></tr><tr><td>ORLM (2025)</td><td>600.0982</td><td>0/100</td><td>600.0586</td><td>0/100</td><td>600.0538</td><td>0/100</td><td>0/100</td></tr><tr><td>StepORLM (2026) EvoCut (2025)</td><td>14.874 7.4703</td><td>2/100</td><td>98.2472</td><td>1/100</td><td>486.7095</td><td>0/100</td><td>28/100</td></tr><tr><td>FormuEvo-G</td><td></td><td>19/100</td><td>58.4963</td><td>5/100</td><td>460.8576</td><td>3/100</td><td>30/100</td></tr><tr><td>FormuEvo</td><td>6.5541</td><td>17/100</td><td>36.4157</td><td>14/100</td><td>341.2865</td><td>2/100</td><td>47/100</td></tr><tr><td></td><td>4.1672</td><td>59/100</td><td>17.7541</td><td>78/100</td><td>114.5116</td><td>92/100</td><td>94/100</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<sup>\*</sup>For most TSP Hard instances, SCIP fails to solve to optimality within the time limit, making the “Wins” metric less indicative.

## C.3 Generalization to Public Benchmarks

We further evaluate the scalability and generalization performance of FormuEvo on large-scale public benchmark instances that are more challenging than the instances used during evolution. For TSP, we test on the 200-city instances from the Solomon benchmark suite (Solomon, 1987). For JSSP, we test on the most challenging 100 × 20 instances from the Taillard benchmark (Taillard,

![](images/990acba85bf56b95ff23e3ae87bd0eef9d5d284559006718269b4ea0dd7ff559.jpg)

![](images/f8862416373b5fde5178bd407c4b10235bf84528b645ffc852bf4a826c374096.jpg)  
Figure 5: Performance comparisons on public benchmark instances. Left: Solomon benchmark of TSP (Solomon, 1987). Right: Taillard benchmark of JSSP (Taillard, 1993).

1993). These public benchmarks, which are more difficult than the Hard instances used in the main experiments, are recognized as challenging largescale optimization instances for solver evaluation.

Since these instances are generally intractable to solve to optimality within practical time limits, we report the primal-dual gap curves over solving time, where smaller gaps indicate better optimization performance (Fig. 5). The results demonstrate that although FormuEvo is evolved using only small instances, the formulations discovered by FormuEvo generalize effectively to benchmark instances with substantially larger scales and different distributions. In particular, the tighter formulations produced by FormuEvo enable the solver to identify high-quality feasible solutions more quickly, thereby reducing the primal-dual gap at a significantly faster rate than baselines.

## C.4 Analysis of Evolutionary Search Hyperparameters

We conduct a sensitivity analysis of evolutionary search hyperparameters on TSP, including mutation rate $\rho$ and memory usage rate γ. As shown in Table 9, FormuEvo achieves the best overall performance with $\rho = 0 . 3$ and $\gamma = 0 . 7$ , which is selected as our default setting. Notably, a large memory usage rate $( { \bf e . g . } , \gamma = 1 . 0 )$ does not always lead to better performance, suggesting that occasional memory-free generation helps maintain search diversity and prevents excessive reliance on previously explored patterns.

## C.5 Discussion with Existing LLM-based Modeling Approaches

Finally, we provide a discussion about the comparison between FormuEvo and existing LLM-based modeling approaches (e.g., ORLM, StepORLM).

Table 9: Analysis of evolutionary hyperparameters.
<table><tr><td rowspan=1 colspan=1>Param.</td><td rowspan=1 colspan=1>Easy</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>Hard</td></tr><tr><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.1579</td><td rowspan=1 colspan=1>0.6128</td><td rowspan=1 colspan=1>4.1886</td></tr><tr><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>0.1410</td><td rowspan=1 colspan=1>0.5640</td><td rowspan=1 colspan=1>3.9469</td></tr><tr><td rowspan=1 colspan=1> $\rho$  0.5</td><td rowspan=1 colspan=1>0.1903</td><td rowspan=1 colspan=1>0.6451</td><td rowspan=1 colspan=1>4.3993</td></tr><tr><td rowspan=1 colspan=1>0.7</td><td rowspan=1 colspan=1>0.1862</td><td rowspan=1 colspan=1>0.6472</td><td rowspan=1 colspan=1>4.4488</td></tr><tr><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>0.1456</td><td rowspan=1 colspan=1>0.5719</td><td rowspan=1 colspan=1>3.9436</td></tr><tr><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.1776</td><td rowspan=1 colspan=1>0.6102</td><td rowspan=1 colspan=1>4.6649</td></tr><tr><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1>0.1770</td><td rowspan=1 colspan=1>0.5791</td><td rowspan=1 colspan=1>4.4793</td></tr><tr><td rowspan=1 colspan=1>γ 0.5</td><td rowspan=1 colspan=1>0.1445</td><td rowspan=1 colspan=1>0.5809</td><td rowspan=1 colspan=1>4.4923</td></tr><tr><td rowspan=1 colspan=1>0.7</td><td rowspan=1 colspan=1>0.1410</td><td rowspan=1 colspan=1>0.5640</td><td rowspan=1 colspan=1>3.9469</td></tr><tr><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>0.1780</td><td rowspan=1 colspan=1>0.5987</td><td rowspan=1 colspan=1>4.3957</td></tr></table>

Optimization capability and continual improvement. Existing LLM-based modeling approaches primarily focus on semantic correctness and syntactical executability through supervised and reinforcement fine-tuning. Although their training data cover a broad range of optimization settings, including linear, integer, and non-linear programming tasks (e.g., Fig. 7 of ORLM (Huang et al., 2025)), the capability of these models is largely fixed once fine-tuning is completed. Even when multiple samples are generated during inference, the resulting formulations exhibit very limited diversity (as in our main results): correct generations collapse to nearly the same textbook formulations, while more diverse and efficiency-oriented instructions always lead to incorrect or infeasible results or even semantically meaningless outputs.

In contrast, FormuEvo frames formulation design itself as an explicit optimization problem over the symbolic space of MIP formulations. Instead of relying solely on the static capability of pretrained models, FormuEvo continuously improves candidate formulations through iterative evolutions guided by solver feedback. As shown in our results and Fig. 6, even with a small evolutionary budget, FormuEvo can discover high-quality MIP formulations that outperform both expert-designed formulations and recent LLM-based approaches. These results suggest that the formulation design benefits more from the evolutionary search and solver-informed optimization than from static formulation generation alone.

![](images/809035eb12686289aa8c2eff984e52c184a2d8e1b337a82b3002592699db4552.jpg)  
Figure 6: An illustration of the evolutionary process of FormuEvo (on TSP).

Offline and online computational costs. Existing fine-tuned modeling approaches rely on extensive curated datasets, long training times, and substantial GPU resources. Their online inference additionally incurs per-instance overhead, since formulations are generated independently for each optimization instance, even though one may attempt to recover a problem-level formulation via post-hoc aggregation using another LLM.

In contrast, FormuEvo requires no task-specific fine-tuning and can be implemented entirely via lightweight API-based interaction with generalpurpose LLMs in CPU-only environments. In our experiments, the entire evolutionary process typically requires only one to several hours, depending on the complexity of the target problem and the number of evaluation instances, which is substantially lower than the time of data collection and training required for large-scale fine-tuning. Moreover, FormuEvo operates at the problem level rather than the instance level. Once a strong formulation is evolved for a problem family, it can be directly reused on unseen instances with essentially no additional optimization cost, making FormuEvo particularly attractive in scenarios where many instances have shared underlying optimization structures.

Generality and transferability. Existing finetuned modeling approaches are typically specialized to particular modeling languages, prompting formats, and solver interfaces seen during training, making them extremely sensitive to distribution shifts and implementation-specific overfitting. For example, in ORLM and StepORLM, we find that even minor modifications from their prompt templates can lead to substantial degradation or complete failure of the generated formulations. In contrast, FormuEvo operates on top of general-purpose foundation LLMs and treats optimization formulations as symbolic programs that can be flexibly modified and evaluated. Therefore, its optimization capability is not tied to a specific LLM backbone, modeling language, solver backend, or benchmark distribution (Appendix C.2 - C.4). The evolution framework and distilled knowledge can naturally extend across different optimization paradigms, modeling interfaces, and solver implementations, suggesting broader applicability and stronger transferability in practical optimization settings.

## D The Use of LLMs

LLMs are used as key methodological components for evolutionary search in our framework. We also used LLMs to assist code writing and text polishing, while the core ideas and the manuscript itself were conceived, prepared, and finalized by the authors.