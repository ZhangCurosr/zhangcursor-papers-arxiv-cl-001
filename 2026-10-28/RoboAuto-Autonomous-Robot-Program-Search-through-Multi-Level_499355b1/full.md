# RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding

Buzhao Liu<sup>∗</sup> Jianing Ye<sup>∗</sup> Xiu Yuan<sup>∗</sup> Yangtong Feng Chengsong Huang Tyler Dinh Yevgeniy Vorobeychik Ning Zhang Chongjie Zhang Washington University in St. Louis

![](images/6b709f4994fa99219968a998b320bbb4f6402703a365849ac0eefded4295dea2.jpg)  
Figure 1. RoboAuto searches for robot programs through physical understanding. Frames from RoboAuto’s programs on LIBERO PRO, LIBERO-PRO Long, LIBERO-Plus, Robosuite, RoboTwin 2.0, and a real robot.

## Abstract

Recently, LLM-based autonomous agents have demonstrated strong capabilities in iteratively generating, evaluating, and refining executable solutions. In robotic manipulation, Code-as-Policy enables these agents to search over modular and editable robot programs. However, effective program search requires physical understanding: agents must understand how objects can be manipulated, how scene geometry constrains actions, and why interactions succeed or fail. Existing methods rely mainly on executionfeedbackfor program refinement, often leaving these physical questions unresolved. To address this limitation, we introduce RoboAuto, a framework for robot program search that explicitly organizes the physical understanding at the object, scene, and interaction levels through three complementary components. First, the hierarchical knowledge base combines executable skills with object-centric knowledge of affordances, spatial relations, and action effects, which can be learned from embodied data and online task solving. Second, scene grounding connects this knowledge to the current environment by identifying relevant objects and resolving action-critical quantities. Third,

multi-stage planning with self-designed experiments checks whether each stage produces its expected physical effects and actively tests competing hypotheses to resolve physical uncertainties. RoboAuto achieves superiorperformance acrossfive benchmarkfamilies, attaining an 89.1% success rate on short-horizon LIBERO-PRO tasks and more than twice the mean success rate of ASPIRE on LIBERO-PRO Long and LIBERO-Plus. It also outperforms ASPIRE and π<sub>0.5</sub> on three challenging long-horizon real-robot tasks requiring precise manipulation. We further demonstrate that the hierarchical knowledge base transfers across simulators and embodiments, and that program search performance scales with the size of the knowledge base. Project page: https://roboauto-ai.github.io/.

## 1. Introduction

Recent advances in autonomous research and coding agents built on large language models (LLMs) [26, 39, 47] highlight the promise of a self-evolving problem-solving paradigm in which an agent generates an executable solution, evaluates its outcome, and uses the resulting evidence to improve subsequent attempts. Code-as-Policy [23] naturally extends this paradigm to robot manipulation by representing robot behaviors as executable programs. Given an instruction such as “put the mug on the plate,” a coding agent can generate a program over perception and control primitives, execute it in the environment, and revise it based on the resulting rollout.

![](images/d7d2774edb28d5f5a751a95e4f2194c3b4674233d850b22221351f445d9b6b76.jpg)  
Figure 2. Overview of RoboAuto. The Hierarchical Knowledge Base (left) stores object-centric physical knowledge, comprising affor dances, spatial relations, and action effects, together with executable skills. Physical Scene Grounding (middle) binds the entries relevant to a task to quantities measured in the current scene. Search and Exploration (right) plans the task as stages that end in verified state changes and designs experiments that separate competing physical hypotheses. Interaction evidence updates the knowledge, and verified programs return to the knowledge base as new skills. RoboAuto is evaluated across five benchmark families, LIBERO-PRO, LIBERO-PRO Long, LIBERO-Plus, Robosuite, and RoboTwin 2.0, where RoboTwin 2.0 tests transfer to a new simulator and embodiment.

(a) Unknown affordance  
(b) Ungrounded geometry  
![](images/a4daec3b98c0fb0f59d480fa51a1ee3a8d59fcc99e742d29ffb13c92e6d5cb90.jpg)  
Figure 3. Unresolved physical assumptions in our baseline runs. In the illustrated ASPIRE and CaP-Agent0 runs, (a) an unsuitable contact strategy persists, (b) a grasp-height rule is transferred without grounding its geometric applicability, and (c) pan placement proceeds without verifying stove activation. The panels show the attempts, outcomes, and retained program choices; Appendix D provides the detailed case studies.

Recent autonomous Code-as-Policy methods instantiate this generate, execute, and revise loop. CaP-Agent0 supports multi-turn program revision with structured feedback [13]; ASPIRE exposes multimodal execution traces and accumulates validated repair knowledge in a skill library [25]; and RHO searches over policy repositories using reward and execution feedback [11]. These methods demonstrate the value of execution feedback and retained experience for robot program search.

However, Figure 3 reveals persistent gaps in physical understanding in our baseline runs. In Figure 3(a), CaP-Agent0 repeatedly attempts a physically unsuitable topdown grasp on a book, indicating that repeated code-level search cannot always compensate for missing knowledge of object affordances. This type of failure motivates a hierarchical knowledge base that complements object-centric physical knowledge acquired from embodied data with executable skills accumulated through online task solving. In Figure 3(b), ASPIRE attempts to grasp a mug by descending to a hardcoded height transferred from another mug, causing the gripper to close on air. This failure shows that even a valid strategy must be instantiated using the geometry of the current scene, motivating a dedicated grounding stage that identifies task-relevant objects and physical quantities with visual tools. In Figure 3(c), CaP-Agent0 proceeds to place a pan despite failing to activate the burner, motivating multi-stage planning and grounded verification of intermediate effects. Moreover, observing a failed knob rotation does not establish how far the knob must turn. Resolving this uncertainty requires active physical exploration through autonomously designed experiments that test specific hypotheses.

Motivated by these observations, we introduce RoboAuto (Fig. 2), a framework for autonomous robot program search through multi-level physical understanding. Its hierarchical knowledge base combines object-centric physical knowledge about affordances, spatial relations, and action effects with executable skills. Physical knowledge is acquired from embodied data and refined through online experience, while the skill library expands as verified programs are discovered through task solving. For each task, RoboAuto retrieves relevant knowledge and skills and grounds them in the current scene, using visual tools, including Molmo2 [10] and SAM 3 [4], to identify task-relevant objects and measure action-critical quantities. Through multi-stage planning with grounded verification, it plans over stage dependencies and verifies each stage’s physical outcome before proceeding. When available evidence leaves a physical uncertainty unresolved, RoboAuto formulates competing hypotheses and conducts self-designed experiments to distinguish them. The resulting evidence refines the physical knowledge, while verified programs expand the skill library, improving both the current solution and future program search.

We evaluate RoboAuto on tasks from five benchmark families. On LIBERO-PRO Long [52], RoboAuto achieves a mean success rate of 73.4%, compared with 30.8% for ASPIRE [25], the strongest coding-agent baseline on average. On LIBERO-Plus [12], it reaches 90.3%, compared with ASPIRE’s 44.7%, while achieving 89.1% on the short-horizon LIBERO-PRO [52] tasks. On contact-rich Robosuite [54] tasks, RoboAuto obtains 90.6% mean success, compared with ASPIRE’s 81.0%. To evaluate knowledge transfer, we apply the hierarchical knowledge base acquired in LIBERO [24] to RoboTwin 2.0 [6], which features a different simulator and a bimanual robot embodiment. RoboAuto achieves a 91.5% success rate, compared with 24.5% without a knowledge base. Knowledge-base scaling experiments further show that larger knowledge bases consistently improve program-search performance. RoboAuto also outperforms ASPIRE and $\pi _ { 0 . 5 }$ on three challenging long-horizon real-robot tasks requiring precise manipulation.

Our contributions are:

• We identify key limitations of existing autonomous Codeas-Policy methods in physical knowledge, scene grounding, intermediate verification, and active exploration.

• We introduce RoboAuto, which addresses these limitations through a hierarchical knowledge base, scene grounding, multi-stage planning with grounded verification, and autonomously designed experiments.

• Across five benchmark families, RoboAuto substantially improves program-search performance and demonstrates knowledge scaling and transfer across simulators and embodiments.

## 2. Related Work

Robotic foundation models. Learned visuomotor policies provide increasingly general foundations for robot control. Diffusion-based policies model multimodal action distributions and have been extended to improve spatial generalization, inference efficiency, and closed-loop responsiveness [8, 9, 14, 27, 45]. At a larger scale, vision-languageaction models learn generalist language-conditioned control from diverse tasks, environments, and robot embodiments [2, 3, 19, 21, 30–32, 50, 55]. World-action models complement this paradigm by coupling action generation with predictions of future visual or latent world states [16, 42, 44, 48, 49, 53]. Whereas these robotic foundation models acquire behavior primarily by training parameterized action policies on large-scale data, RoboAuto introduces a fundamentally different paradigm in which explicit robot programs are autonomously searched and refined through grounded physical understanding, executionbased verification, and active exploration.

Code-as-Policy and autonomous robot selfimprovement. Code as Policies [23] represents robot behavior as programs that compose perception and control primitives, making generated behavior directly inspectable and revisable. CaP-X [13] studies how primitive design, perceptual grounding, and interaction shape this interface, while CaP-Agent0 extends it to multi-turn program revision through execution feedback. More broadly, autonomous research agents [26, 39] and recursive self-improvement systems [34, 47] seek to turn experience from individual executions into persistent capabilities. In robotics, RHO [11] searches over policy repositories, ASPIRE [25] distills validated repairs into reusable skills, RATs [46] acquires skills through self-directed play, and RoboRSI [36] organizes reusable capabilities through skill contracts and validated refinement. OpenETA [7] and ENPIRE [38] similarly improve retained experience or learned policies through physical interaction. Recent work further formalizes physical recursive self-improvement (RSI) and explores recursive improvement through real-to-sim-to-real policy adaptation and simulation-guided refinement of prompts and symbolic skills [22, 35, 51]. Beyond improving policies, prompts, or skill code, RoboAuto makes physical understanding itself a persistent and continually improvable substrate for program search by integrating a hierarchical knowledge base, scene grounding, multi-stage grounded verification, and hypothesis-driven active exploration.

Physical knowledge and scene grounding. Knowledgebased robotics connects reusable object and action knowledge with perception, planning, and execution [1, 5, 33, 40, 41]. MEMORA [43] further transforms egocentric experience into structured, editable memory for grounded robot planning. In parallel, scene-grounding methods construct general-purpose 3D representations [15, 20] or extract task-specific spatial constraints and geometric information [17, 18, 28]. RoboAuto builds on these directions by maintaining reusable object-centric knowledge and selectively grounding task-relevant objects and action-critical quantities during program search, while using execution evidence and autonomously designed experiments to refine its knowledge online.

## 3. Method

RoboAuto enables autonomous robot program search through multi-level physical understanding. It consists of three components (Fig. 2). The Hierarchical Knowledge Base (Sec. 3.1) maintains object-centric physical knowledge learned from embodied data and executable skills acquired through online task solving. Physical Scene Grounding (Sec. 3.2) identifies task-relevant objects and measures action-critical quantities to instantiate candidate programs in the current scene. Search and Exploration (Sec. 3.3) decomposes long-horizon tasks, verifies intermediate outcomes, and designs targeted experiments to resolve physical uncertainties.

## 3.1. Hierarchical Knowledge Base

Robot program search benefits from reusable experience at two different levels of abstraction. General physical knowledge describes what interactions are appropriate and what effects they should produce, while executable skills encode how those interactions are implemented with perception and control primitives. RoboAuto separates these two forms of experience in a hierarchical knowledge base. Its highlevel layer contains object-centric physical knowledge that transfers across tasks and scenes, while its low-level layer contains executable skills that can be directly reused and composed during program search. This separation allows physical knowledge to guide the selection, grounding, and verification of robot programs without tying it to a particular implementation.

Object-centric physical knowledge. Although online task solving can continually expand an agent’s experience, discovering appropriate interaction strategies solely through program search can be difficult. Without object-centric physical knowledge, the agent may continue refining an unsuitable strategy instead of recognizing that a fundamentally different interaction is required. RoboAuto therefore organizes reusable physical knowledge into entries indexed by object class and constructs these entries from both embodied demonstration data and online task-solving experience. Each entry captures three complementary forms of knowledge. Affordances describe suitable interaction strategies for an object; spatial relations characterize task-relevant configurations such as support, containment, and relative position; and action effects and preconditions specify the state changes produced by an action and the conditions required for those changes. RoboAuto constructs these entries from embodied data by analyzing demonstrated interactions, object configurations, and state transitions, and continually refines them using evidence collected during online task solving. To make this knowledge applicable across scenes, each entry also specifies the conditions under which a strategy is applicable and the scene-dependent quantities required to instantiate it. For each quantity, the entry records its purpose and how it can be measured, while leaving its value to be established in the current scene. Physical Scene Grounding (Sec. 3.2) measures these quantities for the current objects and robot, allowing the same objectcentric knowledge to guide program search across different scenes.

Executable skills. The second layer stores verified robot programs as executable skills. Whereas object-centric knowledge captures transferable physical regularities, executable skills implement concrete behaviors using available perception and control primitives. RoboAuto retrieves these skills as candidate implementations or building blocks during program search, reducing the need to rediscover previously successful behaviors. Programs verified through online task solving are added to the skill library, allowing the executable layer to expand continually. Appendix C.1 provides more details.

## 3.2. Physical Scene Grounding

Object-centric knowledge describes reusable interaction strategies and their grounding requirements, but its scenedependent quantities must be established for each task. Given an instruction and the current observation, RoboAuto identifies the task-relevant objects, assigns their roles in the intended interaction, and associates them with the retrieved knowledge entries and executable skills. It then follows the entries’ measurement procedures to check strategy applicability and obtain the required quantities. Rather than constructing an exhaustive representation of the entire scene, this process is scoped by the task and the intended interaction. RoboAuto combines visual perception models, including Molmo2 [10] and SAM 3 [4], with depth-camera observations to identify and segment relevant objects and obtain action-critical information such as grasp locations, metric dimensions, orientations, and spatial relations. Grounding can also involve active viewpoint selection. We introduce active viewpoint selection, which enables the agent to move the robot arm to obtain a more informative camera view when the initial observation is insufficient, thereby improving the accuracy of grounding task-relevant quantities. The resulting grounded representation binds general physical knowledge to concrete objects and measurements, providing the coding agent with the information needed to select a feasible strategy and instantiate it as an executable program. Scene grounding also records which physical conditions have been established and which remain uncertain. Grounded values inform program generation and planning, while unresolved conditions are passed to Search and Exploration (Sec. 3.3) for further investigation. Grounding remains open to revision throughout program search. An execution outcome may invalidate an earlier measurement, reveal a previously unmodeled condition, or change the scene itself. RoboAuto can then remeasure relevant quantities and reassess knowledge applicability to guide the next candidate program. In this way, scene grounding provides a continually updated connection between reusable physical knowledge and the physical state in which the robot acts. Appendix C.2 provides more details.

## 3.3. Search and Exploration

Multi-stage planning with grounded verification. Longhorizon tasks require prerequisite physical changes to be verified before subsequent actions proceed. RoboAuto decomposes each task into stages, each defined by an intended change in object state. Physical dependencies determine their order, while expected effects define the verification criteria. RoboAuto then searches sequentially over the stages. For the current stage, it generates and executes a candidate program together with an executable verifier that evaluates the resulting state from a fresh observation. Search proceeds to the next stage only after the current stage has been verified; otherwise, subsequent execution is stopped and revision remains focused on the incomplete stage. This procedure prevents failures from propagating to later stages, localizes search to a specific physical transition, and preserves the verified program prefix instead of repeatedly regenerating the entire long-horizon solution.

Because the executable verifier may misinterpret an observation, RoboAuto uses an independent auditor to review unsuccessful task executions. Given the stage plan, relevant physical knowledge, execution logs, and synchronized scene and wrist-camera videos, the auditor determines whether each stage achieved its intended object state, and its judgments replace the corresponding verifier results. The coding agent uses this feedback to determine whether the stage program, executable verifier, or overall plan requires revision. When the evidence indicates that a stage is incorrectly defined, ordered under the wrong dependency, or physically infeasible as planned, RoboAuto revises the stage decomposition and regenerates the plan before resuming sequential search, reopening completed stages when their outcomes must be re-established.

Self-designed experiments. Existing autonomous Codeas-Policy methods primarily acquire evidence by executing candidate robot programs and revising them afterward. Because these executions are intended to complete the task rather than investigate a specific uncertainty, their evidence may not isolate the physical assumption responsible for a failure. RoboAuto instead actively explores the scene through autonomously designed experiments. Based on its current physical understanding and scene observations, it formulates physical hypotheses, predicts their observable consequences, and writes dedicated experiment code, separate from the candidate robot program, that specifies the physical intervention and the observations required to evaluate the hypotheses. For example, RoboAuto can test several candidate grasp poses through controlled grasp-and-lift trials and incorporate the successful pose into the robot program without repeatedly executing the full task. The outcome can support, reject, or refine a hypothesis, providing targeted evidence for subsequent program search. It also updates the corresponding object-centric knowledge entry, so that exploration during the current task improves future task solving. Section C.3 provides more details, and Appendix E reproduces the prompts.

## 4. Experiments

## 4.1. Experimental Setup

Benchmarks and evaluation protocol. We evaluate RoboAuto across five benchmark families spanning program robustness, long-horizon manipulation, contactrich control, and cross-embodiment knowledge transfer. LIBERO-PRO [52] evaluates short-horizon Object, Goal, and Spatial tasks under position and task perturbations, while LIBERO-PRO Long focuses on multi-stage goals. LIBERO-Plus [12] introduces camera and scene-layout variations. Robosuite [54] includes contact-rich single- and dual-arm manipulation tasks, and RoboTwin 2.0 [6] evaluates knowledge transfer to a different simulator and bimanual embodiment. Program search and final evaluation use disjoint scenes, testing whether the discovered programs generalize beyond the scenes in which they were developed.

Baselines and implementation. We compare RoboAuto with three autonomous Code-as-Policy methods and a generalist vision-language-action (VLA) model. CaP-Agent0 [13] iteratively revises programs using visual feedback and reuses automatically synthesized skills. AS-PIRE [25], our primary baseline, conducts evolutionary program search and retains skills from prior repairs. RHO [11] improves policy repositories through reward and execution feedback, while the official LIBERO-finetuned $\pi _ { 0 . 5 }$ [19] checkpoint serves as the generalist VLA baseline in simulation. All evaluations build on CaP-Bench [13], which we extend to all five benchmark families, and all autonomous Code-as-Policy methods write programs over the same set of perception and control primitives, including depthcamera observations, Molmo2 [10] pointing, SAM 3 [4] segmentation, and Contact-GraspNet [29] grasp proposals. RoboAuto and all reproduced Code-as-Policy baselines use Claude Opus 4.6 with a 1M-token context window as their coding agent. The initial object-centric knowledge base is built from demonstration videos from the original LIBERO-90 benchmark, disjoint from all evaluation tasks. Further implementation and evaluation details appear in Appendix A.

(f) Robosuite  
![](images/c53088e50d649d6978bfe5903d726d23d5cde492e1af51ded7ea6cb848678bdb.jpg)

![](images/ea0366bd5d8388f6630f964e58f886660139dae3e80a7ac0550692dc587267b3.jpg)

![](images/74b350e5c355d9afe5651e76cdcb53d9b5ddfbc1c3724ef7b31eac9abf85de9c.jpg)

![](images/be535116c221b7935303d941d73c3550e9cf4db9107041efd408c6ccd4f83d09.jpg)

![](images/dbc1ea9a30761ce618eb917229a178fe3c42cfc9016deadd47db1e0009a0f4d1.jpg)

![](images/f72a9d04cfa35e048866c3c41d7037381cd54d2cf2ca97d3ca8958ffc106f207.jpg)  
Figure 4. Program success across benchmarks. (a–c) Each dot is a LIBERO-PRO task under position or task perturbation. (d–f) Horizontal bars give each method’s success on every LIBERO-PRO Long, LIBERO-Plus, and Robosuite setting, ordered from the top by RoboAuto’s success, which is printed beside its bar. Dashed lines and their labels give method means.

## 4.2. Main Results

Figure 4 shows that RoboAuto consistently achieves the highest mean success across all evaluated benchmark suites, substantially outperforming the Code-as-Policy and VLA baselines under a wide range of manipulation settings. This advantage also holds in every category when all evaluation settings are grouped by whether their background and objects match those of the LIBERO-90 demonstrations

(Sec. B.1).

LIBERO-PRO and LIBERO-Plus: generalization across tasks and scenes. LIBERO-PRO evaluates generalization under position and task perturbations across its Object, Spatial, and Goal suites, while LIBERO-Plus introduces camera and scene-layout changes. On LIBERO-PRO Object, RoboAuto and ASPIRE both perform near ceiling at 98.7% and 96.6%, respectively. On Spatial, which requires identifying the target object through its position relative to other objects and placing it at a specified location, RoboAuto achieves 94.8% success compared with 55.3% for ASPIRE. On Goal, which spans diverse interactions such as opening drawers, turning on the stove, and pushing objects, it achieves 73.7% compared with 63.1%. On LIBERO-Plus, RoboAuto achieves 90.3% success compared with 44.7% for ASPIRE. These results show that RoboAuto can retain reusable interaction knowledge while grounding relevant objects and action-critical quantities for new tasks, layouts, and observations.

LIBERO-PRO Long: multi-stage manipulation. LIBERO-PRO Long tests whether a method can manage physical dependencies across a long-horizon task. RoboAuto reaches 73.4% mean success, more than doubling ASPIRE’s 30.8%. RoboAuto decomposes each task into stages, searches over them sequentially, and verifies each intermediate outcome before proceeding, preventing unresolved failures from propagating to later stages.

Robosuite: cross-benchmark transfer to contact and coordination. Robosuite tests precise contact, geometric alignment, and inter-arm coordination in settings distinct from the LIBERO tasks used to initialize RoboAuto. No Robosuite demonstrations are used to construct the knowledge base. RoboAuto nevertheless achieves 90.6% mean success versus 81.0% for ASPIRE, including 74% on nut assembly while every baseline remains below 10%. These results show that object-centric physical knowledge acquired from embodied data can transfer across benchmarks and be re-grounded for contact-sensitive manipulation in substantially different settings.

Generality and transferability of object-centric knowledge. The initial object-centric knowledge is constructed from LIBERO-90 data and is designed to transfer: its entries describe qualitative strategies and the quantities to be measured in each scene, contain no control commands, physical values, task instructions, or specific object instances, and are organized by general affordance, spatial, and effect categories rather than by task or object (Sec. B.1). Grouping evaluation settings by similarity to LIBERO-90, RoboAuto achieves the highest mean success in every group, including settings with both unseen scene layouts and unseen objects (84.3% vs. 68.8%; Table 3). Adding the knowledge base to ASPIRE improves it on several tasks but does not close the gap to RoboAuto (Sec. B.1.6), indicating that RoboAuto’s gains also rely on scene grounding and grounded multistage planning.

## 4.3. Further Experiments and Analysis

This section aims to study the following questions:

• RQ1: How does each component of RoboAuto contribute to robot program search? (Secs. B.3 and 4.3.1)

• RQ2: How transferable is the hierarchical knowledge base across simulators and robot embodiments? (Secs. B.4 and 4.3.2)

• RQ3: How does scaling the number of available hierarchical knowledge-base entries affect program search? (Secs. B.5 and 4.3.3)

• RQ4: How does language model capability affect RoboAuto’s performance? (Sec. B.6)

• RQ5: How sensitive is RoboAuto to randomness across independent program-search runs? (Sec. B.7)

• RQ6: How does RoboAuto’s token usage compare with existing program-search methods? (Sec. B.8)

## 4.3.1. Ablation Studies

The base loop writes a task program, runs it, and revises the whole program based on the execution results. Figure 5 reports mean final-program success over five tasks as the three components are added cumulatively. The base loop makes little progress and reaches only 5.2%. Adding Search and Exploration raises success to 19.2%, supporting stage-wise verification and targeted experiments as a means of turning execution feedback into useful revisions. Adding the Hierarchical Knowledge Base further raises success to 38.4%, indicating that reusable interaction strategies help search progress beyond what structured revision alone achieves. However, this variant still falls well below the full RoboAuto system. The full RoboAuto system adds Physical Scene Grounding, producing the largest gain and reaching 91.2%. This gap highlights the need to ground stored knowledge in the current objects and geometry. Per-task results and case studies appear in Secs. B.3 and D.1.

![](images/9806c648158b8bb869146b7f8ba7f123931faf855f1aac95cd64a1b4851822fa.jpg)  
Figure 5. Ablations of the three components. Mean finalprogram success over five tasks as Search and Exploration, the Hierarchical Knowledge Base, and Physical Scene Grounding are cumulatively added to the base loop.

![](images/52a25342b6295c9763fba98aaae691e9c50c9b155f9ad4b60b871da567dfd8ef.jpg)  
Figure 6. Hierarchical knowledge base transfer to RoboTwin 2.0. Success on four RoboTwin 2.0 tasks without retained knowledge and with the hierarchical knowledge base learned from LIBERO. Dashed lines indicate mean success.

## 4.3.2. Hierarchical Knowledge Base Transferability

The Robosuite results in Fig. 4 demonstrate the crossbenchmark transfer of object-centric knowledge learned from LIBERO. RoboTwin 2.0 [6] provides a more stringent test by changing the simulator, robot embodiment, layouts, tasks, and control primitives. Success increases from 24.5% without retained knowledge to 91.5% with the hierarchical knowledge base (Fig. 6), demonstrating its substantial benefit for program search under cross-simulator and crossembodiment shifts. Per-task success and token usage appear in Secs. B.4 and B.8.

![](images/3df24417c781eed68fd60331fd8ca1e87eef9e14d5a17429c62df57ade8deeb4.jpg)  
Figure 7. Knowledge availability and program search. Mean test success versus search rollouts on the five ablation tasks with 0%, 50%, and 100% of the knowledge base. The 0% and 100% conditions are “+ Search & Exploration” and RoboAuto in Fig. 5.

## 4.3.3. Knowledge Scaling

We study how the availability of hierarchical knowledge base entries affects program search by comparing 0%, randomly sampled 50%, and 100% knowledge availability on five tasks under the same rollout budget (Fig. 7). The half and full knowledge base conditions both begin at 20% mean success but diverge as search proceeds. After 20 rollouts, the full knowledge base condition reaches 90.0%, compared with 57.2% for the half knowledge base condition. Because both conditions start from the same success, the gap reflects how additional entries help the agent interpret execution outcomes and choose subsequent revisions. With the full knowledge base, every task first reaches 80% success within 19 rollouts, whereas over the full budget only three tasks reach this level with half of the knowledge base entries and one task without the knowledge base. Scaling knowledge availability therefore improves both search efficiency and the range of interactions solved within a fixed rollout budget. Per-task results appear in Sec. B.5.

## 4.4. Real-Robot Experiments

We further compare RoboAuto with ASPIRE and the DROID-finetuned $\pi _ { 0 . 5 }$ checkpoint on three long-horizon, precision-manipulation tasks with a WidowX AI arm. RoboAuto reuses its hierarchical knowledge base from LIBERO, ASPIRE reuses its LIBERO-derived skill library, and both use Opus 5 with a 1M-token context window for program search. Sort Balls places two slippery ping-pong balls in their receptacles, Stack Dishes stacks a low-rimmed bowl and a cup, and Erase Board erases “RoboAuto” with a compliant eraser and returns it to its original position. Appendix F describes the hardware, tasks, and evaluation protocol. As shown in Table 1, RoboAuto achieves near-perfect scores on all three tasks, whereas ASPIRE makes limited progress and $\pi _ { 0 . 5 }$ makes none, showing that the hierarchical knowledge base learned in simulation transfers to a real robot and unseen tasks.

![](images/f62b20b17c7ff238e26c63deecfee785d45af6ef7b251e2bb29a4cc63e0351d4.jpg)  
Figure 8. Real-robot rollouts. RoboAuto programs on a WidowX AI arm for three tasks. Each row shows four chronological scenecamera keyframes.

Table 1. Real-robot evaluation. Mean stage-completion score over 20 trials per task. Figure 8 shows example rollouts.
<table><tr><td>Task</td><td>RoboAuto</td><td>ASPIRE</td><td>π0.5</td></tr><tr><td>Sort Balls</td><td>100.00%</td><td>30.00%</td><td>0.00%</td></tr><tr><td>Stack Dishes</td><td>100.00%</td><td>3.75%</td><td>0.00%</td></tr><tr><td>Erase Board</td><td>95.83%</td><td>0.00%</td><td>0.00%</td></tr></table>

## 5. Conclusion

We introduced RoboAuto, a framework for autonomous robot program search that combines a hierarchical knowledge base, scene grounding, stage-wise verified search, and self-designed experiments. Across five benchmark families, RoboAuto consistently improves task success and more than doubles ASPIRE’s performance on LIBERO-PRO Long and LIBERO-Plus. Knowledge acquired in LIBERO also transfers to a new simulator and bimanual embodiment in RoboTwin 2.0, reaching 91.5% success versus 24.5% without it. These results show that transferable physical knowledge and executable robot programs play complementary roles.

Limitations. RoboAuto searches autonomously only in simulation. On real robots, a human operator still resets the scene after each rollout and keeps the arm and its surroundings safe. Automating scene resets and safety monitoring would bring the same autonomy to real-world deployment.

## References

[1] Michael Beetz, Daniel Beßler, Andrei Haidu, Mihai Pomarlan, Asil Kaan Bozcuoglu, and Georg Bartels. Know rob˘ 2.0—a 2nd generation knowledge processing framework for cognition-enabled robotic agents. In 2018 IEEE international conference on robotics and automation (ICRA), pages 512–519. IEEE, 2018. 4

[2] Johan Bjorck, Fernando Castaneda, Nikita Cherniadev,˜ Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 3

[3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π<sub>0</sub>: A vision-languageaction flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 3

[4] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In International conference on learning representations, pages 138846–138923, 2026. 3, 4, 6, 13

[5] Guanqi Chen, Lei Yang, Ruixing Jia, Zhe Hu, Yizhou Chen, Wei Zhang, Wenping Wang, and Jia Pan. Languageaugmented symbolic planner for open-world task planning. arXiv preprint arXiv:2407.09792, 2024. 4

[6] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025. 3, 5, 7, 12

[7] Yitong Chen, Zezheng Huai, Sixian Li, Yubang Wang, Haozhe Zhang, Yifei Zhang, Hechang Chen, Jingjing Gong, Yu-Gang Jiang, and Xipeng Qiu. Eta: A new agentic paradigm for embodied tasks. arXiv preprint arXiv:2608.03924, 2026. 3

[8] Zhuoqun Chen, Xiu Yuan, Tongzhou Mu, and Hao Su. Responsive noise-relaying diffusion policy: Responsive and efficient visuomotor control. arXiv preprint arXiv:2502.12724, 2025. 3

[9] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44 (10-11):1684–1704, 2025. 3

[10] Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Rohun Tripathi, Sangho Lee, Mohammadreza Salehi, Jason Ren, Chris Dongjoo Kim, Yinuo Yang, et al. Molmo2: Open weights and data for vision-language models with video understanding and grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28652–28668, 2026. 3, 4, 6, 13

[11] Karim Elmaaroufi, Justin Svegliato, Sarunas Kalade, Graham Schelle, Sanjit A Seshia, and Matei Zaharia. Rho:

Your coding agent is secretly a roboticist. arXiv preprint arXiv:2606.16458, 2026. 2, 3, 5, 13, 14, 17

[12] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, et al. Libero-plus: A progressive robustness benchmark for visual-language-action models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 38574–38583, 2026. 3, 5, 12

[13] Letian Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, Wenli Xiao, Guanzhi Wang, Dantong Niu, Fei-Fei Li, et al. Cap-x: A framework for benchmarking and improving coding agents for robot manipulation. arXiv preprint arXiv:2603.22435, 2026. 2, 3, 5, 6, 12, 13, 14, 17, 34

[14] Zifeng Gao, Guanxing Lu, Tianxing Chen, Wenxun Dai, Ziwei Wang, Chao Shang, Wenbo Ding, and Yansong Tang. Manicm: Real-time 3d diffusion policy via con sistency model for robotic manipulation. arXiv preprint arXiv:2406.01586, 2024. 3

[15] Qiao Gu, Ali Kuwajerwala, Sacha Morin, Krishna Murthy Jatavallabhula, Bipasha Sen, Aditya Agarwal, Corban Rivera, William Paul, Kirsty Ellis, Rama Chellappa, et al. Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. In 2024 IEEE International Confer ence on Robotics and Automation (ICRA), pages 5021–5028. IEEE, 2024. 4

[16] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2024. 3

[17] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973, 2023. 4

[18] Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. Rekep: Spatio-temporal reasoning of rela tional keypoint constraints for robotic manipulation. arXiv preprint arXiv:2409.01652, 2024. 4

[19] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π<sub>0.5</sub>: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025. 3, 6, 13, 17

[20] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Alaa Maalouf, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, et al. Conceptfusion: Open-set multimodal 3d mapping. arXiv preprint arXiv:2302.07241, 2023. 4

[21] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 3

[22] Haoran Lang, Haotao Lu, Shiyu Sang, Haoyang Luo, Guo Chen, Qun Li, Jingyi Yu, Ye Shi, and Jingya Wang. Embodirsi: Recursive self-improvement for data-efficient robot adaptation. arXiv preprint arXiv:2609.38905, 2026. 3

[23] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In 2023 IEEE International conference on robotics and automation (ICRA), pages 9493–9500. IEEE, 2023. 1, 3

[24] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023. 3

[25] Runyu Lu, Yubo Wu, Ethan Kou, Letian Fu, Wenli Xiao, Ajay Mandlekar, Yinzhen Xu, Guanya Shi, Ken Goldberg, Ang Chen, et al. Aspire: Agentic/skills discovery for robotics. arXiv preprint arXiv:2607.00272, 2026. 2, 3, 5, 13, 14, 17

[26] Alexander Novikov, Ngan Vˆ u, Marvin Eisenberger, Emi-˜ lien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025. 1, 3

[27] Aaditya Prasad, Kevin Lin, Jimmy Wu, Linqi Zhou, and Jeannette Bohg. Consistency policy: Accelerated visuomotor policies via consistency distillation. arXiv preprint arXiv:2405.07503, 2024. 3

[28] Mingyang Sun, Jiude Wei, Qichen He, Donglin Wang, Cewu Lu, and Jianhua Sun. Executable analytic concepts as the missing link between vlm insight and precise manipulation. arXiv preprint arXiv:2510.07975, 2025. 4

[29] Martin Sundermeyer, Arsalan Mousavian, Rudolph Triebel, and Dieter Fox. Contact-graspnet: Efficient 6-dof grasp generation in cluttered scenes. In 2021 IEEE international conference on robotics and automation (ICRA), pages 13438– 13444. IEEE, 2021. 6, 13

[30] Gemini Robotics Team, Abbas Abdolmaleki, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Ashwin Balakrishna, Nathan Batchelor, Alex Bewley, Jeff Bingham, et al. Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342, 2025. 3

[31] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

[32] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 3

[33] Moritz Tenorth and Michael Beetz. Knowrob: A knowledge processing infrastructure for cognition-enabled robots. The International Journal ofRobotics Research, 32(5):566–590, 2013. 4

[34] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandku-

mar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023. 3

[35] Yen-Jen Wang, Haozhe Jiang, Shuying Deng, Haoru Xue, Weirui Ye, Rocky Duan, Nika Haghtalab, S Shankar Sastry, Pieter Abbeel, and Haozhi Qi. Reconstruct, practice, go real: Guided self-improvement for embodied agents. arXiv preprint arXiv:2610.02204, 2026. 3

[36] Zimo Wen, Yijin Chen, Yuxuan Cao, Wendi Chen, Yanwen Zou, Wenye Yu, Fuhang Kuang, Han Xue, Jun Lv, Chuan Wen, et al. Roborsi: Stable, efficient, and reusable robot self-evolution in complex real-world environments. arXiv preprint arXiv:2610.12424, 2026. 3

[37] Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao Zhu, Fangchen Liu, Minghua Liu, Hanxiao Jiang, Yifu Yuan, He Wang, et al. Sapien: A simulated part-based interactive environment. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11094– 11104. IEEE, 2020. 12

[38] Wenli Xiao, Jia Xie, Tonghe Zhang, Haotian Lin, Letian Fu, Haoru Xue, Jalen Lu, Yi Yang, Cunxi Dai, Zi Wang, et al. Enpire: Agentic robot policy self-improvement in the real world. arXiv preprint arXiv:2606.19980, 2026. 3

[39] Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025. 1, 3

[40] Haoming Ye, Yunxiao Xiao, Cewu Lu, and Panpan Cai. Unidomain: Pretraining a unified pddl domain from realworld demonstrations for generalizable robot task planning. Advances in Neural Information Processing Systems, 38: 126668–126713, 2026. 4

[41] Haoming Ye, Yunxiao Xiao, Cewu Lu, and Panpan Cai. Uniplan: Vision-language task planning for mobile manipulation with unified pddl formulation. arXiv preprint arXiv:2602.08537, 2026. 4

[42] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026. 3

[43] Zihao Yu, Xiu Yuan, and Chongjie Zhang. Memora: Embodied action memory from egocentric videos for reasoning and planning. arXiv preprint arXiv:2607.14252, 2026. 4

[44] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026. 3

[45] Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu. 3d diffusion policy: Gen eralizable visuomotor policy learning via simple 3d repre sentations. arXiv preprint arXiv:2403.03954, 2024. 3

[46] Junyi Zhang, Jiaxin Ge, Hanjun Yoo, Letian Fu, Zihan Yang, Yaowei Liu, Raj Saravanan, Shaofeng Yin, Justin Yu, Dan tong Niu, et al. Playful agentic robot learning. arXiv preprint arXiv:2606.19419, 2026. 3

[47] Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. Darwin godel machine: Open-ended evolution¨

of self-improving agents. In International Conference on Learning Representations, pages 104223–104294, 2026. 1, 3

[48] Yuyang Zhang, Wenyao Zhang, Zekun Qi, He Zhang, Haitao Lin, Jingbo Zhang, Yao Mu, Xiaokang Yang, Wenjun Zeng, and Xin Jin. Imagewam: Do world action models really need video generation, or just image editing? arXiv preprint arXiv:2606.19531, 2026. 3

[49] Yang Zhang, Jiangyuan Zhao, Chenyou Fan, Jiayu Hu, Xiu Yuan, Chenjia Bai, and Xiu Li. V-jepa policy: Building effective world-action models on predictive visual latents. arXiv preprint arXiv:2609.37250, 2026. 3

[50] Yang Zhang, Jiangyuan Zhao, Chenyou Fan, Fangzheng Yan, Tian Li, Haitong Tang, Sen Fu, Xuan’er Wu, Qizhen Weng, Weinan Zhang, et al. Prts: A primitive reasoning and tasking system via contrastive representations. arXiv preprint arXiv:2604.27472, 2026. 3

[51] Hai Zhou, Fan Yang, Zeyi Chen, Lejun Zhang, Chenyang Fu, Jianping Li, Jiayi Zhu, Xianyang Liu, Yingxuan Yang, Lili Meng, et al. Who improves the robot? toward physical recursive self-improvement. arXiv preprint arXiv:2610.18842, 2026. 3

[52] Xueyang Zhou, Yangming Xu, Guiyao Tie, Yongchao Chen, Guowen Zhang, Duanfeng Chu, Pan Zhou, and Lichao Sun. Libero-pro: Towards robust and fair evaluation of vision-language-action models beyond memorization. arXiv preprint arXiv:2510.03827, 2025. 3, 5, 12, 14

[53] Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. arXiv preprint arXiv:2504.02792, 2025. 3

[54] Yuke Zhu, Josiah Wong, Ajay Mandlekar, Roberto Mart´ın-Mart´ın, Abhishek Joshi, Kevin Lin, Abhiram Maddukuri, Soroush Nasiriany, and Yifeng Zhu. robosuite: A modular simulation framework and benchmark for robot learning. arXiv preprint arXiv:2009.12293, 2020. 3, 5, 12

[55] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on robot learning, pages 2165–2183. PMLR, 2023. 3

## A. Detailed Experimental Setup

This section specifies the benchmark tasks, RoboAuto search and evaluation protocol, system configuration, and baseline implementations. Quantitative results are reported separately in the main paper and subsequent appendix sections.

## A.1. Simulation Benchmarks

## A.1.1. LIBERO-PRO

LIBERO-PRO [52] evaluates adaptation to controlled task and scene changes. We use its Object, Spatial, and Goal suites. Object requires picking a specified object among distractors and placing it in a basket. Spatial requires identifying the target object through its position relative to other objects and placing it at a specified location. Goal spans diverse interactions such as opening drawers, turning on the stove, and pushing objects within a shared scene. For each suite, we separately evaluate position perturbations, which relocate task-relevant objects, and task perturbations, which modify the goal condition and instruction. Each suite contains ten tasks under each perturbation, producing 60 settings in total. Table 5 reports the suite-level evaluation, while Tables 6–8 provide the complete task-level matrix.

## A.1.2. LIBERO-PRO Long

LIBERO-PRO Long evaluates programs that must produce several dependent physical state changes in the correct order. We randomly select ten task–perturbation settings from the Long suite, covering both task and position perturbations. These tasks require program search to maintain physical progress over an extended horizon and avoid executing later actions before their prerequisites have been established. All methods are evaluated on the same ten settings listed in Table 9.

## A.1.3. LIBERO-Plus

LIBERO-Plus [12] evaluates robustness to visual and geometric changes that preserve the underlying manipulation objective. We randomly select three base tasks from LIBERO-Plus. Each task is evaluated under one camera variation and one layout variation, yielding six settings. For each task and variation type, we use the highest available difficulty level: level 5 for all settings except the camera variation of the stove task, for which the highest available level is 4. Camera variations change the workspace viewpoint and framing, while layout variations change object positions and spatial configurations. These changes require the agent to re-identify task-relevant objects and reestimate scene-dependent quantities rather than reuse parameters tuned to the original scene.

## A.1.4. Robosuite

From Robosuite [54], we use seven single- and dual-arm manipulation tasks. Lift evaluates grasping and transport;

Stack and Restack require precise relative placement; Wipe requires sustained surface contact; and Nut Assembly requires accurate geometric alignment. Two-Arm Lift and Two-Arm Handover additionally test coordination between manipulators. Although LIBERO and Robosuite both use MuJoCo, these tasks introduce different interaction structures, contact requirements, and robot configurations. The complete task set is listed in Table 11.

## A.1.5. RoboTwin 2.0

RoboTwin 2.0 [6] evaluates transfer to a different simulator and robot embodiment. It uses a bimanual robot in a SAPI-EN/PhysX environment [37], changing the gripper geometry, control primitives, contact dynamics, layouts, and task definitions. To evaluate the transfer of the knowledge base, we compare search without retained knowledge and with the hierarchical knowledge base learned from LIBERO under the same search protocol. Table 14 lists the four evaluated tasks and both conditions.

## A.2. RoboAuto Evaluation Protocol

Search and test splits. Unless otherwise specified in Table 2, program search uses 15 training scenes with seeds 110000–110014, and final-program evaluation uses 50 test scenes with seeds 1–50. Robosuite uses 100 test scenes with seeds 1–100. The two sets are disjoint. Program search uses only training-scene evidence; test-scene outcomes are never exposed to the agent or used to update either layer of the knowledge base. After search terminates, the last program is fixed and executed independently on every test scene. The benchmark environment predicate determines task-level success, and suite means weight all listed task settings equally.

Rollout accounting and termination. One rollout is defined as executing a single program in a single training scene. Search runs over a growing subset of the training scenes, the search pool: each candidate is first run on two scenes, and the pool grows from five to ten and then fifteen scenes as each subset is solved. Search terminates when all training scenes are solved or the rollout budget in Table 2 is exhausted. All program executions for active viewpoint selection during scene grounding and for self-designed experiments count toward the same rollout budget. The execution and revision protocols are detailed in Sec. C.3.1.

Final-program evaluation follows the protocol of CaP-Bench and ASPIRE, with 50 test scenes per LIBERO task and 100 per Robosuite task, and all reproduced baselines use the same training and test scenes.

## A.3. RoboAuto Configuration

Code-as-Policy interface. We extend the CaP-Bench evaluation interface [13] to support all five benchmark families. Benchmark-specific wrappers expose perception and control primitives to generated programs and return execution logs, camera observations, and the environment success verdict. RoboAuto writes staged task programs over these primitives. Self-designed experiments use separate experiment and control files but execute through the same benchmark interface.

Table 2. Scene seeds and rollout budgets. Seed intervals include both endpoints. The number of settings counts task–perturbation combinations.
<table><tr><td>Suite</td><td>Settings</td><td>Training seeds</td><td>Test seeds Rollouts</td><td></td></tr><tr><td>LIBERO-PRO Object</td><td>20</td><td>110000-110014</td><td>1-50</td><td>45</td></tr><tr><td>LIBERO-PRO Goal</td><td>20</td><td>110000-110014</td><td>1-50</td><td>45</td></tr><tr><td>LIBERO-PRO Spatial</td><td>20</td><td>110000-110014</td><td>1-50</td><td>45</td></tr><tr><td>LIBERO-PRO Long</td><td>10</td><td>110000-110014</td><td>1-50</td><td>55</td></tr><tr><td>LIBERO-Plus Spatial</td><td>2</td><td>110000-110014</td><td>1-50</td><td>45</td></tr><tr><td>LIBERO-Plus Goal</td><td>2</td><td>110000-110014</td><td>1-50</td><td>45</td></tr><tr><td>LIBERO-Plus Long-derived</td><td>2</td><td>110000-110014</td><td>1-50</td><td>55</td></tr><tr><td>Robosuite</td><td>7</td><td>110000-110014</td><td>1-100</td><td>45</td></tr><tr><td>RoboTwin 2.0</td><td>4</td><td>110000-110014</td><td>1-50</td><td>11</td></tr></table>

Program search uses 15 training scenes; final-program evaluation uses 50 test scenes, except Robosuite, which uses 100. The two LIBERO-Plus Long-derived settings are mugand-pudding tasks under camera and layout perturbations. Ablations use the budget of the corresponding task.

Initial knowledge construction. The initial object-centric knowledge base is constructed from successful LIBERO demonstration videos, using exactly one video per source task in LIBERO-90. Section B.1 groups all evaluation settings by their relation to these demonstrations.

Online knowledge and skill updates. All evaluation tasks begin from the knowledge condition specified for that experiment. Object-centric revisions and executable skills are created only from evidence collected during program search. Before each independently evaluated task, RoboAuto is restored to the corresponding initial condition. Knowledge-scaling experiments vary only the availability of stored entries while keeping the search protocol fixed.

Coding and perception models. RoboAuto uses Claude Opus 4.6 with a 1M-token context window as its coding agent, referred to below as the coder, for program generation and revision. Physical Scene Grounding combines Molmo2 [10] and SAM 3 [4] with depth-camera observations to identify task-relevant objects and estimate action critical quantities. The prompts, knowledge representation, and grounding procedures are described in Appendix E and Secs. C.1 and C.2.

## A.4. Baselines and Comparison Protocol

We compare RoboAuto with three autonomous Code-as-Policy methods and one generalist vision-language-action model. These baselines cover iterative program revision, accumulated executable experience, repository-level policy search, and direct action generation. All autonomous Codeas-Policy methods write programs over the same set of perception and control primitives, including depth-camera observations, Molmo2 [10] pointing, SAM 3 [4] segmentation, and Contact-GraspNet [29] grasp proposals.

CaP-Agent0. CaP-Agent0 [13] generates robot programs over perception and control primitives and revises them across multiple turns using execution logs and visual feedback. It can retain automatically synthesized skills from previous task-solving experience. We compare against its full configuration, which adds a synthesized skill library and parallel reasoning to the multi-turn visual-differencing tier (M4) of CaP-Bench. CaP-Agent0 has no separate search phase and uses no training scenes. Instead, it adapts at test time within each evaluation episode: it executes an initial program on the test scene and may revise it up to ten times based on execution feedback, without resetting the environment, so each evaluation episode allows up to eleven program executions. In our reproduction on LIBERO-PRO Long and LIBERO-Plus, we keep its released configuration unchanged and use Claude Opus 4.6 with a 1M-token context window for all of its model queries.

ASPIRE. ASPIRE [25] performs evolutionary search over candidate robot programs using multimodal execution traces and retains a skill library from repair experience.

RHO. RHO [11] searches over a repository of robot policies and improves the repository using reward and execution feedback.

π<sub>0.5</sub>. π<sub>0.5</sub> [19] is a generalist vision-language-action model that directly predicts robot actions from visual observations and language instructions, providing a non-programmatic comparison with a pretrained visuomotor policy.

Reported and reproduced baseline results. On LIBERO-PRO Object, Goal, and Spatial and on Robosuite, the results of CaP-Agent0, ASPIRE, and RHO are from the published results in CaP-X [13], ASPIRE [25], and RHO [11], respectively. On LIBERO-PRO Long and LIBERO-Plus, we reproduce ASPIRE and CaP-Agent0 from their released codebases using their original experimental configurations without modification. To control for coding-model capability, both use Claude Opus 4.6 with a 1M-token context window, matching RoboAuto. Complete execution records from these runs are retained for the token-usage comparison. RHO has not released the code for its robot experiments, so it cannot be reproduced on LIBERO-PRO Long and LIBERO-Plus. The $\pi _ { 0 . 5 }$ results on LIBERO-PRO and LIBERO-PRO Long are from the published results in LIBERO-PRO [52], which evaluates the official LIBEROfinetuned checkpoint. On LIBERO-Plus, we evaluate the same published LIBERO-finetuned $\pi _ { 0 . 5 }$ checkpoint ourselves on the same 50 test scenes. We do not report $\pi _ { 0 . 5 }$ on Robosuite because no $\pi _ { 0 . 5 }$ checkpoint has been released for Robosuite, and the released LIBERO and DROID checkpoints control a single arm and cannot perform its two-arm tasks. On the real robot, we evaluate all methods ourselves and use the DROID-finetuned $\pi _ { 0 . 5 }$ checkpoint.

## B. Additional Experiment Results

This section provides the complete benchmark results underlying the main comparison and organizes the remaining analyses around six research questions: the contributions of RoboAuto’s components, the transferability of its hierarchical knowledge base, the effect of scaling knowledge-base entries, its performance with a stronger coding model, its sensitivity to search randomness, and its token efficiency.

## B.1. Generality and Transferability of Object-Centric Knowledge

The initial object-centric knowledge is constructed from LIBERO-90 demonstrations, and part of our evaluation uses the LIBERO-PRO and LIBERO-Plus benchmarks. The knowledge base is therefore designed to store physical understanding at a level of abstraction that transfers across tasks, objects, and scenes. Each entry describes how a class of objects can be manipulated and what must be established in the current scene, while the concrete program and its parameters are left to program search and scene grounding. We characterize this abstraction through four properties and verify that every entry satisfies them.

Verification procedure. We read every entry sentence by sentence and checked it against the four properties below. To complement manual inspection, we applied pattern checks to every sentence: (i) numbers, matching any digit sequence $\left( \left[ \ 0 - 9 \right] + \left( \ \cdot \ \left[ \ 0 - 9 \right] + \right) \ ? \ \right)$ and spelled-out numerals; (ii) units, matching length, angle, mass, and force (e.g., mm, cm, m, deg, rad, °, kg, N) following a number or standing alone as a token; (iii) task instructions, matching each instruction from all evaluation suites (LIBERO-PRO, LIBERO-PRO Long, LIBERO-Plus, Robosuite, and RoboTwin 2.0) after lowercasing and punctuation removal, both as a full phrase and as its verb–object sub-phrases (e.g., “put the bowl on the stove” and “bowl on the stove”); and (iv) API names, matching the name of every perception and control primitive exposed to the coder as well as any function-call pattern $( [ \texttt { A - Z a - Z - } ] + \backslash ~ ( )$ . Every match was reviewed manually.

## B.1.1. Control Action Check

Entries describe interaction strategies qualitatively, such as “rim straddle away from the handle,” and leave their implementation to program search. Manual inspection checked every sentence for commanded poses, joint angles, trajectories, gripper commands, and executable code, and the APIname pattern searched for every primitive name and any function-call syntax. No entry contains these elements.

## B.1.2. Physical Quantity Check

Entries identify which physical quantities matter for an object class through their open quantities field, such as the rim radius of a bowl, without recording values for particular objects or scene layouts. This keeps each entry applicable across object sizes and layouts, and the values are measured anew in every scene by Physical Scene Grounding (Sec. 3.2). The number and unit patterns confirmed that no entry stores a physical value.

## B.1.3. Task Instruction Check

Knowledge is organized by object class rather than by task, so entries do not encode task instructions or ordered action sequences for a particular goal. We matched every sentence against all evaluation instructions and their verb– object sub-phrases, and manual inspection checked for tasklevel plans; no entry contains either.

## B.1.4. Object Instance Check

Entries are indexed by general categories defined by affordances, spatial relations, and action effects, rather than by specific object instances such as “cream cheese,” “wine bottle,” or “cookie box.” Manual inspection of every object class and every sentence confirmed this organization. The complete list of categories follows.

Affordance.

• Open-topped vessel wider than the finger opening (low, wide bowl without a handle)

• Open-topped vessel with a protruding side handle (mug)

• Upright short cylinder narrower than the finger opening (can)

• Upright bottle with a narrow cap or neck and a wider body

• Tall, narrow bottle with a long neck

• Small rigid box narrower than the finger opening

• Tall carton, upright or lying flat

• Thin slab standing on its edge (book)

• Wide, shallow pan with a long straight handle

• Upright pot with a protruding side handle

• Rotary knob with a raised ridge

• Sliding drawer (opening or closing)

• Hinged appliance door

• Flat, thin plate relocated without lifting

## Spatial.

• Placing a vessel or mug onto a flat plate

• Placing an item into a walled, open container

• Placing an item onto a raised flat surface

• Inserting a slab into a narrow, open-topped compartment

• Horizontal insertion into a side-opening shelf, onto a high top, or under a shelf

• Nesting one vessel into an identical vessel

• Relative placement beside or in front of a reference object

• Hanging pose of a held object and its propagation to placement

## Effects.

• Swept volume of the arm, the carried object, and moving articulations

• Clearing an articulation’s swept volume in advance

• Articulation state and its visual indicators

• Release height and orientation changes upon release

• Unobservability caused by containment and occlusion

• Identical instances distinguishable only by spatial relations

• Absence of recovery evidence in the corpus

## B.1.5. Results by Similarity to the Knowledge Construction Data

To examine how RoboAuto performs as evaluation tasks move further from the LIBERO-90 tasks used to construct the initial object-centric knowledge, we group all evaluation settings into three categories. Every evaluation task differs from the LIBERO-90 tasks, and the categories further differ in whether the scene layout and the set of manipulated objects also change: (1) same scene layout and same object set; (2) different scene layout, same object set; and (3) different scene layout and different object set. As shown in Table 3, RoboAuto achieves the highest mean success rate in all three categories (74.5%, 90.9%, and 84.3%), outperforming the strongest baseline by 45.5, 18.4, and 15.5 percentage points, respectively. Notably, RoboAuto maintains high performance in the third category, where both the scene layout and the object set are unseen during knowledge construction. This indicates that its gains do not stem from overlap with LIBERO-90, but from the transferable knowledge in the hierarchical knowledge base and the overall agent design.

Table 3. Results by similarity to the LIBERO-90 demonstrations. Mean success rate (%) across evaluation settings, grouped by whether their scene layout and manipulated object set match those in the LIBERO-90 demonstrations used for knowledge construction. #Set. denotes the number of evaluation settings in each row.
<table><tr><td>Benchmark #Set. RoboAuto ASPIRE</td></tr><tr><td>CaP- Agent0 RHO π0.5 Same scene layout, same objects</td></tr><tr><td>LIBERO-PRO Long 10 73.4 30.8 8.6 一 0.4 49.0</td></tr><tr><td>LIBERO-Plus 2 80.0 20.0 1.0 一</td></tr><tr><td>All 12 74.5 29.0 7.3 一 8.5</td></tr><tr><td>Different scene layout, same objects</td></tr><tr><td>LIBERO-PRO Goal 17 77.4 74.0 24.0 50.9 23.1 12.9 10.6</td></tr><tr><td>LIBERO-PRO Spatial 20 94.8 55.3 57.3 LIBERO-PRO Object 16 99.1 96.3 16.3 28.0 5.0</td></tr><tr><td>LIBERO-Plus 4 95.5 57.0 49.5 一 77.5</td></tr><tr><td>All 57 90.9 72.5 19.7 46.4a 17.4</td></tr><tr><td></td></tr><tr><td>Different scene layout, different objects LIBERO-PRO Goal 3 52.7 1.3 5.3 65.3 2.0</td></tr><tr><td>LIBERO-PRO Object 4 97.0 98.0 35.0 11.5 25.5</td></tr><tr><td>Robosuite 7 90.6 81.0 68.3 70.0 一</td></tr><tr><td>RoboTwin 2.0 4 91.5</td></tr><tr><td>一 一 一</td></tr><tr><td>Real robotb 3 98.6 11.3 一 0.0 一</td></tr><tr><td>Allc 14 84.3 68.8 45.3 52.3 15.4ª</td></tr></table>

<sup>a</sup> Mean over the settings in the category for which the method has results. <sup>b</sup> Stage-completion scores.  
<sup>c</sup> Mean over the 14 LIBERO-PRO and Robosuite settings. RoboTwin 2.0 is excluded because no baseline is evaluated on it, and the real robot is excluded because it reports stage-completion scores.

## B.1.6. Hierarchical Knowledge Base with ASPIRE

To test whether the hierarchical knowledge base also improves an existing program-search method and whether it alone accounts for RoboAuto’s gains, we add it to ASPIRE on ten short-horizon LIBERO-PRO tasks. As shown in Table 13, the knowledge base raises ASPIRE’s success on three of the four Spatial tasks, from 8% to 40%, from 16% to 40%, and from 20% to 38%, while picking the bowl on the cookie box changes from 24% to 18%. On the three Object tasks, where ASPIRE already reaches 96% to 100%, success remains near ceiling, and on the three Goal tasks ASPIRE with the knowledge base scores 0%. The knowledge base therefore transfers to a different search procedure but does not close the gap to RoboAuto, which reaches 84.8% mean success on these tasks. On task-perturbed bowl placement on the stove, ASPIRE remains at 0% with the hierarchical knowledge base while RoboAuto reaches 98%, consistent with Table 12, where the hierarchical knowledge base without scene grounding also yields 0%. The knowledge base alone therefore does not reproduce RoboAuto’s performance, which also depends on scene grounding to instantiate the entries in the current scene and on multi-stage planning with grounded verification and self-designed experiments to refine them.

## B.2. Complete Benchmark Results

Table 4 summarizes performance across the five benchmark families. The following tables report every task and perturbation setting used to compute the aggregate results in the main paper.

## B.2.1. Per-Task Results

Table 5 reports the LIBERO-PRO suite-level results. Detailed results for its Object, Goal, and Spatial suites are provided in Tables 6 to 8. Results for LIBERO-PRO Long, LIBERO-Plus, and Robosuite appear in Tables 9 to 11, respectively.

Distribution of program success. We further examine the distribution of final-program success across the 83 settings included in LIBERO-PRO, LIBERO-PRO Long, LIBERO-Plus, and Robosuite. This pool contains 60 LIBERO-PRO settings, 10 LIBERO-PRO Long settings, 6 LIBERO-Plus settings, and 7 Robosuite settings. RoboTwin 2.0 is excluded because it compares different RoboAuto knowledgetransfer conditions rather than RoboAuto with ASPIRE. Each final program is evaluated on held-out test scenes, 100 for Robosuite and 50 elsewhere.

Across these settings, RoboAuto achieves a mean success rate of 87.4%, compared with 65.6% for ASPIRE. The first quartile is 92.0% for RoboAuto but only 30.0% for AS-PIRE, showing that the largest difference lies in the less reliably solved settings. RoboAuto reaches at least 90% success on 64 of 83 settings (77%), compared with 37 settings (45%) for ASPIRE and 3 settings for CaP-Agent0.

The difference is especially pronounced on the 28 settings where ASPIRE achieves less than 50% success. RoboAuto averages 76.6% on this subset, compared with 19.5% for ASPIRE, and reaches at least 80% success on 17 of the 28 settings. On the 37 settings where ASPIRE already reaches at least 90%, both methods average 97.3%. The remaining 18 settings have ASPIRE success rates between 50% and 90%.

In direct per-setting comparisons, RoboAuto outperforms ASPIRE on 56 settings, ties on 9, and underperforms on 18. Using a descriptive tolerance of four percentage points, RoboAuto is higher on 46 settings, comparable on 31, and lower on 6. Against the strongest available baseline for each setting, whose average success is 76.6%, RoboAuto matches or exceeds the baseline on 60 of the 83 settings.

## B.3. RQ1: How Does Each Component Contribute?

Table 12 evaluates the cumulative contributions of Search and Exploration, the Hierarchical Knowledge Base, and Physical Scene Grounding on five tasks. The base coding agent achieves only 5.2% mean success. Adding Search and Exploration raises success to 19.2%, including an improvement from 8% to 82% on the stove task. Sequential stage-wise search and independent verification localize incomplete interactions, while self-designed experiments collect targeted evidence about uncertain physical assumptions. Adding the Hierarchical Knowledge Base further increases mean success to 38.4%. The mug-and-pudding task improves from 4% to 78%, illustrating how object-centric affordance knowledge can introduce an appropriate interaction strategy that repeated parameter revision does not discover. Finally, Physical Scene Grounding raises mean success to 91.2%; drawer retrieval and bowl placement on the stove, which remain unsolved by the partial systems, reach 78% and 98%, respectively. These results show that the three components play complementary roles. Search and Exploration structures the improvement process, the knowledge base supplies reusable interaction strategies, and scene grounding instantiates those strategies for the current objects and geometry.

Figure 9 shows the per-task search trajectories underlying Fig. 5. Section D.1 illustrates how each added component changes the agent’s behavior on the stove task.

## B.4. RQ2: How Transferable Is the Hierarchical Knowledge Base?

We evaluate RoboTwin 2.0 under the transfer conditions defined in Appendix A.1.5.

Success rises from 24.5% without retained knowledge to 91.5% with the hierarchical knowledge base. The knowledge base transfers across the change in simulator and embodiment because its object-centric entries describe the physical requirements of an interaction, which RoboAuto re-grounds in the target scene and implements with the new robot’s control primitives.

## B.5. RQ3: Does Scaling Knowledge-Base Entries Improve Search?

We compare search with no hierarchical knowledge base, a randomly sampled half of the available entries, and the full knowledge base on five tasks. With no entries, scene grounding has nothing to measure, so the 0% condition reuses the “+ Search & Exploration” run of Table 12; only the 50% condition is new. Table 15 reports final-program success, while the search trajectories track test success as a function of the number of search rollouts. Results are averaged equally across tasks. When a rollout does not produce a new candidate program, the previous success is carried forward.

Table 4. Mean success (%) by benchmark and perturbation. Each mean weights task–perturbation settings equally; setting counts are given under each column group. A dash denotes an unavailable result. The last row gives RoboAuto’s difference from ASPIRE. Per-task results follow in Tables 6 to 11.
<table><tr><td></td><td colspan="6">LIBERO-PRO (60)</td><td colspan="3">LIBERO-PRO Long (10)</td><td colspan="3">LIBERO-Plus (6)</td><td>Robosuite</td></tr><tr><td>Method</td><td>Object</td><td>Goal</td><td>Spatial</td><td>Pos</td><td>Task</td><td>All</td><td>Pos</td><td>Task</td><td>All</td><td>Camera Layout</td><td></td><td>All</td><td>(7)</td></tr><tr><td>π0.5 [19]</td><td>9.1</td><td>19.9</td><td>10.6</td><td>25.2</td><td>1.2</td><td>13.2</td><td>0.3</td><td>0.5</td><td>0.4</td><td>38.0</td><td>98.0</td><td>68.0</td><td></td></tr><tr><td>CaP-Agent0 [13]</td><td>20.0</td><td>21.2</td><td>12.9</td><td>19.7</td><td>16.3</td><td>18.0</td><td>9.3</td><td>7.5</td><td>8.6</td><td>24.0</td><td>42.7</td><td>33.3</td><td>68.3</td></tr><tr><td>RHO [11]</td><td>24.7</td><td>53.1</td><td>57.3</td><td>38.6</td><td>51.5</td><td>45.0</td><td></td><td></td><td></td><td></td><td>一</td><td></td><td>70.0</td></tr><tr><td>ASPIRE [25]</td><td>96.6</td><td>63.1</td><td>55.3</td><td>76.5</td><td>66.8</td><td>71.7</td><td>26.7</td><td>37.0</td><td>30.8</td><td>46.0</td><td>43.3</td><td>44.7</td><td>81.0</td></tr><tr><td>RoboAuto (ours)</td><td>98.7</td><td>73.7</td><td>94.8</td><td>90.2</td><td>87.9</td><td>89.1</td><td>60.0</td><td>93.5</td><td>73.4</td><td>84.7</td><td>96.0</td><td>90.3</td><td>90.6</td></tr><tr><td>vs. ASPIRE</td><td>+2.1</td><td>+10.6</td><td>+39.5</td><td>+13.7</td><td>+21.1</td><td>+17.4</td><td></td><td>4+33.3 +56.5</td><td>+42.6</td><td>+38.7</td><td>+52.7</td><td>+45.7</td><td>+9.6</td></tr></table>

Table 5. LIBERO-PRO suite means. Equal-weight mean success (%) across position and task perturbations.
<table><tr><td>Suite</td><td>RoboAuto</td><td>ASPIRE</td><td>CaP-Agent0 RHO</td><td>π0.5</td></tr><tr><td>Object</td><td>98.7</td><td>96.6</td><td>20.0 24.7</td><td>9.1</td></tr><tr><td>Goal</td><td>73.7</td><td>63.1</td><td>21.2 53.1</td><td>19.9</td></tr><tr><td>Spatial</td><td>94.8</td><td>55.3</td><td>12.9 57.3</td><td>10.6</td></tr><tr><td>All settings</td><td>89.1</td><td>71.7</td><td>18.0 45.0</td><td>13.2</td></tr></table>

Table 6. LIBERO-PRO Object. Reported success (%) under position (Pos) and task (Task) perturbations; see Table 2 for RoboAuto’s seed protocol and search budget.
<table><tr><td rowspan="2">ID Task</td><td rowspan="2"></td><td colspan="2">RoboAuto</td><td colspan="2">ASPIRE</td><td colspan="2">CaP-Agent0</td><td colspan="2">RHO</td><td colspan="2">π0.5</td></tr><tr><td>Pos</td><td>Task</td><td>Pos</td><td>Task</td><td>Pos</td><td>Task</td><td>Pos</td><td>Task</td><td>Pos</td><td>Task</td></tr><tr><td>1</td><td>Place(alphabet_soup, basket)</td><td>100</td><td>100</td><td>100</td><td>100</td><td>2</td><td>4</td><td>12</td><td>40</td><td>0</td><td>0</td></tr><tr><td>2</td><td>Place(bbq_sauce, basket)</td><td>92</td><td>98</td><td>100</td><td>100</td><td>12</td><td>42</td><td>32</td><td>88</td><td>100</td><td>2</td></tr><tr><td>3</td><td>Place(butter, basket)</td><td>98</td><td>100</td><td>100</td><td>98</td><td>26</td><td>18</td><td>12</td><td>98</td><td>54</td><td>0</td></tr><tr><td>4</td><td>Place(chocolate_pudding, basket)</td><td>100</td><td>96</td><td>100</td><td>100</td><td>18</td><td>48</td><td>0</td><td>0</td><td>0</td><td>2</td></tr><tr><td>5</td><td>Place(cream_cheese, basket)</td><td>98</td><td>98</td><td>100</td><td>72</td><td>12</td><td>6</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>6</td><td>Place(ketchup, basket)</td><td>98</td><td>100</td><td>92</td><td>96</td><td>32</td><td>12</td><td>0</td><td>4</td><td>20</td><td>2</td></tr><tr><td>7</td><td>Place(milk, basket)</td><td>100</td><td>100</td><td>92</td><td>100</td><td>38</td><td>2</td><td>18</td><td>26</td><td>0</td><td>0</td></tr><tr><td>8</td><td>Place(orange_juice, basket)</td><td>98</td><td>98</td><td>100</td><td>100</td><td>30</td><td>2</td><td>34</td><td>40</td><td>0</td><td>2</td></tr><tr><td>9</td><td>Place(salad_dressing, basket)</td><td>100</td><td>100</td><td>96</td><td>92</td><td>32</td><td>0</td><td>14</td><td>76</td><td>0</td><td>0</td></tr><tr><td>10</td><td>Place(tomato_sauce, basket)</td><td>100</td><td>100</td><td>98</td><td>96</td><td>16</td><td>48</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td colspan="2">Mean</td><td>98.4</td><td>99.0</td><td>97.8</td><td>95.4</td><td>21.8</td><td>18.2</td><td>12.2</td><td>37.2</td><td>17.4</td><td>0.8</td></tr></table>

Figure 10 shows the per-task search trajectories underlying Fig. 7.

The three conditions achieve mean success rates of 21.6%, 33.2%, and 72.0% after 15 rollouts. After 20 roll outs, they reach 22.0%, 57.2%, and 90.0%, respectively, and end at 19.2%, 65.2%, and 91.2%. The full and half knowledge bases begin at the same 20% mean success, but their trajectories separate during revision, showing that additional entries help not only with the initial program but also with interpreting execution outcomes and selecting

subsequent changes.

With the full knowledge base, every task first reaches at least 80% success within 19 rollouts, at rollout indices 0, 13, 14, 18, and 19, with a median of 14; the drawer task’s final program then scores 78%. The half knowledge base reaches this threshold on three tasks, at rollout indices 0, 16, and 32, while search without object-centric knowledge reaches it on only one task, at rollout 44. Final-program success is 19.2%, 65.2%, and 91.2% for the three conditions. Per task, the half knowledge base reaches 98% final success on the ramekin task compared with 94% for the full knowledge base, whereas bowl placement on the stove remains at 0% with half of the entries and reaches 98% with the full set. This experiment measures the effect of providing progressively broader knowledge subsets rather than attributing causal effects to individual entries.

Table 7. LIBERO-PRO Goal. Reported success (%) under position (Pos) and task (Task) perturbations; see Table 2 for RoboAuto’s seed protocol and search budget.
<table><tr><td rowspan="2" colspan="2">ID Task</td><td colspan="2">RoboAuto</td><td colspan="2">ASPIRE</td><td colspan="2">CaP-Agent0</td><td colspan="2">RHO</td><td colspan="2">π0.5</td></tr><tr><td>Pos</td><td>Task</td><td>Pos</td><td>Task</td><td>Pos</td><td>Task</td><td>Pos</td><td>Task</td><td>Pos</td><td>Task</td></tr><tr><td>1</td><td>Open(cabinet, drawer_mid)</td><td>100</td><td>4</td><td>72</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>4</td></tr><tr><td>2</td><td>Put(bowl, drawer_top)</td><td>0</td><td>26</td><td>86</td><td>46</td><td>4</td><td>0</td><td>30</td><td>32</td><td>94</td><td>2</td></tr><tr><td>3</td><td>Push(plate, stove_front)</td><td>0</td><td>6</td><td>0</td><td>76</td><td>0</td><td>10</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>4</td><td>Put(bowl, plate)</td><td>100</td><td>100</td><td>92</td><td>88</td><td>36</td><td>38</td><td>98</td><td>98</td><td>0</td><td>2</td></tr><tr><td>5</td><td>Put(bowl, stove)</td><td>98</td><td>98</td><td>96</td><td>0</td><td>22</td><td>12</td><td>100</td><td>98</td><td>0</td><td>4</td></tr><tr><td>6</td><td>Put(bowl, cabinet_top)</td><td>96</td><td>60</td><td>100</td><td>4</td><td>60</td><td>4</td><td>98</td><td>98</td><td>0</td><td>2</td></tr><tr><td>7</td><td>Put(cream_cheese, bowl)</td><td>92</td><td>100</td><td>74</td><td>74</td><td>4</td><td>34</td><td>10</td><td>10</td><td>98</td><td>2</td></tr><tr><td>8</td><td>Put(wine_bottle, rack)</td><td>100</td><td>100</td><td>98</td><td>86</td><td>2</td><td>12</td><td>58</td><td>62</td><td>88</td><td>2</td></tr><tr><td>9</td><td>Put(wine_bottle, cabinet_top)</td><td>100</td><td>100</td><td>100</td><td>14</td><td>62</td><td>40</td><td>64</td><td>64</td><td>98</td><td>2</td></tr><tr><td>10</td><td>TurnOn(stove)</td><td>100</td><td>94</td><td>90</td><td>66</td><td>66</td><td>18</td><td>48</td><td>94</td><td>0</td><td>0</td></tr><tr><td colspan="2">Mean</td><td>78.6</td><td>68.8</td><td>80.8</td><td>45.4</td><td>25.6</td><td>16.8</td><td>50.6</td><td>55.6</td><td>37.8</td><td>2.0</td></tr></table>

Table 8. LIBERO-PRO Spatial. Reported success (%) under position (Pos) and task (Task) perturbations; see Table 2 for RoboAuto’s seed protocol and search budget.
<table><tr><td rowspan="2">ID Task</td><td rowspan="2"></td><td colspan="2">RoboAuto</td><td colspan="2">ASPIRE</td><td colspan="2">CaP-Agent0</td><td colspan="2">RHO</td><td colspan="2">π0.5</td></tr><tr><td>Pos</td><td>Task</td><td>Pos</td><td>Task</td><td>Pos</td><td>Task</td><td>Pos</td><td>Task</td><td>Pos</td><td>Task</td></tr><tr><td>1</td><td>Pick(between(plate, ramekin), plate)</td><td>92</td><td>100</td><td>94</td><td>8</td><td>22</td><td>14</td><td>92</td><td>60</td><td>2</td><td>0</td></tr><tr><td>2</td><td>Pick(table_center, plate)</td><td>98</td><td>78</td><td>32</td><td>64</td><td>22</td><td>14</td><td>96</td><td>96</td><td>0</td><td>2</td></tr><tr><td>3</td><td>Pick(drawer_top(cabinet_wood), plate)</td><td>78</td><td>92</td><td>10</td><td>96</td><td>2</td><td>10</td><td>6</td><td>4</td><td>0</td><td>0</td></tr><tr><td>4</td><td>Pick(next_to(cookie_box), plate)</td><td>100</td><td>98</td><td>16</td><td>94</td><td>0</td><td>10</td><td>0</td><td>28</td><td>0</td><td>2</td></tr><tr><td>5</td><td>Pick(next_to(plate), plate)</td><td>78</td><td>98</td><td>100</td><td>70</td><td>10</td><td>20</td><td>56</td><td>64</td><td>0</td><td>0</td></tr><tr><td>6</td><td>Pick(next_to(ramekin), plate)</td><td>98</td><td>100</td><td>20</td><td>60</td><td>30</td><td>14</td><td>78</td><td>76</td><td>12</td><td>2</td></tr><tr><td>7</td><td>Pick(on(cookie_box), plate)</td><td>92</td><td>98</td><td>24</td><td>70</td><td>14</td><td>8</td><td>10</td><td>90</td><td>0</td><td>0</td></tr><tr><td>8</td><td>Pick(on(ramekin), plate)</td><td>100</td><td>98</td><td>94</td><td>46</td><td>2</td><td>20</td><td>64</td><td>42</td><td>98</td><td>2</td></tr><tr><td>9</td><td>Pick(on(stove), plate)</td><td>100</td><td>100</td><td>86</td><td>42</td><td>8</td><td>14</td><td>86</td><td>98</td><td>2</td><td>0</td></tr><tr><td>10</td><td>Pick(on(cabinet_wood), plate)</td><td>100</td><td>98</td><td>34</td><td>46</td><td>8</td><td>16</td><td>42</td><td>58</td><td>90</td><td>0</td></tr><tr><td colspan="2">Mean</td><td>93.6</td><td>96.0</td><td>51.0</td><td>59.6</td><td>11.8</td><td>14.0</td><td>53.0</td><td>61.6</td><td>20.4</td><td>0.8</td></tr></table>

## B.6. RQ4: Does RoboAuto Remain Effective with a Stronger Coder?

We repeat five settings from the LIBERO-PRO Goal, Spatial, and Long suites with Opus 5 with a 1M-token context window as the coder for both the base loop and RoboAuto (Table 16). RoboAuto reaches 77.6% mean success against 29.6% for the base loop, showing that its advantage holds with a stronger coder.

Stronger coding does not ensure physical understanding. A stronger model such as Opus 5 has a stronger coding capability, and in particular a stronger ability to control the robot arm through the primitives. It writes its own kinematics when a primitive cannot reach a target, builds more of its own tooling, and, while evolving the program, attends to execution details that Opus 4.6 with a 1M-token context window misses, reading the rollout logs closely enough to catch primitive errors and offsets. These abilities let it reach 78% on picking the bowl from the drawer in the base loop, since controlling the arm reliably and reading the logs carefully are enough to reach the bowl. However, on tasks that cannot be solved without grounded physical understanding, these abilities are not enough. On the long-horizon mugand-pudding task it reaches 58%, failing to pick the pudding reliably near the robot base; on plate pushing it reaches 12%, never establishing how far each push moves the plate. On picking up the two moka pots and opening the cabinet drawer it fails entirely. It does not know how to grasp the moka pot reliably, its grasps repeatedly tip the pot over, and it spends its budget refining execution details and sweeping parameters from the feedback, such as the grasp height along the pot, the hand tilt angle and the closing-axis direction, without understanding what the problem is, how to interact with the object, or how to verify the result. The same unrecognized collision recurs when pulling out the cabinet drawer. Even with longer code and a precise inverse kinematics of its own, Opus 5 loses sight of the broader physical scene, and therefore spends its effort refining code that executes wrong physical assumptions well, with no chance to revise the understanding behind it.

Table 9. LIBERO-PRO Long. Reported success (%); the seed protocol is in Table 2.
<table><tr><td>ID Task</td><td>Perturbation</td><td>RoboAuto</td><td>ASPIRE</td><td>CaP-Agent0</td><td>π0.5</td></tr><tr><td>1</td><td>Place(alphabet_soup, basket) + Place(cream_cheese, basket)</td><td>Pos</td><td>100</td><td>46 40</td><td>0</td></tr><tr><td>2</td><td>Put(white_mug, plate_left) + Put(yellow_white_mug, plate_right)</td><td>Task</td><td>100</td><td>60</td><td>12 2</td></tr><tr><td>3</td><td>Put(white_mug, plate) + Put(chocolate_pudding, right_of(plate))</td><td>Task</td><td>94</td><td>20</td><td>10 0</td></tr><tr><td>4</td><td>TurnOn(stove) + Put(moka_pot, stove)</td><td>Task</td><td>92</td><td>0</td><td>8 0</td></tr><tr><td>5 6</td><td>Put(book, caddy_back)</td><td>Pos</td><td>74</td><td>26</td><td>0 0</td></tr><tr><td></td><td>Place(alphabet_soup, basket) + Place(tomato_sauce, basket)</td><td>Pos</td><td>62</td><td>10</td><td>16 2</td></tr><tr><td>7</td><td>Put(white_mug, plate_left) + Put(yellow_white_mug, plate_right)</td><td>Pos</td><td>58</td><td>22</td><td>0 0</td></tr><tr><td>8</td><td>Put(white_mug, plate)</td><td>Pos</td><td>66</td><td>54</td><td>0 0</td></tr><tr><td>9</td><td>+ Put(chocolate_pudding, right_of(plate)) Put(moka_pot_1, stove)</td><td>Pos</td><td>0</td><td>2</td><td>0 0</td></tr><tr><td>10</td><td>+ Put(moka_pot_2, stove) Put(book, caddy_back)</td><td>Task</td><td>88</td><td>68</td><td>0</td></tr><tr><td>Mean</td><td></td><td></td><td>73.4</td><td>30.8</td><td>0 8.6 0.4</td></tr></table>

Baseline configurations and result sources are described in Appendix A.4.

Table 10. LIBERO-Plus. Reported success (%); the seed protocol is in Table 2.
<table><tr><td>ID</td><td>Task</td><td>Perturbation Level RoboAuto</td><td></td><td></td><td>ASPIRE</td><td>CaP-Agent0</td><td>π0.5</td></tr><tr><td>1</td><td>Pick(next_to(ramekin), plate)</td><td>Camera</td><td>5</td><td>88</td><td>10</td><td>46</td><td>10</td></tr><tr><td>2</td><td>Pick(next_to(ramekin), plate)</td><td>Layout</td><td>5</td><td>94</td><td>18</td><td>60</td><td>100</td></tr><tr><td>3</td><td>Put(white_mug, plate) + Put(chocolate_pudding, right_of(plate))</td><td>Camera</td><td>5</td><td>66</td><td>28</td><td>0</td><td>4</td></tr><tr><td>4</td><td>Put(white_mug, plate) + Put(chocolate_pudding, right_of(plate))</td><td>Layout</td><td>5</td><td>94</td><td>12</td><td>2</td><td>94</td></tr><tr><td>5</td><td>TurnOn(stove)</td><td>Camera</td><td>4</td><td>100</td><td>100</td><td>26</td><td>100</td></tr><tr><td>6</td><td>TurnOn(stove)</td><td>Layout</td><td>5</td><td>100</td><td>100</td><td>66</td><td>100</td></tr><tr><td>Mean</td><td></td><td></td><td></td><td>90.3</td><td>44.7</td><td>33.3</td><td>68.0</td></tr></table>

Each setting uses the highest difficulty level available for its task and perturbation type: level 4 for the stove camera variation and level 5 for all other settings. Baseline configurations and result sources are described in Appendix A.4.

RoboAuto grounds coding in physical understanding. RoboAuto preserves the coder’s execution capabilities while improving its understanding of physical constraints, allowing it to make revisions based on the geometry and mechanics of each scene and task. Before the first rollout, the coder identifies the intended interaction and measures the relevant physical quantities, such as the moka pot’s handle apex or the clearance behind the drawer bar. These measurements let it revise the physical facts it has measured as well as the control parameters. Each failure is then attributed to its physical cause, because stage verification reads the object state after every step and active exploration of the environment tests what the rollouts leave unexplained, so the coder understands why a step failed instead of only that it failed. The moka placement fails and the stage check names a carry fault, so two turns of revision solve the task (0% to 96%); on plate pushing an experiment exposes the primitive’s fingertip floor and a second one measures how far the plate travels per stroke below it (12% to 74%); on the cabinet drawer the coder establishes a pull along the measured slide axis with clearance from the cabinet, and its experiments trace the stalls to collisions with the cabinet, which the base loop had tuned around (0% to 36%); on the mug-and-pudding task the table floor near the robot base is measured and the finger placement verified before closing, which ends the blind retries (58% to 84%). RoboAuto therefore remains effective with a stronger coder.

![](images/75965d4de6830c203dc83823ce53652b8bf550b3c762aadcad600ab7b5270d42.jpg)  
Figure 9. Per-task ablation trajectories. Test success versus the number of search rollouts as Search and Exploration, the knowledge base (KB), and scene grounding are added cumulatively to the base loop.

![](images/3c0c77f3584e6111ff81359b57b0d6029c2f8347e0be2a474fef06230e577c91.jpg)  
Figure 10. Per-task knowledge-availability trajectories. Test success versus the number of search rollouts with 0%, randomly sampled 50%, and 100% of the knowledge base entries.

Table 11. Robosuite. Reported success (%) on seven tasks; see Table 2 for seeds and budget.
<table><tr><td colspan="2">ID Task</td><td></td><td colspan="3">CaP-</td></tr><tr><td></td><td></td><td></td><td>RoboAuto ASPIRE</td><td>Agent0</td><td>RHO</td></tr><tr><td></td><td>1 cube_lift</td><td>100</td><td>97</td><td>97</td><td>98</td></tr><tr><td></td><td>2 cube_stack</td><td>98</td><td>99</td><td>98</td><td>98</td></tr><tr><td></td><td>3 cube_restack</td><td>100</td><td>100</td><td>89</td><td>54</td></tr><tr><td></td><td>4 spill_wipe</td><td>97</td><td>99</td><td>100</td><td>100</td></tr><tr><td></td><td>5 two_arm_handover</td><td>75</td><td>92</td><td>20</td><td>86</td></tr><tr><td></td><td>6 two_arm_lift</td><td>90</td><td>71</td><td>74</td><td>53</td></tr><tr><td></td><td>7 nut_assembly</td><td>74</td><td>9</td><td>0</td><td>1</td></tr><tr><td>Mean</td><td></td><td>90.6</td><td>81.0</td><td>68.3</td><td>70.0</td></tr></table>

## B.7. RQ5: How Sensitive Is RoboAuto to Search Randomness?

To measure variation introduced by program search, we repeat the search on three randomly selected tasks. Run 1 is the main search run reported in the main results, and Runs 2 and 3 are two additional independent searches. Every resulting final program is evaluated on the same 50 test scenes, so differences across runs primarily reflect variation in the search process rather than variation in the test scenes. Table 17 reports the success of each run together with the corresponding mean and sample standard deviation, providing a direct measure of the stability of the final programs discovered by RoboAuto.

Table 12. Component ablation. Reported success (%). KB denotes the hierarchical knowledge base. Each task uses the seeds and budget of its suite in Table 2.
<table><tr><td>Suite</td><td>Task</td><td></td><td></td><td>Perturbation Base loop + Search &amp; Exploration</td><td>+KB</td><td>RoboAuto</td></tr><tr><td>Long</td><td>TurnOn(stove)</td><td>Task</td><td>8</td><td>82</td><td>84</td><td>92</td></tr><tr><td>Long</td><td>+ Put(moka_pot, stove) Put(white_mug, plate)</td><td>Task</td><td>4</td><td>4</td><td>78</td><td>94</td></tr><tr><td></td><td>+ Put(chocolate_pudding, right_of(plate))</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Spatial</td><td>Pick(next_to(ramekin), plate)</td><td>Layout</td><td>14</td><td>10</td><td>30</td><td>94</td></tr><tr><td>(LIBERO-Plus) Spatial</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Pick(drawer_top(cabinet_wood), plate)</td><td>Pos</td><td>0</td><td>0</td><td>0</td><td>78</td></tr><tr><td>Goal</td><td>Put(bowl, stove)</td><td>Task</td><td>0</td><td>0</td><td>0</td><td>98</td></tr><tr><td>Mean</td><td></td><td></td><td>5.2</td><td>19.2</td><td>38.4</td><td>91.2</td></tr></table>

Columns add components cumulatively: Search and Exploration (verified stages and experiments), the Hierarchical Knowledge Base, then Physical Scene Grounding. The last column is the full system used in the main results.

Table 13. Hierarchical knowledge base with ASPIRE. Reported success (%). KB denotes the hierarchica knowledge base. Each task uses the seeds and budget of its suite in Table 2.
<table><tr><td>Suite</td><td>Task</td><td>Perturbation</td><td>ASPIRE</td><td>ASPIRE + KB</td><td>RoboAuto</td></tr><tr><td>Spatial</td><td>Pick(between(plate, ramekin), plate)</td><td>Task</td><td>8</td><td>40</td><td>100</td></tr><tr><td>Spatial</td><td>Pick(next_to(cookie_box), plate)</td><td>Pos</td><td>16</td><td>40</td><td>100</td></tr><tr><td>Spatial</td><td>Pick(next_to(ramekin), plate)</td><td>Pos</td><td>20</td><td>38</td><td>98</td></tr><tr><td>Spatial</td><td>Pick(on(cookie_box), plate)</td><td>Pos</td><td>24</td><td>18</td><td>92</td></tr><tr><td>Goal</td><td>Put(bowl, stove)</td><td>Task</td><td>0</td><td>0</td><td>98</td></tr><tr><td>Goal</td><td>Open(cabinet, drawer_mid)</td><td>Task</td><td>0</td><td>0</td><td>4</td></tr><tr><td>Goal</td><td>Put(bowl, cabinet_top)</td><td>Task</td><td>4</td><td>0</td><td>60</td></tr><tr><td>Object</td><td>Place(chocolate_pudding, basket)</td><td>Task</td><td>100</td><td>100</td><td>96</td></tr><tr><td>Object</td><td>Place(ketchup, basket)</td><td>Task</td><td>96</td><td>98</td><td>100</td></tr><tr><td>Object</td><td>Place(milk, basket)</td><td>Task</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Mean</td><td></td><td></td><td>36.8</td><td>43.4</td><td>84.8</td></tr></table>

Table 14. RoboTwin 2.0. Reported success (%) with an 11-rollout search budget and the seed protocol in Table 2.
<table><tr><td>ID Task</td><td>No retained</td><td>Hierarchical knowledge knowledge base</td></tr><tr><td>1 Stack two bowls</td><td>2</td><td>88</td></tr><tr><td>2 Open microwave</td><td>0</td><td>96</td></tr><tr><td>3 Container on plate</td><td>40</td><td>92</td></tr><tr><td>4 Place empty cup</td><td>56</td><td>90</td></tr><tr><td>Mean</td><td>24.5</td><td>91.5</td></tr></table>

Both conditions use search and experiments.

## B.8. RQ6: How Token-Efficient Is RoboAuto?

We examine the relationship between program success and recorded inference usage. Because each method retains its own search procedure and stopping criteria, this analysis measures the token usage of the complete methods under their native configurations rather than under an artificially matched number of model calls. We report output and total token counts in millions and summarize per-task usage using the geometric mean, which reduces the influence of a small number of exceptionally long searches.

Matched comparison with autonomous Code-as-Policy baselines. All sixteen LIBERO-PRO Long and LIBERO-Plus settings contain complete total-token records for RoboAuto, ASPIRE, and CaP-Agent0. On these settings, RoboAuto achieves 79.8% mean success, compared with 36.0% for ASPIRE and 17.9% for CaP-Agent0. Its geometric-mean token usage is 35.1M per task, compared with 135.6M for ASPIRE and 118.8M for CaP-Agent0, so the two baselines use 3.9 and 3.4 times as many tokens as RoboAuto. The corresponding arithmetic means are 42.7M, 307.5M, and 168.6M tokens, giving ratios of 7.2 and 4.0.

Table 15. Knowledge-base size. Reported success (%). KB denotes the hierarchical knowledge base. Each task uses the seeds and budget of its suite in Table 2.
<table><tr><td>Suite</td><td>Task</td><td>Perturbation</td><td>0%KB</td><td>50% KB</td><td>100% KB</td></tr><tr><td>Long</td><td>TurnOn(stove)</td><td>Task</td><td>82</td><td>86</td><td>92</td></tr><tr><td>Long</td><td>+ Put(moka_pot, stove) Put(white_mug, plate)</td><td>Task</td><td>4</td><td>86</td><td>94</td></tr><tr><td>Spatial</td><td>+ Put(chocolate_pudding, right_of(plate)) Pick(next_to(ramekin), plate)</td><td>Layout</td><td>10</td><td>98</td><td>94</td></tr><tr><td>(LIBERO-Plus)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Spatial</td><td>Pick(drawer_top(cabinet_wood), plate)</td><td>Pos</td><td>0</td><td>56</td><td>78</td></tr><tr><td>Goal</td><td>Put(bowl, stove)</td><td>Task</td><td>0</td><td>0</td><td>98</td></tr><tr><td>Mean</td><td></td><td></td><td>19.2</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Each entry is the last program within the rollout budget, so that all three sizes follow one rule. With no entries, scene grounding has nothing to measure, so the 0% column is the “+ Search & Exploration” run of Table 12.

Table 16. Stronger-model comparison. Reported success (%). Each task uses the seeds and budget of its suite in Table 2.
<table><tr><td>Suite</td><td>Task</td><td>Perturbation</td><td>Base loop</td><td>RoboAuto</td></tr><tr><td>Goal</td><td>Open(cabinet, drawer_mid)</td><td>Task</td><td>0</td><td>36</td></tr><tr><td>Goal</td><td>Push(plate, stove_front)</td><td>Pos</td><td>12</td><td>74</td></tr><tr><td>Spatial</td><td>Pick(drawer_top(cabinet_wood), plate)</td><td>Pos</td><td>78</td><td>98</td></tr><tr><td>Long</td><td>Put(white_mug, plate) + Put(chocolate_pudding, right_of(plate))</td><td>Pos</td><td>58</td><td>84</td></tr><tr><td>Long</td><td>Put(moka_pot_1, stove) + Put(moka_pot_2, stove)</td><td>Pos</td><td>0</td><td>96</td></tr><tr><td>Mean</td><td></td><td></td><td>29.6</td><td>77.6</td></tr></table>

All runs use Opus 5 with a 1M-token context window.

Table 17. Repeated search runs. Each final program is evaluated on test scenes (seeds 1–50). Mean and sample standard deviation are in percentage points.
<table><tr><td>Task / perturbation</td><td>Run 1</td><td>Run 2</td><td>Run 3</td><td>Mean</td><td>Sample s.d.</td></tr><tr><td>Mug + pudding, task perturbed</td><td>94</td><td>92</td><td>88</td><td>91.3</td><td>3.1</td></tr><tr><td>Pick(next_to(ramekin), plate), camera perturbed</td><td>88</td><td>92</td><td>86</td><td>88.7</td><td>3.1</td></tr><tr><td>Place(chocolate_pudding, basket), task perturbed</td><td>96</td><td>100</td><td>92</td><td>96.0</td><td>4.0</td></tr></table>

Each search run uses the training and test intervals in Table 2.

As shown in Fig. 11, RoboAuto more than doubles AS-PIRE’s mean success on both benchmarks while using approximately 26% of its tokens on LIBERO-PRO Long and 25% on LIBERO-Plus. It also achieves higher success with lower token usage than CaP-Agent0. Together with the search trajectories in Sec. B.5, these results show that RoboAuto’s improvement does not require greater recorded inference usage. Its hierarchical knowledge base, stagewise search, grounded verification, and targeted experiments collectively focus program revision on the interactions that remain unresolved.

RoboTwin 2.0 knowledge transfer. The relationship between knowledge and token usage differs in the crossembodiment transfer setting. The condition without retained knowledge uses an average of 9.3M total tokens per task and achieves 24.5% success, whereas transferring the hierarchical knowledge base uses 23.6M tokens and achieves 91.5% success. The knowledge base therefore substantially improves transfer, but the agent also performs more inference while grounding that knowledge and developing successful programs for the new simulator and embodiment.

![](images/b9a64d8b313f6f8848469c8d8e482db30b29073ccfdf2ee17837213052467397.jpg)  
Figure 11. Token usage and program success on matched tasks. Geometric-mean total tokens per task and mean success on LIBERO-PRO Long and LIBERO-Plus settings with complete token records for RoboAuto, ASPIRE, and CaP-Agent0.

Token-record coverage. Tables 18 to 20 report per-task output and total token counts for LIBERO-PRO Long, LIBERO-Plus, and RoboTwin 2.0, the suites for which token records are available. A dash denotes an unreported value and is not treated as zero. Aggregate comparisons include only settings with complete output and total-token records for every method being compared, ensuring that each mean covers the same tasks. We retain the supplied total-token accounting without inferring an unavailable breakdown among input, cache, coder, auditor, or other model calls.

Table 18. RoboTwin 2.0 token usage. Output and total counts are in millions. Task IDs correspond to Table 14.
<table><tr><td></td><td>No retained knowledge</td><td>Hierarchical knowledge base</td><td></td></tr><tr><td>ID Task</td><td>Output Total (M) ↓ (M) ↓</td><td>Output (M) ↓</td><td>Total (M) ↓</td></tr><tr><td>1 Stack two bowls</td><td>0.190 8.780</td><td>0.210</td><td>18.620</td></tr><tr><td>2 Open microwave</td><td>0.280 13.710</td><td>0.420</td><td>33.320</td></tr><tr><td>3 Container on plate</td><td>0.180 8.580</td><td>0.280</td><td>29.890</td></tr><tr><td>4 Place empty cup</td><td>0.080 6.170</td><td>0.120</td><td>12.640</td></tr><tr><td>Mean</td><td>0.183 9.310</td><td>0.258</td><td>23.618</td></tr></table>

Mean rows use only tasks with complete output and total records for every compared method. Total counts reproduce the recorded accounting.

Table 19. LIBERO-PRO Long token usage. Output and total counts are in millions. Task IDs correspond to Table 9.
<table><tr><td colspan="3"></td><td colspan="2">RoboAuto</td><td colspan="2">ASPIRE</td><td colspan="2">CaP-Agent0</td></tr><tr><td>ID Task</td><td></td><td>Perturbation</td><td>Output (M) ↓</td><td>Total (M) ↓</td><td>Output (M) ↓</td><td>Total (M) ↓</td><td>Output (M) ↓</td><td>Total (M) ↓</td></tr><tr><td>1</td><td>Place(alphabet_soup, basket) + Place(cream_cheese, basket)</td><td>Pos</td><td>0.200</td><td>15.700</td><td>0.240</td><td>39.080</td><td>25.500</td><td>176.200</td></tr><tr><td>2</td><td>Put(white_mug, plate_left) + Put(yellow_white_mug, plate_right)</td><td>Task</td><td>1.160</td><td>89.270</td><td>0.763</td><td>2099.000</td><td>34.900</td><td>254.800</td></tr><tr><td>3</td><td>Put(white_mug, plate) + Put(chocolate_pudding, right_of(plate))</td><td>Task</td><td>0.610</td><td>26.030</td><td>0.420</td><td>162.430</td><td>51.600</td><td>276.100</td></tr><tr><td>4</td><td>TurnOn(stove) + Put(moka_pot, stove)</td><td>Task</td><td>0.390</td><td>28.490</td><td>0.690</td><td>300.850</td><td>35.800</td><td>443.700</td></tr><tr><td>5 6</td><td>Put(book, caddy_back) Place(alphabet_soup, basket)</td><td>Pos Pos</td><td>0.520</td><td>34.880</td><td>0.300</td><td>171.080</td><td>40.600</td><td>465.300</td></tr><tr><td></td><td>+ Place(tomato_sauce, basket)</td><td></td><td>0.730</td><td>64.960</td><td>0.150</td><td>95.320</td><td>18.900</td><td>369.900</td></tr><tr><td>7</td><td>Put(white_mug, plate_left) + Put(yellow_white_mug, plate_right)</td><td>Pos</td><td>0.680</td><td>51.940</td><td>1.320</td><td>666.650</td><td>7.530</td><td>75.800</td></tr><tr><td>8</td><td>Put(white_mug, plate) + Put(chocolate_pudding, right_of(plate))</td><td>Pos</td><td>0.700</td><td>61.080</td><td>0.230</td><td>53.190</td><td>9.980</td><td>101.690</td></tr><tr><td>9</td><td>Put(moka_pot_1, stove) + Put(moka_pot_2, stove)</td><td>Pos</td><td>1.030</td><td>86.810</td><td>0.180</td><td>239.380</td><td>25.700</td><td>91.400</td></tr><tr><td>10</td><td>Put(book, caddy_back)</td><td>Task</td><td>0.360</td><td>28.200</td><td>0.120</td><td>20.340</td><td>7.500</td><td>79.410</td></tr><tr><td colspan="3">Mean</td><td>0.638</td><td>48.736</td><td>0.441</td><td>384.732</td><td>25.801</td><td>233.430</td></tr></table>

Mean rows use only tasks with complete output and total records for every compared method. Total counts reproduce the recorded accounting.

Table 20. LIBERO-Plus token usage. Output and total counts are in millions. Task IDs correspond to Table 10.
<table><tr><td></td><td></td><td></td><td colspan="2">RoboAuto</td><td colspan="2">ASPIRE</td><td colspan="2">CaP-Agent0</td></tr><tr><td>ID Task</td><td></td><td>Perturbation</td><td>Output (M) ↓</td><td>Total (M) ↓</td><td>Output (M) ↓</td><td>Total (M) ↓</td><td>Output (M) ↓</td><td>Total (M) ↓</td></tr><tr><td>1</td><td>Pick(next_to(ramekin), plate)</td><td>Camera</td><td>0.590</td><td>26.200</td><td>0.313</td><td>94.480</td><td>4.630</td><td>59.140</td></tr><tr><td>2</td><td>Pick(next_to(ramekin), plate)</td><td>Layout</td><td>0.420</td><td>39.920</td><td>0.509</td><td>624.260</td><td>2.260</td><td>33.280</td></tr><tr><td>3</td><td>Put(white_mug, plate) + Put(chocolate_pudding, right_of(plate))</td><td>Camera</td><td>0.570</td><td>38.500</td><td>0.137</td><td>106.450</td><td>9.910</td><td>95.580</td></tr><tr><td>4</td><td>Put(white_mug, plate) + Put(chocolate_pudding, right_of(plate))</td><td>Layout</td><td>1.530</td><td>70.240</td><td>0.156</td><td>34.180</td><td>9.710</td><td>91.310</td></tr><tr><td>5</td><td>TurnOn(stove)</td><td>Camera</td><td>0.160</td><td>8.640</td><td>0.152</td><td>29.210</td><td>3.310</td><td>44.900</td></tr><tr><td>6</td><td>TurnOn(stove)</td><td>Layout</td><td>0.280</td><td>11.800</td><td>0.323</td><td>183.990</td><td>3.000</td><td>38.390</td></tr><tr><td>Mean</td><td></td><td></td><td>0.592</td><td>32.550</td><td>0.265</td><td>178.762</td><td>5.470</td><td>60.433</td></tr></table>

Mean rows use only tasks with complete output and total records for every compared method. Total counts reproduce the recorded accounting.

## C. Method Details

This section provides the implementation details of RoboAuto’s three components. We first describe how its hierarchical knowledge base is constructed and updated, then explain how physical knowledge is grounded in the current scene, and finally detail the stage-wise search, independent verification, and self-designed experimentation procedures.

## C.1. Hierarchical Knowledge Base

RoboAuto maintains two complementary forms of reusable experience: object-centric physical knowledge and executable skills. Object-centric entries describe transferable properties of physical interactions, while executable skills retain verified program implementations. Both layers continue to expand through online task solving.

Constructing object-centric knowledge from embodied data. Figure 12 illustrates how the object-centric layer is constructed from embodied demonstration videos. A vision-language model converts each video into an evidence record through four steps. It first identifies the entities involved, segments the video into interaction events, describes each event using its preceding state, interaction, resulting state, and visual evidence, and finally evaluates the outcome of the episode. Object-centric knowledge is then distilled across these evidence records.

Each resulting entry is categorized as an affordance, spatial relation, or action effect. It specifies the relevant object class and applicability conditions, describes candidate interaction strategies, and declares the scene-dependent quantities required to instantiate them. The entry also links related knowledge and records the supporting demonstration events. During online task solving, new execution evidence can support, challenge, or refine these entries. Listing 1 shows one stored entry of each type. The knowledge field is abridged. Each of its sentences carries a tag stating whether it was demonstrated, inferred, or remains untested, and the open quantities name what must be measured in each scene before the entry is applied.

Listing 1. One knowledge-base entry of each type.

```jsonl
{"type": "affordance_constraints",
"object_class": "Open-topped vessel with a protruding side handle (mug)",
"strategies": ["RIM STRADDLE AWAY FROM HANDLE: RIM STRADDLE on the rim section diametrically or
laterally away from the handle, so the handle never enters the finger gap.",
"RIM STRADDLE BESIDE HANDLE: RIM STRADDLE on a rim section adjacent to the handle
root, with the handle loop just outside one finger.",
"HANDLE PINCH: close across the handle loop itself."],
"knowledge": "Grasp mugs by the rim, not the handle. Prefer RIM STRADDLE AWAY FROM HANDLE; the
loop visible immediately next to a finger in the wrist view, without collision. Select between
them by the handle's bearing relative to the approach: if the rim section that faces the robot/
carry direction contains the handle root, move the straddle point along the rim until the handle
is outside the finger gap. HANDLE PINCH is untested for this class -- do not assume it; if
tried, watch whether the mug swings about the handle when lifted. Carry the mug upright; it
hangs level from the rim",
"open_quantities": ["handle bearing around the rim", "rim radius"]}
8
{"type": "spatial_relationships",
10 "object_class": "Slab into a narrow open-topped compartment",
11 "strategies": ["IN-PLANE VERTICAL SLOT INSERTION: yaw the held slab so its plane matches the slot'
s long direction, position directly over the slot, and lower straight down."],
12 "knowledge": "Use IN-PLANE VERTICAL SLOT INSERTION with the TOP-EDGE COVER PINCH kept throughout.
Align yaw during the carry, before reaching the caddy, not at the slot. Map the compartment name
to geometry from the caddy layout, not from image position alone: 'front'/'back' referred to
the middle column's front and back slots; 'right' was the tall back-right slot; 'left' the long
left slot. The slab ends standing with its upper part above the walls, hand still on the top
edge; release is not shown.",
13 "open_quantities": ["slot width and long axis", "insertion depth"]}
14
15 {"type": "effects_dependencies",
16 "object_class": "Articulated state and its visual indicators",
17 "strategies": [],
18 "knowledge": "Read states from these cues: drawer OPEN = its slatted/dark interior box protrudes
from the cabinet outline; CLOSED = front flush, interior invisible. Door OPEN = windowed panel
standing out from the body; CLOSED = panel flush against the body. Stove ON = burner coil red;
OFF = grey/dark. States persist without contact: a partly pulled drawer stays where left between
strokes, a partly opened door stays at its angle when the hand repositions, a burner stays red
after the hand leaves the knob. So a later stage may assume the articulation remains where last
observed if nothing touched it. Closing a drawer with an item inside did not visibly disturb the
item, but it becomes unobservable.",
19 "open_quantities": []}
```

Executable skills and online updates. The executable layer stores program components that have been successfully verified across the current search pool. Unlike objectcentric entries, these skills directly implement behaviors using the perception and control primitives of a particular robot interface. After a task is solved, verified program components are consolidated into the skill library, while the physical evidence collected during search updates the corresponding object-centric entries. This separation allows RoboAuto to retain both transferable physical knowledge and reusable executable implementations.

## C.2. Physical Scene Grounding

Physical Scene Grounding determines how retrieved object-centric knowledge applies to the current task and scene. Given the task instruction and initial observations, RoboAuto identifies the task-relevant objects, associates them with retrieved knowledge entries, and checks the entries’ applicability conditions. It then establishes the actioncritical quantities declared by those entries, such as object centers, dimensions, orientations, grasp locations, handle directions, and spatial relations.

RoboAuto combines visual reasoning and segmentation tools, including Molmo2 and SAM 3, with metric observations from scene and wrist depth cameras. The resulting task-specific grounding record stores the quantities that have been established, their supporting observations, and any conditions that remain unresolved. The coder uses this record to select feasible strategies and instantiate their parameters. Grounding remains revisable during search: an execution outcome may challenge an earlier estimate, reveal an overlooked condition, or require an additional measurement or targeted experiment.

## C.3. Search and Exploration

Search and Exploration coordinates three roles: the coder generates and revises programs, the environment executes them, and an independent auditor reviews failed executions. As summarized in Fig. 13, RoboAuto searches sequentially over task stages, verifies their physical outcomes, and designs targeted experiments when execution evidence does not explain a failure.

Multi-stage planning and sequential search. The coder decomposes the task into stages, each defined by an intended change in object state and an executable verifier for that change. The stages are ordered according to their physical dependencies and executed sequentially. A failed verifier stops the remaining task stages, preventing later actions from obscuring or compounding an earlier failure.

Program revision focuses on the earliest stage whose intended state has not been established. Stages that already succeed can therefore be retained while search concentrates on the unresolved interaction. The coder may revise the stage implementation, its verifier, or the broader plan. If execution evidence shows that the current decomposition or dependency structure is invalid, RoboAuto can redesign the stages and replan the task rather than continuing to revise an unrealizable stage.

Independent grounded verification. Program-generated verifiers may themselves be incorrect, so RoboAuto uses a separate language-model session to audit failed executions. The auditor receives the task instruction, stage plan, relevant physical knowledge, execution logs, and synchronized scene and wrist videos. It independently determines whether each stage produced its intended object state and describes the observed interaction.

When the auditor returns a judgment for a stage, that judgment replaces the corresponding program-verifier verdict. If the auditor provides no judgment for a stage, RoboAuto retains the program verifier’s original result. The earliest stage judged incomplete becomes the focus of the next revision. Previously verified stages may also be reopened when later evidence reveals a regression. The environment’s task predicate remains the final criterion for tasklevel success: successful scenes bypass video review, while a verifier failure reported on a successful scene is recorded as a verification error.

Self-designed experiments. Recognizing an incomplete stage does not necessarily reveal which physical assumption caused the failure. When the available evidence is insufficient, RoboAuto formulates competing physical hypotheses and predicts the observations that would support them. It then writes a separate experiment program and corresponding control code to test these predictions. These files are distinct from the candidate robot program because their purpose is to acquire diagnostic evidence rather than complete the task.

Each experiment specifies a hypothesis, a predicted observation, an optional starting-state prefix, and controlled test actions. For example, RoboAuto may generate an experiment that evaluates several candidate grasp poses under the same scene conditions and compares whether each pose establishes stable contact. Prefix stages prepare the required starting state before the test actions. A failed prefix verdict remains visible to the agent but does not prevent the diagnostic actions from running, whereas an execution exception terminates the experiment. The returned logs and camera observations are interpreted relative to the observed starting state and recorded in a persistent diagnosis notebook. The evidence then informs program revision and may update the corresponding object-centric entry.

## C.3.1. Execution and Revision Protocol

Each new or revised candidate is first executed on two scenes, prioritizing scenes in which the current focus stage previously failed. Once that stage succeeds on the initial batch, the same program is evaluated on the remainder of the current search pool. Solving the complete pool expands it from five to ten and then fifteen scenes. If the rollout budget is exhausted while failures remain, the coder produces a final evidence-guided revision.

![](images/00a9b0f2589c70e66cb0de18b0bcae9178f4dd3459f35e4bf992bbce90582dc9.jpg)  
Figure 12. Constructing object-centric knowledge from embodied data. A vision-language model converts each demonstration video into an evidence record containing entities, temporally localized events, and an episode outcome. The example shows the LIBERO-90 task “close the microwave,” in which event E06 moves the door from open to flush. Knowledge distilled across such records is organized into affordances, spatial relations, and action effects. The expanded affordance entry cites supporting events from multiple episodes and declares the scene-dependent quantities, shown in teal, that must be measured again when the entry is applied in a new scene.

![](images/4786df4aa14b77c6efb25bba0e9fadfb9a0d3f54a03f2ff3035862df6365050f.jpg)  
Figure 13. Search and Exploration. The coder decomposes the task into stages with intended object states and executable verifiers (1), and the environment executes these stages sequentially (2). When an execution fails, an independent auditor evaluates the stage outcomes from synchronized videos and may reject a state that the program verifier accepted (3). The earliest incomplete stage is returned to the coder (4), which revises the stage or replans the task before executing the next candidate program (5). If the cause remains uncertain, the coder branches into a separately implemented experiment that collects targeted physical evidence before revision continues. Programs verified across the search pool contribute reusable executable skills.

The grounded knowledge record and diagnosis notebook persist across search iterations. Once the task is solved across the search pool, verified programs are retained as executable skills and relevant physical evidence is consolidated into the object-centric knowledge base.

## D. Behavioral Analysis of RoboAuto

This section analyzes how RoboAuto behaves during program search and how its components change that behavior. We use two complementary forms of evidence. First, baseline case studies examine runs in which program revisions continue without resolving the physical conditions required by an interaction, and compare them with RoboAuto runs on the same task. Second, controlled ablations add Search and Exploration, the Hierarchical Knowledge Base, and Physical Scene Grounding cumulatively while holding the task, rollout budget, and scene seeds fixed.

The cases cover three recurring difficulties. They concern selecting a physically appropriate interaction strategy, grounding the quantities required by that strategy, and verifying that an executed action produced its intended physical effect. Frames are taken from the cited trials. Code and knowledge excerpts are condensed from the generated programs and stored entries while retaining their original identifiers and values.

## D.1. Behavior Across Component Ablations

The component ablation in Table 12 compares four cumulative variants on five tasks. We illustrate the progression on the stove task. The base loop revises complete programs from task-level rollout outcomes. The second variant adds Search and Exploration, including stage decomposition, grounded verification, and self-designed experiments. The third adds the Hierarchical Knowledge Base, and the final variant additionally grounds its object-centric entries in the current scene.

Figure 14 compares the four variants on a single task. Across nine turns, the base loop succeeds on only one training scene and receives no information about which interaction caused the remaining failures. Search and Exploration verifies the knob stage in its first turn and the pan grasp in its second, allowing subsequent revisions to concentrate on placement. Adding the knowledge base supplies workable knob and handle contacts in the first turn, but later revisions must tune unmeasured carry and release constants. The full system measures the pan, knob, and stove before generating its first program. By turn 9, after 36 candidate-program rollouts, it solves 14 of the 15 training scenes. Each component changes the point at which search stops rediscovering the interaction and begins refining its execution.

The remaining four ablation tasks in Table 12 follow the same progression. Without stages, the base loop rewrites the complete program from a per-scene reward and does not establish which interaction failed. Search and Exploration localizes the failed stage, but when that stage admits several explanations it may investigate an incorrect hypothesis first. The Hierarchical Knowledge Base removes much of the search over interaction families, and every task with a matching entry passes its grasp stage within the first one or two turns. What remains unresolved are the scene-dependent quantities required by the entry, which the knowledge-base-only variant approximates from segmentation-mask spread and fixed offsets. These approximations can place the gripper outside a rim or overshoot a placement. Physical Scene Grounding obtains these quantities from geometric estimators cross-checked across cameras and evaluates whether a stored strategy applies. When no stored strategy is feasible, RoboAuto uses a selfdesigned experiment to establish a new interaction.

## D.2. Hierarchical Knowledge Base: Discovering Suitable Interaction Strategies

The book case illustrates how object-centric knowledge supplies a suitable contact strategy when repeated parameter revisions preserve an unsuccessful grasp.

## D.2.1. An Unsuitable Book Grasp Persists Across Revisions

The book begins standing on its edge, as shown in Fig. 15. In CaP-Agent0’s trial $^ { 0 2 , }$ the first top-down approach knocks it flat. The following code block observes that the book is now lying on the table but concludes that another top-down grasp should work. Blocks 2–10 continue adjusting grasp height and orientation around the book center, and only block 11 of 12 attempts an edge grasp.

Across the 44 trials without service errors, every generated program uses top-down grasps, 32 grasp at the book center, 13 explicitly note that the book has fallen, and none completes the task. RoboAuto instead retrieves an entry for manipulating a thin slab standing on its edge. The entry prescribes a top-edge cover pinch supported by 14 bookgrasp demonstrations and declares the book yaw and thickness as quantities to measure. RoboAuto obtains a thickness of 2.8 cm, verifies consistency across the two cameras, and checks that the 5.2 cm caddy slot can receive it. The program pinches the covers before the book is disturbed, verifies the gripper width and the displacement of the book, and inserts it along the slab plane, reaching 74% success on held-out test scenes.

The difference lies in the interaction strategy rather than its parameters. Once the top-down grasp has toppled the book, further adjustment of height or orientation within that contact family does not recover the task.

## D.3. Physical Scene Grounding: Instantiating Knowledge in the Current Scene

The mug case examines the effect of replacing an inherited grasp-height offset with measurements of the current object’s rim.

## D.3.1. A Grasp-Height Rule Is Copied from Another Object

ASPIRE’s skill library contains a manually implemented top-down grasp with a fixed offset below the object top:

$$
\tt g r a s p \mathrm { _ { - } z } = \tt o b j \mathrm { _ { - } z \mathrm { _ { - } t o p } - 0 . 0 2 . }
$$

The rule was distilled from a run involving a different mug in another scene. ASPIRE reuses it for the red mug and grasps at the body center using the stored offset. Because the same mug model appears across scenes, the resulting grasp height is fixed at z = 0.1345 m in all 15 training scenes. The program’s own findings report that this grasp produces a gripper width of approximately 0.015 m, indicating an air grasp, in approximately half of the seeds, while the same code succeeds in others.

The stored rule contains no applicability condition that explains the variation. It does not state the rim radius, finger opening, or geometric relationship for which the offset was valid. ASPIRE’s repair loop ends at 0/50 success; its reported 20% on this setting is obtained through its subsequent evolutionary search.

RoboAuto’s entry for an open vessel wider than the finger opening declares the rim radius and top height as quantities to measure. Across test trials, it obtains a radius between 0.0431 and 0.0433 m and a top height of approximately 0.153 m. The program offsets the pads from the mug center by the measured radius and checks the predicted pad height against the rim before closing, lowering the hand if the pads remain above it. The quantities vary little across these scenes, but they are explicitly tied to and verified against the object being grasped rather than inherited from a procedure that succeeded elsewhere.

## D.4. Search and Exploration: Localizing and Investigating Failures

The stove case illustrates why program search must verify the burner’s state before proceeding to pan placement.

## D.4.1. An Executed Action Is Mistaken for an Achieved Physical State

The instruction is to turn on the stove and place the pan on it. In CaP-Agent0’s trial 18, blocks 1–5 repeat the same routine: descend toward the detected knob point, close the gripper, and rotate the wrist by 90 degrees. Block 5 records that the gripper remains at $z \approx 0 . 1 2 9 \mathrm { m }$ , far above the estimated knob height of $z \approx 0 . 0 1 8$ m. Nevertheless, no block checks whether the burner changed state.

The burner remains off through the first four revisions. Later blocks proceed to pan placement, and the pan ends propped against the stove edge. Across the failed trials, most programs infer the knob geometry from its expected position; only 7 of 93 reason explicitly about the knob ridge or its orientation.

RoboAuto’s knob entry prescribes a ridge pinch and twist and specifies the effect to observe: rotate until the burner appearance changes, rather than by a stored angle. The program descends to the grounded knob height, performs the rotation, and measures the burner’s red fraction. If the burner remains off, it re-grasps and rotates in the opposite direction. It grasps the pan near the handle root only after burner activation has been verified, reaching 92% success on test scenes. Executing the rotation is not equivalent to achieving the intended physical state. Verifying the state localizes the failure to the knob stage rather than allowing it to appear only as a final reward of zero.

## D.5. Summary of Observed Behaviors

The cases illustrate distinct sources of progress during program search: object-centric knowledge supplies a coverpinch strategy for the standing book, scene grounding replaces the mug’s inherited grasp-height offset with measured rim geometry, and stage-wise verification checks burner activation before pan placement. The stove ablation further shows how these components change the revisions made during search, from discovering workable contacts to measuring and refining their execution.

<table><tr><td>0 of 5</td><td>One reward per scene; nothing says which of the three steps failed.</td></tr><tr><td>330 lines changed</td><td>Each turn rewrites knob, grasp, and placement together, 200-390 lines.</td></tr><tr><td>0.06</td><td>The burner offset is re-guessed while the knob turn already works</td></tr></table>

![](images/b18a6f0599a823f7252488adca27f4145eaa81a370d646836d4167184a13c1d3.jpg)  
Figure 14. The four component variants on turning on the stove and placing the pan (LIBERO-PRO Long, task perturbation of turning on the stove and placing the moka pot). Each panel adds one component to the preceding variant. The base loop rewrites the complete program from a per-scene reward. Search and Exploration plans stages whose verdicts localize the failure. The Hierarchica Knowledge Base supplies knob and handle contacts in the first turn but estimates their quantities from segmentation-mask spread and fixed offsets. Physical Scene Grounding measures the knob height and pan geometry and verifies burner activation before the pan is moved.

![](images/ec4166fe0bd5761fbb52aea01f80fda31f7c4ee4f13b17c84c3ccb3ca9244a50.jpg)  
Figure 15. Book into the caddy (LIBERO-PRO Long, position perturbation). CaP-Agent0 continues using top-down grasps after its first approach topples the book. RoboAuto selects a cover pinch from the thin-slab entry and grounds the quantities required to execute it.

![](images/05219e46d6a08812f93e10eb3fc4626c8f2c279becca5a857a89dca303d2eae8.jpg)  
Figure 16. Placing the red mug on the plate and the pudding to its right (LIBERO-PRO Long, task perturbation of placing the white mug on the plate and the pudding to its right). ASPIRE applies a grasp-height rule stored in a skill learned on a different mug, causing the gripper to close on air in approximately half of the training scenes. RoboAuto measures the rim of the current mug and places the pads on it.

![](images/4a4675d54c721c904f434191942c6e3c2e49d22e42f75273c6051fd53423966b.jpg)  
Figure 17. Turning on the stove and placing the pan (LIBERO-PRO Long, task perturbation of turning on the stove and placing the moka pot). CaP-Agent0 rotates the knob by a stored angle and proceeds without checking the burner. RoboAuto’s knob entry specifies the expected observable effect, which the program verifies before beginning pan placement.

## E. Prompts

This section reproduces the prompts that drive program search. The coder prompt is sent once at the start of a task and remains in the session. The auditor prompt starts a fresh session for each audit. Per-turn feedback is assembled from the execution digest, the auditor grid, and a closing instruction. Workspace paths, file-handling instructions, and the primitive API documentation inherited from CaP-X [13] are abridged.

## E.1. Coder Session Prompt

Listing 2 shows the prompt that opens the coder session. It defines the scene pool and batching rules, the stage-file format with per-stage verifiers, the knowledge-base lookup and measurement requirements, the experiment protocol, and the labnotebook discipline. The placeholder GOAL is replaced with the task instruction.

## Listing 2. Coder session prompt.

===== THE TASK =====   
2 You are controlling a robot with API described below.   
3 Goal: {GOAL}   
4 [Documentation of the perception and control primitives follows.]   
5   
===== HOW THIS SESSION WORKS =====   
7 You are the coder in an automated, audited loop, working in this directory.   
8 Each turn you write or revise the program as stage files (below); the loop   
9 then executes it from scratch on a batch of scenes, an independent auditor   
10 watches the videos, and you get a results digest: per scene, the task   
11 outcome, the auditor's verdict on every stage and the auditor's account of   
12 what happened. Nobody else reads your replies: the loop only looks for the   
13 last word, READY or EXPERIMENT.   
14   
15 Scenes   
16 - The run works through 15 fixed scenes with randomized object placements,   
17 in a pool that starts at 5 scenes. Every revision is run on a 2-scene   
18 batch that the loop picks (the scenes where the focus stage last failed   
19 come first). When the focus stage passes on both, the loop runs the other   
20 scenes of the pool with the same program. A stage counts as passed only   
21 when the auditor passes it on every scene of the pool. Once every stage   
22 passes with reward 1.0 on every scene of the pool, the same program is run   
23 on 5 new scenes and the pool grows 5 -> 10 -> 15; the run is accepted at   
24 15 of 15. Every scene run of your program counts against a fixed rollout   
25 budget. After the last rollout you still revise once   
26 more, and that revision is the final program.   
27 - You debug one scene at a time, but you are writing ONE program for all   
28 scenes, and it must complete the ENTIRE task for any placement. Never   
29 write scene-specific code: no branch on a scene id, no constant read off   
30 one scene's placement, no strategy that only makes sense for the layout   
31 in front of you.   
32   
33 Plan   
34 - Design your own high-level plan for the task as an ordered sequence of   
35 stages and write it to plan.md. Then for every stage write (a) what it   
36 does, concretely, for this task and robot, (b) the change in the OBJECTS'   
37 state that proves the stage succeeded, (c) how the verifier will observe   
38 that change. A stage is done when the object is where the plan says it   
39 should be, never because the robot reached some pose.   
40 - Build the plan from the knowledge base first. For every object and fixture   
41 the task involves, and for every relation and state change the goal names,   
42 look up the entries whose object\_class describes it and whose condition   
43 holds in this scene. The affordance\_constraints decide what each stage may   
44 attempt, the spatial\_relationships decide the approach, the   
45 effects\_dependencies decide the order and what each stage's verifier must   
46 check. Cite the entries you used by their object\_class.   
47 - Before implementing, MEASURE. Run a measurement probe that segments each   
48 object with the perception APIs, deprojects it into a point cloud and   
49 measures every quantity the plan will depend on -- above all the   
50 open\_quantities of the knowledge-base entries you rely on -- and record   
51 the results in measurements.json. When a later observation contradicts a

52 recorded value, correct measurements.json and the plan promptly.   
54 Stage files   
55 - Write one file per stage: stages/01\_<a>.py, stages/02\_<b>.py, ... They are   
56 executed strictly in order in ONE shared namespace.   
57 Each stage file must end with exactly one call stage\_done(ok, evidence),   
58 where ok is the verdict of the stage's own verifier: re-detect the object   
59 the task names from a fresh observation and check it is in the state the   
60 stage should leave it. Robot readings alone (gripper opening, EE pose) are   
61 never evidence that a stage succeeded. If ok is False the rollout stops   
62 there -- later stages do not run.   
63   
64 Audit and focus stage   
65 - An independent auditor reviews every verdict against the rollout videos: a   
66 PASS the video does not support counts as FAIL, and a scene without task   
67 success is a failure whatever your verifiers reported.   
68 Each turn the feedback names the focus stage: the earliest stage that did   
69 not PASS on the scenes that ran. Work on that stage. If the evidence shows   
70 an earlier stage is the real cause you may revise that one instead.   
71   
72 Experiments   
Experiment before you fix whenever you are not sure of the cause.   
74 - An experiment is a short program on one scene that does the minimum needed   
75 to reach the situation in question. Write experiments/NN\_<slug>/hypothesis.md   
76 with these lines first:   
77 Hypothesis: <what you believe causes the failure>   
78 Prediction: <what the probe will show if it is true, and if it is false>   
79 Prefix: <a -> b> (optional: stage names to run first)   
80 Scenes: <scene id(s) to run on>   
81 and experiments/NN\_<slug>/experiment.py: plain code over the same APIs.   
82 Keep it small -- a handful of primitive calls, one question per experiment.   
83 End the reply with the single word EXPERIMENT. Record the outcome in   
84 diagnosis.md -- prediction held or refuted -- before acting on it.   
85 Experiments never change stage verdicts.   
86   
87 Lab notebook   
88 Keep diagnosis.md as a running lab notebook, appending one entry per turn.   
Never rewrite or condense earlier entries.   
90 - After each results digest and before writing any code: check every   
91 hypothesis from your previous entry against the new evidence and mark it   
92 supported or refuted. Then record which stage failed on which scenes; the   
93 diagnosis, confirmed in BOTH the full log and video frames you extracted   
94 and looked at yourself; the hypotheses about the cause; and either the   
95 fix, with the concrete prediction it makes for the next rollout, or the   
96 experiment that will decide between the hypotheses.   
97   
===== READ FIRST: UNDERSTAND THE ENVIRONMENT AND THE SCENE =====   
99 setup\_image.png: one training scene one second after the reset, the   
100 agentview camera on the left half and the wrist camera on the right half.   
101 libero\_knowledge.json: the knowledge base, and the most important guiding   
102 file here. Its knowledge list holds entries of three types --   
103 affordance\_constraints (what a shape affords, and what it forbids),   
104 spatial\_relationships (how objects sit relative to each other, the   
105 fixtures and the robot) and effects\_dependencies (what an action really   
106 changes, and what must hold before it works). Every sentence of an entry's   
107 knowledge is tagged [demonstrated], [inferred] or [untested] -- an   
108 [untested] sentence is a hypothesis for an experiment, not a rule.   
109 strategies names the approach; open\_quantities lists what you must   
110 measure per scene, because the base records no numbers.   
111   
112 Your first turn, in this order:   
113 1. Look at setup\_image.png and find the knowledge-base entries that bear on   
114 THIS task: the objects involved, the strategies you will use, the effects   
115 and dependencies that order the stages, and every open\_quantity those   
116 entries say must be measured.   
117 2. Write the measurement probe experiments/00\_measure/ and end your reply   
118 with the single word EXPERIMENT.

119 3. Fill measurements.json from the probe's output.   
120 4. Open diagnosis.md with "Entry 0 -- my understanding of the environment and   
121 the scene", now with your measured numbers.   
122 5. Then write plan.md and the stage files from those measurements, and reply   
123 READY.

## E.2. Auditor Prompt

Listing 3 shows the auditor prompt. Placeholders carry the task goal, the stage plan, the stages already verified in earlier turns, and the execution digest of the current turn. The auditor returns per-stage verdicts and a time-ordered note for every scene.

## Listing 3. Auditor prompt.

```csv
1 You are an independent auditor for a robot-programming loop.
2
3 THE TASK
4 Goal (the robot's instruction, verbatim): {goal}
5 Success is decided by the simulator's own predicate on the TARGET object's
6 final pose; the env reward (1.0 / 0.0) in the results below is that signal.
7 A scene with reward 0 has at least one failing stage.
8
BEFORE JUDGING
10 Read libero_knowledge.json -- the knowledge base: how each kind of object
11 affords grasping and placing, how objects sit relative to fixtures and the
12 robot, and what an action really changes. Use it to recognise what you see
13 in the videos. Do not read the coder's own measurements -- they are not
14 needed to judge the videos.
15
16 WHAT THE CODER WROTE
17 A coder wrote a staged program -- {plan} -- one file per stage. After each
stage the program prints its own verdict, [STAGE <name>] PASS <evidence> or
19 FAIL <evidence>, and a FAIL stops the rollout.
{plan_block}{verified_block}
21 YOUR JOB
22 Decide, for EVERY scene and EVERY stage, whether that stage really
23 succeeded, and describe what actually happened so the coder can fix it. The
24 program's printed verdicts are the coder's CLAIMS, not evidence: a PASS may
25 be false, a FAIL may give a wrong reason. Judge from what actually happened:
26 the rollout videos and the measurements in the logs. Every video frame is
27 two views side by side: the agentview camera on the LEFT and the wrist
28 camera on the RIGHT. Base each verdict and each note on what the two views
29 show combined. A stage passes on a scene only if the TARGET object ends up
30 in the state that stage is supposed to produce. Robot readings alone
31 (gripper opening, end-effector pose) prove nothing. A stage that never ran
32 or was skipped fails.
33
34 For every scene write a note: a careful, factual account of what the video
35 shows, in time order, with the frame times you looked at.
36 Do not fix anything and do not propose fixes.
37
38 Turn {turn}. Scenes that ran and their results:
39 {digest}
40
41 Write your verdicts as a JSON list, one entry per scene, every stage:
42 [{"seed": <int>, "note": "<what happened, in time order, with frame times>",
43 "stages": {"<stage>": {"pass": true|false, "evidence": "<observed facts>"}, ...}}, ...]
```

## E.3. Per-Turn Feedback and Experiment Protocol

Each turn, the coder receives the execution digest, the auditor grid with notes, the rollout budget, and a closing instruction. Listing 4 shows the feedback from an early turn of the LIBERO-PRO Goal task that places the plate on the stove, with the auditor note abridged. When the coder answers with an experiment, the loop runs it and returns its log and video together with the instruction in Listing 5. Listing 6 shows the hypothesis file the coder wrote for its first diagnostic experiment on the same task, which tests whether a reach limit or a collision prevented the gripper from reaching the plate rim.

Listing 4. Per-turn feedback from an early turn on the plate-on-stove task.

Your program.py was executed FROM SCRATCH on 2 scenes (randomized object   
placements). It succeeded on 0 of 2. Per-scene results:   
3 seed 110000: FAILURE (reward 0.000)   
4 [STAGE grasp] FAIL plate dropped during test lift: grip=0.0146   
5 seed 110001: FAILURE (reward 0.000)   
6 [STAGE grasp] FAIL plate dropped during test lift: grip=0.0132   
8 An independent auditor checked your verdicts against the rollout videos.   
9 The auditor's per-stage verdicts (these are ground truth; a PASS of yours   
10 that the auditor marks fail counts as FAIL):   
11 scene 110000 (reward 0.0): grasp: FAIL place: FAIL   
12 note: t=0.0s (f001): Initial scene. Agentview: plate (white with red   
13 concentric ring rim) flat on the table in the lower-center area;   
14 gray patterned bowl above and slightly right of the plate; [...]   
15 t=1.2s (f007): Agentview: arm extended far forward, hand/gripper   
16 low near the table surface beside the plate. Wrist view: the plate   
17 fills the right side of the frame; the finger pads are visible at   
18 the bottom of the wrist view, but the plate rim sits clearly ABOVE   
19 the pad level -- the pads have not reached the rim height. t=1.6s   
20 (f009): gripper is closed near the plate on the table; the fingers   
21 appear to have caught the very top edge of the rim but the hold is   
22 marginal. t=2.0s (f011): the plate appears tipped up onto its edge   
23 momentarily as the fingers tried to lift it, indicating it slipped   
24 out during the test lift. Log confirms: pad\_bot=0.0174 vs   
25 rim\_top=0.0069 (pads 10.5mm above the rim); grip after close was   
26 0.0474 (marginal contact with rim top edge), grip after test lift   
27 was 0.0146 (plate slipped out).   
28 scene 110001 (reward 0.0): grasp: FAIL place: FAIL   
29 note: [...]   
30   
31 Rollout budget: 4 of 45 scene runs used, 41 left.   
32   
33 Focus stage: grasp   
34 [closing instruction as above]

## Listing 5. Instruction returned with each experiment result.

1 Append to diagnosis.md: the hypothesis, the prediction, what was actually   
2 observed (cite log lines and video frames you looked at), and whether the   
3 prediction held or was refuted. Then continue with the focus stage: revise   
4 it and end your reply with READY, or write another experiment and end with   
5 EXPERIMENT. Experiments do not change any stage verdict.

Listing 6. Hypothesis file of the coder’s first diagnostic experiment on the plate-on-stove task.

1 Hypothesis: The arm can't reach the plate rim at x˜0.71 (pad bottom z=0.015,   
2 rim top z=0.007 -> 8mm gap). Straddling from the -x direction (x˜0.65,   
3 6.5cm closer to robot) should allow the arm to reach 3-5cm lower, placing   
4 the pads at the rim level. A diagonal (135 deg) straddle avoids the bowl   
5 collision while bringing the gripper 4.6cm closer.   
6 Prediction: At the 135 deg straddle point (x˜0.668), the pad bottom z will be   
7 at or below 0.007 (the rim top). A grip after close will hold during a 5cm   
8 test lift. At the 90 deg straddle (x=0.714, current approach), the pad   
9 bottom remains above 0.010. At the 180 deg straddle (x=0.649), reach is   
10 best but the bowl may be displaced.   
11 Prefix:   
12 Scenes: 110000

## F. Real-Robot Experimental Details

Hardware setup. We conduct real-robot experiments using a Trossen Robotics WidowX AI arm equipped with an Intel RealSense D405 RGB-D camera and a Logitech C920 RGB camera. The cameras provide complementary geometric and visual observations of the robot, objects, and workspace. All methods use the same hardware, sensing configuration, and control interface.

Knowledge base. For the real-robot experiments, RoboAuto directly reuses its hierarchical knowledge base acquired from LIBERO tasks, which combines reusable object-centric physical knowledge with executable skills. All real-robot tasks are held out and distinct from those used for knowledge-base construction. For a fair comparison, ASPIRE also transfers its skill library acquired from the LIBERO tasks. Despite the domain shift from simulation to the physical world, RoboAuto effectively retrieves, grounds, and composes its accumulated knowledge for previously unseen tasks and configurations. Its substantial advantage over ASPIRE shows that the hierarchical knowledge base acquired in simulation transfers to unseen realrobot tasks.

Experimental setup. We use Opus 5 with a 1M-token context window for the search component of both programsearch methods, RoboAuto and ASPIRE, and allocate each a budget of 10 search rollouts to ensure a controlled comparison. Because no simulator success predicate is available on the real robot, a human operator judges whether each search rollout succeeds; this judgment replaces the environment reward used in simulation and is returned to the search loop. The operator also resets the scene after each rollout. For both search and evaluation, initial object positions are randomly sampled within a fixed reset region, and the positions used for evaluation are sampled independently of and differ from those used during search. We evaluate RoboAuto, $\pi _ { 0 . 5 }$ , and ASPIRE on three long-horizon manipulation tasks using a WidowX AI arm. In Sort Balls, the robot must place a green ping-pong ball on a plate and a red ping-pong ball in a bowl. In Stack Dishes, it must place a low-rimmed bowl on a plate and then place a cup upright in the bowl. In Erase Board, it must pick up a compliant eraser, erase “RoboAuto” from a whiteboard, and return the eraser to its original position. Each method is evaluated over 20 trials per task with initial object positions drawn from the same reset region but not used during search. After each trial, a human evaluator assigns a single stage-completion score, the fraction of stages completed, and performance is the mean score over trials.

Task evaluation. We evaluate performance based on stage completion rather than binary success on the full longhorizon task. Sort Balls consists of four stages: (1) picking up the green ping-pong ball, (2) placing it on the plate, (3) picking up the red ping-pong ball, and (4) placing it in the bowl. Stack Dishes also consists of four stages: (1) picking up the bowl, (2) placing the bowl on the plate, (3) picking up the cup, and (4) placing the cup upright in the bowl. Each completed stage in these two tasks contributes one quarter of the total score. Erase Board consists of three stages: (1) picking up the eraser, (2) erasing the eight letters in “RoboAuto,” and (3) returning the eraser to its original position. Each stage contributes one third of the total score. Because the erasing stage involves eight letters, we assign proportional credit based on the number of letters successfully erased: erasing n letters yields $n / 8$ of the score allocated to this stage. This stage-based metric captures meaningfu partial progress and provides a more informative evaluation than measuring only whether the entire long-horizon task is completed.

![](images/620605d34fe549adbe7c57597eec0436bdec2d2467dd9512fde8786b003b7dd8.jpg)  
Figure 18. Real-robot rollouts in detail. RoboAuto programs on a WidowX AI arm for three tasks. Each row shows nine scene-camera keyframes in time order and one wrist-camera frame. Orange labels mark actions in progress, and checked labels mark stages verified before the next stage begins.