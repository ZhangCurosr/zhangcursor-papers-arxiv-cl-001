# Can LLMs Uncover Hidden Physical Laws through Free Exploration?

Zhen Yang<sup>1</sup>, Wen Wang<sup>3</sup>, Xander Xu<sup>2</sup>\*, Ying-Cong Chen<sup>1,4</sup>\* <sup>1</sup>HKUST(GZ), <sup>2</sup>Alibaba Group, <sup>3</sup>ZJU, <sup>4</sup>HKUST

zheny.cs@gmail.com, yingcongchen@ust.hk

## Abstract

Scientific discovery often begins with an unexpected observation that prompts a new research question. In a world with altered physical laws, can large language models (LLMs) notice unexpected behavior, such as heavier objects falling more slowly, and discover the law through experiments without being told what to investigate? To evaluate this ability, we introduce LongChang, an automated framework that creates simulated worlds with altered physics, where one or more laws differ from ordinary physics. The evaluated LLMs act as explor ers, manipulating objects and building tools to investigate their own questions without research tasks or hints about these laws. Separate LLM agents generate and review scenes, review tool designs, and evaluate reports for correctness and experimental support using the altered physical laws and exploration records. These records guide scene evolution for further exploration, allowing the full cycle to run without human intervention. Among 12 LLMs, GPT-6- Astra achieves a discovery rate of 16.7%, while the other 11 score 0%. The exploration records reveal uninvestigated anomalies and only partial identification of the altered physical laws. Our project page and code are available.

## 1 Introduction

Large language models (LLMs) aid scientific discovery through hypothesis generation, experiment design, and evidence analysis. Scientific discovery often begins with an unexpected observation that prompts a new research question. To pursue such inquiry autonomously, LLMs must decide what to investigate and test their explanations through experiments. This ability could help scientists investigate unexpected phenomena and pursue questions they might otherwise overlook. LongChang’s name reflects independent inquiry, inspired by the philosopher Wang Yangming’s introspection in a cave at Longchang, where he reconsidered how to seek knowledge and reached an insight that helped shape his own School of Mind (Yangming, 2026).

![](images/de599e5b4383005125a21b4ebea05b4b8523c937994d382ea2b8545c06d3650c.jpg)  
Figure 1: LongChang’s glass room. Explorers manipulate objects and build tools. Other LLMs evolve scenes using exploration records without human intervention.

Physical law discovery involves inferring an environment’s laws from observations and experiments. PhysGym (Chen et al., 2025), NewtonBench (Zheng et al., 2026), and DiscoverPhysics (Wiemann et al., 2026) assign lawdiscovery tasks. Tab. 2 compares their exploration guidance and experiments with those of LongChang. Without a research task or hints about altered physical laws, an LLM must notice phenomena worth investigating, choose a research question, and devise experiments to distinguish competing explanations. Can LLMs turn unexpected observations into research questions, conduct experiments, and discover altered physical laws autonomously?

We introduce LongChang, an automated framework that generates simulated worlds with altered physics, where one or more laws differ from ordinary physics. This counterfactual setting tests whether LLMs independently notice departures from ordinary physics and investigate them without research tasks or hints about the altered physical laws. For example, slower falls for heavier objects could lead an explorer to ask how mass affects falling and design controlled comparisons to test this link. These laws are hidden from explorers but available for evaluating their reported explanations.

![](images/e8329c5047ae95c865c8a1fdf93de25a3760313bd8ca7bc0fb2714a326714148.jpg)  
Table 1: Scene 01 uses ordinary physics. Scenes 02–15 each have one altered physical law. Laws are in App. A.1.

LongChang’s base set in Tab. 1 comprises a normal control and 14 scenes with one altered physical law each. In the virtual “glass room” in Fig. 1, explorers manipulate objects and build tools that interact with the scene objects within a shared MuJoCo rigid-body simulation (Todorov et al., 2012). Separate LLM agents review tool designs and evaluate reports against the altered physical laws and exploration records, requiring both correctness and experimental support for discovery. Unresolved questions and corrected errors in these records guide LLM scene designers to generate, refine, and compose scenes for further exploration, continuing the entire discovery cycle without human intervention.

Among 12 evaluated LLMs, GPT-6-Astra achieves a discovery rate of 16.7% under free exploration, while the other 11 score 0%. Exploration records show some LLMs leave observed anomalies uninvestigated, while others experiment but only partially identify the altered physical laws, leaving their conditions or applicability unresolved.

We contribute the LongChang framework for the complete discovery cycle and an empirical study of autonomous physical inquiry. We use model comparisons, controlled interventions, and exploration records to examine how LLMs form research questions and test explanations using experiments.

<table><tr><td>Closest work</td><td>No task</td><td>Interactive 3D scenes</td><td>Tool design</td><td>Scene evolution</td></tr><tr><td>PhysGym</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>NewtonBench</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>DiscoverPhysics</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>LongChang</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 2: LongChang compared with the closest work: PhysGym (Chen et al., 2025), NewtonBench (Zheng et al., 2026), and DiscoverPhysics (Wiemann et al., 2026). No task: no assigned research goal. Interactive 3D scenes: physical worlds in MuJoCo (Todorov et al., 2012). Tool design: devices built to interact with scene objects. Scene evolution: automatically generating and revising scenes guided by exploration records.

## 2 Related Work

## 2.1 Autonomous Scientific Discovery

PhysGym (Chen et al., 2025) and Newton-Bench (Zheng et al., 2026) ask LLMs to recover hidden equations through numerical queries, while DiscoverPhysics (Wiemann et al., 2026) studies law discovery in particle simulations. These benchmarks specify discovery goals before experimentation, as summarized in Tab. 2. Other protocols require explicit initial hypotheses and a minimum number of experiments (Nägele and Marquardt, 2025) or supply the anomaly to explain (Geng et al., 2026). Open-ended systems let LLMs select questions within an assigned research role, using supplied datasets (Agarwal et al., 2025; Chen et al., 2026; Yao et al., 2025) or an optical research setting (Zeng et al., 2026; Yang et al., 2026). LongChang tests whether LLMs notice unusual effects of altered physical laws, initiate investigations, and establish these laws experimentally, without a preassigned research task or hints about these laws.

![](images/5a3c7a027b24c23726dc6c9dd49b695cf099ec373aeab90a9692399abf34d9d0.jpg)  
Figure 2: LongChang’s automated cycle of scene construction, free exploration, assessment, and scene evolution.

Left Alone (Szeider, 2025) studies self-directed behavior without an assigned task or a simulated physical world. Explore-then-Act in Look Before You Leap (Ye et al., 2026) studies task-free exploration, including in ScienceWorld, with modeling instructions and a warning that expectations may fail. It evaluates exploration and downstream task performance, rather than experimentally supported discovery of altered physical laws by the explorer.

Both WorldTest (Warrier et al., 2025) and OneLife (Khan et al., 2026) examine predictive world models and explicitly ask explorers to learn environment rules. LongChang instead asks whether unexpected physical observations prompt LLMs to pose their own questions and establish altered physical laws through experiments, without research tasks or advance hints. Tab. 2 therefore focuses on frameworks for physical law discovery.

## 2.2 Tool Use and Design

DiscoveryWorld supplies instruments and abstracted object interactions in a grid world (Jansen et al., 2024), while ChemWorld exposes chemical procedures through programmable modules (Qiu et al., 2026). Tool-design systems optimize tool geometry and use for a specified manipulation task (Lin et al., 2025) or propose detector layouts under given performance goals and resource constraints (Zoccheddu et al., 2026). LongChang ties tool construction to the explorer’s own investigation: LLMs decide whether a tool is needed, what to build, and how to use it to test a hypothesis. The explorer’s constructed tools interact physically with the other objects in the same simulated world.

## 2.3 Environment Generation and Evolution

Environment-generation methods create tasks and rewards for agent training and adapt environments based on performance or trajectory feedback (Faldor et al., 2025; Kang et al., 2026). Programmable chemical worlds support controlled changes to private laws while preserving public experimental conditions (Qiu et al., 2026). Trajectory-based benchmark construction uses tool-use records to create new tasks with solutions checked by reexecution (Guo et al., 2025). In LongChang, LLM scene designers use unresolved questions and corrected errors in exploration records to generate, refine, and compose scenes. Explorer LLMs choose observations to investigate, while separate LLM agents review tools and assess reports, continuing this discovery process without human intervention.

## 3 LongChang

Fig. 2 presents LongChang’s automated cycle of physical law discovery: environment construction (Sec. 3.1), autonomous exploration (Sec. 3.2), and discovery evaluation (Sec. 3.3). The evaluated LLMs serve as explorers, while separate LLM agents design and review scenes, review apparatus, and evaluate reports. LLMs carry out the entire loop without human intervention, using exploration records to guide the subsequent evolution of scenes.

## 3.1 Environment Construction

Physical Environment. An LLM designer specifies each scene’s objects and geometry, physical program, and any altered physical laws. The framework compiles these definitions into MuJoCo simulations (Todorov et al., 2012) containing manipulable balls and boxes, with colored floor tiles in some scenes. Each base scene with altered physics modifies one physical law, producing effects that depend on object properties, spatial relations, contact events, or past actions. Fig. 3 illustrates an altered physical law: when the balls are close, both rise under an upward force without an upward launch.

![](images/05b763a915ee4006555e3f5ed3328fe56564d6fa8e1a785caa7e5009b2b22d59.jpg)  
Figure 3: An example of an altered physical law. When the two balls are close, an upward force makes them rise together. Labels 1–5 identify successive snapshots.

The designer records each altered physical law’s target objects, conditions, and observable effects for evaluation, withholding this information from the explorer. LongChang’s base set in Tab. 1 comprises a normal control and 14 scenes with one altered physical law each. The normal control assesses whether LLMs ground their conclusions in evidence when no physical law has been altered. App. A.1 details ordinary physics and each altered physical law’s scope, conditions, and parameters.

Scene Evolution. To extend this set, an LLM designer proposes scene changes aimed at more challenging discovery tasks. The designer receives source scenes and any available summaries of earlier explorations. These summaries identify tested conditions, unresolved questions, and explanations supported or corrected by later observations. The designer uses three operations: generation creates new laws and scenes from source mechanisms, refinement modifies an existing scene, and composition combines source laws in a shared environment. Changes guided by these summaries must cite supporting exploration evidence. Automated tests check simulated behavior against the altered physical laws, including interactions with constructed apparatus. An LLM reviewer checks the program and test results against these laws. Failed compilation, tests, or review trigger automatic revision requests to the designer. Only accepted scenes are used in subsequent free-exploration assessments.

## 3.2 Autonomous Exploration

The explorer chooses which observations to investigate, formulates its own research questions, and tests explanations through physical interventions. The initial prompt asks for free exploration without a research task or hints about the altered physics. Structured observations give object positions, motion, and instrument readings, with distances in bricks and elapsed time in counts. The explorer can push or throw objects, grasp and move them, release them, wait, and mark reference locations. App. A.2 documents the complete action interface.

![](images/184a72cbcdbd5fde9bc5b8b033a2069a6b5fa00a868c0cc5733d0033729dec88.jpg)  
Figure 4: An example of an LLM-built tool. The pendulum uses its hinge angle to test whether local gravity is tilted. The instrument reads −8<sup>◦</sup> in the illustrated state.

When an investigation needs additional control or measurement, the explorer can design apparatus such as the pendulum in Fig. 4. It specifies the device’s geometry and mechanical connections, including joints and springs. An LLM reviewer checks the compiled design for physical validity and returns feedback on rejections. The explorer can revise a rejected design and submit it again, with successful designs installed into the current scene. Once installed, the explorer can place objects on the apparatus or adjust its joints and receive updated motion readings with the next observation.

At each turn, the explorer records its observations and action rationale, then acts and receives updated observations. The cycle continues until it stops or reaches the turn limit, retaining notes, actions, observations, and feedback from all attempts.

The final report summarizes the observations and explanations from this record. Each conclusion must cite supporting turns and use only interpretations already recorded in the notes, preserving the uncertainty stated there. The explorer may also report that it found no anomaly during exploration.

## 3.3 Discovery Evaluation

An LLM judge evaluates reports against altered physical laws and full exploration records, requiring law correctness (C) and evidence support (S) for discovery. Law correctness requires identifying each altered physical law, its target objects, conditions, triggers, and constants identifiable in the reported units. Proposed mechanisms may differ if they yield the same observable predictions under allowed interventions. Evidence support requires recorded experiments to justify all claims and reasoning at the stated level of certainty. Correct guesses may pass only C. Supported ordinary observations or partial findings may meet S alone.

<table><tr><td></td><td colspan="3">Discovery (%)</td><td>Control</td><td colspan="5">Exploration</td></tr><tr><td>Model</td><td>D</td><td>C</td><td>S</td><td>R</td><td>L</td><td>ES (%)</td><td>CS</td><td>CF (%)</td><td>AF (%)</td></tr><tr><td>GPT-5.6-Luna</td><td>0.0</td><td>0.0</td><td>38.1</td><td>0</td><td>36.7</td><td>95.6</td><td>0</td><td></td><td>0.8</td></tr><tr><td>GPT-6-Astra</td><td>16.7</td><td>19.0</td><td>97.6</td><td>0</td><td>39.3</td><td>100.0</td><td>11</td><td>48.0</td><td>2.1</td></tr><tr><td>Claude-Opus-5</td><td>0.0</td><td>16.7</td><td>0.0</td><td>0</td><td>85.8</td><td>100.0</td><td>44</td><td>66.6</td><td>18.2</td></tr><tr><td>Gemini-3.7-Flash</td><td>0.0</td><td>0.0</td><td>19.0</td><td>0</td><td>10.9</td><td>100.0</td><td>1</td><td>75.0</td><td>6.1</td></tr><tr><td>GLM-5.3</td><td>0.0</td><td>4.8</td><td>0.0</td><td>3</td><td>91.6</td><td>97.8</td><td>45</td><td>64.0</td><td>16.8</td></tr><tr><td>GLM-5.3-Flash</td><td>0.0</td><td>4.8</td><td>0.0</td><td>1</td><td>87.6</td><td>95.6</td><td>45</td><td>71.4</td><td>20.2</td></tr><tr><td>Kimi-K3</td><td>0.0</td><td>2.4</td><td>0.0</td><td>3</td><td>83.9</td><td>93.3</td><td>42</td><td>82.1</td><td>32.7</td></tr><tr><td>MiMo-V2.6-Flash</td><td>0.0</td><td>4.8</td><td>0.0</td><td>1</td><td>36.2</td><td>100.0</td><td>33</td><td>71.1</td><td>24.2</td></tr><tr><td>MiMo-V2.6-Pro</td><td>0.0</td><td>7.1</td><td>0.0</td><td>2</td><td>42.3</td><td>100.0</td><td>35</td><td>70.1</td><td>19.5</td></tr><tr><td>DeepSeek-Flash</td><td>0.0</td><td>0.0</td><td>0.0</td><td>2</td><td>45.6</td><td>100.0</td><td>42</td><td>77.1</td><td>19.2</td></tr><tr><td>Qwen3-Max</td><td>0.0</td><td>0.0</td><td>9.5</td><td>0</td><td>26.7</td><td>100.0</td><td>16</td><td>63.8</td><td>6.4</td></tr><tr><td>Grok-4.7</td><td>0.0</td><td>0.0</td><td>2.4</td><td>1</td><td>85.2</td><td>24.4</td><td>2</td><td>88.5</td><td>12.2</td></tr></table>

Table 3: Discovery performance and exploration behavior of 12 LLMs in LongChang, with metrics from Sec. 3.4.

## 3.4 Experimental Setup

Main evaluation. All LLMs explore the same 15 scenes from fixed initial states for up to 100 turns. LLMs. We evaluate the 12 LLMs in Tab. 3. GPT-6.1-Sol judges the correctness and experimental support of their reports. GPT-6-Astra reviews apparatus and generates and reviews new scenes. App. A.3 lists the prompts used across the pipeline. Metrics. For scenes with altered physics, C and S show the percentages of runs passing the relevant check. Discovery rate (D) is the percentage passing both checks. Rejection count (R) is the number of rejected normal-control reports, including those with false claims of anomalies or unsupported laws. Exploration statistics. Across all scenes, exploration length (L) is the mean number of turns used, and early stopping (ES) is the percentage of runs ending before the turn limit. Action failure (AF) is the percentage of turns with failed actions. Construction success (CS) counts explorations with at least one successful build, and construction failure (CF) gives the percentage of failed build attempts. These statistics distinguish budget use, access to working apparatus, and action execution reliability.

## 4 Results and Analysis

Tab. 3 shows that GPT-6-Astra achieves a discovery rate of 16.7% under free exploration, while the other 11 LLMs score 0%. Its evidence support rate reaches 97.6%, but law correctness is only

19.0%: supported observations and partial findings often leave the altered physical laws unresolved. For example, a Scene 02 report supports attraction between the tested red balls without establishing its distance dependence or applicability to other red spheres. Claude-Opus-5 shows the opposite pattern, identifying the altered physical laws in 16.7% of runs while no report passes the evidence support check. These results separate correct identification of a law from experimentally supported discovery.

Exploration statistics reveal obstacles during investigation. Gemini-3.7-Flash stops early in every run, averaging only 10.9 turns and leaving little opportunity for follow-up experiments. The GLM models use about 90 turns and build working apparatus in every run, yet make no discoveries. They also fail in more than 60% of build attempts, with failed actions accounting for 16.8–20.2% of turns. Successful construction can thus coexist with repeated unsuccessful attempts. Grok-4.7 stops early least often but makes no discoveries, so persistence and working devices alone do not ensure discovery.

## 4.1 Forming Research Questions

We study whether explorers turn unexpected observations into research questions and pursue them experimentally. Fig. 5 compares two Scene 15 explorations with the same initial state and prompt. When the red ball starts moving while the blue ball is held, GPT-6-Astra postpones its planned collision test to investigate. It measures the restart delay and tests whether pushing blue also resets it, discovering that pushes or throws pause red’s northward force for ten counts. Gemini-3.7-Flash observes a similar restart but continues its planned tests, leaving the unexpected motion unexamined.

We test guidance by giving GPT-6-Astra a law-

Scene 15: The red ball is driven northward. A nonzero push or throw on a movable object pauses the force for 10 counts.

![](images/d7e16ad2c35487ab9bbfac247674598182429ec2ac31305afa5cf922af2fff7a.jpg)  
Figure 5: Case study of research question formation in Scene 15. (a) GPT-6-Astra investigates red’s delayed restart and discovers the altered physical law. (b) Gemini-3.7-Flash observes the restart but still follows its original plan.

<table><tr><td>Condition</td><td>D (%)</td><td>C (%)</td><td>S (%)</td><td>L</td><td>AF (%)</td></tr><tr><td>Task guidance, 100 turns</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Free exploration</td><td>16.7</td><td>19.0</td><td>97.6</td><td>39.3</td><td>2.1</td></tr><tr><td>Task-guided</td><td>28.6</td><td>31.0</td><td>97.6</td><td>30.9</td><td>2.7</td></tr><tr><td>Turn budget</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>100 turns</td><td>16.7</td><td>16.7</td><td>100.0</td><td>42.2</td><td>2.7</td></tr><tr><td>500 turns</td><td>14.3</td><td>14.3</td><td>100.0</td><td>41.4</td><td>3.6</td></tr><tr><td>Apparatus access, 100 turns</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Enabled</td><td>16.7</td><td>19.0</td><td>97.6</td><td>39.3</td><td>2.1</td></tr><tr><td>Disabled</td><td>4.8</td><td>4.8</td><td>100.0</td><td>38.7</td><td>0.9</td></tr></table>

Table 4: Effects of task guidance, turn budget, and apparatus access on discovery and exploration behavior.

discovery task, noting that physical laws may be altered, and supplying a scene-specific clue, with other settings fixed. In Tab. 4, discovery rises from 16.7% to 28.6%, while evidence support remains unchanged. Guidance improves law identification when experimental support is already high. Mean exploration length falls from 39.3 to 30.9 turns, yielding more discoveries with fewer interactions.

## 4.2 Validating Scientific Claims

Distinguishing competing explanations requires experiments for which they predict different outcomes. Fig. 6 illustrates this process in Scene 12. Two successive red–blue collisions change and restore the balls’ colors, consistent with both color exchange and independent red–blue flipping. GPT-6-Astra tests two red balls, for which exchange predicts that both remain red and flipping predicts that both turn blue. Both balls remain red, supporting exchange and ruling out independent flipping.

The normal control tests whether LLMs overinterpret observations under ordinary physics. Across LLMs, 36.1% of reports are rejected for false claims of altered physical laws or unsupported explanations of ordinary physics. For example, DeepSeek-Flash attributes a ball’s longer travel on the decorative gray tile to lower friction after comparing equal pushes. The ball is already moving before each push, so different starting speeds could explain the difference in distance. A fair comparison would start the ball from rest before each push.

## 4.3 Using Experimental Resources

We examine whether larger budgets or apparatus access helps LLMs resolve unanswered questions. We compare budgets of 100 and 500 turns under free exploration without context compaction, with other settings matched. Increasing the limit to 500 turns does not improve discovery: GPT-6-Astra uses about 40 turns under either limit and stops early, sometimes leaving questions unresolved. Extra turns provide little value if explorers stop early.

![](images/d5c8e54cfa3aef373a7997880ce069592689968a90e0215a7ba13c5e6aaf29ec.jpg)  
Figure 6: Case study of testing competing explanations.  
Figure 7: Case study of tool construction and redesign.

The apparatus comparison uses a 100-turn limit. Disabling construction removes its actions and instructions but retains basic object manipulation. With construction enabled, discovery rises from 4.8% to 16.7% and law correctness from 4.8% to 19.0% in Tab. 4, although action failures also increase. The explorer chooses whether to build a tool, so these results concern access to construction rather than the contribution of tool use in each run.

Fig. 7 illustrates the construction and redesign of a weighing platform in Scene 15. GPT-6-Astra builds a flat pan to compare the balls’ loads, but the red ball drifts off after release. It adds retaining walls and increases the pan’s vertical travel, obtaining stable readings after releasing each ball.

## 5 Scene Evolution

Exploration records can also inform what LLMs investigate in subsequent scenes. We study whether corrected errors from successful explorations can guide generation, refinement, and composition of new discovery challenges. Each operation modifies a mechanism involved in the same error to create a scene with altered physics for further exploration.

<table><tr><td colspan="2">Scene 15: A nonzero push or throw pauses the red ball&#x27;s northward force for 10 s.</td></tr><tr><td colspan="2">Corrected error. The explorer initially interpreted this pause as evidence of no lasting force.</td></tr><tr><td>How the scene evolves</td><td>Altered physical laws</td></tr><tr><td>Generation</td><td></td></tr><tr><td>Replace the northward force with mutual repulsion between the red and blue balls, retaining the 10 s pause.</td><td>The red and blue balls are subject to mutual horizontal repulsion. Any nonzero push or throw pauses this mutual repulsion for 10 s.</td></tr><tr><td>Refinement Block the red ball&#x27;s northward force while the blue ball</td><td></td></tr><tr><td>is inside the gray square, even after the 10 s pause ends.</td><td>Blue inside the gray square blocks the red ball&#x27;s northward force. Nonzero pushes or throws also pause the northward force for 10 s.</td></tr><tr><td>Composition Add Scene 14&#x27;s rule: the green ball copies red&#x27;s launch</td><td>A nonzero push or throw on any object pauses red&#x27;s force for 10 s.</td></tr></table>

Table 5: Scene evolution guided by a corrected error in Scene 15, with altered physical laws in three new scenes.

We review successful GPT-6-Astra explorations for explanations corrected after later observations. In Scene 15, red stops after a push, prompting the inference that no persistent northward force exists. When red restarts, the explorer revises this explanation to account for a temporary pause in the force. The designer receives this error, its correction, and the source laws, and uses the same evidence for all three operations. Tab. 5 shows how the three designs retain the pause while modifying these laws. Generation. The designer replaces the fixed northward force on red with mutual repulsion between red and blue, making the force depend on their rela tive positions. Swapping their positions at matched times after a launch should reverse the forces, distinguishing repulsion from a fixed directional force. Refinement. The designer introduces a spatial condition blocking red’s northward force while blue is inside the gray square, so waiting longer than ten seconds may still not restore it. After the pause expires, the planned test moves blue by hand into or out of the square without resetting the timer. Composition. The designer combines the paused northward force with Scene 14’s delayed motion copying: green takes red’s launch velocity twelve seconds later. The planned comparison launches blue after red, postponing red’s force recovery while leaving green’s scheduled velocity copy unchanged. The explorer must identify both altered physical laws and their distinct triggers and timing.

We evaluate GPT-6-Astra on source and evolved scenes using the free-exploration settings in Tab. 3.

Tab. 6 reports discovery rates of 33.3% in Scene 15 and 66.7% in Scene 14, compared with 0% in all three evolved scenes. Evidence support remains at 100%, but no evolved-scene report passes law correctness. In the generated scene, an exploration identifies position-dependent repulsion without recovering its pause and delay. In the refined scene, the explorer observes northward motion without linking it to blue’s position. In the composed scene, it records red’s drift and green’s intermittent motion without identifying the two timing rules. These cases use a corrected error to guide scene evolution, revealing unresolved conditions and dependencies in otherwise supported findings.

<table><tr><td>Scene</td><td>C (%)</td><td>S (%)</td><td>D (%)</td></tr><tr><td>Scene 15</td><td>33.3</td><td>100.0</td><td>33.3</td></tr><tr><td>Scene 14</td><td>66.7</td><td>100.0</td><td>66.7</td></tr><tr><td>Generation</td><td>0.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Refinement</td><td>0.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Composition</td><td>0.0</td><td>100.0</td><td>0.0</td></tr></table>

Table 6: Discovery rates in source and evolved scenes.

## 6 Conclusion

LongChang examines whether LLMs can turn unexpected observations into research questions and discover altered physical laws through experiments. The evaluated LLMs manipulate objects and build tools without research tasks or hints about these laws. Only one of the twelve evaluated LLMs makes discoveries, revealing the difficulty of turning observations into complete, experimentally supported laws. Unanswered questions and corrected errors in exploration records guide separate LLM agents to evolve scenes, continuing the discovery cycle without human intervention. This framework supports research into LLMs that decide what to investigate and design experiments to resolve uncertainties about the physical laws of these worlds.

## 7 Limitations

This study examines whether LLMs initiate investigations and discover altered physical laws in simulated worlds. Our results show that this task remains challenging for current LLMs. The altered physical laws provide a clear basis for evaluating discoveries, while many scientific questions in the real world remain unresolved. Extending the framework to these open questions requires further research on evaluating exploration and its findings in settings where the answers are not yet established.

## 8 Ethical Considerations

LongChang uses no human annotation or testing and evaluates LLMs without human intervention. The evaluation therefore raises no ethical concerns related to participant privacy or informed consent.

## References

Dhruv Agarwal, Bodhisattwa Prasad Majumder, Reece Adamson, Megha Chakravorty, Satvika Reddy Gavireddy, Aditya Parashar, Harshit Surana, Bhavana Dalvi Mishra, Andrew McCallum, Ashish Sabharwal, and Peter Clark. 2025. Autodiscovery: Openended scientific discovery via bayesian surprise. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025.

Jiayao Chen, Shi Liu, and Linyi Yang. 2026. Statefuldiscovery: Evidence-calibrated claim formation in openended scientific discovery. CoRR, abs/2606.11851.

Yimeng Chen, Piotr Piekos, Mateusz Ostaszewski, Firas Laakom, and Jürgen Schmidhuber. 2025. Physgym: Benchmarking llms in interactive physics discovery with controlled priors. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025.

Maxence Faldor, Jenny Zhang, Antoine Cully, and Jeff Clune. 2025. OMNI-EPIC: open-endedness via models of human notions of interestingness with environments programmed in code. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Jiayi Geng, Zhengxuan Wu, Kevin S Chen, Seungone Kim, Joseph Janssen, Zora Zhiruo Wang, Bhupalee Kalita, Runtian Gao, Aaron Ho, Andrew Oakleigh

Nelson, et al. 2026. Eurekabench: Measuring agentic ability to discover new scientific insights. arXiv preprint arXiv:2610.00492.

Dadi Guo, Tianyi Zhou, Dongrui Liu, Chen Qian, Qihan Ren, Shuai Shao, Zhiyuan Fan, Yi R. Fung, Kun Wang, Linfeng Zhang, and Jing Shao. 2025. Towards self-evolving benchmarks: Synthesizing agent trajectories via test-time exploration under validateby-reproduce paradigm. CoRR, abs/2510.00415.

Peter A. Jansen, Marc-Alexandre Côté, Tushar Khot, Erin Bransom, Bhavana Dalvi Mishra, Bodhisattwa Prasad Majumder, Oyvind Tafjord, and Peter Clark. 2024. Discoveryworld: A virtual environment for developing and evaluating automated scientific discovery agents. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Haoqiang Kang, Xiaokang Ye, Yuhan Liu, Siddhant Hitesh Mantri, Lingjun Mao, James Fleming, Drishti Regmi, and Lianhui Qin. 2026. Simworld studio: Automatic environment generation with evolving coding agent for embodied agent learning. CoRR, abs/2605.09423.

Zaid Khan, Archiki Prasad, Elias Stengel-Eskin, Jaemin Cho, and Mohit Bansal. 2026. One life to learn: Inferring symbolic world models for stochastic environments from unguided exploration. In The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026. proceedings.iclr.cc.

Chunru Lin, Haotian Yuan, Yian Wang, Xiaowen Qiu, Tsun-Hsuan Johnson Wang, Minghao Guo, Bohan Wang, Yashraj Narang, Dieter Fox, and Chuang Gan. 2025. Robotsmith: Generative robotic tool design for acquisition of complex manipulation skills. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025.

Maximilian Nägele and Florian Marquardt. 2025. Agentic exploration of physics models. CoRR, abs/2509.24978.

Jiangjie Qiu, Yijun Li, and Xiaonan Wang. 2026. Chemworld: Programmable chemical worlds for controlled and replayable agent experimentation. CoRR, abs/2608.10792.

Stefan Szeider. 2025. What do LLM agents do when left alone? evidence of spontaneous meta-cognitive patterns. CoRR, abs/2509.21224.

Emanuel Todorov, Tom Erez, and Yuval Tassa. 2012. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2012, Vilamoura, Algarve, Portugal, October 7-12, 2012, pages 5026– 5033. IEEE.

Archana Warrier, Dat Nguyen, Michelangelo Naim, Moksh Jain, Yichao Liang, Karen Schroeder, Cambridge Yang, Joshua B Tenenbaum, Sebastian Vollmer, Kevin Ellis, et al. 2025. Benchmarking world-model learning with environment-level queries. arXiv preprint arXiv:2510.19788.

Matt L. Wiemann, Lindsay M. Smith, Peter Melchior, Siddharth Mishra-Sharma, Andrew Gordon Wilson, Pavel Izmailov, and Carolina Cuesta-Lázaro. 2026. Discoverphysics: Benchmarking llms for out-of-thebox scientific thinking. CoRR, abs/2605.26087.

Shuxing Yang, Fujia Chen, Rui Zhao, Junyao Wu, Yize Wang, Haiyao Luo, Ning Han, Qiaolu Chen, Yuze Hu, Wenhao Li, Mingzhu Li, Hongsheng Chen, and Yihao Yang. 2026. End-to-end autonomous scientific discovery on a real optical platform. CoRR, abs/2604.27092.

Wang Yangming. 2026. Instructions for practical living. Arthur Peng.

Lance Yao, Suman Samantray, Ayana Ghosh, Kevin Roccapriore, Libor Kovarik, Sarah I. Allec, and Maxim A. Ziatdinov. 2025. Operationalizing serendipity: Multi-agent AI workflows for enhanced materials characterization with theory-in-the-loop. CoRR, abs/2508.06569.

Ziang Ye, Wentao Shi, Yuxin Liu, Yu Wang, Zhengzhou Cai, Yaorui Shi, Qi Gu, Xunliang Cai, and Fuli Feng. 2026. Look before you leap: Autonomous exploration for LLM agents. CoRR, abs/2605.16143.

Xianrui Zeng, Pengfei Liu, Yirui Zang, Yang Shen, Fei Yu, Chunlei Yu, Minghao Liu, and Yang Du. 2026. Socratic agents for autonomous scientific discovery in high-dimensional physical systems. CoRR, abs/2606.26722.

Tianshi Zheng, Kelvin Kiu-Wai Tam, Newt Hue-Nam K. Nguyen, Baixuan Xu, Zhaowei Wang, Cheng Jiayang, Hong Ting Tsang, Weiqi Wang, Jiaxin Bai, Tianqing Fang, Yangqiu Song, Ginny Y. Wong, and Simon See. 2026. Newtonbench: Benchmarking generalizable scientific law discovery in LLM agents. In The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026. proceedings.iclr.cc.

Sara Zoccheddu, Shah Rukh Qasim, Patrick Haworth Owen, and Nicola Serra. 2026. Large language models for physics instrument design. Mach. Learn. Sci. Technol., 7(4):45005.

## A Appendix

## A.1 Scene Laws

Scene 01 provides the normal control: three movable balls and a decorative gray tile in a room with ordinary rigid-body physics. Motion follows F = ma and p = mv, with constant downward gravitational acceleration independent of mass. Contact with other objects, the floor, and walls produces collisions and friction. Object colors remain fixed, and the gray tile has no physical effect on motion.

Scenes 02–15 each change one physical law of this setting. Tab. 7 and Tab. 8 detail the laws and scope of the 15 base and the three evolved scenes.

## A.2 Action Interface

Tab. 9 specifies the target objects, parameters, and effects of the actions available during exploration.

## A.3 Prompts

The prompts below guide exploration, generate final reports, and evaluate discoveries. Task guidance supplements the original exploration prompt.

## A.4 AI Assistance

We used AI assistants to improve the wording and clarity of the manuscript and to assist with code implementation and debugging. We also used GPT-6.1-Sol as an automated judge to assess the correctness and experimental support of exploration reports, as described in Sec. 3.3. The authors are responsible for the manuscript, code, and findings.

## A.5 Potential Risks

LLM judges may endorse unsupported findings and mislead subsequent research. Our experiments are simulated, but physical use of generated experiments or apparatus could cause harm without independent validation and expert safety review. Repeated LLM inference and simulation also consume energy, with associated environmental costs.

## Free exploration

```markdown
Explore freely and report what you learn.
## Basic operations
- PUSH <object> <heading> <impulse>.
- THROW <object> <heading> <elevation> <impulse>.
- WAIT <counts>: advance the world by 1-30 whole
counts.
- GRASP <object>: hold.
- MOVE <object> <x> <y> <z>: move held object.
- RELEASE <object>: let go.
- MARK <id> <x> <y>: place a floor reference.
```

```markdown
## Parameters and observations
- Heading: degrees (0=east, 90=north) or
east/north/west/south; elevation: 0-90 degrees.
- Impulse: 0-10, added to current motion.
Positions: bricks.
- RELEASE a held target before PUSH/THROW.
- Object motion is in events[name].trajectory:
count, position, state. Counts are relative to the
action start and may be fractional; intervals may
differ. No absolute time calibration is supplied.
- omitted_samples counts compressed samples;
key_points_omitted warns that key changes were
dropped.
```

- Use provided objects/observations, obey   
usage/handling limits, never invent results.

\- DONE ends exploration early and requests a report; empty ACT skips physical action.

## ## Optional construction

\- You may create apparatus with BUILD; state its intended use in NOTE.

\- BUILD <JSON>: submit apparatus; required id, purpose, mjcf; optional name.

\- PLACE <object> <tool> <port>: carry and hold an object at a port.

\- ADJUST <tool> <joint> <target>: use a listed hinge/slide joint; target units are degrees for hinges, bricks for slides.

\- Design your own bodies, geometry, mass/density, friction, joints and sites; no preset part catalogue.

<mujoco><worldbody>...</worldbody></mujoco>. Sections: worldbody with bodies, optional compiler (angle only) and tendon.

\- Allowed geoms: sphere, box, capsule, cylinder, ellipsoid; joints: free, ball, hinge, slide. Joint/tendon springs allowed; tendons need finite limits and may refer only to this apparatus.

\- Unsupported: assets, includes, defaults, actuators, sensors, mocap, gravcomp.

relative. Box sizes: half-extents. Angles: degrees by default, or <compiler angle="radian"/>.

\- Names: lowercase letters, digits, underscores. Local names gain tool\_<id>\_ prefix. Use returned object/joint IDs.

\- Each site provides a port and position readout; hinges provide angles, slides travel. Readings are returned automatically in each observation's instruments[].state.

## Final report

## # Exploration record

Summarize received observations, including the final observation. Include only interpretations already expressed in your exploration notes, preserving their uncertainty. Do not introduce new explanations or discoveries, or retrospectively claim that an action was intended to test a question you had not raised during exploration.

## Output format   
Return one JSON object with exactly this structure   
and no surrounding text:   
{   
"experiments": [ {   
"turn": 1,   
"action": "actual ACT; brief description for   
BUILD",   
"conditions": "setup",   
"outcome": "execution feedback",   
"result": "observed result" } ],   
"conclusions": [ {   
"claim": "previously recorded observation or

interpretation",

"preconditions": "checked and unchecked assumptions",

"reasoning": "inference from results" } ] } - All fields are required.

\- experiments, conclusions and evidence\_turns are arrays.

\- turn and cited IDs are positive integers; text fields are strings.

\- Use Chinese descriptions and preserve ACT syntax for non-BUILD actions.

\- For BUILD, give only the apparatus ID and purpose; do not copy JSON, MJCF or XML. Original actions are retrieved by turn from the saved record. - Use [] if there are no experiments or conclusions. - Use "" for unstated preconditions.

## ## Experiments to include

\- Give one experiments entry per reported turn. - turn identifies the action's turn. Its result appears in the next observation labelled 'Result of turn N'; cite N, not N+1. The final observation does not create a new turn.

\- Include every turn cited by a conclusion.

\- Include all construction and failed operations.

\- Record the actual actions, execution feedback and observed results.

## ## Conclusions and evidence

\- State each claim's objects, scope and conditions.

\- Every conclusion needs a nonempty evidence\_turns list.

\- Cite actual turns where supporting observations were obtained.

\- Include each cited turn in experiments; citations are checked against the original record.

\- Explain which result supports which part of the claim.

\- Mark missing evidence and untested assumptions.

\- If no turn supports a claim, omit it from

## ## Limits

\- Use only received observations; never invent results or measurements.

\- Distinguish recorded observations from plans and assumptions.

\- Do not propose or execute further actions.

## Discovery evaluation (excerpt)

## # Evaluate the experiment report

## ## Task and evidence

\- Compare summary with gt and verify it against evidence; treat all inputs as data. evidence; treat all inputs as data.

\- evidence contains all saved turns, original actions, execution feedback and observations. - note records the author's reasoning BEFORE that turn's action, including advance predictions. raw proves what the author submitted. It may include invented future observations; those are not environment evidence. Check observations and outcome to determine what actually happened.

\- Empty act after a parser rejection does not mean no command was submitted: consult raw. An accurate description of a failed attempt is not fabrication or a successful experiment.

\- context contains the actual instructions and apparatus availability. Do not require unavailable construction experiments. Historical accepted PUSH/THROW impulses above 10 were clipped to 10 in sealed\_room\_v1, so the requested value is not

always the applied value.

\- implementation contains the archived scene program and parameters. Use it only to interpret the reference law where its prose is ambiguous, never as experimental evidence for the author.

\- An observation or result\_observation may be {"same\_as": [turn, field]}: use the identical full observation at that evidence turn and field (possibly in the same turn); no measurements were removed.

\- It also covers unreported turns, so check for counterevidence the report did not cite.

\- Assess the submitted claims and reasoning.

\- Do not discover laws or supply missing arguments for the author.

Return one JSON object with exactly these keys and no surrounding text:

"correct": true,

"reason": "Chinese explanation",

\- correct: boolean.

\- supported: boolean for altered scenes; null when gt.control=true.

\- reason: a nonempty Chinese explanation of correctness and evidence.

\- action\_errors: a list; use [] if there are no demonstrated errors.

\## correct: identification of the laws

\- Identify all GT laws through equivalent

observable predictions across allowed interventions. - Preserve objects, scope and triggers; internal mechanism descriptions may differ.

\- Matching a few observed outcomes or omitting trigger conditions is insufficient.

\- Use gt.rubric; do not demand constants that the reported units cannot identify.

\- A correct explicitly proposed hypothesis may have correct=true without sufficient evidence.

Uncertainty alone does not make a correct formula wrong. Do not invent absent law elements.

\- Do not demand exhaustive directions, distances or arbitrarily small impulses. Apply the same core trigger and scope requirements to every report, rather than changing them between runs.

\- Distinguish target dependencies from numerical guards or contact bookkeeping. For example, the base red-sphere force is inverse-square outside its regularization radius, and the base restitution program selects fixed landing regions. Colours label those regions; proving colour rather than hidden material identity causes the contrast is not required.

\- Treat drift acceleration as an additional drive, not necessarily net floor acceleration. An empirical law need not also prove that every conceivable ordinary mechanism is impossible. - For controls, correct means no retained false anomaly claim.

\## supported: evidence and reasoning

\- Follow each conclusion's evidence\_turns into evidence.

\- Verify that reported actions and results match the original record.

\- Verify that cited interventions, comparisons and reasoning support the claims and their certainty. - Fabricated or contradicted results, missing evidence, or assumptions asserted as established without support mean false. Read note and raw before claiming a prediction or attempt is absent.

\- Conditional estimates may be supported when assumptions are stated, the calculation follows under those assumptions, and the uncertainty is retained. Do not treat that as proof of the assumptions.

\- Check cited observations, not hidden simulator parameters; do not invent the author's argument. - If competing explanations fit the record but differ under allowed interventions, explain the unresolved distinction in reason.

\- Supported ordinary observations or partial findings may have supported=true and correct=false. Full-law identification is checked by correct, not imposed a second time on supported.

\- Check that the report preserves interpretations recorded during exploration, including DONE notes. Statements or invented observations in raw are not substitutes for real experiments. 7

## Task guidance

This environment may contain {rules} beyond ordinary physics, or none. Identify and verify any you find through experiments. Use this clue: {hint}

<table><tr><td>Scene</td><td>Physical laws</td></tr><tr><td>01 Normal control</td><td>Follows ordinary physics (App. A.1).</td></tr><tr><td>02 Red attraction</td><td>Free single-sphere red bodies attract each other with  $\mathbf { F } = 2 0 \mathbf { d } / \operatorname* { m a x } ( r , 0 . 5 ) ^ { 3 }$  , where d points toward the other sphere and  $r = | \mathbf { d } | .$ </td></tr><tr><td>03 Mass-dependent fall</td><td>Each free body has downward acceleration go  $\surd m _ { 0 } / m ,$  with  $m _ { 0 } = 1 $ </td></tr><tr><td>04 Oscillating gravity</td><td>Global gravity follows  $g ( t ) = g _ { 0 } [ 1 + 0 . 3 \sin ( 2 \pi t / 6 ) ]$ </td></tr><tr><td>05 High-speed braking</td><td>Red, blue, and the box receive extra braking  $\mathbf { F } = - 8 ( s - 2 . 5 ) \mathbf { v } / s$  only when  $s = | \mathbf { v } | > 2 . 5 .$ </td></tr><tr><td>06 Automatic side force</td><td>Red and blue receive the horizontal force  $\mathbf { F } = 1 . 2 ( v _ { y } , - v _ { x } , 0 )$  when  $| v _ { x } | + | v _ { y } | > 0 . 0 0 1 .$ </td></tr><tr><td>07 Tile rebounds</td><td>At new floor impacts, both gray balls and movable apparatus parts have vertical restitution  $e = 0 . 9$  on the red tile and  $e = 0 . 1 5$  on the blue tile. Hinged and sliding parts are included.</td></tr><tr><td>08 Speed has a limit</td><td>Each free body obeys  $\mathbf { p } = m \mathbf { v } / ( 1 - | \mathbf { v } | ^ { 2 } / v _ { \operatorname* { m a x } } ^ { 2 } )$  with  $v _ { \mathrm { m a x } } = 2 . 2 .$  Forces and impulses add momentum, while speed remains below this limit.</td></tr><tr><td>09 Slows in, speeds out</td><td>At any height, entering the gray tile&#x27;s footprint  $( | x - 2 | < 1 . 5 , | y | < 1 . 5 )$  scales red&#x27;s and blue&#x27;s linear and angular velocities by 0.4. Leaving scales their current velocities by 2.5.</td></tr><tr><td>10 Acceleration zone</td><td>For red and blue, each step inside the yellow tile&#x27;s footprint  $( | x | < 2 , | y | < 2 )$  , at any height, sets  $v _ { x } = 0 . 3 5$  when  $s = | \mathbf { v } | < 0 . 3 5 ,$  or scales v by 1.0015 when  $0 . 3 5 \leq s < 6 .$ </td></tr><tr><td>11 Collision boost</td><td>When contact between two of red, blue, and green ends, both participants&#x27; remaining linear</td></tr><tr><td>12 Contact color swap</td><td>velocities are multiplied by 1.8. Outgoing speeds can still be lower than incoming speeds. Free single-sphere bodies swap colors once at the onset of each separate contact.</td></tr><tr><td>13 Third-contact jump</td><td>Every third separate contact between red and blue sets both upward velocities to 4. Sustained</td></tr><tr><td>14 Velocity copy</td><td>contact counts once. Only this pair counts, based on object identity, not current color. Each launch of red schedules green to take red&#x27;s 3D linear launch velocity 12 seconds later.</td></tr><tr><td>15 Northward force</td><td>Initially no copies are pending, and later launches do not cancel earlier copies. Red&#x27;s extra northward force  $F _ { y } = 1 . 5 m \ \mathrm { s t a r t s }$  active. Each launch of any movable object</td></tr><tr><td colspan="2">restarts a 10-second pause, after which the force resumes. Conventions. Distances use bricks and masses use simulation units. Time step: up. Launch: nonzero PUSH or THROw. Names denote initial objects. Free bodies include added objects but exclude fixed or</td></tr></table>

Table 7: Altered physical laws of the scenes in Tab. 1.

<table><tr><td>Scene</td><td>Physical laws</td></tr><tr><td>Shared conditions</td><td>Conventions follow Tab. 7. Extra accelerations (in  $\mathrm { \ b r i c { k s } / s ^ { 2 } ) }$  start active and add to ordinary physics Each launch of any movable object restarts a 10-second pause.</td></tr><tr><td>Generation</td><td>When active, red and blue repel horizontally with acceleration 1.5, independent of separation. Repulsion is zero when their horizontal centers coincide.</td></tr><tr><td>Refinement</td><td>After the pause, red accelerates northward at 1.5 exactly when blue is outside the closed horizontal square  $[ 2 . 5 , 3 . 5 ] \times [ 2 . 5 , 3 . 5 ]$  , regardless of blue&#x27;s height or held state. Blue starts outside. Crossing the boundary does not reset the timer.</td></tr><tr><td>Composition</td><td>Red follows Scene 15&#x27;s northward-force rule, while its launches also trigger Scene 14&#x27;s delayed velocity copy to green. The 10-second force pause and 12-second copy delay operate independently.</td></tr></table>

Table 8: Altered physical laws of the evolved scenes in Sec. 5.

<table><tr><td>Action</td><td>Target</td><td>Parameters</td><td>Effect</td></tr><tr><td>PUSH</td><td>Released movable object</td><td>Object ID, heading, impulse [0, 10]</td><td>Adds a horizontal impulse to existing motion.</td></tr><tr><td>THROW</td><td>Released movable object</td><td>Object ID, heading, elevation  $[ 0 ^ { \circ } , 9 0 ^ { \circ } ] ,$  , impulse [0, 10]</td><td>Adds a directed impulse to existing motion.</td></tr><tr><td>WAIT</td><td>Entire scene</td><td>Integer duration: 1–30 counts</td><td>Advances simulation time.</td></tr><tr><td>GRASP</td><td>Reachable movable object</td><td>Object ID</td><td>Holds the object in the hand, one object at a time.</td></tr><tr><td>MOVE</td><td>Object held by hand</td><td>Object ID, target  $( x , y , z )$ </td><td>in bricks Applies limited force toward the target, which may not be reached.</td></tr><tr><td></td><td>RELEASE Object held by hand or Object ID port</td><td></td><td>Removes the hand or port hold.</td></tr><tr><td>MARK</td><td>Floor location</td><td>Marker ID, position  $( x , y )$ </td><td>in bricks Adds a visual reference with no physical effect.</td></tr><tr><td>BUILD</td><td>New apparatus</td><td>id, purpose, mjcf</td><td>Installs the apparatus if validation and review pass.</td></tr><tr><td>PLACE</td><td>Movable object and apparatus port</td><td>Object ID, apparatus ID, port ID</td><td>Moves and holds the object at the port, which transfers the holding reaction to the apparatus.</td></tr><tr><td>ADJUST</td><td>Hinge or slide joint</td><td>in degrees or displacement in bricks which may not be reached.</td><td>Apparatus ID, joint ID, target angle Applies limited torque or force toward the target,</td></tr><tr><td>DONE</td><td>Exploration session</td><td>None</td><td>Ends exploration and requests the final report.</td></tr><tr><td colspan="4">Conventions. Headings use degrees  $( 0 ^ { \circ }$  east,  $9 0 ^ { \circ }$  north) or east/north/west/south. One count equals one simulation second, a</td></tr></table>

Table 9: Targets, parameters, and effects of exploration actions.