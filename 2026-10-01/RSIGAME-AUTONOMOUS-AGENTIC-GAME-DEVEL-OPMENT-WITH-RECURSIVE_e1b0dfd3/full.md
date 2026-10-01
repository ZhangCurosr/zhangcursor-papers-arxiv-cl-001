![](images/341949f419c2eb4c09b911e98c40441d3990fe21eefd97e0721bd7cdcc843c12.jpg)

# RSIGAME: AUTONOMOUS AGENTIC GAME DEVEL-OPMENT WITH RECURSIVE SELF-IMPROVEMENT

Wenyi Wu<sup>1∗</sup> Minghao Fu<sup>1,2∗</sup> Jieyu You<sup>2</sup> Kun Zhou<sup>1†</sup> Siqi Liu<sup>1</sup> Aayush Salvi<sup>1</sup> Yiheng Lin<sup>2</sup> Ce Zhang<sup>3</sup> Xiaohan Lan<sup>2</sup> Jiahui Zhu<sup>2</sup> Yujie Zhong<sup>2†</sup> Qi She<sup>2</sup> Biwei Huang<sup>1</sup>

<sup>1</sup>University of California San Diego <sup>2</sup>ByteDance Inc. <sup>3</sup>Carnegie Mellon University

## ABSTRACT

Recent advances in large language models have made automatic game generation increasingly feasible, yet reliably improving generated games beyond a playable version remains challenging. Naive iterative refinement can easily overfit a small set of test cases, producing fragile games with unresolved bugs, missing behaviors, and poor generalization to broader player interactions. We introduce RSIGAME, an autonomous agentic game development framework with recursive self-improvement. RSIGAME organizes development into complementary local and global loops. Concretely, a local explore-diagnose-improve loop broadly explores the executable game, diagnoses and prioritizes discovered issues, and performs evidence-grounded revision, where an evolving checklist continually accumulates new testing and improvement guidance. A global loop tracks overall quality, preserves the best checkpoint, and detects saturation or regression over long-horizon development. Beyond test-time improvement, RSIGAME further internalizes successful development experience into the generator through training. Across 140 GameCraft-Bench tasks, two game engines, and five generators, RSIGAME consistently improves game quality under matched development budgets. Notably, experience internalization enables Qwen3.8-27B to reach 61.38 on Godot and 58.53 on Phaser, exceeding GPT-5.5 one-shot scores while reducing Qwen’s generation tokens by 11 times.

§ Code Dataset <sup></sup> Project Page

![](images/1989dd558ecd121d1f463ecd3efe10d6dd952f4eca8bfce595c629dde657b415.jpg)  
Figure 1: RSIGAME turns agentic game development into autonomous recursive selfimprovement. It combines a local explore-diagnose-improve loop with global progress monitoring and control, enabling Qwen3.8-27B approach GPT-5.5-level performance.

## 1 INTRODUCTION

Recent advances in large language models (LLMs) and vision-language models (VLMs), have substantially expanded the capability of autonomous agents to perform complex real-world tasks, ranging from software engineering to computer-use automation Yang et al. (2024); Zhu et al. (2026). Automatic game development represents another promising domain, as building a complete game requires jointly designing game mechanics and progression, create source code and multimodal assets, and organize these heterogeneous components into a coherent executable project Luo et al. (2026). It has become increasingly realistic with the rapid improvement of long-horizon generation, multimodal understanding, and tool-use capabilities in foundation models. Accordingly, recent benchmarks show that VLM-based agents can already turn natural-language specifications into playable games with recognizable mechanics and visual content (Luo et al., 2026; Chi et al., 2026).

However, game development remains highly challenging due to its nature as a complex multimodal engineering process. Even in mature human teams, unexpected bugs, missing functionalities, and overlooked corner cases frequently arise during development Roque et al. (2025). A common solu tion is therefore to adopt an iterative development workflow, in which developers repeatedly test the current build, collect feedback, and revise the project until it reaches the desired quality. This process can be naturally viewed as a form of recursive self-improvement Wu et al. (2024), where each development round builds upon the previous version based on newly observed evidence. Inspired by this workflow, we formulate agentic game development as an iterative loop: an agent first generates an initial game, then repeatedly tests and improves it until the target requirements are met.

Despite its natural fit for agentic game development, we find that recursive self-improvement can easily converge to fragile solutions that roughly satisfy the target while leaving many untested bugs, missing behaviors, and corner cases unresolved. In effect, the development loop may overfit a small set of test cases rather than improve the game as a whole, leading to poor generalization to broader player interactions. To address this issue, we strengthen the testing stage to broadly explore remaining failures and potential improvements, with the goal of developing a high-quality game rather than merely a playable one. Concretely, we devise an explore-diagnose-improve loop: the agent first broadly explores the executable game to uncover bugs, weaknesses, and opportunities for enhancement; it then diagnoses the discovered issues and prioritizes them to plan the next round of improvement. Throughout this process, we maintain an evolving checklist that continually accumulates newly identified issues and actionable guidance for subsequent testing and revision. By expanding as new evidence is discovered, this checklist persistently pushes the development process beyond the limited test cases and toward more generalizable improvements in overall game quality.

To this end, we introduce RSIGAME (Figure 2), an autonomous agentic game development framework with recursive self-improvement. At its core, RSIGAME organizes development into complementary local and global loops. The local loop instantiates our explore-diagnose-improve process, repeatedly exploring the executable game, diagnosing and prioritizing discovered issues, and revising the project based on accumulated evidence. The global loop monitors development across iterations, tracks overall game quality, preserves the best checkpoint, and detects saturation or regression, enabling the system to maintain long-horizon progress rather than blindly continue editing. Beyond training-free self-improvement, RSIGAME further introduces training-based experience internalization, converting successful development trajectories, diagnostic decisions, and verified revisions into supervision for the generator. Such a way enables a broader RSI loop to boost the underlying generator. Across 140 GameCraft-Bench tasks, two engines, and five generators, RSIGAME consistently improves games under matched development budgets. With iterative development, Qwen3.8-27B reaches 47.77 on Godot versus Kimi-K2.6’s 44.77 (Moonshot AI, 2026); experience internalization lifts it to 61.38 versus Codex+GPT-5.5’s 50.26 one-shot score (OpenAI, 2025; 2026), while cutting generation tokens by 11 times. On Phaser, it rises to 50.24, past the 49.44 one-shot score of GPT-5.5.

## 2 PRELIMINARIES

Agentic Game Development. It aims to automate game creation with intelligent agents. Given a natural-language game specification, an agent translates user intent into a complete and playable game. Similar to human game development, agentic game development must satisfy two objectives: following user requirements and producing a functionally correct, bug-free game. Formally, given a game specification x, we aim to learn a generator policy $\pi _ { \theta }$ that produces a game project

$$
P _ { 0 } \sim \pi _ { \theta } ( \cdot \mid x ) ,\tag{1}
$$

where $P _ { 0 }$ should both conform to the intended game design and execute correctly. Such a project consists of source code, visual and textual assets, and configuration files. Its construction therefore requires joint generation and integration of coding, text, and visual elements into a coherent interactive system. Importantly, game quality is ultimately reflected in runtime behavior and player experience. Each generated game includes a set of replayable demonstration traces $\mathcal { T } = \{ \tau _ { 1 } , \tau _ { 2 } , \dots , \tau _ { N } \}$ which are replayed during evaluation and scored against a hidden task-specific rubric.

![](images/61e9f25f841001c3dcc080060fef15048d04f67efe268c23b842033b15062575.jpg)  
Figure 2: Overview of RSIGAME. The local loop autonomously evolves the game through direction decision, active exploration, evidence-grounded editing, and agentic verification. The global loop monitors global quality, preserves the best checkpoint, and introduces sparse high-level guidance when progress saturates, enabling progressive multi-stage game evolution.

Our Focus. Because game generation involves many tightly coupled components, producing a fully correct project in one pass is difficult. Existing workflows therefore commonly adopt an iteration-based strategy: $P _ { 0 }  P _ { 1 }  \cdot \cdot \cdot  P _ { T }$ . At each iteration, the current project is executed and tested, observed failures are diagnosed, and the project is revised accordingly. In conventional game development, this loop relies on human developers and playtesters. In this work, we bring the same principle into agentic game development, but replace human-driven testing and revision with afully autonomous process that recursively tests, diagnoses, and improves its own generated game.

## 3 RSIGAME

Our RSIGAME incorporates both training-free and training-based strategies. The training-free strategy builds on local and global loops for iterative development, while the training-based strategy further internalizes accumulated development experience into the backbone game generation model.

## 3.1 TRAINING-FREE RECURSIVE SELF-IMPROVEMENT

Our training-free RSI strategy consists of a local loop that follows an iterative explore-diagnoseimprove process to identify issues and refine the game, and a global loop that monitors long-horizon progress, preserves the best, and detects saturation.

## 3.1.1 LOCAL LOOP: EXPLORE-DIAGNOSE-IMPROVE

The local loop performs fine-grained recursive game improvement through three stages: autonomous exploration, issue diagnosis, and iterative improvement. The loop continuously explores the executable game, accumulates newly discovered issues and improvement opportunities in an evolving checklist, and uses them to guide subsequent revisions.

Controller and Explorer for Autonomous Exploration. The autonomous exploration stage involves two agents with complementary roles. The controller is responsible for deciding what aspect of the game should be explored next, while the explorer interacts with the executable game to collect concrete behavioral evidence. Specifically, given the game specification $x ,$ the current project $P _ { t }$ the development checklist $\mathcal { C } _ { t } ,$ , and optional stage-level guidance $\gamma _ { s } ,$ the controller proposes a development direction $d _ { t } .$ Conditioned on this direction, the explorer performs targeted interaction with the game and produces an exploration trajectory τ<sub>t</sub>:

$$
d _ { t } \sim \pi _ { \mathrm { c o n t r o l l e r } } ( \cdot \mid x , P _ { t } , \mathcal { C } _ { t } , \gamma _ { s } ) , \qquad \tau _ { t } \sim \pi _ { \mathrm { e x p l o r e r } } ( \cdot \mid P _ { t } , d _ { t } ) .\tag{2}
$$

Through this process, the controller provides high-level exploration intent, while the explorer grounds it in observed gameplay, uncovering concrete bugs, missing behaviors, and potential improvement opportunities for subsequent diagnosis and revision.

Editor and Verifier for Issue Diagnosis. The issue diagnosis stage involves two agents with complementary responsibilities. The editor converts the discovered issues and interaction evidence into concrete game modifications, while the verifier evaluates whether these modifications resolve the intended problems without introducing regressions. Specifically, conditioned on the current project, development direction, and observed playtest trajectory, the editor proposes an edit

$$
\Delta _ { t } \sim \pi _ { \mathrm { e d i t o r } } ( \cdot \mid P _ { t } , d _ { t } , \tau _ { t } ) , \qquad P _ { t + 1 } = \mathrm { A p p l y } ( P _ { t } , \Delta _ { t } ) .\tag{3}
$$

The verifier then interacts with the updated project and produces a verification outcome

$$
v _ { t } \sim \pi _ { \mathrm { v e r i f i e r } } ( \cdot \mid P _ { t + 1 } , d _ { t } , \tau _ { t } ) ,\tag{4}
$$

which determines whether the targeted issue has been successfully addressed and whether the edit causes unintended side effects. The verified outcome is subsequently written back to the evolving checklist, providing updated evidence for the controller to plan the next development round.

Iterative Improvement with Checklist Update. To connect exploration, diagnosis, and revision across iterations, RSIGAME maintains a shared development checklist $\mathcal { C } _ { t }$ as an evolving working state. The checklist records discovered issues, improvement opportunities, priorities, and verification outcomes, allowing all agents to operate on a consistent view of the current development status. At each round, the controller reads $\mathcal { C } _ { t }$ to determine the next development direction $d _ { t } ,$ the explorer augments it with newly observed evidence, the editor addresses prioritized items, and the verifier updates their status according to the resulting gameplay behavior. These interactions produce the next-round checklist

$$
\mathcal { C } _ { t + 1 } = U ( \mathcal { C } _ { t } , d _ { t } , \tau _ { t } , \Delta _ { t } , v _ { t } ) ,\tag{5}
$$

which is passed to the controller together with $P _ { t + 1 }$ . In this way, RSIGAME continually accumulates development knowledge and ensures that each iteration builds on verified progress rather than restarting from scratch.

## 3.1.2 GLOBAL LOOP: PROGRESS MONITORING AND CONTROL

While the local loop focuses on improving the game within each development round, the global loop governs the overall development process across stages. It consists of two components: the game quality monitor that tracks accumulated game quality and preserves the strongest checkpoint, and the progress and convergence control mechanism that determines whether development should continue, terminate, or enter a new stage under additional high-level guidance.

Game Quality Monitor. The game quality monitor tracks game-level progress across local development rounds. A stage begins from the checkpoint the previous stage retained, $P _ { s } ^ { \star } \gets P _ { s - 1 } ^ { \star } .$ . At each global evaluation point, the monitor compares the checkpoint just produced against the retained one and updates

$$
P _ { s } ^ { \star } \gets \mathrm { S e l e c t B e s t } \left( P _ { s } ^ { \star } , P _ { t + 1 } \right) .\tag{6}
$$

Unlike the local Verifier, which assesses whether a specific edit resolves its targeted issue, the game quality monitor evaluates the accumulated quality of the game as a whole and maintains a persistent best state throughout development. Importantly, this monitoring process is strictly isolated from the benchmark evaluator, including its scores, rubrics, and feedback. Details are in Appendix C.2.

Progress and Convergence Control. The evolution of the retained checkpoint $P _ { s } ^ { \star }$ provides a natural signal for controlling long-horizon development. If successive global evaluations fail to produce a better checkpoint, RSIGAME treats the persistent lack of progress as convergence. The system then either terminates and returns $P _ { s } ^ { \star }$ as the final game, or optionally introduces new high level guidance $\gamma _ { s + 1 }$ from a human or stronger model. Such guidance provides strategic directions or creative suggestions rather than explicit edit instructions. The next development stage therefore starts from the retained best checkpoint $P _ { s } ^ { \star }$ under $\gamma _ { s + 1 }$ , allowing the local loop to explore new improvement directions without discarding previously verified gains. Repeating this process enables RSIGAME to control long-horizon progress while preserving the strongest solution reached so far.

## 3.2 TRAINING-BASED KNOWLEDGE INTERNALIZATION

During training-free RSI, RSIGAME also accumulates reusable experience about how games are planned, implemented, tested, and refined. Thus, we further internalize successful RSI knowledge into the backbone model, extending self-improvement from context optimization to parameter optimization. This broader RSI loop improves the underlying generator, enabling stronger initial generation and reducing the number of refinement iterations required for high-quality game development.

Development Experience Collection and Curation. We collect three types of supervision: generation traces from GPT-5.5 (OpenAI, 2026) constructing games in Godot (Godot Engine contributors, 2026) and Phaser (Photon Storm, 2026), planning traces collected from successful generations, and improvement traces produced by running RSIGAME with GLM-5.3-Flash (Zhipu AI, 2026). We retain only executable generations and improvements that are verified through post-edit interaction without breaking previously functional behavior. In total, we obtain 2,213 generation traces, 2,108 planning traces, and 2,013 verified improvement rounds from 4,003 candidates.

Experience Internalization. We apply supervised fine-tuning to Qwen3.8-27B (Qwen Team, 2026) on the curated planning, generation, and improvement trajectories. Beyond learning from final game artifacts, the model internalizes the intermediate planning, tool-use, diagnosis, and revision decisions that drive successful development. This transfers reusable RSI knowledge into model parameters, strengthening the refinement ability and reducing the required number of iterations to reach high-quality game generation. Such a way opens the door to continual RSI, in which newly acquired development experience can be repeatedly internalized to further improve the generator over time. Training details are provided in Appendix E.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Benchmark and Engines. We evaluate on GameCraft-Bench (Luo et al., 2026), which contains 140 game-development tasks spanning 15 game families. We instantiate the same tasks in both Godot (Godot Engine contributors, 2026) and Phaser (Photon Storm, 2026), keeping the task specifications, evaluation rubrics, and development protocol fixed across engines. Phaser therefore serves as a cross-engine robustness evaluation of the findings on Godot.

Evaluation Protocol. Following GameCraft-Bench, each generated game is packaged with replayable demonstration traces provided by the game generator, which are replayed and scored against a hidden task-specific rubric over Mechanics (M), Depth (D), Visuals (V), and Art (A):

$$
\begin{array} { r } { Q = \mathrm { B U I L D } \times ( 0 . 1 5 M + 0 . 3 5 D + 0 . 1 5 V + 0 . 3 5 A ) , } \end{array}
$$

where BUILD ∈ {0, 1} assigns zero score to non-playable games. We use Qwen3.8-27B (Qwen Team, 2026) as the judge and average three independent replay-and-score runs. The hidden rubric, scores, and judge feedback are never exposed to the development agents. Evaluation stability and free-play robustness are detailed in Appendices D.3 and D.4.

Baselines and Development Budget. We compare the frozen initial project $P _ { 0 } ,$ , with Play2Code (Huang et al., 2026), and with RSIGAME across multiple generators. For each generator, all development methods start from an identical clone of $P _ { 0 }$ and use the same per-round tool budget. Both Play2Code (Huang et al., 2026) and RSIGAME are allowed up to 26 per-round tool calls and 30 development rounds. Full model configurations, row-specific budgets, cost accounting, and implementation details are provided in Appendix D.1 and Appendix D.2.

Table 1: Main results on GameCraft-Bench (top: Godot; bottom: Phaser). Methods are compared from matched base games under one backbone and budget, averaged over 140 tasks; Tok. and Cost are mean billable tokens and cost per task (Appendix D.1). Bold marks the best value per column within a group; deeper rows internalize development experience.
<table><tr><td>Method</td><td>Mechanics</td><td>Depth</td><td>Visuals</td><td>Art</td><td>Overall ↑</td><td>Tok. ↓</td><td>Cost↓</td></tr><tr><td colspan="8">Godot (Godot Engine contributors, 2026)</td></tr><tr><td colspan="8">Generator: Codex + GPT-5.5 (high)</td></tr><tr><td>Base (frozen  $P _ { 0 } )$ </td><td>58.9</td><td>51.7</td><td>51.4</td><td>44.6</td><td>50.26</td><td>0.26M</td><td></td></tr><tr><td>+ Play2Code</td><td>59.2</td><td>51.3</td><td>52.2</td><td>45.8</td><td>50.74</td><td>3.20M</td><td>$1.06</td></tr><tr><td>+ RSIGAME</td><td>72.8</td><td>60.3</td><td>65.9</td><td>64.5</td><td>64.53</td><td>3.12M</td><td>$0.88</td></tr><tr><td colspan="8">Generator: Codex + Kimi-K2.6</td></tr><tr><td>Base (frozen P0)</td><td>40.2</td><td>30.3</td><td>35.4</td><td>21.9</td><td>29.63</td><td>3.22M</td><td></td></tr><tr><td>+ Play2Code</td><td>44.8</td><td>35.4</td><td>40.2</td><td>28.6</td><td>35.19</td><td>7.07M</td><td>$1.27</td></tr><tr><td>+ RSIGAME</td><td>51.7</td><td>40.1</td><td>47.1</td><td>45.4</td><td>44.77</td><td>7.39M</td><td>$1.18</td></tr><tr><td colspan="8">Generator: Codex + GLM-5.3-Flash</td></tr><tr><td>Base (frozen P0)</td><td>35.7</td><td>30.2</td><td>32.0</td><td>24.9</td><td>29.46</td><td>0.44M</td><td></td></tr><tr><td>+ Play2Code</td><td>47.2</td><td>39.7</td><td>41.1</td><td>32.7</td><td>38.59</td><td>3.18M</td><td>$0.99</td></tr><tr><td>+ RSIGAME</td><td>51.9</td><td>44.4</td><td>49.9</td><td>53.9</td><td>49.72</td><td>5.06M</td><td>$1.25</td></tr><tr><td colspan="8">Generator: Codex + Qwen3.8-27B</td></tr><tr><td>Base (frozen P0)</td><td>41.2</td><td>33.4</td><td>38.6</td><td>38.3</td><td>37.07</td><td>6.41M</td><td></td></tr><tr><td>+ Play2Code</td><td>44.6</td><td>36.1</td><td>42.1</td><td>42.5</td><td>40.53</td><td>9.92M</td><td>$1.32</td></tr><tr><td>+ RSIGAME</td><td>51.2</td><td>40.4</td><td>47.5</td><td>53.8</td><td>47.77</td><td>9.13M</td><td>$1.09</td></tr><tr><td colspan="8">Generator: Codex + Qwen3.8-27B (SFT)</td></tr><tr><td>Base (frozen P0)</td><td>56.1</td><td>47.5</td><td>49.8</td><td>44.8</td><td>48.22</td><td>0.57M</td><td></td></tr><tr><td>+ Play2Code</td><td>57.4</td><td>48.1</td><td>51.6</td><td>47.2</td><td>49.71</td><td>3.74M</td><td>$1.08</td></tr><tr><td>+ RSIGAME</td><td>68.7</td><td>55.8</td><td>62.8</td><td>63.2</td><td>61.38</td><td>3.49M</td><td>$0.94</td></tr><tr><td colspan="8">Phaser (Photon Storm, 2026)</td></tr><tr><td colspan="8">Generator: OpenGame + GPT-5.5</td></tr><tr><td>Base (frozen P0)</td><td>49.1</td><td>38.6</td><td>53.6</td><td>58.6</td><td>49.44</td><td>5.36M</td><td></td></tr><tr><td>+ Play2Code</td><td>57.1</td><td>42.0</td><td>60.0</td><td>65.2</td><td>55.10</td><td>8.34M</td><td>$1.43</td></tr><tr><td>+ RSIGAME</td><td>59.8</td><td>44.2</td><td>62.4</td><td>69.7</td><td>58.21</td><td>8.38M</td><td>$1.44</td></tr><tr><td colspan="8">Generator: OpenGame + Qwen3.8-27B</td></tr><tr><td>Base (frozen P0)</td><td>40.6</td><td>29.5</td><td>43.7</td><td>48.7</td><td>40.03</td><td>9.03M</td><td></td></tr><tr><td>+ Play2Code</td><td>51.0</td><td>36.4</td><td>53.2</td><td>58.0</td><td>48.70</td><td>12.99M</td><td>$1.75</td></tr><tr><td>+ RSIGAME</td><td>52.0</td><td>36.5</td><td>54.6</td><td>61.3</td><td>50.24</td><td>12.78M</td><td>$1.50</td></tr><tr><td colspan="8">Generator: OpenGame + Qwen3.8-27B (SFT)</td></tr><tr><td>Base (frozen  $P _ { 0 } )$ </td><td>47.1</td><td>35.4</td><td>48.9</td><td>52.4</td><td>45.14</td><td>5.42M</td><td></td></tr><tr><td>+ Play2Code</td><td>55.6</td><td>41.7</td><td>57.3</td><td>61.8</td><td>53.15</td><td>9.58M</td><td>$1.61</td></tr><tr><td>+ RSIGAME</td><td>60.4</td><td>46.2</td><td>62.3</td><td>68.4</td><td>58.53</td><td>9.31M</td><td>$1.53</td></tr></table>

## 4.2 GODOT GAME RESULTS

Table 1 reveals three main findings. First, RSIGAME delivers large and consistent gains across generators. Across all five settings, it improves the frozen initial games by 10.7–20.3 Overall points, with gains consistently spanning Mechanics, Depth, Visuals, and Art. Second, under the same compute budget, RSIGAME performs significantly better than other recursive-based methods. Under matched development budgets, RSIGAME outperforms Play2Code by 7.2–13.8 Overall points. Play2Code brings almost no improvement to the strong Codex+GPT-5.5 initialization and Qwen3.8-27B variants, whereas RSIGAME substantially improves the same frozen games. This contrast shows that the benefit comes from how development is organized, rather than from iteration alone. Finally, experience internalization further improves both quality and efficiency. Fine-tuning raises Qwen3.8-27B’s one-shot score from 37.07 to 48.22, within 2.0 points of Codex+GPT-5.5, while reducing generation tokens by 11 times (6.41M to 0.57M). With RSIGAME, the score further rises to 61.38 and total token usage decreases by 2.6 times (9.13M to 3.49M).

![](images/6be1d4e72be29b5c2f3c861c1567cca209ec4993be20af4a4b2eb53fb6f672b2.jpg)

![](images/26b70f30e0b725315801a2465f3422d071b9c563f7319b48918b09b5bd47eda1.jpg)

![](images/4dc793fe7518162c7d12908993fbe1933bf82d7ebbfdbeb7d2f94c4ed86d874d.jpg)

![](images/fee1091ff687167108e1e0e7fd5bb6a5424500895f0353c049aae4d766f8b85c.jpg)

Figure 3: Development-time scaling on GameCraft-Bench (40 tasks). Across Godot and Phaser with strong (GPT-5.5) and weak (Qwen3.8-27B) initial generators, iterative development can plateau or regress, while RSIGAME achieves sustained improvement across development budgets. The Global Quality Monitor preserves the best checkpoint reached so far, improving the quality–cost tradeoff over returning the last checkpoint. Shaded regions denote ±1 standard error.  
![](images/a403623d8bb6863749a7ea8c6c8a05a0154240a49d274fcc202b0af0759785ec.jpg)

![](images/b2c22700fc79ff2601f4cdb99370b9d06704a9c934ecac124650fad7375737d6.jpg)  
Figure 4: Global quality control and experience transfer. (a) Best-checkpoint tracking protects earlier gains from later regressions. (b) Saturation-aware stopping achieves comparable quality with fewer development rounds. (c) Internalizing verified development experience improves one-shot generation across all quality dimensions.

## 4.3 PHASER GAME RESULTS

The Phaser block of Table 1 shows that the gains of RSIGAME transfer across engines, improving all three generators by 8.8–13.4 Overall points over their frozen bases and achieving the strongest final quality in every setting. Interestingly, Play2Code is considerably stronger on Phaser than on Godot. The category breakdown reveals why: Phaser initializations tend to have weaker Mechanic and Depth but stronger Visuals and Art, yielding similar Overall scores while leaving more readily improvable functional headroom. Play2Code can therefore simply find these obvious deficiencies and recover substantial quality, narrowing its gap to RSIGAME. Nevertheless, RSIGAME still consistently produces the best final games, indicating that its advantage extends beyond low-level improvement even when such improvement already captures much of the available headroom.

## 5 FURTHER ANALYSIS

Can Global Quality Monitoring Improve Development-Time Scaling? Figure 3 reveals a clear difference in how development methods scale with additional compute. Simply extending the budget does not guarantee better games: Play2Code quickly plateaus and often regresses as more rounds are added. In contrast, RSIGAME consistently converts additional development rounds into higher game quality across both Godot and Phaser and for both strong and weak initializations. The local loop alone already produces substantial gains, but its trajectory remains volatile; the Global Quality Monitor stabilizes this process by retaining the best state reached so far, yielding sustained improvement throughout the development budget.

![](images/419e1e895db9e5ee49db7b19344991ed0bf802b1e20fca185e0fd004dcc2e0da.jpg)

![](images/ed08420a30fcd2456c8a8ab9cf0bd75f51511305e1ace642c85ccf5d960c9761.jpg)

![](images/1c341b97fea11e26fdf957b6c4965f54e9979e14d392950397280dfd2463d64e.jpg)  
Figure 5: Adaptive development follows the game state and improves final quality. (a, b) Adaptive development reallocates effort according to the current bottleneck, shifting toward improvement after build failures and toward art when visual quality lags. (c, d) Blind pairwise evaluation consistently favors adaptive development over round-robin scheduling.

<table><tr><td>Method</td><td>Grounded Prec. ↑</td><td>Ungrounded / Round↓</td></tr><tr><td>Free critic</td><td>58.6%</td><td>1.93</td></tr><tr><td>Agentic verification</td><td>72.3%</td><td>0.50</td></tr></table>

(a) Pre-improvement target grounding, 40 sessions.

<table><tr><td>Method</td><td>Failure Recall ↑</td><td>Spec. ↑</td><td>Bal. Acc. ↑</td></tr><tr><td>Build only</td><td>0.0%</td><td>100.0%</td><td>50.0%</td></tr><tr><td>Replay + verification</td><td>76.2%</td><td>92.6%</td><td>84.4%</td></tr></table>

(b) Post-improvement failure detection, 48 adjudicated rounds.  
Table 2: Agentic verification improves both target grounding and post-edit reliability. Before editing, it filters unsupported improvement targets; after editing, replay-based verification detects failed changes and regressions that compilation alone cannot reveal.

Figure 4(a)(b) further isolates the benefit of this global control. The retained checkpoint closely tracks the oracle best one within each budget, preventing later edits from erasing earlier gains, while saturation-aware stopping achieves comparable quality to much longer fixed-budget runs with fewer rounds. Together, these results show that the monitor makes development-time scaling both more reliable and more compute-efficient. Detailed stopping criteria are provided in Appendix C.2.

Does Adaptive Evolution Matter? Figure 5 examines whether the local loop benefits from choosing its own development focus adaptively rather than following a fixed round-robin schedule. Adaptive Evolution puts its effort to the current game state: after a build failure it shifts strongly toward improvement, while once visual quality becomes the bottleneck it allocates substantially more rounds to art (Figure 5(a) (b)). This adaptive behavior also leads to better final games: blind pairwise evaluation prefers adaptive development on 59% of comparisons versus 37% for round-robin, with consistent advantages across all four quality dimensions (Figure 5(c)(d)). These results show that long-horizon development benefits from letting the local loop respond to the evolving needs of the game rather than following a fixed schedule. Full evaluation details are provided in Appendix D.6.

Does Agentic Verification Make Local Improvement More Reliable? Table 2 evaluates verification both before and after editing. Evidence-grounded pre-improvement verification raises grounded precision from 58.6% to 72.3% and reduces unsupported targets from 1.93 to 0.50 per round. After editing, replay-based verification detects 76.2% of unsuccessful improvements and reaches 84.4% balanced accuracy, whereas build-only checking detects none of these behavioral failures. Together, these results show that agentic verification improves both the quality of development decisions and the reliability of the feedback returned to subsequent rounds. The audit protocol and metric definitions are given in Appendix D.7.

Can Development Experience Transfer to Future Generation? Figure 4(c) shows that development experience transfers back to the generation model. Training on generation traces alone yields uneven gains across quality dimensions, whereas incorporating planning and verified improvement experience produces consistent improvements in Mechanics, Depth, Visuals, and Art. The gains are particularly pronounced in Mechanics +14.9 and Depth +14.1, leading to a +11.1-point Overall improvement before any test-time development. This suggests that planning and improvement trajectories provide reusable knowledge beyond simply imitating successful game generations.

## 6 RELATED WORK

Game Generation and Development Benchmarks. Recent benchmarks increasingly evaluate agents on complete, executable games rather than isolated code snippets. OpenGame-Bench (Jiang et al., 2026) evaluates web-based games through build validity, visual usability, and instruction alignment; GameCraft-Bench (Luo et al., 2026) extends evaluation to engine-based games with interaction-grounded measures of mechanics, content, visuals, and art; and GameDevBench (Chi et al., 2026) studies multimodal game development over larger codebases and assets. Related work further examines interactions in game-playing agents and broader web- or code-based game generation, through both benchmarks and generation systems (Lin et al., 2026; Hu et al., 2024; Zhang et al., 2025; 2026; Ma et al., 2026). Together, these efforts shift evaluation from static code correctness toward the quality of executable, interactive game artifacts. However, these works score a single submitted build, leaving open how an agent should keep improving a game across rounds of development, which is the setting we study.

Iterative Agentic Game Development. Recent systems extend game generation into iterative development. Play2Code (Huang et al., 2026) alternates between gameplay and code revision, using interaction feedback from each build to guide subsequent edits, while VibeGame (Hu et al., 2026) coordinates multiple specialized agents for continued game development. OpenGame (Jiang et al., 2026) further accumulates reusable development skills across projects to support subsequent generation. These efforts follow broader agentic coding paradigms based on repeated reasoning, execution, and feedback (Yang et al., 2024; Hong et al., 2024; Yao et al., 2023; Shinn et al., 2023). Rather than introducing iteration itself, RSIGAME focuses on how iterative development is sustained reliably over long horizons. It combines adaptive, evidence-grounded local improvement with global bestcheckpoint tracking and saturation detection. High-level guidance can further reopen development after autonomous progress stalls, enabling multi-stage game evolution.

Verification and Learning from Development Experience. Agentic verification has been studied through general LLM-as-judge protocols (Zheng et al., 2023), reliable process control method in agent loop (Wu et al., 2026), and game-specific methods that validate behavior through runtime interaction (Jia et al., 2026; Huang et al., 2026). In RSIGAME, independent verification closes each local development round: the updated game is replayed to confirm the intended improvement and detect new regressions, with verified outcomes written back into the shared development state. Beyond individual tasks, prior work learns from feedback, memory, or verified trajectories (Shinn et al., 2023; Madaan et al., 2023; Chen et al., 2024; Wang et al., 2025; Pan et al., 2026; Zelikman et al., 2022; Zhou et al., 2026), while game systems reuse experience across generations (Jiang et al., 2026; Hu et al., 2026). RSIGAME extends this idea by jointly internalizing planning, generation, and verified improvement experience to strengthen future game generation and development.

## 7 CONCLUSION

We presented RSIGAME, an autonomous agentic game development framework that formulates game creation as a recursive self-improvement process. RSIGAME strengthens the development loop with broad exploration, structured diagnosis, prioritized improvement, and an evolving checklist that continuously expands the set of identified issues and improvement opportunities. Its local explore-diagnose-improve loop enables evidence-grounded revision of the executable game, while the global loop tracks long-horizon progress, preserves the best checkpoint, and detects saturation or regression. We further introduced training-based experience internalization to transfer successful development trajectories back into the backbone game generation model, improving future game creation beyond test-time refinement alone. Experiments across 140 GameCraft-Bench tasks, two game engines, and five generators demonstrated that RSIGAME consistently improves game quality under matched development budgets, enabling smaller open models to reach or even surpass substantially stronger generators after iterative development.

## AI USE STATEMENT

Large language models are integral to the research methodology of this work. GPT-5.5 is used for game generation and planning-data collection; GLM-5.3-Flash is used for game improvement, verification, and improvement-data collection; and Qwen3.8-27B serves as the gameplay test agent, the benchmark evaluation judge, and the base model for experience internalization. Claude Opus 5 is additionally used for the independent verification audit.

LLM-based assistants were also used during research and manuscript preparation for language editing, presentation refinement, and programming assistance. The authors designed the experiments, reviewed the resulting artifacts and measurements, and verified all reported results and scientific claims. The authors take full responsibility for the content of the paper.

## ETHICS STATEMENT

Our study focuses on general-purpose game-generation and game-development tasks. We apply safety constraints during generation and exclude artifacts containing unsafe or sensitive content from evaluation and data curation. The experiments do not use private user data, personal information, or user-generated content.

Human involvement is limited to game evaluation and high-level development guidance. We study the resulting game assessments and guidance rather than participant attributes, and collect no sensitive personal information. Released datasets and artifacts are screened to remove machine-specific paths, credentials, and personal information.

## REPRODUCIBILITY STATEMENT

All reported results are traceable to retained experimental artifacts and documented protocols. Table 3 provides a single index to the released artifacts and to the appendix sections specifying each component of the experimental pipeline.

Controlled comparisons. For each comparison, the initial project $P _ { 0 }$ is frozen and every development method receives an identical clone, the same improvement backbone, and the same per-round tool budget. Row-specific development budgets are reported explicitly in Appendix D.1. Each resulting game is independently replayed and scored three times, while benchmark rubrics, scores, and judge outputs remain hidden from all development agents. Replay and judge variability are quantified in Appendix D.3.

Artifact availability. The full implementation, prompts, and evaluation pipeline are available in our code repository<sup>1</sup>, and representative evolved games are playable from the project page<sup>2</sup>. Frozen base projects, training trajectories, and per-round run trees are undergoing internal review and will be released upon approval. We also release on Hugging Face<sup>3</sup> the complete evaluation artifacts underlying the reported results, including 51,644 scoring files containing per-task scores, judge outputs, and replay reports. Their release status and corresponding protocol documentation are summarized in Table 3.

## REFERENCES

Anthropic. Claude Opus 5 system card. https://www.anthropic.com/ claude-opus-5-system-card, July 2026.

Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. In International Conference on Learning Representations (ICLR), 2024. arXiv:2304.05128.

Wayne Chi, Yixiong Fang, Arnav Yayavaram, Siddharth Yayavaram, Seth Karten, Qiuhong Anna Wei, Runkun Chen, Alexander Wang, Valerie Chen, Ameet Talwalkar, and Chris Donahue. Gamedevbench: Evaluating agentic capabilities through game development. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. arXiv:2602.11103.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2305.14314.

Godot Engine contributors. Godot engine. https://godotengine.org, 2026.

Sirui Hong, Mingchen Zhuge, Jiaqi Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. Metagpt: Meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations (ICLR), 2024. arXiv:2308.00352.

Chengpeng Hu, Yunlong Zhao, and Jialin Liu. Game generation via large language models. In 2024 IEEE Conference on Games (CoG), 2024.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022. arXiv:2106.09685.

Wenbo Hu, Ken Li, Jiazhe Wei, Yukang Cao, Weiyi Hong, Jiayi Dai, Chenjun Bai, Jiajun Liang, Yucheng Liao, Ruichuan An, Zeyu Lou, Haofan Wang, Yueming Lyu, Ziwei Liu, and Chenyang Si. Vibegame: Prompt-to-game development with ai-native engine and self-evolving adversarial agent team. Technical report, 2026. https://vibegame.tettet.org/technical\_ report.pdf.

Yixu Huang, Bo Li, Na Li, Zhe Wang, Kaijie Chen, Haonan Ge, Qingyi Si, Yuanzhe Shen, Ruihan Yang, Guangjing Wang, and Hongcheng Guo. Gui agents for continual game generation, 2026.

Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models, 2023.

Chaobo Jia, Ruipeng Wan, Ting Sun, Weihao Tan, Borui Wan, Yuxuan Tong, Guangming Sheng, and Hong Xu. Gamegen-verifier: Parallel keypoint-based verification for llm-generated games via runtime state injection, 2026.

Yilei Jiang, Jinyuan Hu, Qianyin Xiao, Yaozhi Zheng, Ruize Ma, Kaituo Feng, Jiaming Han, Tianshuo Peng, Kaixuan Fan, Manyuan Zhang, and Xiangyu Yue. Opengame: Open agentic coding for games, 2026.

Mingxian Lin, Shengju Qian, Yuqi Liu, Yi-Hua Huang, Yiyu Wang, Wei Huang, Yitang Li, Fan Zhang, Zeyu Hu, Lingting Zhu, Xin Wang, and Xiaojuan Qi. Omnigamearena: A unified ue5 benchmark for vlm game agents with improvement dynamics, 2026.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019. arXiv:1711.05101.

Tongxu Luo, Rongsheng Wang, Jiaxi Bi, Chenming Xu, Zhengyang Tang, Jianlong Chen, Juhao Liang, Ke Ji, Shuqi Guo, Yuhao Du, Fan Bu, Wenyu Du, Xiaotong Zhang, Kyle Li, Shaobo Wang, Linfeng Zhang, Yuxuan Liu, Xin Lai, Chenxin Li, Yiduo Guo, Zhexin Zhang, Xinyuan Wang, Tianyi Bai, Ziniu Li, and Benyou Wang. Gamecraft-bench: Can agents build playable games end-to-end in a real game engine?, 2026.

Hongnan Ma, Han Wang, Shenglin Wang, Tieyue Yin, Yiwei Shi, Yucong Huang, Yingtian Zou, Muning Wen, and Mengyue Yang. Creativegame: Toward mechanic-aware creative game generation, 2026.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2303.17651.

Moonshot AI. Kimi-K2.6. Model card, Hugging Face, April 2026. https://huggingface. co/moonshotai/Kimi-K2.6.

OpenAI. Introducing Codex. https://openai.com/index/introducing-codex/, May 2025.

OpenAI. GPT-5.5 system card. https://openai.com/index/ gpt-5-5-system-card/, April 2026.

Wenbo Pan, Shujie Liu, Xiangyang Zhou, Shiwei Zhang, Wanlu Shi, Mirror Xu, and Xiaohua Jia. M<sup>⋆</sup>: Every task deserves its own memory harness, 2026.

Photon Storm. Phaser: A fast, free and fun open source html5 game framework. https:// phaser.io, 2026.

Qwen Team. Qwen3.8-27B. Model card, Hugging Face, August 2026. https:// huggingface.co/Qwen/Qwen3.8-27B.

Alejandro Roque, Juan P. Sotomayor, Dionny Santiago, and Peter J. Clarke. A literature review of software testing practices and frameworks in the video gaming industry. Software Testing, Verification and Reliability, 35(2):e70001, 2025. doi: 10.1002/stvr.70001.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2303.11366.

Yinjie Wang, Ling Yang, Ye Tian, Ke Shen, and Mengdi Wang. Co-evolving llm coder and unit tester via reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), 2025. Spotlight. arXiv:2506.03136.

Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, and Sainbayar Sukhbaatar. Meta-rewarding language models: Self-improving alignment with LLM-as-a-meta-judge, 2024. https://arxiv.org/abs/2407.19594.

Wenyi Wu, Sibo Zhu, Kun Zhou, Aayush Salvi, Zixuan Song, and Biwei Huang. Structagent: Harness long-horizon digital agents with unified causal structure, 2026. https://arxiv. org/abs/2607.11388.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2405.15793.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023. arXiv:2210.03629.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. Star: Bootstrapping reasoning with reasoning. In Advances in Neural Information Processing Systems (NeurIPS), 2022. arXiv:2203.14465.

Wei Zhang, Jack Yang, Renshuai Tao, Lingzheng Chai, Shawn Guo, Jiajun Wu, Xiaoming Chen, Ganqu Cui, Ning Ding, Xander Xu, Hu Wei, and Bowen Zhou. V-gamegym: Visual game generation for code large language models, 2025.

Wenyu Zhang, Guoliang You, Tianlun, Haotian Zhao, Tianshu Zhu, Haoran Wang, Xiaoxuan Tang, Mingyang Dai, Jingnan Gu, Daxiang Dong, and Jianmin Wu. Webgamebench: Requirement-toapplication evaluation for coding agents via browser-native games, 2026.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023. arXiv:2306.05685.

Zhipu AI. GLM-5.3-Flash. Model card, Hugging Face, August 2026. https://huggingface. co/zai-org/GLM-5.3-Flash.

Chenyu Zhou, Qiliang Jiang, Shuning Wu, and Xu Zhou. The verifier is the curriculum: Precision sets the return on search in code self-distillation, 2026.

Sibo Zhu, Shicheng Fan, Xinyue Wang, Wenyi Wu, Kun Zhou, and Biwei Huang. Rsiagent: Autonomous exploration for recursive self-improvement in new environments, 2026. https: //arxiv.org/abs/2609.15364.

This appendix provides implementation details, experimental protocols, training specifications, and extended   
analyses supporting the main paper.   
A Reproducibility Index 15   
Artifacts, release status, and where each protocol detail is specified.   
B Limitations 15   
Scope of the autonomous pipeline, the dimensions evaluated, and project size.   
C RSIGAME System Design and Implementation 16   
Complete specification of the recursive development system and its agent interfaces.   
C.1 Recursive Development Procedure 16   
C.2 Global Quality Monitoring 16   
C.3 High-Level Director Interface 17   
C.4 Agent Prompts and Output Contracts 20   
D Experimental Protocols and Evaluation Reliability 24   
Configuration, accounting, and reliability protocols underlying the reported experiments.   
D.1 Experimental Configuration 24   
D.2 Token and Cost Accounting 24   
D.3 Replay and Scoring Stability 25   
D.4 Free-Play Evaluation 28   
D.5 Statistical Reliability of Main Results 28   
D.6 Adaptive Direction Policy Evaluation 29   
D.7 Agentic Verification Audit 30   
D.8 Multi-Stage Guidance Evaluation 31   
E Training Data and Experience Internalization 31   
Construction, curation, and internalization of game-development experience.   
E.1 Training Corpus Construction and Curation 31   
E.2 Fine-Tuning Configuration 33   
E.3 Training-Data Ablation 33   
F Extended Evaluation and Case Studies 34   
Extended comparisons, qualitative trajectories, and task-family breakdowns.   
F.1 Comparison with a Multi-Agent Generate-and-Verify System 34   
F.2 Qualitative Evolution Case Studies 35   
F.3 Per-Family Results 43

## A REPRODUCIBILITY INDEX

Table 3 is the single index referred to by the Reproducibility Statement: for each artifact it gives what the artifact contains, whether it is released, and where the corresponding protocol or implementation detail is specified.

Table 3: Reproducibility index. Artifacts, release status, and locations of the corresponding implementation and protocol details.
<table><tr><td>Item</td><td>Contents / specification</td><td>Location</td></tr><tr><td colspan="3">Released artifacts</td></tr><tr><td>Implementation</td><td>Local/global loops, Global Quality Monitor, director-review tool, and scoring harness</td><td>Code, live</td></tr><tr><td>Prompts and output con- tracts</td><td>Complete prompts and structured outputs used by all agents</td><td>Code, live; App. C.4</td></tr><tr><td>Playable evolution</td><td>Six representative games with retained checkpoints</td><td>Project page, live</td></tr><tr><td>Frozen base projects P0 Evaluation artifacts</td><td>Starting project for each task and generator Per-task rubric scores, judge outputs, and replay re-</td><td>On release HF, live†</td></tr><tr><td></td><td>ports (51,644 files)</td><td></td></tr><tr><td>Training corpus</td><td>2,213 generation trajectories, 2,108 plans, and 2,013 verified improvement rounds</td><td>On release</td></tr><tr><td>Run trees</td><td>Per-round project snapshots for long-horizon de- velopment runs</td><td>On release</td></tr><tr><td colspan="3">Protocol and implementation details</td></tr><tr><td>Recursive development loop</td><td>One local round, global checkpoint update, and</td><td>App. C.1</td></tr><tr><td>Global Quality Monitor</td><td>stopping logic Best-checkpoint selection, saturation criterion, and patience K</td><td>App. C.2</td></tr><tr><td>Model assignment</td><td>Models used for generation, exploration, editing, verification, and judging</td><td>Tab.6</td></tr><tr><td>Hyperparameters</td><td>Loop, monitor, and scoring parameters</td><td>Tab.7</td></tr><tr><td>Development budgets</td><td>Rounds and tool-call budgets for each reported row</td><td>App. D.1</td></tr><tr><td>Token and cost accounting</td><td>Billable tokens, excluded costs, and pricing</td><td>App. D.2</td></tr><tr><td>Scoring stability</td><td>Replay and judge variability</td><td>App. D.3</td></tr><tr><td>Corpus construction</td><td>Extraction, verification filtering, and contamina- tion checks</td><td>App. E.1</td></tr><tr><td>Fine-tuning</td><td>Training mixture and hyperparameters</td><td>App. E.2</td></tr><tr><td>Director review</td><td>Director inputs, interaction budget, and guidance schema</td><td>App. C.3</td></tr><tr><td>Per-family results</td><td>Main-table results broken down by task family</td><td>App. F.3</td></tr></table>

<sup>†</sup> Released on Hugging Face: RSIGame/RSIGame-TableArtifacts. Machine-specific paths and identifying terms are removed before release, and the exported files are re-parsed and re-scanned.

## B LIMITATIONS

The core RSIGAME pipeline is fully autonomous: adaptive local development, global quality monitoring, best-checkpoint tracking, and saturation-aware stopping require no human intervention. We additionally study an optional directed extension in which sparse high-level guidance from a human or a stronger model is introduced only after autonomous improvement saturates. Our current study evaluates this mechanism on a limited set of multi-stage development cases; a broader investigation of when guidance should be invoked, how much guidance is beneficial, and how human and model guidance differ remains future work.

Our evaluation is also centered on the four dimensions of Mechanics, Depth, Visuals, and Art. These dimensions do not fully capture higher-level properties such as originality, narrative quality, longterm player engagement, or subjective enjoyment.

Finally, our experiments focus on relatively compact games that can be iteratively developed within practical agent budgets. Extending autonomous recursive development to substantially larger projects with longer horizons, richer assets, and more complex cross-system dependencies remains an important direction.

## C RSIGAME SYSTEM DESIGN AND IMPLEMENTATION

## C.1 RECURSIVE DEVELOPMENT PROCEDURE

Algorithm 1 is the loop of Section 3 written out: one round of the inner loop, the checkpoint at which the Global Quality Monitor runs, and the two things saturation can lead to – returning the champion, or taking a piece of guidance and opening the next stage.

The multi-stage structure is the counter c and the stage index s. A checkpoint either promotes a new champion, which sets c back to zero, or finds nothing better, which advances it; K checkpoints in a row without a promotion is what the paper calls saturation. Saturation does not by itself end development: it ends the stage. If guidance is available, a brief is written on the champion, the next stage restarts from that champion rather than from the last round’s build, c is reset, and the loop continues with $\gamma _ { s + 1 }$ steering every subsequent round. Development ends only when a stage saturates and no further guidance is given. The runs in this paper use $E = 3$ and $K = 3 .$

Algorithm 1 RSIGAME Recursive Game Development   
Require: Game specification $x ,$ initial project $P _ { 0 } ,$ initial checklist $\mathcal { C } _ { 0 } .$ , checkpoint interval $E ,$ pa  
tience K   
1: $P _ { 0 } ^ { \star }  P _ { 0 } , \gamma _ { 0 }  \infty , t  0 , s  0 , c  0$ ▷ champion, guidance, round, stage, idle   
checkpoints   
2: while development budget remains do   
3: $d _ { t } \sim \pi _ { \mathrm { c o n t r o l l e r } } ( \cdot \mid x , P _ { t } , \mathcal { C } _ { t } , \gamma _ { s } )$ ▷ the checklist and guidance steer the round   
4: $\tau _ { t } \sim \pi _ { \mathrm { e x p l o r e r } } ( \cdot \mid P _ { t } , d _ { t } )$   
5: $\Delta _ { t } \sim \pi _ { \mathrm { e d i t o r } } ( \cdot \mid P _ { t } , d _ { t } , \tau _ { t } ) , P _ { t + 1 }  \mathrm { A p p l y } ( P _ { t } , \Delta _ { t } )$   
6: $v _ { t } \sim \pi _ { \mathrm { v e r i f i e r } } ( \cdot \mid P _ { t + 1 } , d _ { t } , \tau _ { t } )$ ▷ is the improvement observable, and what regressed   
7: $\mathcal { C } _ { t + 1 } \gets U ( \mathcal { C } _ { t } , d _ { t } , \tau _ { t } , \Delta _ { t } , v _ { t } )$ ▷ the working memory the next round builds on   
8: if t mod $E = 0$ then ▷ a checkpoint: the Monitor compares, the loop does not   
9: $P ^ { \prime } \gets \mathrm { S e l e c t B e s t } ( P _ { s } ^ { \star } , P _ { t + 1 } )$   
10: if $P ^ { \prime } \neq P _ { s } ^ { \star }$ then   
11: $\dot { P _ { s } ^ { \star } } \gets \dot { P ^ { \prime } } ; c \gets 0$ ▷ a new champion; the stage is still paying   
12: else   
13: $c \gets c + 1$ ▷ another checkpoint with nothing better   
14: end if   
15: i ${ \bf f } c \geq K$ then ▷ saturated: K checkpoints without a new champion   
16: if no guidance is available then return $P _ { s } ^ { \star }$   
17: $\gamma _ { s + 1 } \gets \mathbf { G u l D A N C E } ( x , P _ { s } ^ { \star } )$ ▷ a brief on the champion opens the next stage   
18: $\dot { P } _ { s + 1 } ^ { \star }  P _ { s } ^ { \star } , P _ { t + 1 }  P _ { s } ^ { \star } , s  s + 1 , c  0$ ▷ the next stage restarts here   
19: end if   
20: end if   
21: $t \gets t + 1$   
22: end while   
23: return $P _ { s } ^ { \star }$

## C.2 GLOBAL QUALITY MONITORING

What it compares. The Monitor never scores a version on its own. At each checkpoint it replays the retained champion $P _ { s } ^ { \star }$ and the candidate build on the same set of demonstrations, pairs the two recordings scenario by scenario, and asks only what changed between them. The comparator is not told which round produced the candidate, what the improvement was trying to achieve, or how much budget has been spent: direction and budget are decided one layer up, and supplying them here would let development intent colour what the model reports seeing.

Scale. Each criterion of the proxy rubric is read on three anchored levels — 1.0 works as described, 0.5 works partly, intermittently, or only in some cases, 0.0 does not work — for the champion and the candidate separately, and only on criteria that both recordings show, so that a difference is a difference in the game rather than in what the two replays happened to reach. Criterion readings are combined into a weighted overall value in [0, 1], reported as two halves: the criteria from the current stage brief, and those from the task document.

Adoption. The candidate replaces the champion when the brief half improves by at least 0.05 and the task half falls by no more than 0.05; otherwise the champion is kept. The task half acts as a floor rather than a target, so a build may be adopted for advancing the stage’s own goals provided it does not regress what the task already required. Without stage criteria the rule reduces to a 0.05 gain on the overall value.

Checkpoints and stopping. Checkpoints are taken every three rounds, which is the shortest interval that contains an improvement, its verification and a replay, and therefore the shortest window in which a difference can be observed at all. Development stops once the champion has gone unchanged for K consecutive checkpoints; we use K = 3. Figure 4 (b) places $K = 1 \dots 4$ against fixed budgets on the same runs: smaller K stops sooner and delivers less, larger K approaches the fixed-budget score at a fraction of its rounds, and K = 3 sits at the knee.

Isolation from the evaluator. The Monitor and the benchmark judge share only the replay mechanism. The Monitor reads a separate proxy rubric, never the benchmark’s held-out rubric, and never its scores or feedback; the module does not import the judge at all. The separation is also internal: the comparator is the only component that sees pixels, and the stage that decides direction and stopping reads its structured counts rather than the frames, so neither layer can both interpret an image and act on its own interpretation.

## C.3 HIGH-LEVEL DIRECTOR INTERFACE

Section 3 describes what a stage brief does to the loop. This appendix describes where briefs come from. Every brief reported in this paper — human and model alike — was written in the interface below, under one protocol, so that “a person directed this run” and “a model directed this run” differ in the director and in nothing else.

## C.3.1 WHAT THE DIRECTOR IS GIVEN

A director reviews one build of one game. Figure 6 is the page: the game runs live in the left panel and takes the keyboard when the frame is focused; the right column holds the budget, the task specification the game was generated from, and a summary of the development that has already happened; the form below is the brief.

The summary is deliberately thin. It reports how many autonomous rounds have run, which of them became champions at the Global Quality Monitor’s three-round checkpoints, which build is under review, and how those rounds divided between functional and presentational work. It reports no score, at any round, for any build.

That omission is the point. The director is asked what the game needs next, and a score would answer a different question — whether the last round worked — in a currency the loop already optimises. Nothing else about the run reaches the page either: no source, no logs, no improvement transcripts, no other director’s review. The instrument shows a playable game and the document it was supposed to become.

## C.3.2 ONE INSTRUMENT, ONE BUDGET

A director gets 60 actions and 2 resets. A key press costs one action however long it is held, a click costs one, and watching and waiting are free, so the budget buys interaction rather than patience. The counters in Figure 6 are the ones the session enforces, and the count each director spent is stored with the brief.

![](images/68790f805bd740ca5536b00a26fb844508497e72a5cdc2b2f080fc7de4fc0f5c.jpg)  
Figure 6: The director review page, mid-session, on the Alley Brawlers build that had survived six monitor checkpoints: the live game, the budget, the task specification, and the development summary — champions at R03 and R09, no update since, and the split of the eighteen rounds between functional and presentational work. The frame is a real moment of play and shows the defect both directors went on to describe in different words: the player’s fighter has come apart into a band of scrambled sprite tiles. The brief form below it is Figure 7.

A person plays through the browser. A model plays through a command-line client — open, press, hold, click, wait, look, reset, state, submit — that drives the same session object on the same server, spends the same budget through the same counters, and returns the frames captured after each action for the model to read. Neither director can see the other’s transport; both are talking to one running game.

The two directors reviewed the same bytes. Each brief records the hash of the project tree it was written against, and for Alley Brawlers both read c12fe90b30b6c176. The human director spent the full 60 actions and no resets; the model spent 25 and no resets.

## C.3.3 HOW THE MODEL DIRECTS

The model director is given one frozen prompt, versioned (director-prompt/2) and unchanged within a run; editing it makes a new run. It casts the model as a senior game director reviewing a build whose autonomous development has plateaued, tells it to decide what the next stage should be rather than to enumerate defects, and forbids implementation-level instructions.

Two rules do the methodological work. Everything the model knows about the game must come from the session: reading, listing or searching any other file — any source, any log, any score, any other review — discards the review. And the brief is submitted through the same endpoint the form posts to, in the same schema, once; submission ends the session.

## C.3.4 WHAT A BRIEF IS

Four fields, identical for both directors: a one-sentence stage\_objective; why\_now, grounded in what the director saw while playing; one to three priorities, stated as directions rather than code changes; and zero to three items to preserve. The loop consumes the brief in three places — the planner, stage ranking, and the improvement and art prompts — and the stage checklist that the Monitor scores against is rebuilt so that the brief’s own criteria carry the most weight (Appendix C.2).

![](images/dda71165d781778e3b3936427675c855d0e51da97ee48255ab26780965999297.jpg)  
Figure 7: The brief form, the lower half of the page in Figure 6. The model director submits the same four fields as JSON through the endpoint this form posts to.

Table 4 puts both briefs for Alley Brawlers side by side. They were written four minutes apart against the same tree, and they do not overlap. The human read the build as a presentation problem and asked for art and onboarding; the model played two full matches, never took a single point of health off the opponent, and read the build as a mechanics problem. Neither is the correct answer, and the outcome was close: from this checkpoint the human-directed branch reached 62.0, the model directed branch 61.5, and a control branch restarted from the same build with no brief reached 60.0. What the comparison shows is that a brief supplies a direction the loop’s own evidence had not produced in eighteen rounds, and that two directors looking at one build through one instrument can supply entirely different ones.

<table><tr><td colspan="2">Human director — 60/60 actions, 0 resets.</td></tr><tr><td>Objective</td><td>The art needs work: the title and game-over screens are cluttered and have no proper background, the character art is ugly and should be regenerated in a comic style, and attacks have no effects. The game also needs to teach how it is played — right now I do not know what to press.</td></tr><tr><td>Why now</td><td>The characters move and there are title and game-over screens, but it is still too rough to be appealing.</td></tr><tr><td colspan="2">Model director (Claude Opus 5) — 25/60 actions, 0 resets. Turn the match from a scripted beatdown into a real two-sided fight: player inputs must visibly</td></tr><tr><td>Objective</td><td>and reliably land, block and escape, against an AI that can be beaten, with two fighters that genuinely look and play differently. Across two full matches (one as each fighter) the AI never lost a pixel of health. Every round</td></tr><tr><td>Why now</td><td>ended in about 18 seconds with me pinned in the left corner in a hit, knockdown, hit loop. Light, heavy, the special input, holding back, crouching and jumping all failed to trade, block or get me out. Nobody can learn to play when nothing they press matters. [...] Seven rounds of visual polish have gone into the frame around the fight (title, background, banners, results screen), but the fight itself is still missing.</td></tr></table>

Table 4: The two briefs written for the same Alley Brawlers build (tree c12fe90b30b6c176), abridged to their first two fields. Priorities and preserve items are omitted for space.

<table><tr><td colspan="6">Director Review Play each game, then write the brief for its next development stage.</td></tr><tr><td>Reviewer ID figure</td><td></td><td colspan="3">Show my cases</td><td></td></tr><tr><td>CASE</td><td>GAME</td><td>BUILD</td><td>STATUS</td><td>ACTIONS</td><td></td></tr><tr><td>pilot1-01</td><td>Maze Muncher</td><td>R06</td><td>not started</td><td>0 /60</td><td>Start</td></tr><tr><td>pilot1-02</td><td>Block Cascade</td><td>R00</td><td>not started</td><td>0 /60</td><td>Start</td></tr><tr><td>pilot1-03</td><td>Alley Brawlers</td><td>R09</td><td>in progress</td><td>3/60</td><td>Resume</td></tr><tr><td>pilot1-04</td><td>Plumber Kingdom</td><td>R18</td><td>not started</td><td>0 /60</td><td>Start</td></tr><tr><td>pilot1-05</td><td>Lawn Guardians</td><td>R12</td><td>not started</td><td>0 / 60</td><td>Start</td></tr><tr><td>pilot1-06</td><td>Critter Clash</td><td>R09</td><td>not started</td><td>0/ 60</td><td>Start</td></tr></table>

Figure 8: The case list a director opens: six games, the build under review for each, and the budget remaining. Case identifiers are internal labels.

## C.3.5 THE CASE SET

Six games were reviewed, each at the build its own run had converged on, which ranges from the generated project to eighteen rounds in (Figure 8). Every game was reviewed independently by the human director and by the model director, and each brief was then given to a branch restarted from that same build, against a control branch given no brief. Three of the six are the case studies of Appendix F.2.

## C.4 AGENT PROMPTS AND OUTPUT CONTRACTS

This appendix reports the prompts of the five modules that carry the loop’s decisions, one per stage of Section 3. Each is abridged to its decision skeleton: fixed catalogues, worked examples, framelabelling conventions and harness housekeeping are cut, and every cut is marked [...]. Braces such as {items} are runtime slots filled with the development state, the playtest record, or the replayed frames.

One design choice recurs across all five and is easier to see stated once. Each prompt is told what it needs to decide its own question and deliberately not told the rest. The planner is given the round’s direction as a conclusion, not as an argument. The Verifier is told what the round was trying to do, because that is the claim it must check, while the comparator is told neither that anything was improved nor which build is newer. The Editor is told that the problem is already established, so that it starts at “where in the code does this live” rather than re-deriving the diagnosis. Withholding is what keeps a stage from confirming its own expectation.

## C.4.1 CONTROLLER

The planner decides what to find out, never what to change. It is given the round’s direction as a settled conclusion, the requirement state, what the build did when last played, the demos available and the budget left; it returns one question and the cheapest way to answer it. Most of the prompt’s length goes to ruling out questions that no observation can settle.

Prompt P1. Controller   
You plan what to investigate next in a game that is being developed one round   
at a time.   
You do not write code, propose patches, or say what should be changed. Another   
agent does that, after the evidence you ask for has been collected. Your output

Table 5: The five LLM-facing modules of the RSIGAME loop, by the stage of Section 3 they implement.
<table><tr><td>Stage</td><td>Module</td><td>Output contract</td><td>Consumed by</td></tr><tr><td>Direction Decision</td><td>Controller</td><td>One development_question; action in {reuse, explore, escalate}; an exploration_plan naming the</td><td>Sets the round&#x27;s objective; escalate re- ports the direction exhausted rather than inventing work.</td></tr><tr><td>Agentic Exploration</td><td>Explorer</td><td>cheapest mode that can answer it. The 1-3 requirements a session should settle, the scenario to boot, and one ac- tionable instruction for the play agent.</td><td>Drives the test agent&#x27;s interaction with Pt, producing the trace τt.</td></tr><tr><td>Evidence-Grounded Editing</td><td>Editor</td><td>Edits to the project under a build con- straint, plus a report of what was changed.</td><td>Produces Pt+1.</td></tr><tr><td>Agentic Verification</td><td>Verifier</td><td>goal_achieved in {yes, no, unclear}; broken_now; next_goal when that list is non-empty.</td><td>Accepts or rejects the round; anything broken re-enters the state as pending work.</td></tr><tr><td>Global Quality Mon- itoring</td><td>Global Monitor</td><td>Quality Per-criterion readings for both builds, changes with direction and magnitude, and regressions — never a score.</td><td>Aggregated into the proxy value that de- cides the champion (Appendix C.2).</td></tr></table>

```jsonl
is a decision about what to FIND OUT.
[... direction, requirement state, last playtest, demos, budget ...]

WHAT TO DECIDE
====================
One primary question. It must be:
- inside the direction above;
grounded in the state and evidence above, not in what a game like this
usually has;
- answerable by watching the game: someone replays it and looks. Not by
reading the source, which nobody will do for you;
- concrete enough that you can say what that observation would be;
- useful for deciding whether something needs repairing.
Not a question: "Can the game be improved?" -- nothing observable answers it.
Not a question: "Improve the HUD layout." -- that is an instruction to change
something, and nothing has been observed yet.
Then choose one action:
reuse the evidence above already settles what the problem is.
explore the evidence is missing or ambiguous, and a specific observation
would settle it.
escalate there is no grounded unresolved question left in this direction.
Say so rather than inventing one.
If exploring, choose the cheapest mode that can answer the question:
replay_existing_demo a demo already exercises what you need to see
extend_existing_demo a demo reaches the state but stops before the answer
custom_targeted_probe no demo can reach it. The expensive option.
If a question has already been asked several rounds running without being
settled, asking it again the same way is not a plan. Either say what would be
different this time, or pick something else.
Reply with strict JSON and nothing else:
{"action": "reuse|explore|escalate",
"development_question": {"question": "...", "why_now": "..."},
"exploration_plan": {"mode": "...", "demo_id": "...", "focus": "...",
"stop_when": "..."}}
```

## C.4.2 EXPLORER

Exploration is aimed, not generic. The probe selector turns the requirements the stage cannot yet answer into one instruction a play agent can carry out, and is asked to weigh importance against what a short session can reach.

Prompt P2. Explorer   
You are deciding what ONE play session should go and find out about this game   
next.   
These are the requirements this stage still cannot answer, with where each   
could be checked:   
[... candidate requirements, with status and scenario ...]   
Pick the 1 to {cap} that matter most RIGHT NOW and write a single instruction   
for the person who will play the build. Judge importance -- a requirement the   
rest of the game depends on beats a detail, and something you can settle in a   
few actions beats something that needs a long grind.   
If the ones you pick share a place to look, say so; a session that has to boot   
two different states spends half its steps travelling.   
The instruction has to be ACTIONABLE: name the input to make or the screen to   
reach and what to watch. "Check whether the plane rotates" is weak. "In a   
sortie, press Left and then Right for about a second each and report whether   
the plane's heading changes" can be carried out.   
Reply with JSON and nothing else:   
{"items": ["T04"], "scenario": "", "ask": "...", "why": "one sentence"}

## C.4.3 EDITOR

The Editor receives a problem that has already been established, and the prompt’s first job is to stop it from establishing it again. It is given the source, the build command that must pass, and a way to run and watch the build; it is not given the authority to change the harness or the evaluation traces.

Prompt P3. Editor   
You are improving a small game. This is round {r} of at most {R}.   
You did not play this build. Someone else did: a play agent drove it, the   
frames were read, and what follows is what that evidence supports and what was   
chosen for this round.   
So the problem below has already been established. You do not have to find out   
whether it is real, and you should not replay the game to confirm it -- that   
work is done, and repeating it spends the calls you need for the repair. Your   
job starts at "where in the code does this live".   
[... the round's evidence packet, recent history, project map ...]   
WHAT YOU HAVE   
- the source, in the directory you are working in   
- \`{build\_cmd}\` -- it must pass when you stop   
- you can run the build and watch it: [...]   
RULES   
- Change the game, not the harness. Do not edit \`node\_modules\`, \`dist\`,   
\`demo\_outputs/\`, or anything under \`\_repair\_evidence/\`.   
- Do not edit \`demo\_outputs/\`. Those are the input traces the evaluator   
replays; changing them changes the exam, not the game.   
[... process and server hygiene ...]

## C.4.4 VERIFIER

After the edit, every shipped demo is replayed and one reader decides whether the round earned its commit. The prompt separates two failures that are easy to conflate: a feature the game never had is not a defect, and an input that visibly does something has been received even if the script never reaches the outcome it was written for. Only an input that changes nothing counts as broken.

Prompt P4. Verifier   
A repair agent has just edited a game. Your job is to look at what the game   
does now and answer two questions. You are the only check on this edit: after   
you, it ships.   
WHAT THE ROUND WAS TRYING TO DO   
{goal}   
[... the requirements this goal came from; the replayed frames ...]   
ANSWER THESE TWO, IN THIS ORDER.   
1. goal\_achieved -- yes / no / unclear.   
Is the thing the round set out to do visibly true now? "unclear" is a real   
answer: say it when the demos never reach the part of the game the goal is   
about, rather than guessing.   
2. broken\_now -- a list, possibly empty.   
Anything the game does that is plainly wrong to a player watching these   
frames. Be concrete about what you SAW and in which frame. The kinds that   
matter most, because they are invisible in source code:   
- an input produces no change at all -- the before and after frames are   
the same, again and again, so the game is not listening   
- something covers the playfield: one sprite, one effect, one panel that   
hides the characters, the HUD or the action   
- the screen is frozen: the last frame equals the first   
- text is unreadable, clipped, or drawn on top of itself   
Do not list a missing feature here. A feature the game never had is a task   
requirement, not something that is broken.   
AND ONE THING NOT TO REPORT. An input that visibly DOES something -- a mode   
label appears, a prompt is shown -- has been received, even when the demo never   
goes on to reach the outcome it was written for. That is the game asking for   
more steps than this fixed script performs, and it is not a fault in the game.   
Only an input that changes NOTHING belongs in broken\_now.   
Then, only if broken\_now is not empty, write next\_goal: one sentence telling   
the repair agent what to fix first. Fix the worst thing, not all of them.

## C.4.5 GLOBAL QUALITY MONITOR

The comparator is the only component in the method that reads pixels. It is shown the champion and the candidate replayed under the same fixed script and asked what changed; it is told neither which build is newer nor that an improvement took place. Most of the prompt defends against the two ways a pairwise reading goes wrong: assuming the newer build is better, and mistaking recorder timing for a behavioural change.

Prompt P5. Global Quality Monitor   
Two builds of the same game were replayed with the SAME fixed input script.   
Your job is to say what changed in this scenario, and nothing else.   
[... the scenario, the criteria it can speak to, the paired frames ...]   
The script is fixed, so both builds received the same inputs at the same   
moments. For each moment below you get the OLD build just before and just after   
the input, then the NEW build at the same two moments.   
The recording is not frame-exact: replaying one build twice cuts the same   
moment up to a frame or two apart, which by itself makes a score read one point   
different, an animation look half a beat behind, or a falling piece sit one row   
lower. That is the recorder, not the build.   
You get two frames after each input for exactly this reason. A difference that   
is gone by the second frame, or that later moments undo, is timing. A   
difference that is still there at the next moment and at the FINAL frame is the   
build. Say \`same\` for the first kind.   
RULES   
- Compare only what these frames support.   
- Do not assume the newer build is better, and do not assume a difference you

cannot explain is a defect. Both tilts are errors.   
- If a behaviour is not exercised here, its criterion is \`unobserved\`. Not   
seeing something is not seeing it missing.   
- Report anything that got worse separately, however small the rest.   
- Judge relative change. Do not score either build.   
- Do not propose code edits, repairs, or next steps. That is not your job.

## D EXPERIMENTAL PROTOCOLS AND EVALUATION RELIABILITY

## D.1 EXPERIMENTAL CONFIGURATION

This appendix lists, in one place, which model plays which role, how each is served and decoded, and every budget a run is given. Table 6 covers the models, and Table 7 the loop and scoring parameters.

Scoring protocol. Every number in Table 1 is the mean over the 140 tasks of a score produced by the Qwen3.8-27B judge from three independent replay-and-score runs of the same artefact. A task whose generator produced no project scores 0; so does a task whose development has not finished. Base rows score the frozen $P _ { 0 } .$ , Play2Code rows the version thirty rounds deliver, and RSIGAME rows the version the Global Quality Monitor is holding when the saturation stop fires.

Table 6: Models by role. The generator differs per row group; every other role is the same model in every row, so that a comparison between rows is a comparison of development methods and not of the models inside them. Decoding is greedy wherever the answer is a claim about the game: a coordinate, a verdict, a rubric item. Served by matters for cost and for reproducibility — one model on OpenRouter is sold by a dozen providers, so the judge and the monitor pin theirs.
<table><tr><td>Role</td><td>Model</td><td>Served by</td><td>Decoding</td></tr><tr><td>Generator</td><td>one per row group of Table 1</td><td>harness default</td><td>harness default</td></tr><tr><td>Controller</td><td>GLM-5.3-Flash</td><td>OpenRouter</td><td>T = 0, top-p 1</td></tr><tr><td>Explorer</td><td>Qwen3.8-27B</td><td>local vLLM (TP = 2)</td><td>T = 0, top-p 1</td></tr><tr><td>Verifier</td><td>GLM-5.3-Flash</td><td>OpenRouter</td><td>T = 0, top-p 1</td></tr><tr><td>Editor</td><td>GLM-5.3-Flash</td><td>OpenRouter</td><td>T = 0, top-p 1</td></tr><tr><td>Global Quality Monitor</td><td>Qwen3.8-Flash</td><td>OpenRouter</td><td>T = 0</td></tr><tr><td>Asset generation</td><td>Seedream-5.0-lite</td><td>OpenRouter (images)</td><td></td></tr><tr><td>Evaluation judge</td><td>Qwen3.8-27B</td><td>local vLLM</td><td>T = 0, thinking off, ≤2048 output</td></tr></table>

## D.2 TOKEN AND COST ACCOUNTING

Scope. We count two phases per task. Generation is the generator’s single agent run that produces the frozen base $P _ { 0 }$ . Development is every model call made during the development rounds: the controller that determines the direction, the explorer that plays the game, the verifier that inspects it, the editor that edits it, and the Global monitor’s periodical check. Scoring by the evaluation judge is excluded for all methods.

Tokens. The Tok. column counts billable tokens: input excluding cache reads, plus output. Cache reads are left out of this count because long agent sessions re-read the same prefix on every call; they are still paid, at the cache price, in Cost.

Cost. Per call,

$$
\mathrm { c o s t } = n _ { \mathrm { i n } } p _ { \mathrm { i n } } + n _ { \mathrm { c a c h e } } p _ { \mathrm { c a c h e } } + n _ { \mathrm { o u t } } p _ { \mathrm { o u t } } ,
$$

with $n _ { \mathrm { i n } }$ the uncached input tokens and OpenRouter list prices (Table 8). Development is not priced this way but summed from the amount the API returns with each call, which the proxy records for every call the Controller, Explorer, Verifier, Editor, and Monitor make; the formula above is used where only token counts survive, which is generation. Cost covers development only; generation is reported separately in Table 9. Generation cost is dominated by the harness rather than the model: the same GPT-5.5 hits 94% cache under Codex and 42% under OpenGame, and the resulting bills differ by a factor of eight.

Table 7: Loop, monitor and scoring parameters. One value per knob, the same in every run reported here.
<table><tr><td>Parameter</td><td></td><td>Value</td></tr><tr><td rowspan="5">Round</td><td>Tool calls the Editor may make per round</td><td>26</td></tr><tr><td>Wall-clock budget per improvement round</td><td>1800 s</td></tr><tr><td>Improvement attempts per round before the round is given up</td><td>3</td></tr><tr><td>Verification samples per claimed fix</td><td>3</td></tr><tr><td>Probe steps taken before an edit is proposed</td><td>8</td></tr><tr><td rowspan="3">Run</td><td>Generated assets per run</td><td>20</td></tr><tr><td>Project snapshot kept</td><td>every round</td></tr><tr><td>Replay recording kept</td><td>every 3rd round</td></tr><tr><td rowspan="5">Monitor</td><td>Checkpoint interval</td><td>every 3 rounds</td></tr><tr><td>Demos replayed per checkpoint</td><td>≤3</td></tr><tr><td>Frames per demo the comparison reads</td><td>16</td></tr><tr><td>Margin a candidate must beat the champion by</td><td>0.05</td></tr><tr><td>Checkpoints with an unchanged champion before the saturation stop (K)</td><td>3</td></tr><tr><td rowspan="3">Scoring</td><td>Replay-and-score runs per artefact</td><td>3</td></tr><tr><td>Uniform frames sampled per demo</td><td>16</td></tr><tr><td>Rubric items per task (M/D/V/A)</td><td>12–21 (median 18)</td></tr></table>

Which price. A model on OpenRouter is served by a dozen providers whose prices differ by up to a factor of two, and the model page quotes only one of them, which for several models is the dearest on the board. We therefore price each model at its cheapest standard endpoint (Table 8). Two exclusions make that well-defined. Latency tiers are not alternative sellers of the same service and are left out: the flex tier halves GPT-5.5 by deferring the request. And endpoints are compared on what these runs would actually cost there rather than on input price alone, because a quoted cache price of zero means the endpoint offers no prompt caching, not that cache reads are free – read literally it would make a provider without caching the cheapest seller of a workload that is 42–96% cache.

Sources. Generation usage comes from each generation run’s own record (input, cache-read and output tokens), released with the base-game corpora; for Codex + GPT-5.5 on Godot only corpuslevel means are recorded. Development usage is logged per call by a proxy between the agents and the API.

Table 8: OpenRouter list prices (\$ per million tokens), retrieved 2026-09-16. Each model is priced at its cheapest standard endpoint for the cache mix of the runs reported here; the serving provider is named because prices for one model differ by up to a factor of two across providers.
<table><tr><td>Model</td><td>Provider</td><td>Input</td><td>Cache read</td><td>Output</td></tr><tr><td>GPT-5.5</td><td>OpenAI</td><td>5.000</td><td>0.500</td><td>30.00</td></tr><tr><td>Kimi-K2.6</td><td>Baidu</td><td>0.408</td><td>0.069</td><td>1.72</td></tr><tr><td>Qwen3.8-27B</td><td>DeepInfra</td><td>0.150</td><td>0.037</td><td>1.88</td></tr><tr><td>GLM-5.3-Flash</td><td>DeepInfra</td><td>0.075</td><td>0.015</td><td>0.25</td></tr></table>

## D.3 REPLAY AND SCORING STABILITY

Replay Variability. GameCraft-Bench scores a game from its observed gameplay rather than its source code alone. Even for the same frozen project, repeated playbacks can produce slightly different trajectories because game execution, input scheduling, and frame capture are not perfectly synchronized. The magnitude of this variation depends on the game, and Table 11 measures which kind of game it depends on. Therefore, repeatedly judging the same recording does not capture the full uncertainty of the evaluation process.

Table 9: Generation of the frozen base P , mean per task. Billable is the Tok. of the Base rows; Total adds cache reads; Cost applies Table 8 to all three token counts. The two GPT-5.5 rows are the same model under two harnesses and differ in price: Codex re-reads a cached prefix, OpenGame does not. Over all 140 tasks.
<table><tr><td>Engine</td><td>Generator</td><td>Billable</td><td>Total</td><td>Cache</td><td>Output</td><td>Cost</td></tr><tr><td>Godot</td><td>Codex + GPT-5.5 (high)</td><td>0.26M</td><td>3.83M</td><td>94%</td><td>28k</td><td>$3.77</td></tr><tr><td>Godot</td><td>GLM-5.3-Flash</td><td>0.44M</td><td>11.43M</td><td>96%</td><td>36k</td><td>$ 0.20</td></tr><tr><td>Godot</td><td>Kimi-K2.6</td><td>3.22M</td><td>8.98M</td><td>65%</td><td>123k</td><td>$1.87</td></tr><tr><td>Godot</td><td>Qwen3.8-27B</td><td>6.41M</td><td>11.62M</td><td>46%</td><td>179k</td><td>$1.47</td></tr><tr><td>Godot</td><td>Qwen3.8-27B (SFT)</td><td>0.57M</td><td>8.21M</td><td>95%</td><td>140k</td><td>一</td></tr><tr><td>Phaser</td><td>OpenGame + GPT-5.5</td><td>5.36M</td><td>9.21M</td><td>42%</td><td>97k</td><td>$31.17</td></tr><tr><td>Phaser</td><td>Qwen3.8-27B</td><td>9.03M</td><td>26.69M</td><td>67%</td><td>292k</td><td>$ 2.51</td></tr></table>

Evaluation Protocol. To account for this variability, every artifact is evaluated with three independent end-to-end runs. Each run starts from a fresh replay, generates its own gameplay recording, and is scored independently by the same Qwen3.8-27B judge used for every score in this paper. We report the mean score across the three runs. The judge configuration is held fixed across all evaluations, so the repeated runs primarily capture variation introduced by gameplay replay rather than changes in the evaluation setup.

Observed Stability. Table 10 summarizes the variation across 529 frozen artifacts, every Godot artifact in this paper that was scored three times: the frozen base of two generators, the versions Play2Code delivered, and the versions RSIGAME delivered.

Most artifacts are close to deterministic. The median range across three independent runs is 0.73 points, 225 of the 529 score identically in all three, and a quarter of them vary by less than a hundredth of a point. Improved artifacts vary somewhat more than frozen ones (median 0.87 against 0.36), which is what a build with more of a game in it should do.

Table 10: Score variation across three independent end-to-end evaluations of the same frozen artifact. Each run uses a fresh gameplay replay and fresh judging by the Qwen3.8-27B judge. Range denotes the difference between the highest and lowest Overall scores across the three runs; the columns are its quartiles.

<table><tr><td>Subset</td><td>n</td><td>p25</td><td>Median</td><td>p75</td><td>Identical 3/3</td></tr><tr><td>All artifacts</td><td>529</td><td>0.00</td><td>0.73</td><td>3.33</td><td>225</td></tr><tr><td>Base P0</td><td>266</td><td>0.00</td><td>0.36</td><td>2.33</td><td>131</td></tr><tr><td>Improved</td><td>263</td><td>0.00</td><td>0.87</td><td>3.50</td><td>94</td></tr></table>

Table 11: Replay variation by task family, for the families with at least twelve scored artifacts, ordered by median range. Same 529 artifacts and same definition of range as Table 10.
<table><tr><td>Family</td><td>n</td><td>Median</td><td>Family</td><td>n</td><td>Median</td></tr><tr><td>sports</td><td>14</td><td>4.66</td><td>shooter</td><td>24</td><td>0.44</td></tr><tr><td>horror</td><td>19</td><td>3.37</td><td>puzzle</td><td>32</td><td>0.42</td></tr><tr><td>openworld</td><td>62</td><td>1.41</td><td>simulation</td><td>24</td><td>0.28</td></tr><tr><td>visualnovel</td><td>32</td><td>1.17</td><td>strategy</td><td>63</td><td>0.00</td></tr><tr><td>platformer</td><td>77</td><td>0.87</td><td>rhythm</td><td>21</td><td>0.00</td></tr><tr><td>tycoon</td><td>58</td><td>0.73</td><td>idle</td><td>16</td><td>0.00</td></tr><tr><td>roguelike</td><td>53</td><td>0.62</td><td>racing</td><td>16</td><td>0.00</td></tr><tr><td>cardgame</td><td>18</td><td>0.50</td><td></td><td></td><td></td></tr></table>

Which Games Vary. Splitting the 529 artifacts by task family locates the variance rather than leaving it as a property of “game dynamics” (Table 11). The split is not fast against slow: the three most variable families are sports, horror and openworld, while racing and rhythm — as real-time as anything in the benchmark — have a median range of exactly zero, alongside strategy. What the variable families share is that their rubric is answered late. Whether a match was won, whether a haunting resolved, whether a world was traversed is decided by where twenty seconds of play happens to end up; whether a beat landed on time or a lap timer ran is decided by frames the replay reproduces every time. Variance tracks how far into a session the evidence for a criterion arrives, not how fast the game moves.

This also says where a single-run comparison is least safe. A one-point difference is meaningful for a rhythm game and is noise for a sports game, and is one reason the per-family tables of Appendix F.3 should be read with their family’s variance in mind.

Cross-Judge Robustness. To test whether the main results depend on a particular evaluation model, we re-score 40 family-stratified Godot tasks with GPT-5.5, using exactly the same replays, sampled frames, task-specific rubrics, and scoring formula as the Qwen3.8-27B judge; only the judge model is changed.

As shown in Table 12, the two judges produce the same method-level ordering, RSIGAME > Play2Code > Base, with similar improvement margins. The gain of RSIGAME over the frozen base is +14.04 points under Qwen3.8-27B and +12.04 under GPT-5.5, while its gain over Play2Code is +13.47 and +11.76, respectively; all four bootstrap intervals remain well above zero. The judges also agree on the direction of the RSIGAME-over-base comparison for 37 of 40 tasks. Although their absolute scores are only moderately correlated (ρ = 0.64), the conclusions drawn from the relative comparisons remain stable across model families.

Table 12: Cross-judge robustness on 40 family-stratified Godot tasks. Qwen3.8-27B and GPT-5.5 score the same replay recordings with the same rubric. Both judges preserve the method ordering and yield similar pairwise improvement margins. Confidence intervals are 95% task-level bootstrap intervals.
<table><tr><td></td><td colspan="2">Overall</td><td colspan="3"></td></tr><tr><td>Method</td><td>Qwen3.8-27B</td><td>GPT-5.5</td><td></td><td></td><td></td></tr><tr><td>Base (frozen P0)</td><td>49.07</td><td>50.34</td><td></td><td></td><td></td></tr><tr><td>+ Play2Code</td><td>49.63</td><td>50.62</td><td></td><td></td><td></td></tr><tr><td>+ RSIGAME</td><td>63.10</td><td>62.38</td><td></td><td></td><td></td></tr><tr><td>Comparison</td><td>∆ Qwen</td><td>∆ GPT-5.5</td><td>95% CI Qwen</td><td>95% CI GPT-5.5</td><td>Agreement</td></tr><tr><td>RSIGAME — Base</td><td>+14.04</td><td>+12.04</td><td>[+9.84, +18.59]</td><td>[+9.53, +14.60]</td><td>37/40</td></tr><tr><td>RSIGAME — Play2Code</td><td>+13.47</td><td>+11.76</td><td>[+8.95, +18.19]</td><td>[+7.79, +16.26]</td><td>34/40</td></tr><tr><td>Play2Code – Base</td><td>+0.56</td><td>+0.28</td><td>[−3.45, +4.21]</td><td>[−3.83, +3.83]</td><td>24/40</td></tr></table>

Agreement counts tasks on which the two judges prefer the same method.

## D.4 FREE-PLAY EVALUATION

Protocol. GameCraft-Bench evaluates a game by replaying the demonstration traces packaged with its submitted project. These traces are visible development resources, rather than hidden benchmark trajectories, and RSIGAME reuses them for controlled before–after comparisons across checkpoints. Although the hidden rubric, scores, and judge feedback are never exposed during development, repeated use of the same demonstrations could still favor improvements specific to those interaction trajectories. We therefore evaluate whether the gains persist under independently generated gameplay.

We sample 20 family-stratified Godot tasks. For each task, the frozen base, Play2Code, and RSIGAME builds are independently played by blind agents with 40 actions and one reset. The agents see only the public game specification and the running game—not the submission demonstrations, rubric, score, source code, development history, or method identity—and return factual reports of their observations. A separate judge compares these reports pairwise, yielding 60 comparisons in total.

Evaluation. The pairwise judge follows the same four quality dimensions and relative weighting used in the main evaluation, while judging only evidence observed during free interaction. Importantly, the replay scripts are never shown to the play agents: they choose their own actions, timing, and trajectories. This changes the interaction distribution while keeping the definition of game quality aligned with the main evaluation.

Results. As shown in Table 13, RSIGAME is preferred over the frozen base on 17 of 20 tasks and over Play2Code on 18 of 20, while Play2Code does not show a clear advantage over the base. Across all decided comparisons, the free-play evaluation agrees with the benchmark ordering in 45 of 59 cases. These results indicate that the gains of RSIGAME persist under independently chosen gameplay trajectories rather than being confined to the fixed replay scripts used during development and benchmark evaluation.

Table 13: Free-play evaluation on 20 family-stratified Godot tasks. Blind agents independently play each build, and a separate judge compares their reports using the same quality dimensions and weighting as the main evaluation. p is a two-sided sign test over decided comparisons.
<table><tr><td>Comparison</td><td>W/L/T</td><td>p</td><td>Agreement</td></tr><tr><td>RSIGAME – Base</td><td>17/3/0</td><td>0.003</td><td rowspan="3">45/59</td></tr><tr><td>RSIGAME – Play2Code</td><td>18/2/0</td><td>&lt; 0.001</td></tr><tr><td>Play2Code – Base</td><td>9/11/0</td><td>0.824</td></tr></table>

W/L/T denotes wins/losses/ties for the first method. Agreement counts decided comparisons for which free-play evaluation and the benchmark score prefer the same build.

## D.5 STATISTICAL RELIABILITY OF MAIN RESULTS

Paired task-level uncertainty. All main-table comparisons are paired by task. We estimate uncertainty in the mean Overall difference ∆ using 20,000 bootstrap resamples of the 140 benchmark tasks, and additionally report a Wilcoxon signed-rank test over the paired task-level differences. We separately resample the three replay-and-score runs of each fixed task to measure evaluation noise; these intervals are substantially narrower than the task-bootstrap intervals, indicating that benchmark-task variation dominates replay and judge noise.

Main comparisons. RSIGAME significantly improves over its frozen starting point in every engine–generator setting inlcuded in Table 14, with gains of +8.8 to +14.3 Overall points and task-bootstrap intervals well above zero. Against Play2Code, the improvement is significant for both Godot generators and for GPT-5.5 on Phaser; the Qwen3.8-27B Phaser comparison remains unresolved (+1.55, p = 0.394).

Table 14: Paired significance of the main results. ∆ is the mean task-level Overall difference. Confidence intervals are obtained from 20,000 paired bootstrap resamples of benchmark tasks; p is from a Wilcoxon signed-rank test. W/L counts tasks on which the first method scores higher/lower.
<table><tr><td>Comparison</td><td>∆</td><td>95% CI</td><td>p</td><td>W/L</td></tr><tr><td>Godot</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.5: RSIGAME – Base</td><td>+14.26</td><td>[11.49, 17.06]</td><td>&lt; 10−4</td><td>112/16</td></tr><tr><td>GPT-5.5: RSIGAME – Play2Code</td><td>+13.79</td><td>[10.89, 16.65]</td><td>&lt; 10−4</td><td>118/19</td></tr><tr><td>Qwen3.8-27B: RSIGAME – Base</td><td>+10.70</td><td>[7.75, 13.77]</td><td>&lt; 10 -4</td><td>100/31</td></tr><tr><td>Qwen3.8-27B: RSIGAME – Play2Code</td><td>+7.24</td><td>[4.49, 10.04]</td><td>&lt; 10−4</td><td>92/41</td></tr><tr><td>Phaser</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.5: RSIGAME – Base</td><td>+8.77</td><td>[6.13, 11.51]</td><td>&lt; 10−4</td><td>101/35</td></tr><tr><td>GPT-5.5: RSIGAME – Play2Code</td><td>+3.11</td><td>[0.53, 5.61]</td><td>0.008</td><td>83/53</td></tr><tr><td>Qwen3.8-27B: RSIGAME – Base</td><td>+10.21</td><td>[6.99, 13.70]</td><td>&lt; 10−4</td><td>95/36</td></tr><tr><td>Qwen3.8-27B: RSIGAME – Play2Code</td><td>+1.55</td><td>[-1.50, 4.66]</td><td>0.394</td><td>73/59</td></tr></table>

## D.6 ADAPTIVE DIRECTION POLICY EVALUATION

Experimental Setup. We compare adaptive development with a round-robin direction policy on 16 paired games initialized from the same frozen base projects. Each policy is run for 30 development rounds with the same agent models, tools, and per-round budget; the only difference is how the development focus is selected. Figure 5a,b analyzes the resulting trajectories using signals that are not exposed to either policy during development. We define a broken build as a round whose resulting project fails to produce a runnable game; the following round is then aligned at offset +1 relative to this failure event. For state-conditioned analysis, we additionally use a held-out diagnostic rubric to identify the currently lagging quality dimension. This rubric is used only for post-hoc analysis and is never provided to the development loop. Shaded areas in Figure 5a are ±1 standard error over the corresponding break events or grouped development rounds.

Blind Pairwise Evaluation. To compare the final games produced by the two policies, we perform 63 blind pairwise comparisons using their round-30 checkpoints. For each comparison, the same demo trace is replayed on both builds and the resulting observations are presented side by side to the evaluator. The left–right assignment is randomized independently for each comparison, and no information about the underlying direction policy is revealed. Evaluators report an overall preference as well as preferences along the Mechanics, Depth, Visuals, and Art dimensions; ties are allowed. To check for position bias, a subset of comparisons is evaluated again with the two sides swapped. These repeated judgments are used only as a consistency check and are excluded from the reported 63 comparisons. The pairwise evaluator is independent of the GameCraft-Bench benchmark evaluator and has no access to its scores, rubrics, or feedback.

## D.6.1 PAIRWISE JUDGING CRITERIA

The criteria given to the blind pairwise judge of Section 5, reproduced in full. They are deliberately not the benchmark’s rubric: the judge compares two builds on what its frames show, while the benchmark scores one build against the task. Rule 1 is the one that does the most work — without it a broken build collects credit for the frames it does manage to render.

Comparison Rubrics. Blind pairwise judge   
You are an expert game reviewer. You compare two builds of the same game, Game   
1 and Game 2. Both were developed from the same starting game and the same   
design document, which is given below. Each build was played with exactly the   
same scripted input (a "demo"), and you see frames sampled at the same fixed   
interval from that recording, in time order. Judge only what is visible in the   
frames.   
For each dimension below, decide which build is better: "1", "2", or "tie".   
## Core Mechanics

Does the scripted input visibly produce the gameplay the design document   
describes?   
- Better: actions have clear on-screen effects (movement, attacks, placement,   
state changes, score or resource changes); the core loop of the design   
document can be observed.   
- Worse: input has no visible effect; the game stays on a title or menu screen;   
a scene fails to load: the screen is blank or frozen.   
## Content Depth   
How much of the designed game is present and reached in this demo?   
- Better: more of the distinct mechanics, enemies, levels, events, progression,   
or win/lose states from the design document appear.   
- Worse: a single repeated screen; placeholder content; systems that are   
announced in the UI but never shown.   
## Functional Visuals   
Can a player read the game state?   
- Better: HUD, feedback, and important objects are legible and clearly   
separated from the background; text is not clipped or overlapping; sprites   
and animations are stable from frame to frame.   
- Worse: unreadable or overlapping text; flickering or inconsistent sprites;   
key objects hidden behind others; no visible feedback for important events.   
## Presentation & Art   
Does it look like a coherent, finished game?   
- Better: a consistent art style across screens; intentional composition;   
assets that match the theme of the design document.   
Worse: primitive shapes where art is expected; mixed or clashing styles;   
visual noise; decoration that makes the game harder to read.   
## Overall   
Which build would a player of this design document prefer, all things   
considered?   
## Rules   
1. A build that is broken in these frames (blank, crashed, frozen, stuck on a   
screen the demo should have left) loses every dimension it fails to show,   
however good its other frames look.   
2. More assets, more effects, or busier frames are not better by themselves.   
Prefer them only when they serve the design and keep the game readable.   
3. Ignore which side a build is shown on. Game 1 and Game 2 are in random   
order.   
4. Answer "tie" when the two builds are not visibly different on a dimension,   
or when both fail it equally.   
5. Each reason must point to something visible, naming the build and the frame   
(for example "Game 2, frame 7: the score counter overlaps the timer").

## D.7 AGENTIC VERIFICATION AUDIT

Audit Protocol. We evaluate the reliability of verification on development trajectories from 40 games initialized by GPT-5.5. We sample pre- and post-improvement decisions from these trajectories and provide the complete interaction evidence associated with each decision to Claude Opus 5 (Anthropic, 2026) for independent adjudication. Each case is evaluated twice; disagreements between the two judgments are manually reviewed to obtain the final label. This evaluator is used only for the verification audit and is fully isolated from the GameCraft-Bench evaluator and its scores, rubrics, and feedback.

Pre-Improvement Grounding. We evaluate whether proposed improvement targets are supported by the evidence available before editing. For each audited session, a target is labeled as grounded when the observed gameplay evidence supports the claimed problem or improvement opportunity, and ungrounded when it is unsupported or contradicted by the evidence. We report grounded precision, the fraction of decidable targets that are grounded, and the average number of ungrounded targets per round. We compare the agentic evidence-grounding procedure used by RSIGAME against a free-form critic operating on the same development context.

Post-Improvement Failure Detection. We further evaluate whether verification can detect unsuccessful edits after the updated game is executed. An improvement is labeled unsuccessful when the intended change is not observable in gameplay or when the edit introduces a regression. Treating unsuccessful improvements as the positive class, we report failure recall, specificity, and balanced accuracy over 48 adjudicated rounds. We compare replay-based agentic verification with a buildonly baseline that checks whether the updated project executes successfully but does not inspect its interactive behavior.

## D.8 MULTI-STAGE GUIDANCE EVALUATION

Section 5 tests one stage of development. This appendix specifies how we ask whether a second and a third stage keep paying: each pair of stage champions is compared by blind pairwise play, so a later stage has to be preferred as a game rather than merely score higher.

What is compared. Three pairs of builds, each the champion the Global Quality Monitor delivered at the end of a stage: stage 2 against stage 1, stage 3 against stage 2, and stage 3 against stage 1. Both builds of a pair come from the same game, so each comparison is within-task, with the same generator, the same task specification and the same per-stage budget on both sides. The result shows that later stages are consistently preferred over earlier ones, indicating that high-level guidance can repeatedly reopen development headroom after saturation and support continued multi-stage evolution.

![](images/9758ee76348fcee64b5eddec97f29d90a935899ae0f8e8653e0055c87d54dd66.jpg)  
Figure 9: Blind pairwise win rate of the later stage in each pair, with ±1 s.e.

Why pairwise. A stage that repairs a broken mechanic and a stage that adds an unused menu can move the benchmark score by the same amount. This evaluation therefore asks which build a player would rather have, and do not use the benchmark score as the outcome.

Three agents, one build each. A comparison uses three agents. Two play agents each receive one build and never see the other, nor any score, source file, round number or development history; each returns a fixed-format factual report — whether the game starts, whether the core loop can be completed, how inputs respond, what rendered, and what is missing or broken — and is explicitly forbidden to give a verdict. A third judge agent reads only the two reports, in an order randomised per comparison and recorded, and returns report 1 better, report 2 better or tie with a confidence level. The judge is instructed to weigh a concretely named defect above a general impression, and to treat anything about the review harness — how much game time a call advances, how long a build takes to boot, the action budget — as shared by both builds and therefore not evidence about either.

Equal budgets. Each play agent drives its build through the review tool of Appendix C.3 under a fixed budget of 40 actions and one reset, with wait, look and state free. Sessions are keyed by reviewer id and a reopened session resumes with the budget it has already spent, so each agent is given an id unique to its comparison and checks its own action counter immediately after opening; a session that reports a non-zero count is discarded and that comparison re-run. Two builds compared under unequal budgets yield a verdict that follows the budget rather than the build, which is why this check is part of the protocol rather than an afterthought.

Reporting. A comparison counts as a win for the later stage when the judge prefers the report belonging to it. Ties are reported separately and excluded from the win rate rather than split, since a tie states that the two reports do not separate the builds. Win rates carry ±1 standard error in Wilson form, which keeps the interval inside [0, 1] near the ends; at a few dozen games per pair these intervals are wide, and the text says so rather than reading a ranking out of overlapping bars.

## E TRAINING DATA AND EXPERIENCE INTERNALIZATION

## E.1 TRAINING CORPUS CONSTRUCTION AND CURATION

The released corpus has two parts. Base games are the generated projects themselves, with the run that produced each one: 1,998 de-duplicated games (1,446 Godot, 552 Phaser), 1,070 of them paired with the generation trajectory, plus 910 complete per-trial run trees over three generation batches. Supervision is what we train on, extracted from those runs in three stages and summarised in Table 15: S1 game generation, S2 planning, and S3 improvement. Generation trajectories are produced by GPT-5.5 and improvement trajectories by GLM-5.3-Flash. Plans are not free-written: 1,237 are distilled from the file tree of a finished game, and 870 come from an earlier plan set and are kept only where every path they name exists in the matching game.

Table 15: The released training corpus, counted from the published tables. A decision row is one assistant turn together with the history that preceded it, so decision rows repeat context and their token counts are not independent; the trajectory rows are the same recordings un-windowed and are what to count tokens over. Verified improvement rounds are those an independent post-improvement verification judged to have achieved the stated goal; the remainder are released as negatives rather than dropped.
<table><tr><td>Stage</td><td>Unit</td><td>Teacher</td><td>Godot</td><td>Phaser</td><td>Total</td></tr><tr><td>S1 generation</td><td>trajectory</td><td>GPT-5.5</td><td>1,929</td><td>284</td><td>2,213</td></tr><tr><td>S1 generation</td><td>decision row</td><td>GPT-5.5</td><td>66,312</td><td>28,097</td><td>94,409</td></tr><tr><td>S2 planning</td><td>plan</td><td>distilled</td><td>1,979</td><td>129</td><td>2,108</td></tr><tr><td>S3 improvement</td><td>round</td><td>GLM-5.3-Flash</td><td>3,730</td><td>273</td><td>4,003</td></tr><tr><td>of which verified</td><td>round</td><td></td><td>1,869</td><td>144</td><td>2,013</td></tr><tr><td>S3 improvement agent</td><td>decision row</td><td>GLM-5.3-Flash</td><td>111,302</td><td>9,930</td><td>121,232</td></tr><tr><td>of which verified</td><td>decision row</td><td></td><td>48,341</td><td>5,098</td><td>53,439</td></tr><tr><td>Games improved</td><td>game</td><td></td><td>345</td><td>43</td><td>388</td></tr><tr><td>Distinct tasks</td><td>task</td><td></td><td>2,189</td><td>322</td><td>2,511</td></tr><tr><td>Training rows</td><td>row</td><td></td><td>183,063</td><td>38,689</td><td>221,752</td></tr></table>

Trajectory Serialization. Each generation session is serialized as a single multi-turn agent trajectory containing the task specification, assistant outputs, tool calls, and tool results. Training loss is applied only to assistant-generated outputs, avoiding repeated supervision over the shared interaction context. Planning examples are extracted from the agent’s initial implementation plan before coding, while later progress updates to the same plan are discarded. A decision row keeps a pinned prefix (system prompt, task, and the plan where one exists), then as many recent whole decisions as fit, then the target decision last, within a 32k-token budget.

What is filtered out. A generation trajectory is retained only when the session produced an executable artifact; artifact existence is used rather than the agent SDK’s reported success status, which does not reliably indicate whether a runnable project was produced. Empty trajectories and duplicated tasks are removed. A plan is kept only if every file path it references exists in the finished game and its engine matches the brief. An improvement round is dropped when its session failed or its diff is empty, and is labelled positive only on an independent post-improvement verification — 1,869 of 3,730 Godot rounds and 144 of 273 Phaser attempts. Verbose tool outputs such as build logs are middle-truncated, preserving the executed command and its verdict. Whole-project diffs are compacted to the source hunks the improvement actually changed, which shrinks an improvemen example by roughly 48× at the median.

Evaluation contamination. Nothing from a baseline, ablation or evaluation run enters the corpus, and the 140 distinct benchmark task ids are excluded at build time. The published tables were rescanned against those ids after the fact: zero overlaps in all 13 configurations, over a union of 2,511 task ids.

Recording gaps are flagged, not hidden. One agent SDK stored a one-line UI summary of a tool result instead of the content the model actually received. The affected rows are kept and carry explicit flags (row\_has\_summary\_only, target\_follows\_summary\_only, tool\_results\_source, trajectory\_incomplete), so a clean training subset is a filter rather than a different release. This affects all 284 Phaser generation trajectories (20.2% of their rows have the target decision immediately after a summary) and one of the two Godot improvement arms (31.7% of multi-turn improvement rows); the other Godot arm is complete.

Data sanitization. Every row, metadata column and document was rewritten member by member. Absolute machine paths became /workspace/game, /workspace/harness or <path> (403,280 rewrites); user, host, account and internal project names became <redacted> (13,610); model-gateway hosts, internal mirrors and cluster names were replaced; e-mail addresses and nonloopback IP literals were removed; owner columns in captured directory listings were rewritten; and 26,577 configuration fields were neutralised in place, so the schema is unchanged. Tool outputs that dumped the build host’s process table were deleted rather than scrubbed (26 listings, 379 lines). Scientific content — game code and assets, rubric text, per-requirement scores, model identifiers, reasoning effort, timings — is untouched. The cleaner was verified by a second, independently written scanner carrying 47 identity and credential patterns plus a planted-leak self-test that requires every planted leak to be removed; both report zero hits.

## E.2 FINE-TUNING CONFIGURATION

We fine-tune Qwen3.8-27B (Qwen Team, 2026) using parameter-efficient supervised fine-tuning with LoRA (Hu et al., 2022) over 4-bit quantized base weights (Dettmers et al., 2023). Each agentic trajectory is treated as one long training document, allowing the model to learn planning, coding, tool-use, and improvement decisions within their original development context. Unless otherwise specified, all training configurations use the same hyperparameters; only the composition of the training corpus changes.

Training Mixture. The three variants are not three separate mixtures but three nested corpora: each adds one stage of supervision to the one before it, so ${ \bf S } 1 \subset { \bf S } 2 \subset { \bf S } 3$ and any difference between two rows of Table 18 is attributable to the stage that was added. Both engines are trained together; Table 16 gives the composition.

Table 16: The training corpus of each variant in Table 18. Each stage adds rows to the previous corpus; the cumulative column is what that variant was trained on. Tasks in the held-out brief list are filtered from every stage.
<table><tr><td>Stage added</td><td>Unit</td><td>Godot</td><td>Phaser</td><td>Rows added</td><td>Cumulative</td></tr><tr><td>S1 generation</td><td>agent trajectory</td><td>876</td><td>418</td><td>1,294</td><td>1,294</td></tr><tr><td>S2 planning</td><td>brief → file plan</td><td>216</td><td>92</td><td>308</td><td>1,602</td></tr><tr><td>S3 improvement</td><td>improvement round</td><td>509</td><td>0</td><td>509</td><td>2,111</td></tr><tr><td>Total</td><td></td><td>1,601</td><td>510</td><td>2,111</td><td></td></tr></table>

Long-Context Training. Agentic development trajectories can span tens of thousands of tokens. We therefore use a maximum context length of 57,344 tokens and distribute each sequence across four GPUs using Ulysses-style sequence parallelism (Jacobs et al., 2023). Gradient checkpointing and a fused cross-entropy implementation are used to reduce activation and output-head memory consumption.

Training Objective. For each trajectory, the original interaction context is preserved, while the supervised loss is applied only to assistant-generated outputs. This trains the model on the sequence of development decisions without treating tool observations or environment feedback as prediction targets. The full RSIGAME model is trained on the union of planning, generation, and independently verified improvement examples.

## E.3 TRAINING-DATA ABLATION

Table 18 gives every variant compared in Section 5 (Figure 4c draws +GEN and the full model).   
BASE and the full model are the Qwen3.8-27B and Qwen3.8-27B (SFT) rows of Table 1.

Table 17: Fine-tuning configuration used for experience internalization. These are the settings of the released adapter; every variant in Table 18 is trained with them and differs only in the corpus of Table 16.
<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Base model</td><td>Qwen3.8-27B</td></tr><tr><td>Adaptation</td><td>LoRA (Hu et al., 2022)</td></tr><tr><td>LoRA rank r</td><td>16</td></tr><tr><td>LoRA α</td><td>32</td></tr><tr><td>LoRA dropout</td><td>0.05</td></tr><tr><td>LoRA targets</td><td>All linear projections</td></tr><tr><td>Rank-stabilized LoRA (rsLoRA)</td><td>Yes</td></tr><tr><td>Base-weight precision</td><td>4-bit NF4 (Dettmers et al., 2023)</td></tr><tr><td>Double quantization</td><td>Yes</td></tr><tr><td>Compute precision</td><td>bfloat16</td></tr><tr><td>Maximum context length</td><td>57,344 tokens</td></tr><tr><td>Epochs</td><td>1</td></tr><tr><td>Optimizer</td><td>AdamW (Loshchilov &amp; Hutter, 2019), fused</td></tr><tr><td>Learning rate</td><td>5 × 10 -5</td></tr><tr><td>Learning-rate schedule</td><td>Cosine</td></tr><tr><td>Warmup ratio</td><td>0.03</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Per-device batch size</td><td>1</td></tr><tr><td>Gradient accumulation</td><td>None</td></tr><tr><td>Sequence-parallel degree</td><td>4</td></tr><tr><td>Hardware</td><td>4× H100 80 GB</td></tr></table>

Table 18: One-shot Godot generation by training variant. Every row is over the 140 tasks.
<table><tr><td>Variant</td><td>Mechanics</td><td>Depth</td><td>Visuals</td><td>Art</td><td>Overall ↑</td></tr><tr><td>BASE</td><td>41.2</td><td>33.4</td><td>38.6</td><td>38.3</td><td>37.07</td></tr><tr><td>+GEN</td><td>49.0</td><td>41.4</td><td>42.0</td><td>38.0</td><td>41.45</td></tr><tr><td>+GEN+PLAN</td><td>48.5</td><td>40.8</td><td>42.6</td><td>39.5</td><td>41.77</td></tr><tr><td>+GEN+PLAN+IMPROVE</td><td>56.1</td><td>47.5</td><td>49.8</td><td>44.8</td><td>48.22</td></tr></table>

## F EXTENDED EVALUATION AND CASE STUDIES

## F.1 COMPARISON WITH A MULTI-AGENT GENERATE-AND-VERIFY SYSTEM

Why this comparison. This paper does not claim that an agent can improve a game it generated — several systems do that. It claims that doing so reliably needs a controller: something that decides where the headroom is, keeps the best version, and stops. The closest public system we could run is VibeGame<sup>4</sup>, an eight-role agent team whose pipeline already contains verification and optimisation roles. If a team of that shape closed the gap on its own, a controller would be an ornament.

Setup. Forty Phaser tasks of GameCraft-Bench, the VibeGame code unmodified, all eight of its roles set to GPT-5.5, one trial per task, fully automatic, no human intervention. The builds are scored with the same verifier, rubrics and judge as every other Phaser arm in this paper.

What it shows. An agent team that already verifies and optimises scores 30.5, against 46.0 for a single generation pass on the same tasks: its loop does not convert rounds into quality. On the twelve tasks we share it reaches 23.3, where the frozen $P _ { 0 }$ of our Phaser arm is already at 50.1 and RSIGAME brings those same twelve to 62.4. The presence of verification and optimisation roles in a pipeline is not what produces sustained improvement.

Table 19: VibeGame against one-shot generation and against RSIGAME. Left: the twelve tasks shared by VibeGame’s forty and our forty-task Phaser development subset, so every column is the same twelve games. Right: VibeGame’s own forty tasks, against the one-shot reference its release quotes on them.
<table><tr><td></td><td>the 12 shared tasks</td><td>VibeGame&#x27;s own 40</td></tr><tr><td>VibeGame</td><td>23.3</td><td>30.5</td></tr><tr><td>OpenGame + GPT-5.5, one shot</td><td>50.1</td><td>46.0</td></tr><tr><td>+ Play2Code</td><td>56.4</td><td></td></tr><tr><td>+ RSIGAME</td><td>62.4</td><td></td></tr></table>

Reading it fairly. One trial per task, and a system we configured only to the extent of setting its role models; we make no claim about VibeGame under other models or a larger budget.

## F.2 QUALITATIVE EVOLUTION CASE STUDIES

We follow three games from the build a code model generates to the build we ship. Each case opens with a timeline: the official score at every third round of the 30-round autonomous run, the version the Global Quality Monitor holds, the round at which the saturation stop ends the run, and, past an axis break, the passes the loop did not run — the stage briefs written once the champion stopped being beaten, and the development passes after them. Each build on that timeline then gets a card of its own: what that round or pass found, what it changed, and the frames it is judged on, with boxes placed by eye over the screenshots, green where something arrived and red where it is still wrong. Every build of a game is driven through the same script, so its panels are comparable.

## F.2.1 HOLDING THE BEST VERSION THROUGH REGRESSIONS

![](images/970aa2d5c761c7a17c749e783c3b570c3adfa9a5e6e86f1e65514a9c9b0191c5.jpg)

![](images/0dc3933f6c59e150c4b787cc9e3049f9e73e45fa2c578c800024ef31d90dfeae.jpg)  
Figure 10: Lawn Guardians, from the generated game to the build we shipped. The score band gives the official score at every third round of the 30-round autonomous run, the version the Global Quality Monitor holds, and the star where the saturation stop delivers it at round 21, nine rounds early. The round band gives one cell per round — improvement or art — with the post-improvement verdict above it. Past the axis break are the passes the loop did not run: the stage brief written once the champion had survived three checkpoints, and the three iterations after it, which are development passes rather than loop rounds and so carry no round numbers. Each build below is opened up as a card in Figures 11–F.2.1.

Development does not end where the run does. The Global Quality Monitor’s stop criterion is about the run’s own scores, not about the game: once the champion survives three checkpoints, nothing in the loop’s evidence points anywhere, and a stage brief written from outside supplies the direction the loop no longer has. Three further passes follow it — controls, animation, and set dressing with a written soundtrack — each changing what the game is like to play rather than what it does. The six builds are shown one card each in Figures 11–F.2.1.

![](images/d9fa11e64ab02fcd4b9d2ea189c211cd30efd4454cc3780e701da376aa65f633.jpg)  
Figure 11: Lawn Guardians from the generated build to the stage brief. Each card carries what that round or pass found, what it changed, and the frames it is judged on; boxes over the screenshots are placed by eye, green where something arrived and red where it is still wrong, and every build is driven through the same script so the scenes are comparable. Base game: the generator has the rules right and the presentation wrong — the art it shipped is on disk and unused. Autonomous: twelve rounds wire that art in and add hit feedback, and the champion that emerges is never beaten. Stage 1: with the champion held for three checkpoints the loop has run out of its own evidence, and a brief written from outside — by Claude Opus 5, alongside the human’s — puts sun on the lawn and zombies inside their lanes.

![](images/ba0dba882bc0f4616bd7dfa9bcdacf0248b338a6fa43fddfcb648d4fe0ebb0fd.jpg)  
Figure 12: The three development passes that follow the stage brief. Iteration 1 is a fix no still of the board can show, captured instead as two before/after pairs under the same script: at a stretched canvas the click was transformed twice, so a plant landed a cell from the cursor, and peas stayed on the lawn after connecting. Iteration 2 replaces the title screen and the walk cycle, the one step a still frame does carry. Iteration 3 is set dressing and a rewritten soundtrack; the seed bar and verge are in the frames, and the audio is the half no figure can show.

In Lawn Guardians (Figure 10), progress is not monotone: an art round (R6) and a feedback round (R18) each lower the score, and the Global Quality Monitor adopts neither. It holds R12 (86.1, up from 61.7) until the saturation stop delivers it at round 21. Two limits also show: a verification replay can be too short to see a real fix (R26), and the official judge can reward a build whose art covers the play field (R27). Play2Code peaks at 81.5 and ends at 65.7.

## F.2.2 TWO BRIEFS ON THE SAME CHECKPOINT

Alley Brawlers is the case where the loop’s own evidence buys the scenery and loses the actors. Thirty autonomous rounds draw a lit street behind the fight — the largest single change in the run — and in the same rounds the fighters stop resolving: in the delivered build a body in motion is a torn smear. The two cancel, and the champion the saturation stop delivers scores exactly what the base game scored.

What the loop could not produce is the observation that nobody can win. Both stage-1 directors played the build and said so, and the run that followed the model’s brief moved the score for the first time. A control branch, restarted from the same checkpoint with no brief at all, reached 60.0; the briefs reached 61.5 and 62.0. A second brief, written at the stage-1 champion, then went after what still made the match unplayable — two bodies standing in the same space — and bought a readable exchange rather than a higher score.

![](images/3afeb078c7528ddddf3b4b06bafb98bfca2b14aaa82f6b0edce5c6b7a295954a.jpg)  
Figure 13: Alley Brawlers, from the generated game to the build we shipped. Score band, round band and frames as in Figure 10. The saturation stop ends the autonomous run at round 18 and delivers round 9. Past the axis break are the passes the loop did not run: two stage briefs, each written by a director who played the build in front of it, and three development passes. Stage 2 and the iterations both continue from the stage-1 champion, so they are alternatives rather than a sequence.

![](images/abf432dbd2436ca097eed9d1136c8eb90889ecf472ff96c1eaac1e41b1695cdf.jpg)  
Figure 14: Alley Brawlers from the generated build to the first brief. Cards, boxes and script as in Figure 11. Base game: a working match on a flat strip under falling squares. Autonomous: the arena arrives and the fighters break, and the delivered build scores exactly what the base game scored. Stage 1: a director who played two matches asks for a fair exchange and for a fighter that stays one intact figure; the score moves for the first time in the run.

![](images/bb4b578883e3532da6301e2a83066dd64634907f0d87ea45c007a34b71e55a55.jpg)  
Figure 15: The second brief and the development passes, both continuing from the stage-1 champion. Stage 2: a second director names spacing as the reason a player cannot act, and the exchange becomes legible from both sides. Iterations 1–2 rebuild the front end — the title becomes steps rather than a wall of text, and the HUD names and shows both fighters — and add hit sparks and damage numbers. Iteration 3 is the feel of contact: hitstop, screen shake, K.O. slow motion, a combo counter and a soundtrack.

## F.2.3 A GAME THE LOOP CANNOT IMPROVE

Block Cascade is the opposite starting point. Seven-bag randomiser, next-three preview, ghost piece, hold swap, wall kicks, line clears, top-out and a best score all work in the generated build, and thirty rounds of evidence-led improvement find nothing worth swapping to. The loop is not stuck on a defect; it has run out of anything to call one.

Restarting from that build is what pays, and the briefs are not what makes it pay: the two brief-guided branches and the control branch with no brief all reach 50.0, and what the briefs buy is arriving there at round 6 rather than round 12. The second brief then asks for stakes — a named mode, a live clock, a reward ladder — and gets them, along with a collision between the new readouts and the ones already on screen, which the pass after it lays out.

![](images/230528b1646e15061e6824470cdd2ff47bb545a626b93e61f50f0f6b2fc43471.jpg)

![](images/db65dd1ae210600614ba0000558ac7998b585bb4bd50230b51dafb059ae00692.jpg)  
Figure 16: Block Cascade, from the generated game to the build we shipped. The generated game already implements every mechanic in the spec, so the improvement rounds have almost nothing to improve: through nine checkpoints the Global Quality Monitor never prefers a later build, the saturation stop ends the run at round 9, and what it delivers is round 0 itself. Everything past the axis break follows a restart from that build.

![](images/e4866578e46ccf18ea65354e25aa08a3acddff2d0dadbc894c18c7039c0c27e6.jpg)  
Figure 17: Block Cascade from the generated build through both briefs. Cards, boxes and script as in Figure 11. Base game: every mechanic in the spec works first try, and nine rounds later the loop delivers this same build. Stage 1: the well gets a painted ground, faceted blocks and a danger line — and a control branch with no brief reaches the same score. Stage 2: a named mode and a live speed readout arrive, printed on top of the readouts already there.

![](images/57ddf325905b23d99f807f820ee11899f5c74e374fe9432ffbb2a077baade398.jpg)

![](images/60b568f51769c8986c9130181ab87a4d29a985dbdad8783f2c622b7b41d1106a.jpg)  
Figure 18: The two development passes that close the run. Iteration 1 gives every number a place of its own and a landing its feedback. Iteration 2 adds a centred result card over a dimmed board, a HOLD box that says which key fills it, and a written soundtrack in place of a 1.8-second fragment.

## F.3 PER-FAMILY RESULTS

Table 20: Per-family breakdown on Godot for the Codex + GPT-5.5 (high) generator (Table 1, first row group), by mean Overall (↑), Qwen3.8-27B judge. N is the number of tasks in the family (scored/total while scoring is in progress); families are ordered by base score; a task with no generated project scores 0. Abbreviations: +P2C Play2Code; –: not yet available on all tasks.
<table><tr><td>Family</td><td>N</td><td>Base ↑</td><td>+P2C</td><td>+RSIGAME</td></tr><tr><td>idle</td><td>4</td><td>81.9</td><td>86.8</td><td>87.5</td></tr><tr><td>sports</td><td>4</td><td>72.8</td><td>70.5</td><td>81.2</td></tr><tr><td>horror</td><td>5</td><td>67.2</td><td>71.4</td><td>75.1</td></tr><tr><td>openworld</td><td>15</td><td>59.2</td><td>57.9</td><td>62.2</td></tr><tr><td>tycoon</td><td>16</td><td>53.3</td><td>54.4</td><td>73.2</td></tr><tr><td>visualnovel</td><td>11</td><td>52.0</td><td>55.8</td><td>71.5</td></tr><tr><td>roguelike</td><td>14</td><td>46.1</td><td>38.5</td><td>56.8</td></tr><tr><td>shooter</td><td>7</td><td>46.0</td><td>49.7</td><td>64.1</td></tr><tr><td>strategy</td><td>17</td><td>46.0</td><td>46.3</td><td>61.5</td></tr><tr><td>rhythm</td><td>5</td><td>45.4</td><td>44.7</td><td>67.9</td></tr><tr><td>cardgame</td><td>5</td><td>44.2</td><td>44.3</td><td>71.3</td></tr><tr><td>puzzle</td><td>8</td><td>43.4</td><td>45.1</td><td>53.9</td></tr><tr><td>platformer</td><td>19</td><td>43.3</td><td>44.4</td><td>58.4</td></tr><tr><td>racing</td><td>4</td><td>43.3</td><td>53.4</td><td>68.4</td></tr><tr><td>simulation</td><td>6</td><td>38.1</td><td>37.7</td><td>48.8</td></tr><tr><td>All</td><td>140</td><td>50.26</td><td>50.74</td><td>64.53</td></tr></table>

Table 21: Per-family breakdown on Godot for the Kimi-K2.6 generator (Table 1, second row group).
<table><tr><td>Family</td><td>N</td><td>Base ↑</td><td>+P2C</td><td>+RSIGAME</td></tr><tr><td>idle</td><td>4</td><td>53.5</td><td>50.1</td><td>60.0</td></tr><tr><td>sports</td><td>4</td><td>45.9</td><td>57.3</td><td>64.4</td></tr><tr><td>horror</td><td>5</td><td>43.1</td><td>56.6</td><td>63.4</td></tr><tr><td>tycoon</td><td>16</td><td>36.1</td><td>40.7</td><td>46.4</td></tr><tr><td>rhythm</td><td>5</td><td>33.7</td><td>37.1</td><td>45.4</td></tr><tr><td>shooter</td><td>7</td><td>33.4</td><td>35.9</td><td>39.1</td></tr><tr><td>simulation</td><td>6</td><td>30.7</td><td>29.8</td><td>34.9</td></tr><tr><td>visualnovel</td><td>11</td><td>30.1</td><td>36.8</td><td>45.3</td></tr><tr><td>roguelike</td><td>14</td><td>29.4</td><td>35.1</td><td>44.5</td></tr><tr><td>racing</td><td>4</td><td>29.1</td><td>39.4</td><td>41.8</td></tr><tr><td>openworld</td><td>15</td><td>27.0</td><td>33.6</td><td>42.5</td></tr><tr><td>cardgame</td><td>5</td><td>26.2</td><td>23.6</td><td>48.8</td></tr><tr><td>strategy</td><td>17</td><td>24.1</td><td>33.1</td><td>47.7</td></tr><tr><td>platformer</td><td>19</td><td>23.3</td><td>29.2</td><td>41.4</td></tr><tr><td>puzzle</td><td>8</td><td>15.3</td><td>19.4</td><td>29.2</td></tr><tr><td>All</td><td>140</td><td>29.63</td><td>35.19</td><td>44.77</td></tr></table>

Table 22: Per-family breakdown on Godot for the GLM-5.3-Flash generator (Table 1, third row group).
<table><tr><td>Family</td><td>N</td><td>Base ↑</td><td>+P2C</td><td>+RSIGAME</td></tr><tr><td>sports</td><td>4</td><td>49.6</td><td>61.5</td><td>72.6</td></tr><tr><td>idle</td><td>4</td><td>39.0</td><td>59.2</td><td>68.6</td></tr><tr><td>tycoon</td><td>16</td><td>38.8</td><td>43.1</td><td>56.1</td></tr><tr><td>racing</td><td>4</td><td>34.9</td><td>36.7</td><td>53.0</td></tr><tr><td>horror</td><td>5</td><td>34.1</td><td>43.6</td><td>49.9</td></tr><tr><td>visualnovel</td><td>11</td><td>32.7</td><td>38.8</td><td>59.0</td></tr><tr><td>shooter</td><td>7</td><td>32.6</td><td>42.5</td><td>54.9</td></tr><tr><td>strategy</td><td>17</td><td>32.2</td><td>39.2</td><td>42.5</td></tr><tr><td>simulation</td><td>6</td><td>32.0</td><td>38.6</td><td>46.5</td></tr><tr><td>rhythm</td><td>5</td><td>26.3</td><td>37.2</td><td>52.9</td></tr><tr><td>cardgame</td><td>5</td><td>25.5</td><td>33.1</td><td>52.7</td></tr><tr><td>platformer</td><td>19</td><td>25.3</td><td>35.7</td><td>48.7</td></tr><tr><td>roguelike</td><td>14</td><td>24.9</td><td>37.5</td><td>40.9</td></tr><tr><td>openworld</td><td>15</td><td>19.2</td><td>28.1</td><td>43.2</td></tr><tr><td>puzzle</td><td>8</td><td>16.9</td><td>33.3</td><td>40.8</td></tr><tr><td>All</td><td>140</td><td>29.46</td><td>38.59</td><td>49.72</td></tr></table>

Table 23: Per-family breakdown on Godot for the Qwen3.8-27B generator (Table 1, fourth row group).
<table><tr><td>Family</td><td>N</td><td>Base ↑</td><td>+P2C</td><td>+RSIGAME</td></tr><tr><td>horror</td><td>5</td><td>58.7</td><td>56.6</td><td>57.0</td></tr><tr><td>idle</td><td>4</td><td>48.8</td><td>53.2</td><td>73.9</td></tr><tr><td>cardgame</td><td>5</td><td>48.3</td><td>53.1</td><td>59.3</td></tr><tr><td>sports</td><td>4</td><td>44.8</td><td>51.2</td><td>55.6</td></tr><tr><td>shooter</td><td>7</td><td>42.3</td><td>44.7</td><td>61.5</td></tr><tr><td>visualnovel</td><td>11</td><td>42.0</td><td>46.0</td><td>44.9</td></tr><tr><td>tycoon</td><td>16</td><td>41.0</td><td>52.0</td><td>62.6</td></tr><tr><td>strategy</td><td>17</td><td>39.5</td><td>35.2</td><td>44.9</td></tr><tr><td>openworld</td><td>15</td><td>34.5</td><td>39.7</td><td>36.4</td></tr><tr><td>rhythm</td><td>5</td><td>33.6</td><td>35.0</td><td>54.7</td></tr><tr><td>simulation</td><td>6</td><td>32.1</td><td>36.4</td><td>33.3</td></tr><tr><td>platformer</td><td>19</td><td>31.6</td><td>31.7</td><td>43.5</td></tr><tr><td>roguelike</td><td>14</td><td>31.0</td><td>36.7</td><td>42.4</td></tr><tr><td>puzzle</td><td>8</td><td>23.5</td><td>32.5</td><td>43.5</td></tr><tr><td>racing</td><td>4</td><td>23.3</td><td>22.7</td><td>27.9</td></tr><tr><td>All</td><td>140</td><td>37.07</td><td>40.53</td><td>47.77</td></tr></table>

Table 24: Per-family breakdown on Godot for the Qwen3.8-27B (SFT) generator (Table 1, fifth row group).
<table><tr><td>Family</td><td>N</td><td>Base ↑</td><td>Play2Code ↑</td><td>RSIGAME ↑</td></tr><tr><td>idle</td><td>4</td><td>80.4</td><td>78.6</td><td>84.1</td></tr><tr><td>sports</td><td>4</td><td>73.3</td><td>68.7</td><td>76.1</td></tr><tr><td>horror</td><td>5</td><td>60.6</td><td>62.4</td><td>71.5</td></tr><tr><td>tycoon</td><td>16</td><td>54.7</td><td>56.4</td><td>66.3</td></tr><tr><td>strategy</td><td>17</td><td>52.1</td><td>51.0</td><td>62.0</td></tr><tr><td>visualnovel</td><td>11</td><td>50.2</td><td>48.9</td><td>58.4</td></tr><tr><td>openworld</td><td>15</td><td>48.1</td><td>46.7</td><td>58.2</td></tr><tr><td>ròguelike</td><td>14</td><td>46.4</td><td>47.8</td><td>60.2</td></tr><tr><td>platformer</td><td>19</td><td>42.3</td><td>46.9</td><td>56.0</td></tr><tr><td>racing</td><td>4</td><td>41.3</td><td>38.5</td><td>52.1</td></tr><tr><td>puzzle</td><td>8</td><td>40.6</td><td>44.8</td><td>60.3</td></tr><tr><td>simulation</td><td>6</td><td>40.4</td><td>45.1</td><td>61.2</td></tr><tr><td>rhythm</td><td>5</td><td>37.4</td><td>43.9</td><td>63.8</td></tr><tr><td>cardgame</td><td>5</td><td>36.1</td><td>41.8</td><td>58.9</td></tr><tr><td>shooter</td><td>7</td><td>35.3</td><td>39.8</td><td>55.2</td></tr><tr><td>All</td><td>140</td><td>48.22</td><td>49.71</td><td>61.38</td></tr></table>

Table 25: Per-family breakdown on Phaser for the OpenGame + GPT-5.5 generator (Table 1, Phaser block, first row group).
<table><tr><td>Family</td><td>N</td><td>Base ↑</td><td>+P2C</td><td>+RSIGAME</td></tr><tr><td>idle</td><td>4</td><td>72.7</td><td>73.8</td><td>78.2</td></tr><tr><td>cardgame</td><td>5</td><td>68.9</td><td>68.9</td><td>61.8</td></tr><tr><td>visualnovel</td><td>11</td><td>68.0</td><td>68.6</td><td>73.5</td></tr><tr><td>tycoon</td><td>16</td><td>59.9</td><td>62.9</td><td>74.3</td></tr><tr><td>sports</td><td>4</td><td>57.2</td><td>59.8</td><td>64.8</td></tr><tr><td>roguelike</td><td>14</td><td>56.5</td><td>58.2</td><td>66.7</td></tr><tr><td>horror</td><td>5</td><td>53.5</td><td>52.7</td><td>57.6</td></tr><tr><td>simulation</td><td>6</td><td>51.8</td><td>57.7</td><td>55.4</td></tr><tr><td>rhythm</td><td>5</td><td>50.8</td><td>52.5</td><td>58.2</td></tr><tr><td>strategy</td><td>17</td><td>45.7</td><td>56.8</td><td>54.8</td></tr><tr><td>shooter</td><td>7</td><td>45.2</td><td>58.6</td><td>60.4</td></tr><tr><td>racing</td><td>4</td><td>39.8</td><td>60.4</td><td>52.8</td></tr><tr><td>platformer</td><td>19</td><td>38.1</td><td>43.7</td><td>45.0</td></tr><tr><td>openworld</td><td>15</td><td>32.8</td><td>41.0</td><td>39.5</td></tr><tr><td>puzzle</td><td>8</td><td>32.5</td><td>40.6</td><td>51.6</td></tr><tr><td>All</td><td>140</td><td>49.44</td><td>55.10</td><td>58.21</td></tr></table>

Table 26: Per-family breakdown on Phaser for the Qwen3.8-27B generator (Table 1, Phaser block, second row group).
<table><tr><td>Family</td><td>N</td><td>Base ↑</td><td>+P2C</td><td>+RSIGAME</td></tr><tr><td>sports</td><td>4</td><td>56.1</td><td>68.8</td><td>65.4</td></tr><tr><td>horror</td><td>5</td><td>55.6</td><td>52.9</td><td>57.6</td></tr><tr><td>simulation</td><td>6</td><td>55.3</td><td>53.3</td><td>67.8</td></tr><tr><td>visualnovel</td><td>11</td><td>53.3</td><td>54.2</td><td>52.4</td></tr><tr><td>rhythm</td><td>5</td><td>50.9</td><td>58.4</td><td>61.4</td></tr><tr><td>tycoon</td><td>16</td><td>47.5</td><td>58.9</td><td>56.7</td></tr><tr><td>cardgame</td><td>5</td><td>47.0</td><td>72.3</td><td>61.4</td></tr><tr><td>puzzle</td><td>8</td><td>42.6</td><td>44.2</td><td>39.5</td></tr><tr><td>idle</td><td>4</td><td>41.3</td><td>61.5</td><td>65.2</td></tr><tr><td>racing</td><td>4</td><td>40.9</td><td>45.3</td><td>51.6</td></tr><tr><td>strategy</td><td>17</td><td>37.3</td><td>50.0</td><td>50.6</td></tr><tr><td>platformer</td><td>19</td><td>37.1</td><td>40.2</td><td>46.2</td></tr><tr><td>roguelike</td><td>14</td><td>29.6</td><td>40.7</td><td>48.8</td></tr><tr><td>shooter</td><td>7</td><td>27.7</td><td>46.0</td><td>48.1</td></tr><tr><td>openworld</td><td>15</td><td>21.1</td><td>32.0</td><td>29.3</td></tr><tr><td>All</td><td>140</td><td>40.03</td><td>48.70</td><td>50.24</td></tr></table>

Table 27: Per-family breakdown on Phaser for the Qwen3.8-27B (SFT) generator (Table 1, Phaser block, third row group).
<table><tr><td>Family</td><td>N</td><td>Base ↑</td><td>Play2Code ↑</td><td>RSIGAME ↑</td></tr><tr><td>idle</td><td>4</td><td>73.0</td><td>75.2</td><td>78.5</td></tr><tr><td>horror</td><td>5</td><td>56.0</td><td>60.7</td><td>67.5</td></tr><tr><td>tycoon</td><td>16</td><td>53.3</td><td>62.0</td><td>66.5</td></tr><tr><td>sports</td><td>4</td><td>51.6</td><td>49.9</td><td>56.8</td></tr><tr><td>shooter</td><td>7</td><td>41.8</td><td>51.4</td><td>55.4</td></tr><tr><td>rhythm</td><td>5</td><td>46.8</td><td>61.3</td><td>66.2</td></tr><tr><td>simulation</td><td>6</td><td>46.0</td><td>50.7</td><td>60.1</td></tr><tr><td>roguelike</td><td>14</td><td>45.5</td><td>56.8</td><td>60.2</td></tr><tr><td>openworld</td><td>15</td><td>42.2</td><td>46.0</td><td>50.7</td></tr><tr><td>puzzle</td><td>8</td><td>43.8</td><td>58.7</td><td>64.8</td></tr><tr><td>strategy</td><td>17</td><td>42.9</td><td>51.3</td><td>54.9</td></tr><tr><td>cardgame</td><td>5</td><td>41.6</td><td>45.1</td><td>58.7</td></tr><tr><td>visualnovel</td><td>11</td><td>40.0</td><td>46.2</td><td>54.3</td></tr><tr><td>platformer</td><td>19</td><td>39.9</td><td>50.4</td><td>54.1</td></tr><tr><td>racing</td><td>4</td><td>32.8</td><td>39.1</td><td>49.6</td></tr><tr><td>All</td><td>140</td><td>45.14</td><td>53.15</td><td>58.53</td></tr></table>