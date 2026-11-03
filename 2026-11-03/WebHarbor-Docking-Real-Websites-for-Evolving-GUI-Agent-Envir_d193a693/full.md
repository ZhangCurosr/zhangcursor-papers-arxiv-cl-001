# WebHarbor: Docking Real Websites for Evolving GUI Agent Environments

WebHarbor Team and Contributors<sup>∗</sup>

<sup></sup> Project Page Data & Models <sup>§</sup> Code

## ABSTRACT

GUI web agents require realistic and interactive environments that remain stable during both evaluation and training. Live website evaluation suffers from changing content, access restrictions, and limited coverage of authenticated workflows, while existing synthetic environments for training often lack functional depth and visual fidelity. We propose WebHarbor, a suite of 153 executable and dockerized website environments that faithfully reconstruct the visual interfaces, content, and functional workflows of real-world web services under deterministic control. We build WebHarbor through task-driven human-agent collaboration, where environments evolve as new tasks expand their functional scope, coding agents implement the required changes, and human contributors audit visual fidelity, functional correctness, and task validity. The resulting suite comprises 3,518 tasks spanning multi-step and post-authentication workflows, where existing works often fail to cover. Beyond reliable evaluation, WebHarbor provides high-fidelity synthetic web environments for cost-efficient reinforcement learning (RL). We demonstrate that a Qwen 4B-level GUI agent trained with RL in WebHarbor generalizes to unseen tasks and performs competitively with leading open-source GUI web agents across four benchmarks. The code and data are publicly available at https://github.com/aiming-lab/WebHarbor.

## 1 INTRODUCTION

Web agents powered by large language models (LLMs) with multimodal capability interpret rendered webpages and complete tasks through browser interactions (Deng et al., 2023; He et al., 2024; Zheng et al., 2024; Qin et al., 2025). Their progress depends on environments that support reliable evaluation and training. Evaluation requires stable and reproducible outcomes, while reinforcement learning (RL) additionally requires executable state transitions and repeatable resets (Qi et al., 2025; Bai et al., 2024; Wei et al., 2025; Lai et al., 2025). Live websites offer realistic interfaces, but changing content, bot detection, and regional restrictions undermine experimental control. Moreover, live-web benchmarks rarely cover authenticated workflows, despite their importance to many everyday user requests (He et al., 2024; Xue et al., 2025). In our experiments, about 38% of Online-Mind2Web tasks were blocked by bot detection, and 22% of WebVoyager tasks were outdated due to website changes. These failures introduce uncontrolled variation into evaluation and prevent agents from learning the many workflows hidden behind authentication.

An alternative should combine realistic visual observations, functional interactions, and persistent authenticated state with offline execution and reliable resets. Achieving this balance is challenging: commercial websites encompass extensive content and functionality, making complete reconstruction impractical, while visually similar replicas may lack correct search, account management, booking, or transaction logic. Even apparently functional environments may expose shortcuts, omit plausible alternative actions, or leak answers through their interfaces, undermining task validity. The goal is to match both what agents observe and how websites respond to their actions, while ensuring reproducibility and practical utility for both evaluation and RL.

Human crafted benchmarks such as WebArena and VisualWebArena offer reproducible applications but cover very few websites (Zhou et al., 2024; Koh et al., 2024). Manually reconstructing real websites can improve fidelity, but requires substantial effort and remains difficult to scale. Recent approaches automatically synthesize diverse functional websites (Zhou, 2026; Zhang et al., 2026; Wu et al., 2026; Cao et al., 2026; Wang et al., 2026a). However, they typically emphasize newly generated worlds over task-grounded reconstructions of real websites, or offer limited multimodal content and post-authentication functionality. Table 1 summarizes this coverage gap: existing released environments do not jointly provide broad website coverage, offline execution, image-rich content, and login-based workflows. The case study in Figure 1 makes the visual-fidelity limitation concrete: the compared generated environments either omit the target website or simplify its layout, merchandising hierarchy, and product-specific imagery.

(a) AutoWebWorld  
![](images/aa0604700329fa318d0af1111cbd335607fae8019469672f9a4c9e8f623cd11d.jpg)

(b) CUA-Gym  
![](images/5810745d80efad54d64954e50fcb5c9029a4a7404b27ae85c15e6b6f63056a30.jpg)

![](images/2e35f69baf2d220c6725bec54a91c62424550174da7e6fabc4c0cd9d3b77cc73.jpg)

(d) WebHarbor  
![](images/6fbc0f5fc4fd34e17ca764e8c28c9ad34f037c9487b22501965d25fd4bd0e8df.jpg)  
Figure 1: Comparison on visual fidelity. (a) AutoWebWorld captures broad Amazon cues, but its hero and card layout depart from the origin. (b) CUA-Gym recovers the storefront structure while filling product slots with generic scenery images. (c) WebArena-Infinity provides no Amazon website, its closest page is a lightweight Handshake template without images. (d) WebHarbor combines target-like navigation, dense merchandising hierarchy, and product-specific imagery, yielding the most coherent and visually faithful environment. See more cases in Appendix D. Zoom in for details.

<table><tr><td>Method</td><td>Curation</td><td># Envs.</td><td>Offline</td><td>Images</td><td>Login</td></tr><tr><td colspan="6">Evaluation</td></tr><tr><td>WebVoyager (He et al., 2024)</td><td>Human+LLM</td><td>15</td><td>×</td><td>√</td><td>X</td></tr><tr><td>WebArena (Zhou et al., 2024)</td><td>Human</td><td>6</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Online-Mind2Web (Xue et al., 2025)</td><td>Human</td><td>136</td><td>×</td><td>√</td><td>×</td></tr><tr><td>DeepShop (Lyu et al., 2025)</td><td>Agent</td><td>1</td><td>X</td><td>√</td><td>X</td></tr><tr><td colspan="6">Training</td></tr><tr><td>WebArena-Infinity (Zhou, 2026)</td><td>Agent</td><td>13</td><td>√</td><td>X</td><td>X</td></tr><tr><td>AutoWebWorld (Wu et al., 2026)</td><td>Agent</td><td>29</td><td>√</td><td>△</td><td>△</td></tr><tr><td>CUA-Gym (Wang et al., 2026a)</td><td>Agent</td><td>98</td><td>√</td><td>△</td><td>X</td></tr><tr><td>WebHarbor</td><td>Human+Agent</td><td>153</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1: Comparison of released web agent evaluation and training environments. Images denotes image-rich website content. △ denotes generated images or partial coverage. Login requires the agent to authenticate through the interface. # Envs. follow the actual releases.

To address this gap, we introduce WebHarbor, a suite of 153 executable website environments drawn from the website spaces covered by WebVoyager and Online-Mind2Web. WebHarbor uses taskconditioned construction: target tasks specify the pages, data, and interaction workflows that each environment must support, providing a tractable scope for reconstructing large web content. Coding agents then implement the frontend, backend, database, and visual content through a standardized set of environment-construction skills. Human contributors inspect the rendered websites, execute the target workflows, and revise failures in visual fidelity, functional correctness, data realism, and task validity, where fix implementations are done by coding agents. This human-agent process produces Dockerized environments that run without access to the target services, reset to controlled initial states, and support both public information-seeking tasks and in-depth workflows behind authentication. The final resource contains 3,518 tasks covering information retrieval, account management, bookings, and transactions, and can be used as a stable benchmark or as an executable training environment.

In summary, our contributions are threefold.

(1) We introduce WebHarbor, a suite of 153 controllable web environments and 3,518 tasks with authenticated workflows and persistent state changes largely absent from live-web benchmarks.

(2) We develop a human-agent collaborative construction pipeline that combines task-conditioned website implementation by coding agents with iterative human auditing of visual fidelity, functional correctness, and task validity until approval.

(3) Experiments across three web agent benchmarks demonstrate that a Qwen 4B agent trained with RL in WebHarbor generalizes to unseen tasks and achieves performance competitive with leading open-source web agents on live web benchmarks.

## 2 RELATED WORK

GUI Web Agents Web agents have evolved from systems that operate on HTML or accessibility representations (Zheng et al., 2024; He et al., 2024) to multimodal agents that integrate webpage understanding, reasoning and planning, action grounding, and sequential decision-making to execute user instructions (Gur et al., 2024; Koh et al., 2025; Putta et al., 2024). While prior work has largely focused on learning from demonstrations (Hong et al., 2024; Cheng et al., 2024; Xu et al., 2024; Qin et al., 2025; Wang et al., 2025), recent approaches such as WebRL, WebAgent-R1, and OpenWebRL, further optimize agent policies through reinforcement learning based on environment interactions (Qi et al., 2025; Wei et al., 2025; Yang et al., 2026). These approaches motivate the need for environments that provide realistic visual observations, executable state transitions, and reproducible initial conditions. WebHarbor provides such an infrastructure for efficient training and reliable evaluation of GUI web agents across diverse website interfaces.

Web Agent Benchmarks Existing benchmarks trade off website coverage, interaction depth, and experimental control. MiniWoB++ and WebShop provide controlled settings for browser interaction and shopping (Liu et al., 2018; Yao et al., 2023), while Mind2Web expands website coverage but relies on cached snapshots that limit interactive exploration (Deng et al., 2023). Live-web benchmarks such as WebVoyager, Online-Mind2Web, and DeepShop offer realistic interactions across diverse websites (He et al., 2024; Xue et al., 2025; Lyu et al., 2025). Online-Mind2Web, even with periodic task updates, however, it remains vulnerable to website changes, bot detection, CAPTCHA, and network or geographic restrictions, while offering limited coverage of authenticated and statechanging workflows. In contrast, self-hosted benchmarks such as WebArena, VisualWebArena, and WorkArena support complex interactions under controlled settings (Zhou et al., 2024; Koh et al., 2024; Drouin et al., 2024). However, their limited website coverage, heavyweight deployments, and costly environment resets complicate repeated evaluation and online RL training. REAL provides deterministic replicas of 11 real websites with persistent state changes, but starts from pre-authenticated states (Garg et al., 2025). WebHarbor bridges these trade-offs with 153 image-rich and executable websites and 3,518 tasks, covering both information seeking and authenticated and state-changing workflows. Its lightweight SQLite-backed architecture enables fast resets, facilitating reproducible evaluation and scalable online RL training across many websites.

Synthetic Web Environments Recent work scales web agent training through environment and task synthesis. WebArena-Infinity and InfiniteWeb synthesize functional browser environments from documentation, specifications, and visual references (Zhou, 2026; Zhang et al., 2026). AutoWebWorld,

![](images/9998841b359a1dfb2b62f4b74645ba12a73495b639498f1baf82a7fef64c2715.jpg)  
Figure 2: Overview of WebHarbor. (a) Live websites provide target interfaces and workflows but introduce access barriers. (b) Target tasks scope full-stack reconstruction, then coding agents implement and evolve each environment, while human reviewers audit visual fidelity, functional correctness, and task validity until approval. (c) Dockerized environments support deterministic and fast state resets and evidence-grounded rewards for both evaluation and RL.

CUA-Gym, and WebForge further explore verifiable state transitions, task generation, and difficulty control (Wu et al., 2026; Wang et al., 2026a; Yuan et al., 2026). GUI-GENESIS and VeriEnv reconstruct real applications as executable environments for agent training (Cao et al., 2026; Chae et al., 2026). However, synthesis pipeline with coding agent alone does not guarantee visual fidelity and realistic workflow depth. In contrast, WebHarbor emphasizes task-grounded reconstruction of real websites with human-validated quality. We cover all websites from WebVoyager and Online-Mind2Web, extending task coverage to complex authenticated and state-changing workflows. These tasks guide coding agents in iteratively implementing and expanding website functionality, while human reviewers audit visual fidelity, functional correctness, and task validity. This approach yields 153 image-rich environments that combine broad website coverage with human-audited quality, supporting both reliable evaluation and scalable RL training.

## 3 WEBHARBOR

Given a target website and a set of relevant tasks, WebHarbor reconstructs an executable local environment that preserves task-relevant interfaces, content, and state transitions. We call this process docking: replacing live dependencies with local assets, seeded state, and controlled resets while retaining recognizable visual and functional behavior. Figure 2 illustrates the iterative workflow. Target tasks define the reconstruction scope, coding agents use reusable skills to implement the frontend, backend, database, and visual content, and human contributors audit the resulting environments. Human feedback guides coding agents in repairing visual and functional defects. As new tasks reveal missing functionality or shortcuts, the same process expands and refines the environments while maintaining deterministic execution and reliable resets.

## 3.1 TASK-DRIVEN DOCKING

Reconstructing an entire commercial website is impractical, given the significant scale of services such as Amazon’s product catalog or YouTube’s video library. Instead, we adopt a task-driven reconstruction process that scopes each environment to the functionality and content required by its target tasks. Each task specifies entry and terminal pages, observable entities and attributes, required browser actions, and persistent state changes. These requirements define the task-relevant support surface: the pages, routes, image content, and data needed for realistic interactions.

We begin with WebVoyager’s 15 websites and 643 tasks (He et al., 2024), whose live-web evaluation has become less reliable due to website changes. We then extend coverage to all websites in Online-Mind2Web (Xue et al., 2025), but exclude its original tasks, which primarily focus on information seeking rather than authenticated and state-changing workflows. For these websites, human contributors identify core functional areas, and an LLM expands them into candidate workflows using its knowledge of common web services. Human contributors then review and refine these tasks for functional coverage, feasibility, and interaction depth. The resulting tasks cover account management, bookings, purchases, saved items, and other persistent operations. Task-driven scoping also facilitates visual and functional fidelity by explicitly defining the observations and state transitions that must be reproduced. Beyond task completion, environments preserve natural navigation paths and surrounding content, including plausible alternatives and near-miss items that prevent trivial shortcuts. This approach bounds reconstruction effort while retaining the interaction complexity needed for evaluation and training. The full list of 153 websites is provided in Tables 19 and 20 in the Appendix. Synthesis prompts are provided in Appendix B.

## 3.2 SKILL-DRIVEN SYNTHESIS AND EVOLUTION

Task-driven docking defines what an environment should support, while skill-driven synthesis and evolution transform these requirements into an executable offline website. For each website, the coding agent first explores the target service through a rendered browser, recording its task-relevant page hierarchy, navigation paths, URL patterns, interactive controls, and visual appearance. The rendered DOM and resolved asset URLs provide references for reconstructing client-rendered pages and lazy-loaded content. Guided by these observations and reusable construction skills (see Appendix B), the coding agent builds a self-contained web application backed by SQLite and bundled with local visual assets. These assets are either downloaded from the target website or replaced with visually similar alternatives obtained from public sources. The initial reconstruction is then refined through repeated execution of target tasks. For each task, the coding agent interacts with the rendered interface and identifies the first unsupported observation or state transition. Missing pages prompt the creation of new routes and views, incomplete workflows require additional form handlers and persistent state updates, and insufficient content motivates new database records and task-relevant alternatives. After each revision, the environment is reset to its seeded state, and the complete workflow is executed again. This iterative process allows increasingly diverse and challenging tasks to expand and refine the environment until all target workflows can be completed end to end through the visible interface with the intended persistent state changes.

## 3.3 HUMAN-IN-THE-LOOP ITERATIVE REFINEMENT

Automated synthesis provides weak quality control for visual fidelity, long-horizon behavior, and task validity. Coding agents may complete target workflows while leaving surrounding interfaces visually inaccurate, functionally shallow, or artificially easy. In our pilot study of 26 websites, 13 environments exhibited visual or asset defects (e.g., images mismatched with descriptions), and 14 had broken or shallow interactions (e.g., unresponsive buttons). Such issues are often overlooked by coding agents during long-horizon environment construction but are readily identified by human reviewers. Figure 1 illustrates how fully automated reconstructions may capture coarse website layouts while simplifying functionality or omitting important visual content.

To improve the environment quality, we introduce human-in-the-loop auditing and iterative collaboration. Specifically, We organize human participation into two roles. Contributors specify target websites and high-level functional requirements and guide coding agents during construction. Desig nated reviewers then compare reconstructed environments against the target websites and execute each task end to end, checking task feasibility, intended interaction sequences, and visual and functional fidelity. Reviewers translate identified defects into actionable requests for coding agents and re-evaluate affected workflows until approval. On average, each environment undergoes 2–3 review rounds, each taking 1–2 hours. Reviewers also develop task-specific verifiers and judging rubrics that define the grading criteria for LLM-as-a-Judge (Zheng et al., 2023) or agent-as-a-Judge (Zhuge et al., 2024). All contributors and reviewers are community volunteers included in the author list.

## 3.4 ENVIRONMENT SUITE

Through approximately five months of community collaboration involving 57 contributors, iterative construction and auditing yielded 153 executable website environments with 3,518 tasks. These environments support reliable resets, image-rich observations, and complex authenticated workflows for RL training and evaluation. We construct the training set by holding out all 15 WebVoyager websites, 15 additional websites, and 20% of tasks from each of the remaining 123 websites, yielding 2,288 training tasks. The remaining tasks form the test candidate pool. To ensure the reliability of test set, we execute these candidates using five models: GPT-4.1 (OpenAI, 2025a), GPT-5.5 (OpenAI, 2026a), GPT-5.6-Sol (OpenAI, 2026b), GPT-6-Astra (OpenAI, 2026c), and Qwen3.6-35B-A3B (Qwen Team, 2026) across sizes and model capabilities. We retain only tasks for which scripted verifiers and an OpenCode agent powered by GPT-5.6-Sol (anomalyco, 2025) agree on most executions. To promote diversity, we further limit each website to two test tasks, producing a final test set of 306 tasks. Figure 3 summarizes this final split.

![](images/cfd339773b82b778ba19786de0d9b70df780023bde104ce202743e80934d8f91.jpg)  
Figure 3: Train/test split: 123 train and 30 heldout sites (inner), and 2,288 train, 306 test, and 924 unallocated tasks (outer).

<table><tr><td>Task capability</td><td>Tasks</td><td>Share (%)</td><td>Sites</td></tr><tr><td>Information/search</td><td>2,282</td><td>64.9</td><td>151</td></tr><tr><td>Account/profile</td><td>161</td><td>4.6</td><td>98</td></tr><tr><td>Content/communication</td><td>100</td><td>2.8</td><td>46</td></tr><tr><td>Transaction/booking</td><td>448</td><td>12.7</td><td>95</td></tr><tr><td>Settings/other state</td><td>527</td><td>15.0</td><td>119</td></tr><tr><td>Authentication required</td><td>1,117</td><td>31.8</td><td>135</td></tr><tr><td>Persistent state change</td><td>1,253</td><td>35.6</td><td>141</td></tr><tr><td>Cross-page reasoning</td><td>1,888</td><td>53.7</td><td>151</td></tr></table>

Table 2: Task coverage by LLM-based classification. Rows 1–5 are exclusive categories, and rows 6–8 are non-exclusive properties.

<table><tr><td colspan="7">Community composition</td></tr><tr><td>Members 57</td><td colspan="2">Affiliations 26</td><td>Sites Contrib. 153 33</td><td>Reviewers 29</td><td></td><td colspan="2">Both roles 22</td></tr><tr><td colspan="4">Project activity</td><td colspan="3">Per-site collaboration</td></tr><tr><td>Activity</td><td>N</td><td>Merged / median</td><td></td><td>Metric</td><td>Mean</td><td>Median P90</td></tr><tr><td>GitHub PRs</td><td>348</td><td>313 (89.9%)</td><td>PRs</td><td>3.18</td><td>3</td><td>5</td></tr><tr><td>HF asset PRs</td><td>192</td><td>149 (77.6%)</td><td>Participants</td><td>3.33</td><td>2</td><td>5</td></tr><tr><td>PR comments</td><td>316</td><td>132 words</td><td>Handoffs</td><td>1.49</td><td>1</td><td>3</td></tr></table>

Table 3: Community collaboration in WebHarbor. Some early-stage contributors are included in the total but may not be reflected in the role-specific counts. In Project activity, the third column reports the merged PR count (rate) for repository rows and median word count for PR comments. Handoffs count transitions between different authors within the same website.

## 3.5 DATASET ANALYSIS

We analyze WebHarbor along four dimensions. (1) Effective community collaboration: As shown in Table 3, 57 members across 26 affiliations collaboratively developed 153 website environments in five months. Substantial code contributions, cross-role participation, and iterative reviews demonstrate the feasibility of scaling environment construction through coordinated community effort. (2) Broad and deep task coverage: Figure 4 demonstrates that WebHarbor achieves both broad semantic diversity and substantial interaction depth, rather than relying on repetitive information-seeking tasks. Under a size-controlled comparison, WebHarbor exhibits the lowest median nearest-neighbor similarity (0.58 versus 0.66–0.76), with its least-similar half covering approximately 91% of websites. The dashed cross-set curve shows a median nearest-task similarity of only 0.39 to Online-Mind2Web, indicating minimal task overlap and mitigating the risk of data leakage. Moreover, Table 2 presents that the suite includes 448 transaction/booking tasks, with 31.8% requiring authentication, 35.6% modifying persistent state, and 53.7% involving cross-page reasoning, which remain largely absent from existing live-web and synthetic benchmarks. (3) Rich environment implementations: Table 4 compares WebHarbor to existing synthetic environment suites. WebHarbor leads the compared suites in total and per-environment pages, images, asset size, and source lines, with an average of 37.3 pages, 1,319.9 images, 104.8 MiB of image assets, and 6,610.6 source lines of code per environment.

<table><tr><td>Metric</td><td>WebArena-Infinity</td><td>AutoWebWorld</td><td>CUA-Gym</td><td>WebHarbor</td></tr><tr><td colspan="5">Total</td></tr><tr><td>Pages</td><td>115</td><td>636</td><td>1,401</td><td>5,711</td></tr><tr><td>Images</td><td>0</td><td>2,303</td><td>56</td><td>201,946</td></tr><tr><td>Assets (MiB)</td><td>0.0</td><td>183.3</td><td>3.2</td><td>16,030.5</td></tr><tr><td>Lines</td><td>61,536</td><td>104,074</td><td>579,474</td><td>1,011,419</td></tr><tr><td colspan="5">Mean (per environment)</td></tr><tr><td>Pages</td><td>8.8</td><td>21.9</td><td>14.3</td><td>37.3</td></tr><tr><td>Images</td><td>0.0</td><td>79.4</td><td>0.6</td><td>1,319.9</td></tr><tr><td>Assets (MiB)</td><td>0.00</td><td>6.3</td><td>0.03</td><td>104.8</td></tr><tr><td>Lines</td><td>4,733.5</td><td>3,588.8</td><td>5,913.0</td><td>6,610.6</td></tr></table>

Table 4: Environment complexity and implementation breadth. Pages, Images, Assets, and Lines denote website routes, bundled image files, bundled image size (MiB), and source lines of code.
<table><tr><td></td><td colspan="4">Model performance</td><td colspan="3">Ranking consistency</td></tr><tr><td>Benchmark</td><td>Qwen3-VL- 235B-A22B</td><td>Qwen3-VL- 4B</td><td>GPT-5.1</td><td>Orchard-4B (Peng et al., 2026)</td><td>Pairwise orderings</td><td>Spearman ρ</td><td>Pearson r</td></tr><tr><td>WebVoyager</td><td>66.4</td><td>52.6</td><td>55.9</td><td>74.1</td><td rowspan="2">6/6</td><td rowspan="2">1.000</td><td rowspan="2">0.802</td></tr><tr><td>WebHarbor</td><td>76.4</td><td>62.8</td><td>74.7</td><td>78.5</td></tr><tr><td>∆</td><td>+10.0</td><td>+10.2</td><td>+18.8</td><td>+4.4</td><td rowspan="2">(100%)</td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 5: Model performance and ranking consistency on WebVoyager and the same sites of WebHarbor. Pairwise orderings counts model pairs whose order is preserved.

(4) Consistent benchmark rankings: Table 5 shows that four agents achieve 4.4–18.8 points higher scores on WebHarbor than on live Web-Voyager, reflecting reduced interference from content drift and access restrictions. Crucially, all six pairwise agent rankings are preserved, with a Spearman rank correlation of 1.0 and a Pearson score correlation of 0.8, demonstrating strong consistency in relative agent performance across the two benchmarks. These findings highlight the feasibility of community-driven construction, the diversity and depth of the resulting environments, and their value for both reliable evaluation and scalable RL training.

![](images/ad59faf9d9e0c046c29a385f2d73b95545e8fe6682ed45a88b019861d9acf981.jpg)  
Figure 4: Task semantic diversity: nearestneighbor similarity within each suite and to Online-Mind2Web (left axis) and cumulative site coverage (right axis).

## 4 EXPERIMENTS

## 4.1 EXPERIMENT SETUP

RL training & Baseline We use OpenWebRL-4B (Yang et al., 2026) as our main baseline because it is a SOTA open-source agent trained through online RL on live websites. The synthetic environment suites WebArena-Infinity, AutoWebWorld, and CUA-Gym do not release trained checkpoints. OpenWebRL is trained on 2,102 WebGym tasks across 704 live websites (Bai et al., 2026). Our mirror-trained policy starts from the same SFT checkpoint, OpenWebRL-4B-SFT, and reuses the same multimodal multi-turn GRPO (MM-GRPO) objective and ReAct-style GUI harness. At each turn, the policy observes the current state screenshot, textual action feedback, and full interaction history, and acts through browser tools (e.g., MOUSE\_CLICK). For WebHarbor training, each instance runs in an isolated and deterministically seeded sandbox through Orchard interface (Peng et al., 2026). Table 7 measured latencies show that our environment supports efficient stateful episode turnover and timely verification suitable for large-scale RL training. To keep the training framework and agent configuration consistent, we retain OpenWebRL’s format-gated GPT-4.1 final-answer reward. We isolate the environments and tasks as the primary difference, investigating whether RL in offline synthetic environments can match RL on live websites. Appendix E gives more details.

<table><tr><td>Model</td><td>WebVoyager</td><td>Online- Mind2Web</td><td>Avg.</td></tr><tr><td>Proprietary models</td><td></td><td></td><td></td></tr><tr><td>GPT-40* (OpenAI et al., 2024)</td><td>65.1</td><td>34.6</td><td>49.9</td></tr><tr><td>Gemini-3-flash* (Google, 2025a)</td><td>85.6</td><td>44.8</td><td>65.2</td></tr><tr><td>GPT-5* (OpenAI, 2025b)</td><td>90.6</td><td>57.7</td><td>74.2</td></tr><tr><td>OpenAI CUA preview* (OpenAI, 2025c)</td><td>70.9</td><td>58.3</td><td>64.6</td></tr><tr><td>Gemini CUA preview* (Google, 2025b)</td><td>88.6</td><td>57.3</td><td>73.0</td></tr><tr><td>Open-source models</td><td></td><td></td><td></td></tr><tr><td>UI-TARS-1.5-7B* (Qin et al., 2025)</td><td>66.4</td><td>31.3</td><td>48.9</td></tr><tr><td>GLM-4.1V-9B-Thinking* (Team et al., 2026)</td><td>66.8</td><td>33.9</td><td>50.4</td></tr><tr><td>Fara-7B* (Awadallah et al., 2025)</td><td>73.5</td><td>34.1</td><td>53.8</td></tr><tr><td>MolmoWeb-4B* (Gupta et al., 2026)</td><td>75.2</td><td>31.3</td><td>53.3</td></tr><tr><td>MolmoWeb-8B* (Gupta et al., 2026)</td><td>78.2</td><td>35.3</td><td>56.8</td></tr><tr><td>4B model comparison</td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-4B-Thinking (base)</td><td>52.6*</td><td>32.0*</td><td>42.3</td></tr><tr><td>OpenWebRL-4B-SFT</td><td>60.2*</td><td>47.0*</td><td>53.6</td></tr><tr><td>OpenWebRL-4B (RL)</td><td>74.1*</td><td>67.0*</td><td>70.6</td></tr><tr><td>WebHarbor-4B</td><td>81.2</td><td>70.4</td><td>75.8</td></tr></table>

Table 6: Performance on WebVoyager and Online-Mind2Web, excluding invalid trials (e.g., re-CAPTCHA). <sup>∗</sup> marks results reported by OpenWebRL (Yang et al., 2026). Bold and underlined values are the best and second-best results in each column.

Benchmarks We evaluate generalization on four benchmarks. Online-Mind2Web contains 300 realistic tasks across 100+ live websites and provides broad coverage of sites and user domains (Xue et al., 2025). WebVoyager comprises 643 tasks on 15 popular live websites (He et al., 2024). WebArena contains complex tasks in reproducible self-hosted websites (Zhou et al., 2024), where we use a subset of 226 tasks adopted by Sun et al. (2025) and Wang et al. (2026b). To test authenticated workflows, WebArena-NL disables automatic login and provides the account credentials in the task description. We also evaluate on the WebHarbor test set, which is free from live-web disruptions and covers diverse websites.

<table><tr><td>Metric</td><td>Mean (s)</td><td>Median (s)</td><td>P90 (s)</td></tr><tr><td>Initialization</td><td>7.43</td><td>6.75</td><td>9.37</td></tr><tr><td>Reset</td><td>6.68</td><td>6.00</td><td>9.00</td></tr><tr><td>Verifier</td><td>4.04</td><td>4.00</td><td>7.00</td></tr></table>

Table 7: Latency of WebHarbor operations including overhead of data transfer and Orchard management.

Evaluation Protocol Following the evaluation protocol of OpenWebRL, we use a 1280×1000 viewport and decode with temperature 0.6, top-p 0.95, and top-k 20. Each task receives a fresh browser context and a 30-step budget except for WebArena, which uses a 50-step budget. Online-Mind2Web uses an o4-mini AgentTrek judge (Xu et al., 2025; Xue et al., 2025). WebVoyager and DeepShop use benchmark-specific GPT-4.1 judges checking both the screenshots and the final answers. WebArena uses official programmatic state validators. For WebHarbor benchmark, we use the verifier to inspect state changes and the final answer.

## 4.2 MAIN RESULTS

Unseen Benchmarks Tables 6 and 9 report four findings. (1) Consistent RL gains. In the controlled 4B comparison, WebHarbor-4B obtains the highest live-web average of 75.8, exceeding the shared SFT checkpoint by 22.2 points and the live-web-trained OpenWebRL-4B policy by 5.2 points. Its gains over OpenWebRL-4B hold across all four evaluation settings. (2) Open-source competitiveness. WebHarbor-4B achieves the strongest results among the listed open-source agents on both live-web benchmarks. (3) Complementary generalization. All 15 WebVoyager websites are excluded from WebHarbor training, the original Online-Mind2Web tasks are excluded from the WebHarbor task suite, and WebArena uses stateful environments that were developed independently of WebHarbor and do not appear in the training data. The results therefore demonstrate transfer to unseen websites, unseen task instances, and a distinct environment family. (4) Complex-workflow transfer. Table 9 breaks WebArena down by site. WebHarbor-4B leads on four of the five sites in both settings. When automatic login is disabled, its overall score drops by only 2.7 points, compared with 4.9 and 3.6 points for the OpenWebRL SFT and RL variants, and its lead widens on GitLab (30.4 vs. 19.6) and Map (25.0 vs. 15.6). Reddit is the exception, where all models stay below 12% and WebHarbor-4B solves only one of 26 tasks. The overall gain points to the authenticated workflows seen during training. These findings suggest that training in WebHarbor helps agents learn GUI policies that generalize well beyond the synthesized environments.

<table><tr><td>Model</td><td>Harness</td><td>Verifier</td><td>Agent-Judge</td><td>Consistency</td></tr><tr><td>GPT-4.1</td><td>GUI</td><td>26.4</td><td>37.9</td><td>93.5</td></tr><tr><td></td><td>OpenCode GUI</td><td>1.0</td><td>1.3</td><td>99.6</td></tr><tr><td>GPT-5.5</td><td>OpenCode</td><td>26.9 79.7</td><td>27.4 90.7</td><td>98.6 90.2</td></tr><tr><td>GPT-6-Astra</td><td>GUI</td><td>86.7</td><td>95.4</td><td>90.3</td></tr><tr><td></td><td>OpenCode</td><td>83.0</td><td>95.7</td><td>88.1</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>GUI</td><td>36.8</td><td>43.4</td><td>92.5</td></tr><tr><td></td><td>OpenCode</td><td>57.4</td><td>66.0</td><td>88.7</td></tr><tr><td>4B web agents</td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>GUI</td><td>19.7</td><td>24.5</td><td>94.9</td></tr><tr><td>OpenWebRL-4B-SFT</td><td>GUI</td><td>29.1</td><td>35.5</td><td>90.6</td></tr><tr><td>OpenWebRL-4B</td><td>GUI</td><td>19.7</td><td>25.7</td><td>91.6</td></tr><tr><td>WebHarbor-4B</td><td>GUI</td><td>30.0</td><td>41.5</td><td>87.0</td></tr></table>

Table 8: Results on the 306-task WebHarbor test set. Verifier and Agent-Judge report pass rates under different judge methods, and Cons. reports their judge agreement on trials with both verdicts.

<table><tr><td>Model</td><td>Shopping</td><td>Admin</td><td>Reddit</td><td>GitLab</td><td>Map |</td><td>Overall</td></tr><tr><td>WebArena (auto-login)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-4B-Thinking</td><td>25.5</td><td>15.8</td><td>15.4</td><td>14.3</td><td>12.5</td><td>17.3</td></tr><tr><td>OpenWebRL-4B-SFT</td><td>36.4</td><td>24.6</td><td>23.1</td><td>23.2</td><td>18.8</td><td>26.1</td></tr><tr><td>OpenWebRL-4B (RL)</td><td>29.1</td><td>21.1</td><td>7.7</td><td>23.2</td><td>9.4</td><td>20.4</td></tr><tr><td>WebHarbor-4B</td><td>32.7</td><td>33.3</td><td>26.9</td><td>25.0</td><td>21.9 1</td><td>28.8</td></tr><tr><td>WebArena-NL (no auto-login)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-4B-Thinking</td><td>30.9</td><td>12.3</td><td>7.7</td><td>8.9</td><td>12.5</td><td>15.5-1.8</td></tr><tr><td>OpenWebRL-4B-SFT</td><td>30.9</td><td>21.1</td><td>11.5</td><td>19.6</td><td>15.6</td><td>21.2-4.9</td></tr><tr><td>OpenWebRL-4B (RL)</td><td>23.6</td><td>17.5</td><td>7.7</td><td>16.1</td><td>12.5</td><td>16.8-3.6</td></tr><tr><td>WebHarbor-4B</td><td>34.6</td><td>24.6</td><td>3.9</td><td>30.4</td><td>25.0</td><td>26.1-2.7</td></tr></table>

Table 9: Success rates (%) of 4B models on WebArena by site, with and without automatic login (NL gives the credentials in the task instead). Subscripts give the drop from WebArena to WebArena-NL. Bold and underlined values are the best and second-best results in each column of each setting.

WebHarbor Benchmark Table 8 evaluates GPT-4.1, GPT-5.5, GPT-6-Astra, Qwen3.6-35B-A3B, and 4B baselines with a ReAct-style GUI harness and OpenCode equipped with GUI tools. High verifier–judge agreement supports test-set reliability. GPT-4.1 favors the GUI harness (26.4 vs. 1.0), whereas GPT-5.5 and Qwen3.6-35B-A3B benefit from OpenCode. WebHarbor-4B outperforms all 4B baselines, reaching verifier and agent-judge pass rates of 30.0 and 41.5, respectively. GPT-6-Astra exceeds 95% under both measures, indicating saturation for frontier models, yet the environments remain useful for training competitive small models, transferring gains to live websites, and synthesizing more complex computer-use tasks (Wang et al., 2026c).

## 4.3 ANALYSIS ON RL GAINS

To locate where RL training helps, we classify each task with GPT-4.1 into capability groups. Figure 5 shows that WebHarbor-4B improves over OpenWebRL-4B on every group in both benchmarks. Both overall differences are significant under paired tests (p<0.01). The largest gains fall on authentication (+13.3), multi-page navigation (+12.6), and stateful operations (+10.8) on WebHarbor, and on transactions (+21.4) and stateful operations (+10.5) on WebArena. On Online-Mind2Web, 128 of the 139 tasks valid for all three agents are information retrieval. WebHarbor-4B exceeds SFT checkpoint by 15.1 points on these tasks. The gains concentrate on authenticated and stateful workflows, which WebHarbor offers and live web benchmarks largely lack in their released task collections.

![](images/bcf5b605db7bef553940fd12cd98a3427face576222397dfd2b0b1ba7dc4125a.jpg)  
Figure 5: RL gains by task capability.

## 5 CONCLUSION

We introduce WebHarbor, a comprehensive suite of 153 executable website environments and 3,518 tasks that bring real-world web interactions into controllable offline settings. Through task-driven docking and human-agent collaboration, these environments continuously evolve to support richer functionality while preserving visual fidelity and task validity. The resulting suite offers broad website coverage, image-rich interfaces, and complex authenticated workflows, with lightweight state resets enabling reliable evaluation and efficient RL training. Our held-out test set further provides a stable benchmark for assessing GUI web agent capabilities often overlooked by existing benchmarks. Furthermore, we demonstrate that a Qwen 4B agent trained entirely in WebHarbor generalizes effectively to unseen tasks and achieves competitive performance against leading open-source web agents across four benchmarks. We believe WebHarbor can facilitate reproducible evaluation, scalable RL training, and sustained community-driven evolution of GUI web agent environments.

## LIMITATIONS

As an environment suite for GUI web agents, WebHarbor has several limitations.

(1) WebHarbor imitates target website rather than equaling it, and our claims are bounded by the proposed tasks and fidelity auditing by human reviewers. Client-heavy behavior (interactive maps, real-time tickers) is approximated or replaced with static surrogates.

(2) Despite judge consistency checks for the test set of WebHarbor, scripted verifiers and model-based judges may overlook valid alternative solutions or subtle failures. This common limitation of web agent evaluation calls for more robust evaluation protocols in future work.

(3) To validate the training value of our synthesized offline environments, we mainly compare with OpenWebRL (Yang et al., 2026) which is a SOTA open-source agent trained through online RL on live websites. We do not systematically explore alternative supervised finetuning (SFT) data, general-purpose harnesses such as OpenCode (anomalyco, 2025) and CLI-based agents, or other RL algorithms. Due to computational constraints, we train only one 4B model and leave generalization across model families and scales to future work.

## ETHICS STATEMENT

WebHarbor is developed to support open research on GUI web agent evaluation and training. Environment construction involves referencing real websites and obtaining visual assets from original services or publicly available alternatives. We acknowledge that website designs, trademarks, and visual assets may be subject to third-party rights. Our environments are intended for research and benchmarking rather than commercial replication of the original services. Researchers reusing or redistributing these materials should respect applicable licenses and intellectual property rights. By reconstructing websites as self-contained environments, WebHarbor enables agents to practice complex interactions without repeatedly accessing or modifying live services. Authenticated workflows, account management, bookings, and transactions operate within controlled local states, reducing risks associated with experiments on live websites. However, synthetic environments cannot fully capture the diversity and dynamics of real websites, including evolving content, personalized interfaces, access restrictions, and external dependencies. Strong performance in WebHarbor does not guarantee equivalent performance or safe behavior on live websites. Agents intended for real-world deployment require additional validation and appropriate safeguards. We hope that accessible and reproducible environments will support more transparent and responsible research on GUI web agents.

## REFERENCES

anomalyco. OpenCode: The open source ai coding agent, 2025. URL https://github.com/a nomalyco/opencode.

Ahmed Awadallah, Yash Lara, Raghav Magazine, Hussein Mozannar, Akshay Nambi, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Vibhav Vineet, Spencer Whitehead, and Andrew Zhao. Fara-7b: An efficient agentic model for computer use, 2025. URL https: //arxiv.org/abs/2511.19663.

Hao Bai, Yifei Zhou, Mert Cemri, Jiayi Pan, Alane Suhr, Sergey Levine, and Aviral Kumar. Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning, 2024. URL https://arxiv.org/abs/2406.11896.

Hao Bai, Alexey Taymanov, Tong Zhang, Aviral Kumar, and Spencer Whitehead. Webgym: Scaling training environments for visual web agents with realistic tasks, 2026. URL https://arxiv. org/abs/2601.02439.

Yuan Cao, Dezhi Ran, Mengzhou Wu, Yuzhe Guo, Xin Chen, Ang Li, Gang Cao, Gong Zhi, Hao Yu, Linyi Li, et al. Gui-genesis: Automated synthesis of efficient environments with verifiable rewards for gui agent post-training. arXiv preprint arXiv:2602.14093, 2026.

Hyungjoo Chae, Jungsoo Park, and Alan Ritter. Safe and scalable web agent learning via recreated websites, 2026. URL https://arxiv.org/abs/2603.10505.

Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Li YanTao, Jianbing Zhang, and Zhiyong Wu. SeeClick: Harnessing GUI grounding for advanced visual GUI agents. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9313–9332, Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.acl-long.505.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023.

Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H. Laradji, Manuel Del Verme, Tom Marty, Léo Boisvert, Megh Thakkar, Quentin Cappart, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. Workarena: How capable are web agents at solving common knowledge work tasks?, 2024. URL https://arxiv.org/abs/2403.07718.

Divyansh Garg, Shaun VanWeelden, Diego Caples, Andis Draguns, Nikil Ravi, Pranav Putta, Naman Garg, Tomas Abraham, Michael Lara, Federico Lopez, James Liu, Atharva Gundawar, Prannay Hebbar, Youngchul Joo, Jindong Gu, Charles London, Christian Schroeder de Witt, and Sumeet Motwani. Real: Benchmarking autonomous agents on deterministic simulations of real websites, 2025. URL https://arxiv.org/abs/2504.11543.

Google. Gemini 3 flash model card, December 2025a. URL https://deepmind.google/mo dels/model-cards/gemini-3-flash/.

Google. Introducing the gemini 2.5 computer use model, October 2025b. URL https://blog.g oogle/innovation-and-ai/models-and-research/google-deepmind/gemi ni-computer-use-model/.

Tanmay Gupta, Piper Wolters, Zixian Ma, Peter Sushko, Rock Yuren Pang, Diego Llanes, Yue Yang, Taira Anderson, Boyuan Zheng, Zhongzheng Ren, Harsh Trivedi, Taylor Blanton, Caleb Ouellette, Winson Han, Ali Farhadi, and Ranjay Krishna. Molmoweb: Open visual web agent and open data for the open web, 2026. URL https://arxiv.org/abs/2604.08516.

Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. A real-world webagent with planning, long context understanding, and program synthesis, 2024. URL https://arxiv.org/abs/2307.12856.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. WebVoyager: Building an end-to-end web agent with large multimodal models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6864–6890, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.371. URL https://aclanthology.org/2024.acl-long. 371/.

Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for gui agents, 2024. URL https://arxiv.org/abs/2312.08914.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks, 2024. URL https://arxiv.org/abs/24 01.13649.

Jing Yu Koh, Stephen Marcus McAleer, Daniel Fried, and Ruslan Salakhutdinov. Tree search for language model agents. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=QF0N3x2XVm.

Hanyu Lai, Xiao Liu, Yanxiao Zhao, Han Xu, Hanchen Zhang, Bohao Jing, Yanyu Ren, Shuntian Yao, Yuxiao Dong, and Jie Tang. Computerrl: Scaling end-to-end online reinforcement learning for computer use agents, 2025. URL https://arxiv.org/abs/2508.14040.

Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration, 2018. URL https://arxiv. org/abs/1802.08802.

Yougang Lyu, Xiaoyu Zhang, Lingyong Yan, Maarten de Rijke, Zhaochun Ren, and Xiuying Chen. Deepshop: A benchmark for deep research shopping agents, 2025. URL https://arxiv.or g/abs/2506.02839.

OpenAI. Introducing GPT-4.1 in the api, April 2025a. URL https://openai.com/index/g pt-4-1/.

OpenAI. Gpt-5 system card, August 2025b. URL https://openai.com/index/gpt-5-s ystem-card/.

OpenAI. Computer-using agent, January 2025c. URL https://openai.com/index/compu ter-using-agent/.

OpenAI. Introducing GPT-5.5: A new class of intelligence for real work, April 2026a. URL https://openai.com/index/introducing-gpt-5-5/.

OpenAI. GPT-5.6: Frontier intelligence that scales with your ambition, July 2026b. URL https: //openai.com/index/gpt-5-6/.

OpenAI. GPT-6 Astra: A new generation of intelligence, September 2026c. URL https://open ai.com/index/gpt-6-astra/.

OpenAI, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander M ˛adry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, Alex Paino, Alex Renzin, Alex Tachard Passos, Alexander Kirillov, Alexi Christakis, Alexis Conneau, Ali Kamali, Allan Jabri, Allison Moyer, Allison Tam, Amadou Crookes, Amin Tootoochian, Amin Tootoonchian, Ananya Kumar, Andrea Vallone, Andrej Karpathy, Andrew Braunstein, Andrew Cann, Andrew Codispoti, Andrew Galu, Andrew Kondrich, Andrew Tulloch, Andrey Mishchenko, Angela Baek, Angela Jiang, Antoine Pelisse, Antonia Woodford, Anuj Gosalia, Arka Dhar, Ashley Pantuliano, Avi Nayak, Avital Oliver, Barret Zoph, Behrooz Ghorbani, Ben Leimberger, Ben Rossen, Ben Sokolowsky, Ben Wang, Benjamin Zweig, Beth Hoover, Blake Samic, Bob McGrew, and Others. Gpt-4o system card, 2024. URL https://arxiv.org/abs/2410.21276.

Baolin Peng, Wenlin Yao, Qianhui Wu, Hao Cheng, Xiao Yu, Rui Yang, Tao Ge, Alessandro Sordoni, Xingdi Yuan, Yelong Shen, et al. Orchard: An open-source agentic modeling framework. arXiv preprint arXiv:2605.15040, 2026.

Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, and Rafael Rafailov. Agent q: Advanced reasoning and learning for autonomous ai agents, 2024. URL https://arxiv.org/abs/2408.07199.

Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Wenyi Zhao, Yu Yang, Xinyue Yang, Jiadai Sun, Shuntian Yao, Tianjie Zhang, Wei Xu, Jie Tang, and Yuxiao Dong. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning, 2025. URL https://arxiv.org/abs/2411.02337.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

Qwen Team. Qwen3.6-35B-A3B: Agentic coding power, now open to all, April 2026. URL https://qwen.ai/blog?id=qwen3.6-35b-a3b.

Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, Ben Kao, Guohao Li, Junxian He, Yu Qiao, and Zhiyong Wu. OS-genesis: Automating GUI agent trajectory construction via reverse task synthesis. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5555–5579, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.277. URL https: //aclanthology.org/2025.acl-long.277/.

V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Haochen Li, Jiale Zhu, Jiali Chen, Jiaxing Xu, Jiazheng Xu, Jing Chen, Jinghao Lin, and Others. Glm-4.5v and glm-4.1vthinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2026. URL https://arxiv.org/abs/2507.01006.

Bowen Wang, Dunjie Lu, Junli Wang, Tianyi Bai, Shixuan Liu, Zhipeng Zhang, Haiquan Wang, Hao Hu, Tianbao Xie, Shuai Bai, Dayiheng Liu, Que Shen, Junyang Lin, and Tao Yu. Cuagym: Scaling verifiable training environments and tasks for computer-use agents, 2026a. URL https://arxiv.org/abs/2605.25624.

Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Wu, Zhennan Shen, Zhuokai Li, Ryan Li, Xiaochuan Li, Junda Chen, Boyuan Zheng, LI PEIHANG, Fangyu Lei, Ruisheng Cao, Yeqiao Fu, Dongchan Shin, Martin Shin, Hu Jiarui, Yuyan Wang, Jixuan Chen, Yuxiao Ye, Danyang Zhang, Yipu Wang, Heng Wang, Diyi Yang, Victor Zhong, Y.Charles, Zhilin Yang, and Tao Yu. Opencua: Open foundations for computer-use agents. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 139756–139806. Curran Associates, Inc., 2025. doi: 10.52202/085713-4669.

Zhaoyang Wang, Yiming Liang, Xuchao Zhang, Qianhui Wu, Siwei Han, Anson Bastos, Rujia Wang, Chetan Bansal, Baolin Peng, Jianfeng Gao, Saravan Rajmohan, and Huaxiu Yao. SynthAgent: Adapting web agents with synthetic supervision. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 15730–15752, San Diego, California, United States, July 2026b. Association for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/2026.acl-long.716. URL https://aclanthology.org/2026.acl-l ong.716/.

Zhaoyang Wang, Qianhui Wu, Haorui Wang, Simon Yu, Baolin Peng, Canwen Xu, Hao Cheng, Wenlin Yao, Boyi Liu, Yite Wang, Jianfeng Gao, and Huaxiu Yao. Agent world model in the wild: Computer-use agent environment synthesis. 2026c. URL https://arxiv.org/abs/2610 .26828.

Zhepei Wei, Wenlin Yao, Yao Liu, Weizhi Zhang, Qin Lu, Liang Qiu, Changlong Yu, Puyang Xu, Chao Zhang, Bing Yin, Hyokun Yun, and Lihong Li. Webagent-r1: Training web agents via end-to-end multi-turn reinforcement learning, 2025. URL https://arxiv.org/abs/2505 .16421.

Yifan Wu, Yiran Peng, Yiyu Chen, Jianhao Ruan, Zijie Zhuang, Cheng Yang, Jiayi Zhang, Man Chen, Yenchi Tseng, Zhaoyang Yu, et al. Autowebworld: Synthesizing infinite verifiable web environments via finite state machines. arXiv preprint arXiv:2602.14296, 2026.

Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454, 2024.

Yiheng Xu, Dunjie Lu, Zhennan Shen, Junli Wang, Zekun Wang, Yuchen Mao, Caiming Xiong, and Tao Yu. Agenttrek: Agent trajectory synthesis via guiding replay with web tutorials, 2025. URL https://arxiv.org/abs/2412.09605.

Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, and Yu Su. An illusion of progress? assessing the current state of web agents. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=6jZi4HSs6o.

Rui Yang, Qianhui Wu, Yuxi Chen, Hao Bai, Wenlin Yao, Hao Cheng, Baolin Peng, Huan Zhang, Tong Zhang, and Jianfeng Gao. Openwebrl: Demystifying online multi-turn reinforcement learning for visual web agents, 2026. URL https://arxiv.org/abs/2606.02031.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents, 2023. URL https://arxiv.org/ abs/2207.01206.

Peng Yuan, Yuyang Yin, Yuxuan Cai, and Zheng Wei. Webforge: Breaking the realism-reproducibilityscalability trilemma in browser agent benchmark, 2026. URL https://arxiv.org/abs/ 2604.10988.

Ziyun Zhang, Zezhou Wang, Xiaoyi Zhang, Zongyu Guo, Jiahao Li, Bin Li, and Yan Lu. Infiniteweb: Scalable web environment synthesis for gui agent training, 2026. URL https://arxiv.org/ abs/2601.04126.

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v(ision) is a generalist web agent, if grounded, 2024. URL https://arxiv.org/abs/2401.01614.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Shuyan Zhou. Webarena-infinity: Generating browser environments with verifiable tasks at scale. shuyanzhou.com, March 2026. URL https://webarena.dev/webarena-infinity/.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pp. 15585–15606, 2024.

Mingchen Zhuge, Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, et al. Agent-as-a-judge: Evaluate agents with agents. arXiv preprint arXiv:2410.10934, 2024.

# A AUTHORS & CONTRIBUTION

Project leads<sup>†</sup> Zhaoyang Wang<sup>1</sup>, Qianhui Wu<sup>2</sup>

Core contributors\* Boyu Gou<sup>3</sup>, Demi Wang<sup>4</sup>, Jiacheng Zhang<sup>5</sup>, Peichun Hua<sup>6</sup>, Qize Li<sup>7</sup>, Shi Qiu<sup>1</sup>, Yifan Jiang<sup>8</sup>, Yifan Zhang<sup>9</sup>, Zexu Jin<sup>7</sup>

Contributors<sup>\*</sup> Baolin Peng<sup>2</sup>, Bolin Shen<sup>10</sup>, Chetan Bansal<sup>2</sup>, Dehan Kong<sup>7</sup>, Deren Lei<sup>11</sup>, Dongshen Peng<sup>12</sup>, Fengqing Jiang<sup>13</sup>, Hao Cheng<sup>2</sup>, Haoqing Wang<sup>14</sup>, Jiaming Shan<sup>15</sup>, Jicheng Wang<sup>16</sup>, Jingguang Zhou<sup>4</sup>, Jixuan He<sup>1</sup>, Kaide Zeng<sup>1</sup>, Kangyu Zhu<sup>1</sup>, Lama Moukheiber<sup>17</sup>, Mufan Qiu<sup>1</sup>, Peng Xia<sup>1</sup>, Qihan Sun<sup>14</sup>, Qingwei Lin<sup>2</sup>, Qiushi Sun<sup>18</sup>, Ruichen Zhang<sup>1</sup>, Shuqing Luo<sup>1</sup>, Si Qin<sup>2</sup>, Sicheng Fan<sup>19</sup>, Tao Xiong<sup>20</sup>, Wenlin Yao<sup>2</sup>, Xinyu Zhang<sup>6</sup>, Xinyue Zeng<sup>21</sup>, Xuanrui Li<sup>14</sup>, Xuchao Zhang<sup>2</sup>, Yian Wang<sup>22</sup>, Yichao Liu<sup>23</sup>, Yiming Liang<sup>21</sup>, Yiyang Zhou<sup>1</sup>, Yizhu Jiao<sup>24</sup>, Yuetai Li<sup>13</sup>, Yurun Chen<sup>20</sup>, Yuxuan Zhang<sup>25</sup>, Zheyuan Yang<sup>26</sup>, Zhiyuan Yao<sup>20</sup>, Zhongyang Li<sup>3</sup>, Zihao Li<sup>24</sup>, Zili Wang<sup>16</sup>, Jianfeng Gao<sup>2</sup>, Huaxiu Yao<sup>1</sup>

## AFFILIATIONS

1. University of North Carolina at Chapel Hill   
2. Microsoft   
3. The Ohio State University   
4. Carnegie Mellon University   
5. University of Michigan   
6. The Chinese University of Hong Kong   
7. WebAgentLab   
8. University of Southern California   
9. University of Delaware   
10. Florida State University   
11. Meta Superintelligence Labs   
12. Stanford University   
13. University of Washington   
14. Queen Mary University of London   
15. University of California, Santa Barbara   
16. University of California, Davis   
17. Georgia Institute of Technology   
18. The University of Hong Kong   
19. Fudan University   
20. Zhejiang University   
21. Virginia Tech   
22. National University of Singapore   
23. Nankai University   
24. University of Illinois Urbana-Champaign   
25. University of British Columbia   
26. Tongji University

## B SYNTHESIS PROMPTS

WebHarbor synthesizes each environment from observations of the target website and a task specification. Website observations define the page hierarchy, navigation, visual design, content structure, and interaction patterns. Tasks define the routes, records, account state, and state transitions required for evaluation. The synthesis process uses both sources to produce one executable environment.

Stage 1 reconstructs the interface, backend, data model, authentication flows, and initial seed state. Stage 2 defines tasks that cover the website’s main functions and include multi-step workflows. Stage 3 executes every task and adds any missing routes, data, forms, or state transitions. Stage 4 removes answer leakage, adds near-miss alternatives, expands result sets, and checks cross-field consistency. Stage 5 finalizes the deterministic seed database and validates reset behavior.

Table 10 presents the Stage 1 prompt for reconstructing the target interface, backend, assets, authentication flows, and deterministic seed state. Table 11 gives the Stage 2 prompt for defining executable task contracts with broad functional coverage, calibrated difficulty, and grounded answers. Table 12 details the Stage 3 execution loop that identifies missing routes, data, forms, and state transitions and then replays each repaired workflow. Table 13 specifies the Stage 4 checks for answer leakage, near-miss distractors, catalog breadth, and cross-field consistency. Table 14 provides the Stage 5 prompt for idempotent seeding, benchmark account state, asset pinning, and reset verification. Each prompt also states its procedure, constraints, required checks, and output artifacts.

<table><tr><td>Stage 1: Clone the target website Skill: clone-website</td></tr><tr><td>objective. Build a new executable WebHarbor mirror from the observed website, with browser-captured evidence, aligned real assets, working interactions, authenticated state, and deterministic seed data. Target site: &lt;REAL_URL&gt;. Site slug: &lt;SLUG&gt;. Scaffold and register. Run ./scripts/new_site.py &lt;SLUG&gt;. Keep the prescribed sites/&lt;SLUG&gt;/ layout and register the site consistently in websyn_start.sh, control_server.py, and Dockerfile. Derive the port from the current registry. Do not leave reports, review notes, or one-off harvest scripts inside the site directory. Observe the live site. Define a tracked snapshot contract with capture ID and time, locale, timezone, viewport, source URLs, requested and final states, and retained screenshot/DoM evidence. Drive a real Chromium browser with Playwright. Capture the post-hydration DOM, resolved image URLs, navigation hierarchy, URL patterns, listing and detail views, search, authentication, account pages, and state-changing forms. Do not use requests or curl as a substitute for rendered-page observation. Build the mirror. Harvest real target images and brand assets; record stable entity keys, upstream URLs, local paths, and SHA-256 digests.</td></tr><tr><td>Stage 2: Design task contracts Skill: design-tasks</td></tr><tr><td>objective. Create 15-20 benchmark tasks for sites/&lt;SLUG&gt;/tasks.jsonl. Treat every task as an executable contract between the natural-language instruction, the mirror's interface, its database state, and the verifier. Reuse applicable WebVoyager tasks verbatim, then add tasks grounded in the target site's real workflows. Schema.Write one JSON object per line with web_name, a unique id of the form &lt;web_name&gt;-&lt;index&gt;, the user-facing ques, the local mirror URL in web, and the real upstream_url. Keep instructions natural and include demo credentials directly when authentication is required. Do</td></tr><tr><td>not place expected answers or grading hints in the task text. Functional coverage.Cover the site's complete feature surface rather than repeating one search flow. Include search and filtering, browsing and detail inspection, account management, saved state, history, comparison, and the site's characteristic state-changing operations such as cart, checkout, booking, messaging, review, subscription, or collection management. Use several tasks from each major functional area. Difficulty. Include easy tasks of two or three steps, medium tasks of four to six steps, and at least 3-5 hard tasks requiring five or more steps. Hard tasks should require cross-page evidence, visual comparison, constraint composition, arithmetic or ranking, or a persistent action followed by confirmation. Include at least two genuine disambiguation tasks for which the seeded account contains multiple plausible items and the agent must request clarification instead of quessing. Grounding. Ensure the database contains a stable, page-specific</td></tr><tr><td>answer for every non-disambiguation task. Natural searches must return at least six plausible results, including near misses. The answer must not be exposed in a result-card title, count label, snippet, sort order, or unique single result when the task is intended to require detail-page reasoning. Verification.Hand-walk every task through the rendered mirror with Playwright. Confirm that all required routes and controls are</td></tr><tr><td>Stage 3: Task-driven environment evolution</td></tr><tr><td>objective. Execute every task in sites/&lt;SLUG&gt;/tasks.jsonl and evolve the mirror until each workflow can be completed naturally through the rendered interface. Preserve the target site's interaction conventions and the byte-identical reset invariant while adding missing capabilities. fields, or the DOM seen by the agent. Task loop. For each task, begin from the deterministic seed, perform</td></tr><tr><td>Browser execution. Use Playwright with a real Chromium browser for every search, navigation, form submission, login, and state verification. Save evidence at the important steps. Do not substitute curl, requests, Flask's test client, or static template inspection for browser interaction; those methods do not expose client-side validation, route guards, asynchronous redirects, hidden</td></tr><tr><td></td></tr><tr><td>the instruction end to end, and identify every missing route, template, database record, form handler, validation rule, or state transition. Extend app.py, Jinja templates, and seed_data.py only as required by consistent site behavior. Rebuild and replay the complete workflow after each change. Persistent actions must remain visible after navigation and reload, and unrelated account state must remain unchanged. task-answer leaks, precomputed answers, unique obvious targets, or result ordering that reveals the solution. Exercise malformed and and back-navigation. Replace exact-match-only search with scored</td></tr><tr><td>Detect shortcuts. Inspect the rendered search and listing pages for empty forms, partial queries, multiple detail pages, login persistence, matching where needed. Add realistic category-adjacent records when more than half of the returned items satisfy all task constraints. Data discipline. Gate every seed function at the function level.</td></tr><tr><td>Stage 4: Environment hardening</td></tr><tr><td>objective. Audit every executable task along four dimensions: answer de-leaking, near-miss distractors, catalog breadth, and cross-field consistency. The environment must preserve task feasibility while preventing an agent from succeeding through a visible answer, a single obvious result, or inconsistent data. A. De-leak answers. Search with the task's natural query and inspect the rendered DoM and screenshot before opening any detail page. Remove constraint values from item names, answer text from snippets, precomputed comparisons, visible list counts, question-shaped headings, pinned answer callouts, and algorithm-revealing labels. Check all known leak patterns, including fuzzy search that floods an answer bucket, a sort order that places the target first, and a pre-curated lookup table that reproduces the task wording. Move required evidence into consistent specs, descriptions, histories, or detail views. B. Add near misses. For each task, retain 1-3 full matches and add at least five plausible items that match the category but fail exactly one constraint, such as price, size, color, date, brand, location, availability, or required feature. Keep the full-match density at or below 50%. The alternatives must look realistic and support meaningful comparison; do not add arbitrary noise.</td></tr><tr><td>Stage 5: Finalize the seed database Skill: seed-database</td></tr><tr><td>objective. Finalize the deterministic seed database and the asset side of the mirror. The accepted runtime state must survive ordinary restart, return exactly after reset, support realistic search and authenticated histories, and remain pinned to the matching heavy assets. Idempotent seeding. Audit every seed_*() function in app.py and seed_data.py. Each function must return before any insert, update, or commit when its data already exists. Per-row guards are insufficient because a no-op commit can still change SQLite metadata. Boot from the proposed seed, run all seed functions again, reset, restart, and compare the complete runtime and seed database bytes after each operation. Benchmark state. Seed the four standard users alice.j@test.com, bob.c@test.com, carol.d@test.com, and david.k@test.com with password TestPass123!. Give each account realistic profile data and the site-appropriate histories needed for authenticated tasks, including several saved items, cart or bag entries, prior orders or bookings,</td></tr></table>

Table 10: Skill prompt used in Stage 1 to construct an executable clone of the target website.

Table 11: Skill prompt used in Stage 2 to define executable task contracts for the cloned website.

Table 12: Skill prompt used in Stage 3 to evolve the environment through complete task execution.

Table 13: Skill prompt used in Stage 4 to strengthen environment fidelity, consistency, and task difficulty.

Table 14: Skill prompt used in Stage 5 to produce a deterministic and resettable seed database.

<table><tr><td>Website</td><td>ID</td><td>Workflow</td><td>Authenticated task</td></tr><tr><td rowspan="2">Airbnb</td><td>6</td><td>Compare and book</td><td>Use dana.k@test.com / TestPass123! to help me choose from my &#x27;Winter cabins&#x27; wishlist. Inspect both saved cabins, comparing their nightly rates, ratings and hot-tub amenities; check the second cabin&#x27;s top review theme and count as well. Book the first saved cabin for two guests using its displayed dates, and report the selected name, booked dates, total and confirmation code alongside the comparison.</td></tr><tr><td>14</td><td>Compare and book</td><td>Help me book a New York cultural tour for three guests using bob.c@test.com / TestPass123!. Compare the two cheapest Cultural tours by name, price per guest, minimum age and activity level. Choose the cheaper tour and book its first available date, then report the comparison, booked date, total and confirmation code.</td></tr><tr><td rowspan="2">Carvana</td><td>7</td><td>Appraise and accept</td><td>Help me understand the demo valuation for VIN 1HGCM82633A004352 before accepting an offer. Compare Good and Excellent condition at 45,000 miles, and Good condition at 60,000 miles, reporting each simulated offer and the differences. Log in as bob.c@test.com with password TestPass123! and accept the 45,000-mile Good-condition offer. Confirm its saved offer code and amount.</td></tr><tr><td>15</td><td>Reconcile and shortlist</td><td>I want to compare another Civic with the one I already ordered Log in as alice.j@test.com with password TestPass123! and report my scheduled order&#x27;s vehicle, monthly payment, term, APR and down payment. Find the first 2023 Civic in the Honda Civic results and estimate its payment with $1,500 down over 60 months using Great credit. Save that alternative without cancelling my order, and report its price, amount financed, payment and the final saved-car list.</td></tr><tr><td rowspan="2">Compass</td><td>7</td><td>Curate and share</td><td>Log in as david.kim@test.com (password webharbor123). Create a new collection called &#x27;Austin top picks&#x27; and add exactly these two homes: 3305 Dolphin Drive and 1036 Liberty Park Drive, Unit 38A. Open the collection&#x27;s share page to confirm both homes appear, then report the share token.</td></tr><tr><td>20</td><td>Submit seller inquiry</td><td>Log in as alice.j@test.com (password webharbor123). Read the name, email, and phone currently shown on your account overview. Use those contact details to submit exactly one Sell With Us inquiry for ZIP code 94107 in this local Compass environment. Read the confirmation, then report its reference number and the submitted name, email, phone, and ZIP. Keep your account profile unchanged.</td></tr></table>

Table 15: Authenticated and multi-step tasks from five WebHarbor environments (1/2, continued in Table 16). Each task requires authentication, a persistent state change, and cross-page reasoning.

## C TASK CASE STUDIES AND BENCHMARK COMPARISON

Tables 15 and 16 present ten authenticated tasks from Airbnb, Carvana, Compass, Thumbtack, and YouTube. Each task requires cross-page evidence, an authenticated action, and a persistent state change whose result the agent must then verify through the rendered interface.

The examples cover booking and reservation changes on Airbnb. They cover vehicle research, financing, and saved inventory on Carvana, as well as property collections and seller inquiries on Compass. The Thumbtack examples cover service requests, hiring, reviews, and messages. The YouTube examples cover subscriptions, comments, and Watch Later state. These workflows test whether an environment implements account-dependent operations.

<table><tr><td>Website Thumbtack</td><td>ID 12</td><td>Workflow Review and</td><td>Authenticated task</td></tr><tr><td rowspan="2"></td><td>14</td><td>message Quote, hire, and</td><td>Log in with the demo account (email: alice.j@test.com, password: TestPass123!). My house cleaning project is finished but I never left a review. Open the project, note the hired pro&#x27;s price and their response note, and check their profile&#x27;s current reviews. Then leave them a 5-star review saying they were thorough, including the word &quot;spotless&quot;. Reopen their profile to confirm your review is the most recent one, and message them asking whether they could return for a move-out clean next month — follow up asking whether they&#x27;d bring their own supplies. Report both replies. Log in with the demo account (email: bob.c@test.com,</td></tr><tr><td></td><td>review</td><td>password: TestPass123!). Ants have invaded my kitchen. Check the cost guide for what an exterminator typically runs, then find the exterminators who are Top Pros and request quotes for indoor ant treatment in zip 98101 within a week, describing the problem. Hire the responder with the most reviews, mark the project complete, and leave them a 5-star review mentioning the ants. Confirm on their profile that your review shows, and tell me their name, their quote, and the guide&#x27;s typical range.</td></tr><tr><td rowspan="2">YouTube</td><td>2</td><td>Subscribe and save Comment and</td><td>Sign in as carol.d@test.com using password TestPass123!. Open the AI Uncovered channel and inspect its public playlists and uploads. Open its most-viewed upload, subscribe to the channel and save that upload to Watch Later. Confirm the saved video and the new channel subscription on your account page.</td></tr><tr><td>18</td><td>verify</td><td>Sign in as alice.j@test.com using password TestPass123!. Find Top 17 New Technology Trends That Will Define 2026 and open its watch page. Add the comment &quot;Insightful breakdown of trend priorities.&quot;, save the video to Watch Later, then verify the exact new comment and the saved video in Watch Later.</td></tr></table>

Table 16: Authenticated and multi-step tasks from WebHarbor environments (2/2, continued from Table 15).

## C.1 TASK EXAMPLES ACROSS BENCHMARKS

Tables 17 and 18 place two released tasks from each suite side by side. The task instructions show the operations evaluated within an episode and the account state available to the agent.

WebArena provides a strong controlled comparison. Its examples combine conditional reasoning, cross-site information gathering, and persistent writes. WebArena begins each episode from an authenticated browser state and evaluates subsequent account-dependent operations.

Online-Mind2Web and WebVoyager provide broad coverage of live websites, while their released tasks concentrate on interactions that are publicly accessible and require no authentication. All 300 Online-Mind2Web records are marked as not requiring login and include no stored session. Among WebVoyager’s 643 released instructions, one explicitly asks the agent to sign in with credentials. The selected tasks illustrate their main scope: retrieval, filtering, and comparison over public content.

The synthesized suites also contain complex state-changing tasks. CUA-Gym supplies task-specific initial state, and WebArena-Infinity starts each generated application from seeded state. AutoWeb-World contains two explicit-login instructions among 4,322 unique released instructions. These systems demonstrate scalable generation of stateful interactions, but explicit authentication is rare.

WebHarbor systematically includes login as part of task execution. The selected tasks require the agent to enter credentials, access account-specific state, complete multi-page actions, persist changes, and verify the final state from the interface. Under our fixed GPT-4.1 audit protocol, 1,117 of 3,518

<table><tr><td>Benchmark / method</td><td>Representative released tasks</td></tr><tr><td>WebArena (Zhou et al., 2024)</td><td>Task 1. Go to the merge request on verification functions I have to review, find if the author of the merge request responded at the end, and reply &quot;Thank you&quot; if he did. Otherwise remind him with a simple @. Task 2. Create a folder named real_space in gimmiethat.space repo. Within it, create a file named urls.txt that contains the URLs of the 5 most recent posts from the space?</td></tr><tr><td>Online- Mind2Web (Xue et al., 2025)</td><td>Task 1. Find the weight of baggage allowance for economy class on Qatar Airways. Task 2. Find the cheapest used Honda Civic that meets all the following criteria: under $25,000, fewer than 30,000 miles, includes Apple CarPlay, adaptive cruise control, and blind-spot sensors.</td></tr><tr><td>WebVoyager (He et al., 2024)</td><td>Task 1. Find a recipe for a vegan smoothie bowl on Allrecipes that includes bananas and leaves, has more than 20 reviews, and a rating of at least 4 stars. Provide a list of ingredients, preparation time, and a summary of the recipe steps. Task 2. Compare business class flight options from Lisbon to Singapore for a one-way trip on March 15, 2024, select one of the flights and see which websites offer its booking options. Which one is the cheapest.</td></tr><tr><td>CUA-Gym (Wang et al., 2026a)</td><td>Task 1. Check all open PRs across our repos and report on CI status. For any PR with failing checks, add a comment tagging the author: &#x27;@ {author} — CI is failing on this PR. Please check the {check_name} step. Run details: {detail }&#x27;. Only comment on PRs that don&#x27;t already have a CI-failure comment. Task 2. I&#x27;m preparing for a collaboration announcement. Create a new post with image https://picsum.photos/800/800?random=700, caption &#x27;Big announcement coming soon! Stay tuned #collaboration&#x27;, location &#x27;Downtown Studio&#x27;. Then send this post to the group chat with Alice and Bob, and also send a text message saying &#x27;Excited to share this with you both!</td></tr></table>

Table 17: Representative released tasks from controlled, live, synthesized, and WebHarbor suites (1/2, continued in Table 18).

<table><tr><td>Benchmark / method</td><td>Representative released tasks</td></tr><tr><td>WebArena- Infinity (Zhou, 2026)</td><td>Task 1. Approve the pharmacy&#x27;s Atorvastatin-to-Rosuvastatin substitution, then prescribe Rosuvastatin 10mg tablet at bedtime for hyperlipidemia — qty 30, 5 refills, send to CVS #4521. Task 2. Create a new invoice for Wellington &amp; Partners in NZD for 16 hours of consulting, and send it.</td></tr><tr><td>AutoWebWorld (Wu et al., 2026)</td><td>Task 1. Book a flight from London Heathrow to New York JFK on March 17, 2028 in economy class, find American Airlines options, select the standard seat with standard baggage, fill out booking with first name Alex, last name Johnson, contact email alex.johnson@example.com, card number 4111111111111111, review and confirm the booking, then view the trip summary Task 2. Log into my account using email user@example.com and password SecurePass123, then go to payment methods, select PayPal, authorize payment with PayPal email paypaluser@example.com</td></tr><tr><td>WebHarbor</td><td>Task 1. Sign in as alice.j@test.com using password TestPass123!. Find Top 17 New Technology Trends That Will Define 2026 and open its watch page. Add the comment &quot;Insightful breakdown of trend priorities.&quot;, save the video to Watch Later, then verify the exact new comment and the saved video in Watch Later. Task 2. Log in as david.kim@test.com (password webharbor123). Create a new collection called &#x27;Austin top picks&#x27;and add exactly these two homes: 3305 Dolphin Drive and 1036 Liberty Park Drive, Unit 38A. Open the collection&#x27;s share page to confirm both homes appear, then report the share token.</td></tr></table>

Table 18: Representative released tasks (2/2, continued from Table 17).

WebHarbor tasks are classified as authentication-dependent. Among them, 457 also require a state change and cross-page reasoning. Authentication is therefore evaluated as part of the task workflow.

(a) AutoWebWorld  
![](images/2589e3b7df060b1758bb42b8a77ff71da9811b73456e6f4a9292f0ba8ac68dca.jpg)

(b) CUA-Gym  
![](images/a40a58035c59e9e3bb7e10119be0e4f4c751757886db6f2f8eccf7116f0e941e.jpg)

![](images/c6171cb17c907c99f5ca723c42a381a8eee3c1dd81380952f34677cf2ff09ee2.jpg)

(d) WebHarbor  
![](images/ae248ff56cceecfd278553ddb1a00b0ddfdc3845b25914fc421c4a35a6afd0e3.jpg)  
Figure 6: Comparison of YouTube-related environments. (a) AutoWebWorld uses a welcome template with placeholder recommendations. (b) CUA-Gym reproduces the video-feed structure with generic thumbnails and irrelevant images. (c) WebArena-Infinity has no YouTube website. Its closest YouTube-specific page is an image-free Linked Services template in the Gmail Accounts & Contacts environment. (d) WebHarbor provides target-like navigation, video-specific thumbnails, channel identity, and engagement metadata such as view counts.

## D ENVIRONMENT COVERAGE AND VISUAL EXAMPLES

A GUI agent observes the rendered interface, including navigation, images, layout hierarchy, and metadata. Figure 6 compares these properties across four YouTube-related environments. The comparison covers the browsing layout, video thumbnails, channel information, duration labels, and engagement metadata available to the agent.

Figures 7 and 8 show representative pages from 30 WebHarbor environments across commerce, travel, media, entertainment, developer tools, and information services. The gallery records the variation in page structure, content density, imagery, navigation, and controls covered by the suite.

## D.1 ENVIRONMENT INVENTORY

Tables 19 and 20 list all 153 WebHarbor environments and their source-benchmark website coverage. Online-Mind2Web groups websites by domain and reports 136 websites. We separate services that share a domain into individual environments, yielding 147 environments from Online-Mind2Web. Adding the 15 WebVoyager environments, nine of which are shared, gives 153 in total.

![](images/d9f3114133ba0cf21c0eac6773d74840d966a7c4d7fbc82f710dd27132846c6e.jpg)

![](images/08dc27cd846bec604d056fb078aef621de51dfe5d7b22c3af836e105e3414e95.jpg)

![](images/a18478b17fdea629e15eb53c65fc6413dbd076bb36cd4d64ee0f5ed17638c3de.jpg)

![](images/369cd0c05b53cecf86a42bbafdbb399d5c83dacb5e8f1e4160aa1c51d0a18881.jpg)

![](images/9fba67d9b555b0dece91fce4c3d915e106a5a941317645ce0323cd7d3917e951.jpg)

![](images/58e6a60d50484b0b2f5daf55c8fed4b5d89c17a9ce7d6bbd393d8cc95cf164d3.jpg)

![](images/f5b233f5096fb41231e203a20f2220f2e87b5b237ceb17ab4d2e4fdb6c55b9b7.jpg)

Allrecipes  
![](images/0b0276e3f61fffaa7cb7d1ddd1c89a898be72b9b4e2969e222060dfdb9f59acd.jpg)  
ArXiv

Apple  
Amazon  
![](images/2be33ba982d79f8fcfb59dcf7d279af31074c3854777a00b3c3a436454ce0f30.jpg)

![](images/bba91e2c48d389809832ff2cd7d1b83e3e06c08e052ce873be87eb6e91df4cc6.jpg)

BBC News  
![](images/c0522252e0cd328b01027436b0cfdac05e8530f70f1ba2f1984af4a260293bfb.jpg)  
Booking

![](images/f15b85d71c2d8edef175ca218de0f7c2312a619ff062a383e7134ba594d0218b.jpg)

![](images/34e2d661f90ca2321537f0931bfb2abfbbf4e1d89b6daa625a2705bdfc4be65c.jpg)

![](images/ef1ce667bdcd5f1427cccfe291df39ecef975be0c3542a0a9f1c47f364500f98.jpg)

Cambridge Dict.  
![](images/cb2d866906b98102e1248b834753e5f3f694f733b62479fc6dea3902d7959fee.jpg)  
Coursera

![](images/b0b73406be080062de011a30967634169a642e9c2a09308d9aaadd3be71796ee.jpg)

![](images/5b5e64d1717d54e476e4f9fc352471de91cdd2d178545116706d3ba4f1325ec9.jpg)

ESPN  
![](images/9c1e3241c924df4f94a244bfec016b45c009ef1841639ed37f7b762f90fa8e1f.jpg)  
GitHub

![](images/b4a61cadb355f1fee6c5ece4d2a375d8d431b68d423b3b9363ecc0d2e4a9f7a4.jpg)

![](images/e41e47c1bb2a90082e5e017e986674d9964539d37b37e7ed0f7a4dfbf7227ed7.jpg)  
Google Search

![](images/bca1eb14fcc3627147b98b9f81e55744aaf84c19715ea3a78be3019280d2fdb0.jpg)  
Hugging Face

![](images/3aca83b4aa58fd5b93d3126b30377332d60049aabe1d7f9b4fc05176f65c0d7e.jpg)  
Wolfram Alpha  
Figure 7: Representative pages from 30 WebHarbor environments across commerce, travel, media, entertainment, developer tools, and information services (1/2, continued in Figure 8). The examples show the page structures, imagery, navigation, metadata, and controls implemented across the suite.

![](images/47fac06ae34565751555c72e3b083e2d2a0df1e95a1bfc1458b83371a8a53098.jpg)

![](images/d6a94ba2ea3c2cebb3900457d75ad6486fc31a6ad563533d0ef83edfb88f5d39.jpg)  
Carnival Cruise

![](images/9dbcf47e2b1840bd08484ccc11782740fde8f4b0b3a0b8d7bb3a535a535c0702.jpg)

![](images/11270c7fdfc506b19766371403fc465bd9e63de53e1398a112fc9fd592bea7d0.jpg)  
League of Legends

![](images/4d0f19cf118650b350bad2362edcc3a690b9c5b52924e19c59f314a2aab58f08.jpg)  
Imgur

![](images/286baab464d18379ae9219bda7468c11b430343b0b87c51851b80260b6c87030.jpg)

![](images/482f67f862dd3c3606c8d252d9a8b01875c7f43a000b04c8e0d3f6d9f7ea37bb.jpg)  
SoundCloud

![](images/241c24c852fa4e604248be63a9149e6f326320347939911008eb3b3914aa1c48.jpg)  
Ticketmaster

![](images/262116255510846f53d9d1a75307f78163a72d2703a349ca7ff9ccc6f71c962f.jpg)  
Tumblr

![](images/8fa5a3be2829466e6818ecab3363bc833b8e3d2046689eddb9dd13e8d8f182f6.jpg)

![](images/3f21fe823b9d63b6277d2bf7fda5a41922d9e436d2a52c6952b4a2a21d89b4be.jpg)

![](images/bbacdbe427e6f3530b94b567f9ab4be19bbb08230a1ce7bf9935e11a23d47291.jpg)

![](images/0c195bfcbadc9e3e6939ea417b74bb2083e25c6d43fa832d602a90eb5e81f85a.jpg)

![](images/a81f8f497e3f7dd11f35e4f85c03d62b2e4b3eb2b5663987b7694311e3762dd1.jpg)  
Wine Access

![](images/bc4e96f36a077282e6d39dcf96db860bde8681d9415b8a4d9ae2526b74a5c9d2.jpg)

![](images/1c48aca09e420abc047bdcb2b5ce1fe8a1615c4067ddbfec14f3756579b7615d.jpg)  
Figure 8: Representative pages from WebHarbor environments (2/2, continued from Figure 7).

<table><tr><td>Environment</td><td>WV</td><td>OM2W</td><td>Environment</td><td>WV</td><td>OM2W</td></tr><tr><td>4shared</td><td></td><td></td><td>Coolmath4Kids</td><td></td><td></td></tr><tr><td>9GAG</td><td></td><td></td><td>Coursera</td><td></td><td></td></tr><tr><td>AccuWeather</td><td></td><td></td><td>Craigslist</td><td></td><td></td></tr><tr><td>Adopt a Pet</td><td></td><td></td><td>CVS</td><td></td><td></td></tr><tr><td>Airbnb</td><td></td><td></td><td>dblp</td><td></td><td></td></tr><tr><td>AKC</td><td></td><td></td><td>Dillards</td><td></td><td></td></tr><tr><td>Allrecipes</td><td></td><td></td><td>Discogs</td><td></td><td></td></tr><tr><td>Amazon</td><td></td><td></td><td>Disney</td><td></td><td></td></tr><tr><td>Amazon Jobs</td><td></td><td></td><td>Drugs.com</td><td></td><td></td></tr><tr><td>America&#x27;s Health Rankings</td><td></td><td></td><td>ESPN</td><td></td><td></td></tr><tr><td>American Express</td><td></td><td></td><td>Eventbrite</td><td></td><td></td></tr><tr><td>Amtrak</td><td></td><td></td><td>Fandom</td><td></td><td></td></tr><tr><td>Apartments.com</td><td></td><td></td><td>FedEx</td><td></td><td></td></tr><tr><td>Apple</td><td></td><td></td><td>FlightAware</td><td></td><td></td></tr><tr><td>ArXiv</td><td></td><td></td><td>FOX Sports</td><td></td><td></td></tr><tr><td>B&amp;H Photo</td><td></td><td></td><td>GitHub</td><td></td><td></td></tr><tr><td>BabyCenter</td><td></td><td></td><td>Google Finance</td><td></td><td></td></tr><tr><td>Backcountry</td><td></td><td></td><td>Google Flights</td><td></td><td></td></tr><tr><td>Bandcamp</td><td></td><td></td><td>Google Map</td><td></td><td></td></tr><tr><td>BBC News</td><td></td><td></td><td>Google Search</td><td></td><td></td></tr><tr><td>Best Buy</td><td></td><td></td><td>Google Shopping</td><td></td><td></td></tr><tr><td>Better Business Bureau</td><td></td><td></td><td>GOV.UK</td><td></td><td></td></tr><tr><td>Birkenstock</td><td></td><td></td><td>Healthgrades</td><td></td><td></td></tr><tr><td>BoardGameGeek</td><td></td><td></td><td>Healthline</td><td></td><td></td></tr><tr><td>Booking</td><td></td><td></td><td>Huggingface</td><td></td><td></td></tr><tr><td>CA.gov</td><td></td><td></td><td>iclr</td><td></td><td></td></tr><tr><td>Cambridge Dictionary</td><td></td><td></td><td>IGN</td><td></td><td></td></tr><tr><td>CarMax</td><td></td><td></td><td>IKEA</td><td></td><td></td></tr><tr><td>Carnival Cruise</td><td></td><td></td><td>IMDb</td><td></td><td></td></tr><tr><td>Cars.com</td><td></td><td></td><td>Imgur</td><td></td><td></td></tr><tr><td>Carvana</td><td></td><td></td><td>Instructure</td><td></td><td></td></tr><tr><td>Cboe</td><td></td><td></td><td>IRS Refund Tracker</td><td></td><td></td></tr><tr><td>CBP.gov</td><td></td><td></td><td>JCPenney</td><td></td><td></td></tr><tr><td>Chase</td><td></td><td></td><td>Kaggle</td><td></td><td></td></tr><tr><td>Chess.com</td><td></td><td></td><td>Kelley Blue Book</td><td></td><td></td></tr><tr><td>Chronicle Jobs</td><td></td><td></td><td>LandWatch</td><td></td><td></td></tr><tr><td>CoinMarketCap</td><td></td><td></td><td>League of Legends</td><td></td><td></td></tr><tr><td>Compass</td><td></td><td></td><td>Macy&#x27;s Wine Shop</td><td></td><td></td></tr><tr><td>Cookpad</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 19: Provenance of the 153 WebHarbor environments (1/2, continued in Table 20). A bullet under WV or OM2W marks that the target site appears in the corresponding WebVoyager or Online-Mind2Web source task set. Memberships are non-exclusive: the suite covers 15 WebVoyager and 147 Online-Mind2Web sites.

<table><tr><td>Environment</td><td>WV</td><td>OM2W</td><td>Environment</td><td>WV</td><td>OM2W</td></tr><tr><td>Marriott</td><td></td><td></td><td>Statista</td><td></td><td></td></tr><tr><td>Mayo Clinic</td><td></td><td></td><td>Steam</td><td></td><td></td></tr><tr><td>Medicare.gov</td><td></td><td></td><td>StubHub</td><td></td><td></td></tr><tr><td>MEGA</td><td></td><td></td><td>Student.com</td><td></td><td></td></tr><tr><td>Megabus</td><td></td><td></td><td>Super Lawyers</td><td></td><td></td></tr><tr><td>Merriam-Webster</td><td></td><td></td><td>Target</td><td></td><td></td></tr><tr><td>Michaels</td><td></td><td></td><td>TED</td><td></td><td></td></tr><tr><td>Micro Center</td><td></td><td></td><td>The Weather Network</td><td></td><td></td></tr><tr><td>Microsoft Azure</td><td></td><td></td><td>Thumbtack</td><td></td><td></td></tr><tr><td>MTA</td><td></td><td></td><td>Ticketmaster</td><td></td><td></td></tr><tr><td>NBA</td><td></td><td></td><td>TourRadar</td><td></td><td></td></tr><tr><td>NFL</td><td></td><td></td><td>Trader Joe&#x27;s</td><td></td><td></td></tr><tr><td>NVIDIA</td><td></td><td></td><td>Trip.com</td><td></td><td></td></tr><tr><td>NYSE</td><td></td><td></td><td>Tumblr</td><td></td><td></td></tr><tr><td>Ohio State University</td><td></td><td></td><td>U.S. DOJ</td><td></td><td></td></tr><tr><td>Ohio.gov</td><td></td><td></td><td>UC Berkeley</td><td></td><td></td></tr><tr><td>OhioMeansJobs</td><td></td><td></td><td>Uniqlo</td><td></td><td></td></tr><tr><td>Parkers</td><td></td><td></td><td>United Airlines</td><td></td><td></td></tr><tr><td>Petfinder</td><td></td><td></td><td>University of Michigan</td><td></td><td></td></tr><tr><td>PhET Interactive Simulations</td><td></td><td></td><td>UPS</td><td></td><td></td></tr><tr><td>Phys.org</td><td></td><td></td><td>US Appliance</td><td></td><td></td></tr><tr><td>Porsche</td><td></td><td></td><td>USCIS.gov</td><td></td><td></td></tr><tr><td>Public Storage</td><td></td><td></td><td>USPS.com</td><td></td><td></td></tr><tr><td>Qatar Airways</td><td></td><td></td><td>Verizon</td><td></td><td></td></tr><tr><td>Raising Cane&#x27;s</td><td></td><td></td><td>Versus</td><td></td><td></td></tr><tr><td>Recreation.gov</td><td></td><td></td><td>Virginia DMV</td><td></td><td></td></tr><tr><td>Red Bull</td><td></td><td></td><td>Walmart Careers</td><td></td><td></td></tr><tr><td>REMAX</td><td></td><td></td><td>Wanderlog</td><td></td><td></td></tr><tr><td>RottenTomatoes</td><td></td><td></td><td>Weather</td><td></td><td></td></tr><tr><td>Ryanair</td><td></td><td></td><td>WebMD</td><td></td><td></td></tr><tr><td>Samsung</td><td></td><td></td><td>WebMD Doctor</td><td></td><td></td></tr><tr><td>SEC.gov</td><td></td><td></td><td>Wine Access</td><td></td><td></td></tr><tr><td>SmartAsset</td><td></td><td></td><td>Wolfram Alpha</td><td></td><td></td></tr><tr><td>SoundCloud</td><td></td><td></td><td>Y Combinator</td><td></td><td></td></tr><tr><td>SourceForge</td><td></td><td></td><td>Yahoo Finance</td><td></td><td></td></tr><tr><td>Speedo</td><td></td><td></td><td>YouTube</td><td></td><td></td></tr><tr><td>SpotHero</td><td></td><td></td><td>Zara</td><td></td><td></td></tr><tr><td>Stanford University</td><td></td><td></td><td>ZipRecruiter</td><td></td><td></td></tr></table>

Table 20: Provenance of the 153 WebHarbor environments (2/2, continued from Table 19).

## E RL TRAINING IMPLEMENTATION DETAILS

We use the established OpenWebRL framework (Yang et al., 2026) to evaluate WebHarbor as an RL training substrate. The released live-web-trained reference policy and our mirror-trained policy use the same OpenWebRL-4B-SFT initialization, multimodal multi-turn GRPO objective, browser agent interface, and GPT-4.1 reward definition. The WebHarbor adaptation replaces the live task pool and browser backend with WebHarbor tasks executed in deterministic mirror sandboxes. This design isolates the training signal of the mirrors from any change in policy optimization.

## E.1 BROWSER AGENT HARNESS

At each turn, the policy receives the current screenshot, active URL, viewport and tab metadata, textual feedback from the previous action, and the complete reasoning and tool-call history. Only the current screenshot remains in the multimodal context. Textual history preserves earlier observations and action outcomes. Table 21 lists the 13 browser tools shared by live-web and mirror rollouts. The harness may execute several tool calls from one response before returning the next observation.

<table><tr><td>Tool</td><td>Arguments and function</td></tr><tr><td>Pointer click</td><td>point_2d, clicks, button. Click a viewport point with the chosen mouse button.</td></tr><tr><td>hover drag</td><td>point_2d. Move the pointer to reveal hover states, tooltips, or menus. start_point_2d, end_point_2d, steps. Drag between two viewport coordinates.</td></tr><tr><td>Keyboard write</td><td>message. Clear the focused field and enter text.</td></tr><tr><td>press_keys Navigation</td><td>keys, is_hotkey, presses. Execute sequential keys or a keyboard shortcut</td></tr><tr><td>scroll</td><td>direction, amount, point_2d. Scroll the page or an element by a viewport fraction.</td></tr><tr><td>goto_url</td><td>ur1. Navigate the active tab directly to a URL.</td></tr><tr><td>go_back</td><td>No arguments. Return to the previous browser history entry.</td></tr><tr><td>wait</td><td></td></tr><tr><td></td><td>seconds. Pause for page loading or animation.</td></tr><tr><td>Tabs</td><td></td></tr><tr><td></td><td></td></tr><tr><td>new_tab</td><td>No arguments. Open a blank browser tab.</td></tr><tr><td>switch_tab</td><td>tab_index. Activate a tab by its zero-based index.</td></tr><tr><td>close_tab</td><td></td></tr><tr><td></td><td>No arguments. Close the active tab and select a remaining tab.</td></tr><tr><td>Termination</td><td></td></tr><tr><td>done</td><td>response. End the episode and return the final user-facing answer.</td></tr></table>

Table 21: Browser action space shared by live-web and WebHarbor mirror rollouts.

## E.2 TASK VERIFICATION AND AGENT JUDGE

Each episode executes one task in an isolated image for its website. At startup, the image copies the frozen seed into a fresh live database and stores a root-only initial snapshot before the agent begins. The harness records the action–observation trajectory, final answer, termination state, browser state, and timestamped screenshots. Root-owned evidence separately preserves the site-access log and the initial and post-episode database states. The verifier assembles these records after the agent phase so that grading uses the observed interaction and protected environment state.

The reviewer authors a task-specific verifier and a natural-language rubric as two forms of the grading contract. The verifier produces the official binary outcome. Depending on the task, it checks required answer facts, task-relevant visited routes, the relevant final browser state, and the exact database delta between the initial and final snapshots. Mutation checks require the requested additions, updates, or deletions and reject changes to unrelated tables, rows, or fields. Read-only tasks require state preservation. The rubric specifies the success conditions, acceptable semantic variants, required interaction or sequence evidence, and failure cases for semantic review. It is used only during grading and is never shown to the acting agent.

<table><tr><td>Parameter</td><td>WebHarbor RL setting</td></tr><tr><td colspan="2">Initialization and environment</td></tr><tr><td>Initialization</td><td>OpenWebRL/OpenWebRL-4B-SFT</td></tr><tr><td>Training set</td><td>2,288 tasks from 123 WebHarbor websites, with 15 WebVoyager and 15 other websites held out</td></tr><tr><td>Episode environment Viewport</td><td>One isolated WebHarbor mirror sandbox initialized from a frozen seed 1280 × 1000 pixels, device-pixel ratio 1</td></tr><tr><td colspan="2">Rollout and context</td></tr><tr><td>RL iterations</td><td>100</td></tr><tr><td>Effective task groups per iteration</td><td>48 groups retained after dynamic sampling</td></tr><tr><td>Trajectories per task Maximum browser steps</td><td>5</td></tr><tr><td>Screenshots in policy context</td><td>50 Current screenshot only</td></tr><tr><td>History in policy context</td><td>Full prior reasoning, tool calls, and textual environment feedback</td></tr><tr><td>Maximum context length</td><td>32,768 tokens</td></tr><tr><td>Maximum response length</td><td></td></tr><tr><td>Rollout sampling</td><td>1,024 tokens per policy turn Temperature 0.8, top-p 1.0, top-k disabled</td></tr><tr><td colspan="2">MM-GRPO optimization</td></tr><tr><td>Advantage estimator</td><td>Group-relative trajectory-reward advantage, applied to all assistant turns</td></tr><tr><td>Dynamic sampling</td><td>Discard task groups with zero reward variance</td></tr><tr><td>PPO epochs</td><td>2</td></tr><tr><td>Asymmetric clipping</td><td> $\epsilon _ { \mathrm { l o w } } = 0 . 2 0 , \epsilon _ { \mathrm { h i g h } } = 0 . 2 8$ </td></tr><tr><td>KL / entropy coefficients</td><td> $_ { 0 / 0 }$ </td></tr><tr><td>Global optimization batch size</td><td>256 assistant-turn samples</td></tr><tr><td>Optimizer</td><td>Adam,  $\beta = ( 0 . 9 , 0 . 9 8 )$  , weight decay 0.1</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 6 }$  with a constant schedule</td></tr><tr><td>Training and rollout backends</td><td>Megatron (TP2, DP4) and eight single-GPU SGLang engines</td></tr><tr><td colspan="2">Reward</td></tr><tr><td>Optimized reward</td><td>OpenWebRL format gate followed by a binary GPT-4.1 success judgment</td></tr><tr><td>Judge evidence</td><td>Final answer, complete action history, and three recent screenshots</td></tr><tr><td>Task-specific WebHarbor verifier</td><td>Logged as a diagnostic metric only and excluded from the policy objective</td></tr></table>

Table 22: RL training configuration of WebHarbor-4B, reusing the OpenWebRL MM-GRPO recipe and browser interface.

The deterministic verifier runs first. The Agent Judge then runs in the same sandbox before it is released and audits the completed trial. It receives the task instruction, reviewer-authored rubric, complete normalized trajectory, final answer, and screenshots. It may also inspect the verifier code and recorded verdict, pristine seed database, live final database, and site code. It returns agent\_success, agent\_failed, or task\_error, together with a confidence score, a concise explanation, verified hard facts, and a verifier-disagreement flag. The task\_error label is reserved for a concrete environment defect that makes the task impossible, never for an agent mistake. The Judge verdict is an audit signal and cannot alter the verifier result. A missing or malformed Judge output is recorded as a Judge error while the verifier result remains unchanged. The two signals therefore cover different requirements: the verifier provides a reproducible final-outcome check, while the Agent Judge can assess semantic equivalence and sequence-dependent rubric items from the trajectory. Both remain separate from the optimized OpenWebRL reward described below.

## E.3 TRAINING CONFIGURATION

Training starts from the released OpenWebRL SFT checkpoint and applies MM-GRPO to groups of five trajectories per task. A trajectory-level reward is converted to a group-relative advantage for every assistant turn, and groups with identical rewards are removed by dynamic sampling. The reward is OpenWebRL’s format gate followed by a GPT-4.1 success judgment over the final answer, action history, and recent screenshots. The task-specific WebHarbor verifier serves only as a diagnostic metric. Table 22 lists the full rollout, optimization, and reward settings.